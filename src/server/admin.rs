use super::{
    auth_error_response, auth_source_key, authenticate_request, error_response, record_audit,
    record_swap_execution, request_id_from_headers, with_request_id, ServerState,
};
use crate::config::Config;
use crate::resources::{budget_plan, snapshot};
use crate::worker::{PlannedWorker, StartupPlan, SwapBudget, SwapMode, WorkerId};
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub(super) struct AdminSwapRequest {
    active: String,
    replacement: String,
    mode: SwapMode,
}

pub(super) async fn admin_swap(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<AdminSwapRequest>,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    let principal = match authenticate_request(
        &state,
        &headers,
        auth_source_key(&state.cfg, &headers, connect_info),
    ) {
        Ok(principal) => principal,
        Err(err) => {
            return with_request_id(auth_error_response(err), request_id);
        }
    };

    if !principal.has_scope("admin") {
        record_audit(
            &state,
            Some(request_id),
            principal,
            "admin.swap",
            &request.replacement,
            "denied",
            json!({ "reason": "missing admin scope" }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing admin scope".to_string(),
            ),
            request_id,
        );
    }

    let Some(worker_control) = state.worker_control.clone() else {
        record_audit(
            &state,
            Some(request_id),
            principal,
            "admin.swap",
            &request.replacement,
            "failed",
            json!({ "reason": "model swap is not supported in native in-process mode" }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::NOT_FOUND,
                "native_swap_unavailable",
                "model swap is not supported in native in-process mode; restart the server to apply model changes".to_string(),
            ),
            request_id,
        );
    };

    let active = WorkerId::new(request.active);
    let Some(replacement) = replacement_worker(&state.cfg, &request.replacement) else {
        record_audit(
            &state,
            Some(request_id),
            principal,
            "admin.swap",
            &request.replacement,
            "rejected",
            json!({ "reason": "replacement worker is not in startup plan" }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::BAD_REQUEST,
                "unknown_worker",
                "replacement worker is not in startup plan".to_string(),
            ),
            request_id,
        );
    };

    // A hot swap loads the replacement while the active model is still resident.
    // Verify both fit the resource budget before allowing that double-allocation;
    // if they cannot co-reside, `execute_swap_with_budget` downgrades to a cold
    // swap rather than risking OOM. `None` (e.g. model files unavailable for
    // sizing) leaves the requested mode unchanged.
    let budget = swap_budget(&state.cfg, &active, &replacement);
    let execution = {
        let mut supervisor = worker_control.lock().await;
        supervisor
            .execute_swap_with_budget(request.mode, &active, &replacement, budget)
            .await
    };

    record_swap_execution(&state, Some(request_id), principal, &execution).await;
    let status = if execution.success {
        StatusCode::OK
    } else {
        StatusCode::CONFLICT
    };
    with_request_id((status, Json(execution)).into_response(), request_id)
}

fn replacement_worker(cfg: &Config, replacement: &str) -> Option<PlannedWorker> {
    let replacement_id = WorkerId::new(replacement);
    StartupPlan::from_config(cfg)
        .workers
        .into_iter()
        .find(|planned| planned.worker.id == replacement_id)
}

/// Builds the co-residency budget for a hot swap: the active and replacement
/// model footprints against the planned resource budget. Returns `None` when
/// any input can't be sized (e.g. the active worker is not in the plan or a
/// model file is unreadable), in which case the swap runs in its requested mode.
fn swap_budget(cfg: &Config, active: &WorkerId, replacement: &PlannedWorker) -> Option<SwapBudget> {
    let plan = StartupPlan::from_config(cfg);
    let active_planned = plan
        .workers
        .iter()
        .find(|planned| &planned.worker.id == active)?;
    let active_bytes = model_footprint_bytes(&active_planned.worker.model.path)?;
    let replacement_bytes = model_footprint_bytes(&replacement.worker.model.path)?;
    let budget_bytes = swap_resource_budget_bytes(cfg)?;
    Some(SwapBudget {
        active_bytes,
        replacement_bytes,
        budget_bytes,
    })
}

/// Estimates a model's resident footprint as the on-disk size of its weights.
fn model_footprint_bytes(model_path: &std::path::Path) -> Option<u64> {
    std::fs::metadata(model_path)
        .ok()
        .map(|metadata| metadata.len())
        .filter(|len| *len > 0)
}

/// The resource budget both models must fit within to co-reside during a hot
/// swap: the tightest GPU VRAM budget when GPUs are present, otherwise the
/// system-memory budget.
fn swap_resource_budget_bytes(cfg: &Config) -> Option<u64> {
    let plan = budget_plan(&snapshot(&cfg.resources), cfg.resources.budget);
    plan.gpu_budgets
        .iter()
        .map(|gpu| gpu.vram_budget_bytes)
        .min()
        .filter(|budget| *budget > 0)
        .or(Some(plan.memory_budget_bytes))
}
