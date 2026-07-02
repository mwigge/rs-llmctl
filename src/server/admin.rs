use super::{
    auth_error_response, auth_source_key, authenticate_request, error_response, record_audit,
    record_swap_execution, request_id_from_headers, with_request_id, ServerState,
};
use crate::config::Config;
use crate::worker::{PlannedWorker, StartupPlan, SwapMode, WorkerId};
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
            json!({ "reason": "native in-process runtime does not expose external worker swap" }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::NOT_FOUND,
                "native_swap_unavailable",
                "native Candle serving uses model start/stop/upgrade commands; external worker swap is not available".to_string(),
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

    let execution = {
        let mut supervisor = worker_control.lock().await;
        supervisor
            .execute_swap(request.mode, &active, &replacement)
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
