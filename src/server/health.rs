use super::{active_routed_models, error_response, with_request_id, ServerState};
use crate::config::{is_external_host, Config};
use crate::storage::Storage;
use crate::worker::StartupPlan;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use uuid::Uuid;

pub(super) async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

pub(super) async fn livez() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

pub(super) fn draining_response(state: &ServerState, request_id: Uuid) -> Option<Response> {
    if state.draining.load(Ordering::SeqCst) {
        Some(with_request_id(
            error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_draining",
                "server is draining; retry on another node".to_string(),
            ),
            request_id,
        ))
    } else {
        None
    }
}

pub(super) async fn readyz(State(state): State<Arc<ServerState>>) -> Response {
    let storage_ready = storage_ready(&state.storage).await;
    let draining = state.draining.load(Ordering::SeqCst);
    let active_models = active_routed_models(&state.cfg).len();
    // When an external worker supervisor is present, readiness must reflect
    // ACTUAL live workers, not just configured models: `Some(false)` here means
    // every supervised worker is down, so the node is not ready even though the
    // config still lists models.
    let live_ready = live_worker_readiness(&state).await;
    let ready = storage_ready && !draining && active_models > 0 && live_ready.unwrap_or(true);
    let http_status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        http_status,
        Json(readiness_status_for(
            &state.cfg,
            storage_ready,
            draining,
            live_ready,
        )),
    )
        .into_response()
}

/// Reports whether at least one supervised worker is live and ready. Returns
/// `None` when there is no external worker supervisor (native in-process
/// serving), in which case readiness falls back to configured models.
///
/// Prefers the lock-free admission registry: a worker admits requests only while
/// ready, so "any worker admitting" is exactly "at least one live-ready worker",
/// computed without contending on the supervisor mutex a swap may hold.
async fn live_worker_readiness(state: &ServerState) -> Option<bool> {
    if let Some(registry) = state.worker_admissions.as_ref() {
        let any_ready = registry
            .read()
            .map(|registry| registry.values().any(|admission| admission.is_admitting()))
            .unwrap_or(false);
        return Some(any_ready);
    }
    let worker_control = state.worker_control.as_ref()?;
    let supervisor = worker_control.lock().await;
    Some(supervisor.ready_worker_count() > 0)
}

pub async fn readiness_status(cfg: &Config, storage: &Storage) -> Value {
    readiness_status_for(cfg, storage_ready(storage).await, false, None)
}

pub(super) fn readiness_status_for(
    cfg: &Config,
    storage_ready: bool,
    draining: bool,
    live_ready: Option<bool>,
) -> Value {
    let aliases: Vec<_> = active_routed_models(cfg)
        .into_iter()
        .map(|model| model.alias.as_str())
        .collect();
    let worker_plan = StartupPlan::from_config(cfg);
    let ready = storage_ready && !draining && !aliases.is_empty() && live_ready.unwrap_or(true);
    let status = if ready {
        "ready"
    } else if draining {
        "draining"
    } else if aliases.is_empty() {
        "no_models"
    } else if live_ready == Some(false) {
        "no_ready_workers"
    } else {
        "unavailable"
    };

    json!({
        "status": status,
        "mode": cfg.mode,
        "draining": draining,
        "models": {
            "configured": aliases.len(),
            "aliases": aliases
        },
        "workers": {
            "planned": worker_plan.workers.len(),
            "live_ready": live_ready
        },
        "storage": {
            "ready": storage_ready
        },
        "auth": {
            "required": cfg.security.require_auth
        },
        "external_bind": {
            "enabled": cfg.security.bind_external || is_external_host(&cfg.server.host)
        }
    })
}

async fn storage_ready(storage: &Storage) -> bool {
    sqlx::query_scalar::<_, i64>("SELECT 1")
        .fetch_one(storage.pool())
        .await
        .is_ok()
}
