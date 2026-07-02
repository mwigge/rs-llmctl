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
    let ready = storage_ready && !draining && active_models > 0;
    let http_status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        http_status,
        Json(readiness_status_for(&state.cfg, storage_ready, draining)),
    )
        .into_response()
}

pub async fn readiness_status(cfg: &Config, storage: &Storage) -> Value {
    readiness_status_for(cfg, storage_ready(storage).await, false)
}

pub(super) fn readiness_status_for(cfg: &Config, storage_ready: bool, draining: bool) -> Value {
    let aliases: Vec<_> = active_routed_models(cfg)
        .into_iter()
        .map(|model| model.alias.as_str())
        .collect();
    let worker_plan = StartupPlan::from_config(cfg);
    let ready = storage_ready && !draining && !aliases.is_empty();

    json!({
        "status": if ready { "ready" } else if draining { "draining" } else if aliases.is_empty() { "no_models" } else { "unavailable" },
        "mode": cfg.mode,
        "draining": draining,
        "models": {
            "configured": aliases.len(),
            "aliases": aliases
        },
        "workers": {
            "planned": worker_plan.workers.len()
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
