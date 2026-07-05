#[cfg(test)]
use crate::audit::UsageEvent;
use crate::config::{is_external_host, Config};
#[cfg(test)]
use crate::config::{Mode, ModelConfig};
use crate::native;
#[cfg(test)]
use crate::quota::Principal;
use crate::storage::Storage;
use crate::worker::{StartupPlan, TokioWorkerRunner, WorkerSupervisor};
use axum::body::Body;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
#[cfg(test)]
use axum::http::HeaderMap;
use axum::http::{HeaderValue, Method};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::Router;
#[cfg(test)]
use serde_json::Value;
#[cfg(test)]
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
#[cfg(test)]
use std::net::IpAddr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::trace::{DefaultOnResponse, TraceLayer};
#[cfg(test)]
use uuid::Uuid;

mod accounting;
mod admin;
mod admin_ui;
mod auth;
mod chat;
mod chat_dispatch;
mod chat_json;
mod chat_stream;
mod chat_types;
mod embeddings;
mod genai_span;
mod health;
mod lifecycle;
mod lineage;
mod local;
mod models;
mod native_chat;
mod responses;
mod routing;
mod sse;
mod state;
mod tls;
mod traffic;
mod upstream;
use accounting::{
    audit_reject, audit_reject_response, gen_ai_system_for_provider, record_audit,
    record_quota_decision, record_swap_execution, record_usage, slo_status, UsageRecordInput,
};
#[cfg(test)]
use accounting::{usage_span_attributes, webhook_payload};
use auth::{auth_source_key, authenticate_request, authenticate_with_chat_scope};
#[cfg(test)]
use auth::{authenticate, forwarded_client_ip, is_trusted_proxy};
#[cfg(test)]
use chat::quota_admission_scopes;
use chat_dispatch::{dispatch_chat_request, DispatchFailure, UpstreamRequestContext};
use chat_json::json_upstream;
#[cfg(test)]
use chat_stream::stream_status;
use chat_stream::stream_upstream;
use chat_types::{
    chat_audit_detail, chat_route_audit_detail, gen_ai_params_from_request,
    sanitize_native_chat_message, ChatCompletionRequest, GenAiRequestParams, ToolAuditDetail,
};
use genai_span::emit_gen_ai_inference_span;
use health::draining_response;
pub use health::readiness_status;
#[cfg(test)]
use health::readiness_status_for;
pub use lifecycle::{
    serve, serve_with_shutdown, serve_with_storage, serve_with_storage_and_native_engine,
    serve_with_storage_and_native_engines, serve_with_storage_and_shutdown,
    serve_with_storage_worker_control_and_shutdown, shutdown_signal, spawn_worker_reaper,
};
use lineage::{record_request_lineage_joins, runtime_lineage_from_headers_and_metadata};
#[cfg(test)]
use models::*;
use native_chat::{dispatch_native_chat, token_accounting_label, NativeChatContext};
use responses::{
    auth_error_response, build_response, corpus_header_name, error_response,
    lineage_id_header_name, lineage_ids_header_name, model_count_header_name, model_header_name,
    quota_decision_header_name, request_id_from_headers, request_id_header_name, response_headers,
    upstream_model_header_name, with_chat_metadata, with_model_count, with_request_id,
};
#[cfg(test)]
use routing::ModelRouteError;
use routing::{
    active_routed_models, apply_message_redactions, model_route_error_response,
    resolve_model_route, rewrite_chat_model, routed_models, ResolvedExternalProvider,
    ResolvedModelRoute,
};
use sse::{usage_tokens, SseUsageParser};
use state::ServerState;
pub use state::{NativeEngineRegistry, ServingLimits};
use tls::{build_tls_acceptor, serve_tls};
use traffic::{
    AdmissionController, AdmissionError, AdmissionPermit, AuthFailureLimiter, CircuitBreakers,
};
use upstream::{
    model_upstream_timeout, record_admission_busy_telemetry, record_circuit_breaker_state,
    record_upstream_failure, record_upstream_telemetry, upstream_error_status,
    upstream_request_error,
};

const DEFAULT_MAX_IN_FLIGHT: usize = 128;
#[cfg(test)]
const DEFAULT_UPSTREAM_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_SLO_LATENCY_MS: u64 = 10_000;
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub fn router(cfg: Config, storage: Storage) -> Router {
    let limits = ServingLimits::from_config(&cfg);
    router_with_serving_limits(cfg, storage, limits)
}

pub fn router_with_serving_limits(
    cfg: Config,
    storage: Storage,
    serving_limits: ServingLimits,
) -> Router {
    router_with_worker_control_and_native_engines(
        cfg,
        storage,
        serving_limits,
        None,
        NativeEngineRegistry::new(),
    )
}

pub fn router_with_native_engine(
    cfg: Config,
    storage: Storage,
    native_engine: Arc<dyn native::NativeEngine>,
) -> Router {
    let limits = ServingLimits::from_config(&cfg);
    let mut native_engines = NativeEngineRegistry::new();
    native_engines.insert(native_engine.model_alias().to_string(), native_engine);
    router_with_worker_control_and_native_engines(cfg, storage, limits, None, native_engines)
}

pub fn router_with_native_engines(
    cfg: Config,
    storage: Storage,
    native_engines: NativeEngineRegistry,
) -> Router {
    let limits = ServingLimits::from_config(&cfg);
    router_with_worker_control_and_native_engines(cfg, storage, limits, None, native_engines)
}

pub fn router_with_serving_limits_and_native_engines(
    cfg: Config,
    storage: Storage,
    serving_limits: ServingLimits,
    native_engines: NativeEngineRegistry,
) -> Router {
    router_with_worker_control_and_native_engines(
        cfg,
        storage,
        serving_limits,
        None,
        native_engines,
    )
}

pub fn router_with_worker_control(
    cfg: Config,
    storage: Storage,
    serving_limits: ServingLimits,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
) -> Router {
    router_with_worker_control_and_native_engines(
        cfg,
        storage,
        serving_limits,
        worker_control,
        NativeEngineRegistry::new(),
    )
}

fn router_with_worker_control_and_native_engines(
    cfg: Config,
    storage: Storage,
    serving_limits: ServingLimits,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    native_engines: NativeEngineRegistry,
) -> Router {
    router_with_worker_control_native_engine_and_drain(
        cfg,
        storage,
        serving_limits,
        worker_control,
        None,
        native_engines,
        Arc::new(AtomicBool::new(false)),
    )
}

#[allow(clippy::too_many_arguments)]
fn router_with_worker_control_native_engine_and_drain(
    cfg: Config,
    storage: Storage,
    serving_limits: ServingLimits,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    worker_admissions: Option<crate::worker::WorkerAdmissionRegistry>,
    native_engines: NativeEngineRegistry,
    draining: Arc<AtomicBool>,
) -> Router {
    let upstreams = serving_upstreams(&cfg);
    let native_engines = scheduled_native_engines(native_engines, &cfg.runtime.scheduler);
    let admission = AdmissionController::new(serving_limits.max_in_flight);
    let client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(serving_limits.max_in_flight.min(128))
        .build()
        .expect("reqwest client configuration is valid");
    let cors = cors_layer(&cfg);
    let state = ServerState {
        cfg: Arc::new(cfg),
        storage,
        started_at: Instant::now(),
        client,
        upstreams,
        admission,
        serving_limits,
        native_engines,
        worker_control,
        worker_admissions,
        draining: draining.clone(),
        circuit_breakers: CircuitBreakers::default(),
        auth_failures: AuthFailureLimiter::default(),
    };

    Router::new()
        .route("/playground", get(playground))
        .route("/ui", get(admin_ui::admin_ui))
        .route("/v1/admin/status", get(admin_ui::admin_status))
        .route("/v1/admin/models", get(admin_ui::admin_models))
        .route("/v1/admin/quotas", get(admin_ui::admin_quotas))
        .route("/v1/admin/usage", get(admin_ui::admin_usage))
        .route("/v1/admin/audit", get(admin_ui::admin_audit))
        .route("/v1/admin/keys", get(admin_ui::admin_keys))
        .route("/healthz", get(health::healthz))
        .route("/livez", get(health::livez))
        .route("/readyz", get(health::readyz))
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/embeddings", post(embeddings::proxy_embeddings))
        .route("/v1/local/search", post(local::local_search))
        .route(
            "/v1/local/recommendations",
            post(local::local_recommendations),
        )
        .route("/v1/admin/swap", post(admin::admin_swap))
        .layer(cors)
        .layer(trace_layer())
        .with_state(Arc::new(state))
}

fn scheduled_native_engines(
    native_engines: NativeEngineRegistry,
    scheduler: &crate::config::NativeSchedulerRuntimeConfig,
) -> NativeEngineRegistry {
    native_engines
        .into_iter()
        .map(|(alias, engine)| {
            let scheduled: Arc<dyn native::NativeEngine> =
                Arc::new(native::NativeSchedulerEngine::new(
                    engine,
                    native::NativeSchedulerConfig {
                        max_concurrent_requests: scheduler.max_concurrent_requests,
                        max_queued_requests: scheduler.max_queued_requests,
                        max_batch_size: scheduler.max_batch_size,
                        max_batch_wait_ms: scheduler.max_batch_wait_ms,
                        kv_cache_budget_bytes: scheduler.kv_cache_budget_bytes,
                    },
                ));
            (alias, scheduled)
        })
        .collect()
}

fn trace_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    impl Fn(&axum::http::Request<Body>) -> tracing::Span + Clone,
> {
    TraceLayer::new_for_http()
        .make_span_with(|request: &axum::http::Request<Body>| {
            let request_id = request
                .headers()
                .get(request_id_header_name())
                .and_then(|value| value.to_str().ok())
                .unwrap_or("generated");
            tracing::info_span!(
                "http.request",
                http.method = %request.method(),
                http.route = %request.uri().path(),
                llmctl.request_id = %request_id
            )
        })
        .on_response(DefaultOnResponse::new())
}

fn cors_layer(cfg: &Config) -> CorsLayer {
    let mut layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([
            AUTHORIZATION,
            CONTENT_TYPE,
            request_id_header_name(),
            lineage_id_header_name(),
            lineage_ids_header_name(),
            corpus_header_name(),
        ])
        .expose_headers([
            request_id_header_name(),
            model_count_header_name(),
            model_header_name(),
            upstream_model_header_name(),
            quota_decision_header_name(),
        ]);

    if cfg.security.production || cfg.security.bind_external || is_external_host(&cfg.server.host) {
        let origins = cfg
            .server
            .cors_allowed_origins
            .iter()
            .filter_map(|origin| HeaderValue::from_str(origin).ok())
            .collect::<Vec<_>>();
        if !origins.is_empty() {
            layer = layer.allow_origin(AllowOrigin::list(origins));
        }
        layer
    } else {
        // Dev default (not production, not external bind). Restrict CORS to
        // loopback origins rather than `Any`: this layer allows the
        // `AUTHORIZATION` header, and when `require_auth = false` anonymous
        // `chat` + `models.read` are served. Combining `Any` origin with an
        // authorized/credentialed local endpoint would let ANY website loaded
        // in the operator's browser silently drive this local server. Loopback
        // origins keep the bundled playground working without that exposure.
        layer.allow_origin(AllowOrigin::predicate(|origin, _request_parts| {
            is_loopback_origin(origin)
        }))
    }
}

/// True when a CORS `Origin` header names a loopback host (localhost, IPv4
/// `127.0.0.1`, or IPv6 `::1`), regardless of scheme or port. Used to scope the
/// dev-default CORS policy to the operator's own machine.
fn is_loopback_origin(origin: &HeaderValue) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    let Some((_scheme, rest)) = origin.split_once("://") else {
        return false;
    };
    let host = if let Some(after_bracket) = rest.strip_prefix('[') {
        // IPv6 literal, e.g. http://[::1]:8080 — host is inside the brackets.
        match after_bracket.split_once(']') {
            Some((host, _)) => host,
            None => return false,
        }
    } else {
        rest.split([':', '/']).next().unwrap_or("")
    };
    host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1"
}

/// Static HTML+JS chat page for exercising the OpenAI-compatible endpoints
/// directly from a browser — model picker against `/v1/models`, chat against
/// `/v1/chat/completions`, with a user-supplied bearer API key. No build step,
/// no framework: served verbatim from `assets/playground.html`.
fn playground_html() -> &'static str {
    include_str!("../assets/playground.html")
}

async fn playground() -> impl IntoResponse {
    Html(playground_html())
}

#[cfg(test)]
fn normalize_upstream(raw: &str) -> String {
    let raw = raw.trim().trim_end_matches('/');
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    }
}

fn serving_upstreams(cfg: &Config) -> BTreeMap<String, String> {
    StartupPlan::from_config(cfg)
        .workers
        .into_iter()
        .map(|planned| {
            (
                planned.worker.model.alias.clone(),
                planned.worker.upstream(),
            )
        })
        .collect::<BTreeMap<_, _>>()
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
