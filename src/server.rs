use crate::audit::{AuditEvent, UsageEvent};
use crate::config::{
    is_external_host, Config, ExternalProviderKind, Mode, ModelConfig, NativeEmbeddingMode,
};
use crate::guardrails;
use crate::native;
use crate::observability::{
    emit_runtime_telemetry, inject_trace_context, RuntimeTelemetryEvent, TelemetryEventName,
    TelemetrySignal,
};
use crate::quota::{
    check_quota, matching_quota_policies, quota_admission_scope, quota_is_subject_scoped, Principal,
};
use crate::storage::{QuotaDecisionRecord, RequestLineageJoinRecord, Storage};
use crate::worker::{
    PlannedWorker, StartupPlan, SwapExecution, SwapMode, TokioWorkerRunner, WorkerId,
    WorkerSupervisor,
};
use anyhow::{Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::ConnectInfo;
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use futures_util::{pin_mut, FutureExt, StreamExt};
use hyper::body::Incoming;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as HyperBuilder;
use hyper_util::service::TowerToHyperService;
use opentelemetry::global;
use opentelemetry::trace::{Span, SpanKind, Tracer};
use opentelemetry::KeyValue;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::future::Future;
#[cfg(test)]
use std::net::IpAddr;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_rustls::TlsAcceptor;
use tower::ServiceExt;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::{DefaultOnResponse, TraceLayer};
use uuid::Uuid;

mod auth;
mod local;
mod models;
mod sse;
mod traffic;
use auth::{auth_source_key, authenticate_request, authenticate_with_chat_scope};
#[cfg(test)]
use auth::{authenticate, forwarded_client_ip, is_trusted_proxy};
use models::*;
use sse::{usage_tokens, SseUsageParser};
use traffic::{
    AdmissionController, AdmissionError, AdmissionPermit, AuthFailureLimiter, CircuitBreakers,
};

const DEFAULT_MAX_IN_FLIGHT: usize = 128;
#[cfg(test)]
const DEFAULT_UPSTREAM_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_SLO_LATENCY_MS: u64 = 10_000;
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub type NativeEngineRegistry = BTreeMap<String, Arc<dyn native::NativeEngine>>;

#[derive(Clone)]
pub struct ServerState {
    cfg: Arc<Config>,
    storage: Storage,
    client: reqwest::Client,
    upstreams: BTreeMap<String, String>,
    admission: AdmissionController,
    serving_limits: ServingLimits,
    native_engines: NativeEngineRegistry,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    draining: Arc<AtomicBool>,
    circuit_breakers: CircuitBreakers,
    auth_failures: AuthFailureLimiter,
}

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
        native_engines,
        Arc::new(AtomicBool::new(false)),
    )
}

fn router_with_worker_control_native_engine_and_drain(
    cfg: Config,
    storage: Storage,
    serving_limits: ServingLimits,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
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
        client,
        upstreams,
        admission,
        serving_limits,
        native_engines,
        worker_control,
        draining: draining.clone(),
        circuit_breakers: CircuitBreakers::default(),
        auth_failures: AuthFailureLimiter::default(),
    };

    Router::new()
        .route("/playground", get(playground))
        .route("/healthz", get(healthz))
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(proxy_embeddings))
        .route("/v1/local/search", post(local::local_search))
        .route(
            "/v1/local/recommendations",
            post(local::local_recommendations),
        )
        .route("/v1/admin/swap", post(admin_swap))
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
        layer.allow_origin(Any)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ServingLimits {
    max_in_flight: usize,
    upstream_timeout: Duration,
}

impl ServingLimits {
    pub fn new(max_in_flight: usize, upstream_timeout: Duration) -> Self {
        Self {
            max_in_flight: max_in_flight.max(1),
            upstream_timeout: upstream_timeout.max(Duration::from_millis(1)),
        }
    }

    fn from_config(cfg: &Config) -> Self {
        let configured_max = cfg
            .quotas
            .iter()
            .filter_map(|quota| usize::try_from(quota.max_concurrency).ok())
            .filter(|limit| *limit > 0)
            .fold(0usize, usize::saturating_add);
        let max_in_flight = if configured_max > 0 {
            configured_max
        } else {
            DEFAULT_MAX_IN_FLIGHT
        };

        Self::new(
            max_in_flight,
            Duration::from_secs(cfg.server.upstream_timeout_seconds),
        )
    }

    fn upstream_timeout(&self) -> Duration {
        self.upstream_timeout
    }
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

async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn livez() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

fn draining_response(state: &ServerState, request_id: Uuid) -> Option<Response> {
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

async fn readyz(State(state): State<Arc<ServerState>>) -> Response {
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

fn readiness_status_for(cfg: &Config, storage_ready: bool, draining: bool) -> Value {
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

async fn list_models(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Some(response) = draining_response(&state, request_id) {
        return response;
    }
    let principal = match authenticate_request(
        &state,
        &headers,
        auth_source_key(&state.cfg, &headers, connect_info),
    ) {
        Ok(principal) => principal,
        Err(err) => {
            record_audit(
                &state,
                Some(request_id),
                Principal::anonymous(),
                "models.list",
                "models",
                "denied",
                json!({ "reason": err }),
            )
            .await;
            return with_request_id(auth_error_response(err), request_id);
        }
    };

    if !principal.has_scope("models.read") {
        record_audit(
            &state,
            Some(request_id),
            principal,
            "models.list",
            "models",
            "denied",
            json!({ "reason": "missing models.read scope" }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing models.read scope".to_string(),
            ),
            request_id,
        );
    }

    record_audit(
        &state,
        Some(request_id),
        principal,
        "models.list",
        "models",
        "allowed",
        json!({}),
    )
    .await;

    let snap = CapabilitySnapshot::current();
    let response = with_request_id(
        Json(ModelList {
            object: "list",
            data: routed_models(&state.cfg)
                .into_iter()
                .map(|m| build_model_object(m, snap))
                .collect(),
        })
        .into_response(),
        request_id,
    );
    with_model_count(response, state.cfg.models.len())
}

#[derive(Debug, Deserialize)]
struct AdminSwapRequest {
    active: String,
    replacement: String,
    mode: SwapMode,
}

async fn admin_swap(
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

async fn proxy_embeddings(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    native_embeddings(state, connect_info, headers, body).await
}

#[derive(Debug, Deserialize)]
struct EmbeddingRequest {
    model: String,
    input: EmbeddingInput,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    String(String),
    StringArray(Vec<String>),
}

impl EmbeddingInput {
    fn into_strings(self) -> Vec<String> {
        match self {
            Self::String(input) => vec![input],
            Self::StringArray(input) => input,
        }
    }
}

async fn native_embeddings(
    state: Arc<ServerState>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Some(response) = draining_response(&state, request_id) {
        return response;
    }
    let started = Instant::now();
    let principal = match authenticate_with_chat_scope(
        &state,
        &headers,
        connect_info,
        request_id,
        "embeddings",
    )
    .await
    {
        Ok(principal) => principal,
        Err(response) => return response,
    };

    let request: EmbeddingRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            return audit_reject(
                &state,
                request_id,
                principal,
                "embeddings",
                "unknown",
                "rejected",
                StatusCode::BAD_REQUEST,
                "bad_request",
                "request body must be valid JSON".to_string(),
                json!({ "reason": err.to_string() }),
            )
            .await;
        }
    };

    if request
        .encoding_format
        .as_deref()
        .is_some_and(|format| format != "float")
    {
        return audit_reject(
            &state,
            request_id,
            principal,
            "embeddings",
            request.model,
            "rejected",
            StatusCode::BAD_REQUEST,
            "unsupported_encoding_format",
            "native embeddings support only float encoding_format".to_string(),
            json!({ "reason": "native embeddings support only float encoding_format" }),
        )
        .await;
    }

    let route = match resolve_model_route(&state.cfg, &request.model, request_id) {
        Ok(route) => route,
        Err(err) => {
            let response = model_route_error_response(&err);
            return audit_reject_response(
                &state,
                request_id,
                principal,
                "embeddings",
                request.model,
                "rejected",
                response,
                json!({ "reason": err.to_string() }),
            )
            .await;
        }
    };

    let embedding_selection = match native_embedding_selection(&state.cfg, &route) {
        Ok(selection) => selection,
        Err(err) => {
            return audit_reject(
                &state,
                request_id,
                principal,
                "embeddings",
                route.requested_alias,
                "rejected",
                StatusCode::SERVICE_UNAVAILABLE,
                "native_embedding_model_unavailable",
                err.clone(),
                json!({ "reason": err }),
            )
            .await;
        }
    };

    let metadata = native_embedding_metadata(
        request.metadata,
        request_id,
        &route,
        embedding_selection.mode,
        &embedding_selection.model_alias,
    );
    let native_request = native::NativeEmbeddingRequest {
        model: embedding_selection.model_alias.clone(),
        input: request.input.into_strings(),
        metadata,
    };
    let native_response = match embedding_selection.mode {
        NativeEmbeddingMode::Semantic => {
            let Some(engine) = state
                .native_engines
                .get(&embedding_selection.model_alias)
                .cloned()
            else {
                return audit_reject(
                    &state,
                    request_id,
                    principal,
                    "embeddings",
                    route.requested_alias,
                    "error",
                    StatusCode::SERVICE_UNAVAILABLE,
                    "native_embedding_model_unavailable",
                    "semantic native embedding model is not loaded".to_string(),
                    json!({
                        "reason": "native_embedding_model_unavailable",
                        "runtime_backend": "candle-native",
                        "embedding_mode": embedding_selection.mode.as_str(),
                        "embedding_model_alias": embedding_selection.model_alias
                    }),
                )
                .await;
            };

            match engine.embeddings(native_request).await {
                Ok(response) => response,
                Err(err) => {
                    tracing::warn!(error = %err, "native embedding runtime failed");
                    return audit_reject(
                        &state,
                        request_id,
                        principal,
                        "embeddings",
                        route.requested_alias,
                        "error",
                        StatusCode::SERVICE_UNAVAILABLE,
                        "native_embedding_runtime_error",
                        "native runtime failed to serve semantic embeddings".to_string(),
                        json!({
                            "reason": "native_embedding_runtime_error",
                            "runtime_backend": "candle-native",
                            "embedding_mode": embedding_selection.mode.as_str(),
                            "embedding_model_alias": embedding_selection.model_alias
                        }),
                    )
                    .await;
                }
            }
        }
        NativeEmbeddingMode::DevFallback => {
            match native::deterministic_native_embeddings(native_request) {
                Ok(response) => response,
                Err(err) => {
                    tracing::warn!(error = %err, "native embedding fallback failed");
                    return with_request_id(
                        error_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "native_embedding_error",
                            "native embedding fallback failed".to_string(),
                        ),
                        request_id,
                    );
                }
            }
        }
    };

    native_embedding_response(
        &state,
        NativeEmbeddingResponseInput {
            request_id,
            principal,
            route,
            embedding_mode: embedding_selection.mode,
            embedding_model_alias: embedding_selection.model_alias,
            started,
            native_response,
        },
    )
    .await
}

fn native_embedding_metadata(
    metadata: Option<Value>,
    request_id: Uuid,
    route: &ResolvedModelRoute,
    mode: NativeEmbeddingMode,
    model_alias: &str,
) -> BTreeMap<String, Value> {
    let mut metadata: BTreeMap<String, Value> = metadata
        .and_then(|value| match value {
            Value::Object(object) => Some(object.into_iter().collect()),
            _ => None,
        })
        .unwrap_or_default();
    metadata.insert(
        "llmctl.request_id".to_string(),
        Value::String(request_id.to_string()),
    );
    metadata.insert(
        "llmctl.requested_model".to_string(),
        Value::String(route.requested_alias.clone()),
    );
    metadata.insert(
        "llmctl.upstream_model".to_string(),
        Value::String(route.upstream_alias.clone()),
    );
    metadata.insert(
        "llmctl.embedding_mode".to_string(),
        Value::String(mode.as_str().to_string()),
    );
    metadata.insert(
        "llmctl.embedding_model_alias".to_string(),
        Value::String(model_alias.to_string()),
    );
    metadata
}

#[derive(Debug, Clone)]
struct NativeEmbeddingSelection {
    mode: NativeEmbeddingMode,
    model_alias: String,
}

struct NativeEmbeddingResponseInput {
    request_id: Uuid,
    principal: Principal,
    route: ResolvedModelRoute,
    embedding_mode: NativeEmbeddingMode,
    embedding_model_alias: String,
    started: Instant,
    native_response: native::NativeEmbeddingResponse,
}

fn native_embedding_selection(
    cfg: &Config,
    route: &ResolvedModelRoute,
) -> std::result::Result<NativeEmbeddingSelection, String> {
    let mode = cfg.runtime.embeddings.mode;
    let model_alias = cfg
        .runtime
        .embeddings
        .model_alias
        .clone()
        .unwrap_or_else(|| route.requested_alias.clone());
    if cfg.models.iter().any(|model| model.alias == model_alias) {
        Ok(NativeEmbeddingSelection { mode, model_alias })
    } else {
        Err(format!(
            "native embedding model alias '{model_alias}' is not configured"
        ))
    }
}

async fn native_embedding_response(
    state: &ServerState,
    input: NativeEmbeddingResponseInput,
) -> Response {
    let NativeEmbeddingResponseInput {
        request_id,
        principal,
        route,
        embedding_mode,
        embedding_model_alias,
        started,
        native_response,
    } = input;
    let embedding_count = native_response.embeddings.len();
    let embedding_dimensions = native_response
        .embeddings
        .first()
        .map(Vec::len)
        .unwrap_or(0);
    let usage_status = if native_response.semantic {
        "native_embedding_semantic"
    } else {
        "native_embedding_dev_fallback"
    };
    record_usage(
        state,
        UsageRecordInput {
            request_id,
            principal: &principal,
            model: &route.requested_alias,
            input_tokens: native_response.usage.input_tokens,
            output_tokens: 0,
            latency_ms: elapsed_ms(started),
            status: usage_status,
            accounting_mode: token_accounting_label(&native_response.usage.accounting_mode),
            gen_ai_system: gen_ai_system_for_provider(None),
        },
    )
    .await;
    record_audit(
        state,
        Some(request_id),
        principal,
        "embeddings",
        route.requested_alias.clone(),
        "ok",
        json!({
            "runtime_backend": "candle-native",
            "embedding_backend": native_response.backend.clone(),
            "embedding_status": native_response.status.clone(),
            "embedding_mode": embedding_mode.as_str(),
            "embedding_model_alias": embedding_model_alias.clone(),
            "embedding_count": embedding_count,
            "embedding_dimensions": embedding_dimensions,
            "token_accounting": native_response.usage.accounting_mode.clone(),
            "semantic": native_response.semantic
        }),
    )
    .await;

    let data = native_response
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({
                "object": "embedding",
                "embedding": embedding,
                "index": index
            })
        })
        .collect::<Vec<_>>();
    let body = Json(json!({
        "object": "list",
        "model": native_response.model,
        "data": data,
        "usage": {
            "prompt_tokens": native_response.usage.input_tokens,
            "total_tokens": native_response.usage.total_tokens()
        },
        "llmctl": {
            "embedding_backend": native_response.backend,
            "embedding_status": native_response.status,
            "embedding_mode": embedding_mode.as_str(),
            "embedding_model_alias": embedding_model_alias,
            "embedding_dimensions": embedding_dimensions,
            "semantic": native_response.semantic,
            "token_accounting": native_response.usage.accounting_mode
        }
    }))
    .into_response();
    with_chat_metadata(
        with_request_id(body, request_id),
        &route.requested_alias,
        &route.upstream_alias,
        "allowed",
    )
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Some(response) = draining_response(&state, request_id) {
        return response;
    }
    let started = Instant::now();
    let principal = match authenticate_with_chat_scope(
        &state,
        &headers,
        connect_info,
        request_id,
        "chat.completions",
    )
    .await
    {
        Ok(principal) => principal,
        Err(response) => return response,
    };

    let mut request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            return audit_reject(
                &state,
                request_id,
                principal,
                "chat.completions",
                "unknown",
                "rejected",
                StatusCode::BAD_REQUEST,
                "bad_request",
                "request body must be valid JSON".to_string(),
                json!({ "reason": err.to_string() }),
            )
            .await;
        }
    };
    let lineage = runtime_lineage_from_headers_and_metadata(&headers, request.metadata.as_ref());

    let route = match resolve_model_route(&state.cfg, &request.model, request_id) {
        Ok(route) => route,
        Err(err) => {
            let response = model_route_error_response(&err);
            return audit_reject_response(
                &state,
                request_id,
                principal,
                "chat.completions",
                request.model,
                "rejected",
                response,
                json!({ "reason": err.to_string() }),
            )
            .await;
        }
    };
    let model = route.requested_alias.clone();

    if state.cfg.guardrails.is_active() {
        let message_texts: Vec<(usize, String)> = request
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| (index, native::message_content_text(message)))
            .collect();
        let verdict = guardrails::evaluate(&message_texts, &state.cfg.guardrails);

        if verdict.is_blocked() {
            let message = format!(
                "request blocked by guardrails: {}",
                verdict.block_reasons.join(", ")
            );
            return audit_reject(
                &state,
                request_id,
                principal,
                "chat.completions",
                model,
                "denied",
                StatusCode::BAD_REQUEST,
                "guardrail_blocked",
                message,
                json!({
                    "reason": "guardrail_violation",
                    "guardrails": verdict.block_reasons,
                    "findings": verdict.findings.audit_detail(),
                }),
            )
            .await;
        }

        if verdict.has_findings() {
            record_audit(
                &state,
                Some(request_id),
                principal.clone(),
                "chat.completions",
                model.clone(),
                "flagged",
                json!({
                    "reason": "guardrail_match",
                    "findings": verdict.findings.audit_detail(),
                    "redacted": !verdict.redactions.is_empty(),
                }),
            )
            .await;
        }

        for (index, redacted_text) in verdict.redactions {
            if let Some(message) = request.messages.get_mut(index) {
                message.content = Some(Value::String(redacted_text));
            }
        }
    }

    record_request_lineage_joins(
        &state,
        request_id,
        &lineage,
        Some(model.as_str()),
        "chat.completions",
    )
    .await;
    let admission_scope = quota_admission_scope(&principal);
    let quota = match state
        .storage
        .with_quota_admission(&admission_scope, || async {
            check_quota(&state.storage, &state.cfg.quotas, &principal, &model).await
        })
        .await
    {
        Ok(decision) => decision,
        Err(err) => {
            return audit_reject(
                &state,
                request_id,
                principal,
                "chat.completions",
                model,
                "rejected",
                StatusCode::INTERNAL_SERVER_ERROR,
                "quota_error",
                "quota admission is unavailable".to_string(),
                json!({ "reason": err.to_string() }),
            )
            .await;
        }
    };
    record_quota_decision(
        &state,
        Some(request_id),
        &principal,
        &model,
        &quota,
        json!({ "configured_quotas": state.cfg.quotas.len() }),
    )
    .await;

    if !quota.allowed {
        let reason = quota.reason.clone();
        return audit_reject(
            &state,
            request_id,
            principal,
            "chat.completions",
            model,
            "denied",
            StatusCode::TOO_MANY_REQUESTS,
            "quota_exceeded",
            quota.reason,
            json!({ "reason": reason }),
        )
        .await;
    }

    record_audit(
        &state,
        Some(request_id),
        principal.clone(),
        "chat.completions",
        model.clone(),
        "allowed",
        chat_route_audit_detail(
            &request.tool_audit_detail(),
            json!({ "stream": request.stream, "upstream_model": route.upstream_alias }),
            route.external_provider.as_ref(),
        ),
    )
    .await;

    let admission = match state
        .admission
        .try_acquire_for_all(quota_admission_scopes(&state.cfg, &principal))
    {
        Ok(permit) => permit,
        Err(AdmissionError::Busy) => {
            record_admission_busy_telemetry(&model, &principal, request.stream);
            return audit_reject(
                &state,
                request_id,
                principal,
                "chat.completions",
                model,
                "denied",
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
                "server is busy; retry later".to_string(),
                json!({ "reason": "admission_limit_exceeded" }),
            )
            .await;
        }
    };

    let has_subprocess_upstream =
        state.upstreams.contains_key(&route.upstream_alias) || state.upstreams.contains_key("*");

    if route.external_provider.is_some() || has_subprocess_upstream {
        let tool_audit = request.tool_audit_detail();
        let gen_ai = gen_ai_params_from_request(&request);
        let upstream_timeout = model_upstream_timeout(&state, &model);
        return match dispatch_chat_request(
            &state, &route, &body, request_id, &principal, &model, started,
        )
        .await
        {
            Ok((upstream_response, upstream_model)) if request.stream => {
                stream_upstream(
                    state,
                    upstream_response,
                    UpstreamRequestContext {
                        request_id,
                        principal,
                        model,
                        upstream_model,
                        upstream_timeout,
                        external_provider: route.external_provider,
                        tool_audit,
                        started,
                        admission,
                        gen_ai,
                    },
                )
                .await
            }
            Ok((upstream_response, upstream_model)) => {
                json_upstream(
                    state,
                    upstream_response,
                    UpstreamRequestContext {
                        request_id,
                        principal,
                        model,
                        upstream_model,
                        upstream_timeout,
                        external_provider: route.external_provider,
                        tool_audit,
                        started,
                        admission,
                        gen_ai,
                    },
                )
                .await
            }
            Err(DispatchFailure::BadRequest(err)) => with_request_id(
                error_response(StatusCode::BAD_REQUEST, "bad_request", err),
                request_id,
            ),
            Err(DispatchFailure::NoUpstream(message)) => with_request_id(
                error_response(StatusCode::BAD_GATEWAY, "upstream_unavailable", message),
                request_id,
            ),
            Err(DispatchFailure::Request {
                status,
                code,
                message,
                usage_status,
            }) => {
                record_usage(
                    &state,
                    UsageRecordInput {
                        request_id,
                        principal: &principal,
                        model: &model,
                        input_tokens: 0,
                        output_tokens: 0,
                        latency_ms: elapsed_ms(started),
                        status: usage_status,
                        accounting_mode: "none",
                        gen_ai_system: gen_ai_system_for_provider(None),
                    },
                )
                .await;
                with_request_id(error_response(status, code, message), request_id)
            }
        };
    }

    dispatch_native_chat(
        state,
        NativeChatContext {
            request_id,
            principal,
            model,
            upstream_model: route.upstream_alias,
            tool_audit: request.tool_audit_detail(),
            started,
            _admission: admission,
        },
        request,
    )
    .await
}

async fn dispatch_native_chat(
    state: Arc<ServerState>,
    context: NativeChatContext,
    request: ChatCompletionRequest,
) -> Response {
    let Some(engine) = state.native_engines.get(&context.upstream_model).cloned() else {
        return native_chat_runtime_not_ready(&state, context, request.stream).await;
    };

    let native_request = native::NativeChatRequest {
        model: context.upstream_model.clone(),
        messages: request
            .messages
            .into_iter()
            .map(sanitize_native_chat_message)
            .collect(),
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        tools: request.tools,
        tool_choice: request.tool_choice,
        metadata: native_chat_metadata(request.metadata, &context, request.stream),
    };

    if request.stream {
        let native_response = match engine.chat_stream(native_request).await {
            Ok(response) => response,
            Err(err) => {
                return native_chat_runtime_error(&state, context, err).await;
            }
        };
        return native_chat_stream_response(&state, context, native_response).await;
    }

    let native_response = match engine.chat(native_request).await {
        Ok(response) => response,
        Err(err) => {
            return native_chat_runtime_error(&state, context, err).await;
        }
    };

    native_chat_response(&state, context, native_response).await
}

struct NativeChatContext {
    request_id: Uuid,
    principal: Principal,
    model: String,
    upstream_model: String,
    tool_audit: ToolAuditDetail,
    started: Instant,
    _admission: AdmissionPermit,
}

fn token_accounting_label(mode: &native::TokenAccountingMode) -> &'static str {
    match mode {
        native::TokenAccountingMode::NativeExact => "native-exact",
        native::TokenAccountingMode::Estimated => "estimated",
    }
}

async fn native_chat_response(
    state: &ServerState,
    context: NativeChatContext,
    native_response: native::NativeChatResponse,
) -> Response {
    record_usage(
        state,
        UsageRecordInput {
            request_id: context.request_id,
            principal: &context.principal,
            model: &context.model,
            input_tokens: native_response.usage.input_tokens,
            output_tokens: native_response.usage.output_tokens,
            latency_ms: elapsed_ms(context.started),
            status: "ok",
            accounting_mode: token_accounting_label(&native_response.usage.accounting_mode),
            gen_ai_system: gen_ai_system_for_provider(None),
        },
    )
    .await;
    let detail = chat_audit_detail(
        &context.tool_audit,
        json!({
            "runtime_backend": "candle-native",
            "token_accounting": native_response.usage.accounting_mode.clone()
        }),
    );
    record_audit(
        state,
        Some(context.request_id),
        context.principal,
        "chat.completions",
        context.model.clone(),
        "ok",
        detail,
    )
    .await;

    let mut message = json!({
        "role": "assistant",
        "content": native_response.content
    });
    if let Some(tool_calls) = native_response.tool_calls {
        if let Some(object) = message.as_object_mut() {
            object.insert("tool_calls".to_string(), tool_calls);
        }
    }

    let body = Json(json!({
        "id": format!("chatcmpl-{}", context.request_id),
        "object": "chat.completion",
        "created": Utc::now().timestamp(),
        "model": native_response.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": native_response.finish_reason
        }],
        "usage": {
            "prompt_tokens": native_response.usage.input_tokens,
            "completion_tokens": native_response.usage.output_tokens,
            "total_tokens": native_response.usage.total_tokens()
        }
    }))
    .into_response();
    with_chat_metadata(
        with_request_id(body, context.request_id),
        &context.model,
        &context.upstream_model,
        "allowed",
    )
}

async fn native_chat_stream_response(
    state: &ServerState,
    context: NativeChatContext,
    native_response: native::NativeChatResponse,
) -> Response {
    record_usage(
        state,
        UsageRecordInput {
            request_id: context.request_id,
            principal: &context.principal,
            model: &context.model,
            input_tokens: native_response.usage.input_tokens,
            output_tokens: native_response.usage.output_tokens,
            latency_ms: elapsed_ms(context.started),
            status: "ok",
            accounting_mode: token_accounting_label(&native_response.usage.accounting_mode),
            gen_ai_system: gen_ai_system_for_provider(None),
        },
    )
    .await;
    let detail = chat_audit_detail(
        &context.tool_audit,
        json!({
            "runtime_backend": "candle-native",
            "stream": true,
            "token_accounting": native_response.usage.accounting_mode.clone()
        }),
    );
    record_audit(
        state,
        Some(context.request_id),
        context.principal,
        "chat.completions",
        context.model.clone(),
        "ok",
        detail,
    )
    .await;

    let request_id = context.request_id;
    let model = native_response.model.clone();
    let content = native_response.content;
    let finish_reason = native_response.finish_reason;
    let stream = async_stream::stream! {
        if !content.is_empty() {
            let chunk = json!({
                "id": format!("chatcmpl-{request_id}"),
                "object": "chat.completion.chunk",
                "created": Utc::now().timestamp(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": { "content": content },
                    "finish_reason": Value::Null
                }]
            });
            yield Ok::<Bytes, std::io::Error>(Bytes::from(format!("data: {chunk}\n\n")));
        }

        let done_chunk = json!({
            "id": format!("chatcmpl-{request_id}"),
            "object": "chat.completion.chunk",
            "created": Utc::now().timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        });
        yield Ok::<Bytes, std::io::Error>(Bytes::from(format!("data: {done_chunk}\n\n")));
        yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n"));
    };

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    let response = build_response(
        StatusCode::OK,
        headers,
        Body::from_stream(stream),
        context.request_id,
    );
    with_chat_metadata(response, &context.model, &context.upstream_model, "allowed")
}

async fn native_chat_runtime_error(
    state: &ServerState,
    context: NativeChatContext,
    err: anyhow::Error,
) -> Response {
    tracing::warn!(error = %err, "native runtime failed");
    let message = err.to_string();
    let queue_full = message.contains("native scheduler queue is full");
    let status_text = if queue_full {
        "native_scheduler_queue_full"
    } else {
        "native_runtime_error"
    };
    let http_status = if queue_full {
        StatusCode::TOO_MANY_REQUESTS
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let error_code = if queue_full {
        "rate_limit_exceeded"
    } else {
        "native_runtime_error"
    };
    let error_message = if queue_full {
        "native scheduler queue is full; retry later"
    } else {
        "native runtime failed to serve chat completions"
    };
    record_usage(
        state,
        UsageRecordInput {
            request_id: context.request_id,
            principal: &context.principal,
            model: &context.model,
            input_tokens: 0,
            output_tokens: 0,
            latency_ms: elapsed_ms(context.started),
            status: status_text,
            accounting_mode: "none",
            gen_ai_system: gen_ai_system_for_provider(None),
        },
    )
    .await;
    record_audit(
        state,
        Some(context.request_id),
        context.principal,
        "chat.completions",
        context.model,
        "error",
        json!({
            "reason": status_text,
            "runtime_backend": "candle-native"
        }),
    )
    .await;
    with_request_id(
        error_response(http_status, error_code, error_message.to_string()),
        context.request_id,
    )
}

async fn native_chat_runtime_not_ready(
    state: &ServerState,
    context: NativeChatContext,
    stream: bool,
) -> Response {
    record_usage(
        state,
        UsageRecordInput {
            request_id: context.request_id,
            principal: &context.principal,
            model: &context.model,
            input_tokens: 0,
            output_tokens: 0,
            latency_ms: elapsed_ms(context.started),
            status: "native_runtime_not_ready",
            accounting_mode: "none",
            gen_ai_system: gen_ai_system_for_provider(None),
        },
    )
    .await;
    record_audit(
        state,
        Some(context.request_id),
        context.principal,
        "chat.completions",
        context.model,
        "error",
        json!({
            "reason": "native_runtime_not_ready",
            "runtime_backend": "candle-native",
            "stream": stream
        }),
    )
    .await;
    with_request_id(
        error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "native_runtime_not_ready",
            "native runtime is not ready to serve chat completions".to_string(),
        ),
        context.request_id,
    )
}

fn native_chat_metadata(
    metadata: Option<Value>,
    context: &NativeChatContext,
    stream: bool,
) -> BTreeMap<String, Value> {
    let mut metadata: BTreeMap<String, Value> = metadata
        .and_then(|value| match value {
            Value::Object(object) => Some(object.into_iter().collect()),
            _ => None,
        })
        .unwrap_or_default();
    metadata.insert(
        "llmctl.request_id".to_string(),
        Value::String(context.request_id.to_string()),
    );
    metadata.insert(
        "llmctl.requested_model".to_string(),
        Value::String(context.model.clone()),
    );
    metadata.insert(
        "llmctl.upstream_model".to_string(),
        Value::String(context.upstream_model.clone()),
    );
    metadata.insert("llmctl.stream".to_string(), Value::Bool(stream));
    metadata.insert(
        "llmctl.tool_schema_count".to_string(),
        Value::from(context.tool_audit.tool_schema_count),
    );
    metadata.insert(
        "llmctl.tool_choice".to_string(),
        context.tool_audit.tool_choice.clone(),
    );
    metadata.insert(
        "llmctl.tool_call_count".to_string(),
        Value::from(context.tool_audit.tool_call_count),
    );
    metadata
}

fn quota_admission_scopes(cfg: &Config, principal: &Principal) -> Vec<(String, usize)> {
    let mut scopes = BTreeMap::<String, usize>::new();
    for quota in matching_quota_policies(&cfg.quotas, principal) {
        let Ok(limit) = usize::try_from(quota.max_concurrency) else {
            continue;
        };
        if limit == 0 {
            continue;
        }
        let scope = if quota_is_subject_scoped(quota, principal) {
            format!("subject:{}", principal.subject)
        } else {
            format!("team:{}", principal.team)
        };
        scopes
            .entry(scope)
            .and_modify(|existing| *existing = (*existing).min(limit))
            .or_insert(limit);
    }
    scopes.into_iter().collect()
}

fn upstream_for_alias(
    state: &ServerState,
    upstream_alias: &str,
) -> std::result::Result<String, String> {
    state
        .upstreams
        .get(upstream_alias)
        .or_else(|| state.upstreams.get("*"))
        .cloned()
        .ok_or_else(|| format!("no upstream configured for model {upstream_alias}"))
}

#[derive(Debug)]
enum DispatchFailure {
    NoUpstream(String),
    BadRequest(String),
    Request {
        status: StatusCode,
        code: &'static str,
        message: String,
        usage_status: &'static str,
    },
}

async fn dispatch_chat_request(
    state: &ServerState,
    route: &ResolvedModelRoute,
    original_body: &[u8],
    request_id: Uuid,
    principal: &Principal,
    model: &str,
    started: Instant,
) -> std::result::Result<(reqwest::Response, String), DispatchFailure> {
    let mut aliases = vec![route.upstream_alias.clone()];
    aliases.extend(route.fallback_aliases.clone());
    let mut last_failure = None;

    for alias in aliases {
        let external_provider = if alias == route.upstream_alias {
            route.external_provider.clone()
        } else {
            None
        };
        let attempt_route = ResolvedModelRoute {
            requested_alias: route.requested_alias.clone(),
            upstream_alias: alias.clone(),
            fallback_aliases: Vec::new(),
            external_provider,
        };
        let body = rewrite_chat_model(original_body, &attempt_route)
            .map_err(DispatchFailure::BadRequest)?;
        let target = dispatch_target(state, &attempt_route)?;
        let upstream_base = target.base_url.clone();
        if !state.circuit_breakers.allow_request(
            &upstream_base,
            Duration::from_secs(state.cfg.server.circuit_breaker_reset_seconds),
        ) {
            last_failure = Some(DispatchFailure::Request {
                status: StatusCode::BAD_GATEWAY,
                code: "upstream_circuit_open",
                message: "upstream circuit breaker is open".to_string(),
                usage_status: "upstream_circuit_open",
            });
            continue;
        }
        let upstream = format!("{upstream_base}/v1/chat/completions");
        let mut request_builder = inject_trace_context(
            state
                .client
                .post(upstream)
                .header(CONTENT_TYPE, "application/json")
                .body(body),
        );
        if let Some(api_key) = target.api_key {
            request_builder = request_builder.bearer_auth(api_key);
        }
        match timeout(
            model_upstream_timeout(state, &route.requested_alias),
            request_builder.send(),
        )
        .await
        {
            Ok(Ok(response)) if should_retry_upstream_status(response.status()) => {
                state
                    .circuit_breakers
                    .record_failure(&upstream_base, state.cfg.server.circuit_breaker_failures);
                retry_after_delay(&response).await;
                last_failure = Some(DispatchFailure::Request {
                    status: StatusCode::BAD_GATEWAY,
                    code: "upstream_error",
                    message: "upstream request failed".to_string(),
                    usage_status: "upstream_error",
                });
                continue;
            }
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    state.circuit_breakers.record_success(&upstream_base);
                }
                return Ok((response, alias));
            }
            Ok(Err(err)) => {
                state
                    .circuit_breakers
                    .record_failure(&upstream_base, state.cfg.server.circuit_breaker_failures);
                let (status, code, message, usage_status) = upstream_request_error(&err);
                record_upstream_failure(state, request_id, principal, model, started, usage_status)
                    .await;
                last_failure = Some(DispatchFailure::Request {
                    status,
                    code,
                    message,
                    usage_status,
                });
            }
            Err(_) => {
                state
                    .circuit_breakers
                    .record_failure(&upstream_base, state.cfg.server.circuit_breaker_failures);
                record_upstream_failure(state, request_id, principal, model, started, "timeout")
                    .await;
                last_failure = Some(DispatchFailure::Request {
                    status: StatusCode::GATEWAY_TIMEOUT,
                    code: "timeout",
                    message: "upstream request timed out".to_string(),
                    usage_status: "timeout",
                });
            }
        }
    }

    Err(last_failure.unwrap_or_else(|| {
        DispatchFailure::NoUpstream(format!(
            "no upstream configured for model {}",
            route.upstream_alias
        ))
    }))
}

#[derive(Debug)]
struct DispatchTarget {
    base_url: String,
    api_key: Option<String>,
}

fn dispatch_target(
    state: &ServerState,
    route: &ResolvedModelRoute,
) -> std::result::Result<DispatchTarget, DispatchFailure> {
    if let Some(provider) = route.external_provider.as_ref() {
        let api_key = env::var(&provider.api_key_env).map_err(|_| DispatchFailure::Request {
            status: StatusCode::BAD_GATEWAY,
            code: "provider_api_key_unavailable",
            message: format!(
                "external provider {} API key is not available from configured environment reference",
                provider.id
            ),
            usage_status: "provider_api_key_unavailable",
        })?;
        if api_key.trim().is_empty() {
            return Err(DispatchFailure::Request {
                status: StatusCode::BAD_GATEWAY,
                code: "provider_api_key_unavailable",
                message: format!(
                    "external provider {} API key is empty in configured environment reference",
                    provider.id
                ),
                usage_status: "provider_api_key_unavailable",
            });
        }
        return Ok(DispatchTarget {
            base_url: provider.base_url.clone(),
            api_key: Some(api_key.trim().to_string()),
        });
    }

    Ok(DispatchTarget {
        base_url: upstream_for_alias(state, &route.upstream_alias)
            .map_err(DispatchFailure::NoUpstream)?,
        api_key: None,
    })
}

fn should_retry_upstream_status(status: StatusCode) -> bool {
    status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS
}

async fn retry_after_delay(response: &reqwest::Response) {
    let delay = response
        .headers()
        .get(axum::http::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|seconds| Duration::from_secs(seconds.min(2)))
        .unwrap_or_else(|| Duration::from_millis(100));
    tokio::time::sleep(delay).await;
}

/// Extracted gen_ai semantic-convention request parameters for lifecycle span
/// instrumentation.  Fields are bounded to prevent large allocations from
/// prompt-heavy payloads.
#[derive(Debug, Clone, Default)]
struct GenAiRequestParams {
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    /// First `system` role message body, truncated to 1 000 chars.
    system_message: Option<String>,
    /// Last `user` role message body, truncated to 1 000 chars.
    user_message: Option<String>,
}

struct UpstreamRequestContext {
    request_id: Uuid,
    principal: Principal,
    model: String,
    upstream_model: String,
    upstream_timeout: Duration,
    external_provider: Option<ResolvedExternalProvider>,
    tool_audit: ToolAuditDetail,
    started: Instant,
    admission: AdmissionPermit,
    gen_ai: GenAiRequestParams,
}

/// Extracts gen_ai observability parameters from a [`ChatCompletionRequest`].
///
/// Message bodies are truncated to 1 000 chars to bound span payload size.
fn gen_ai_params_from_request(request: &ChatCompletionRequest) -> GenAiRequestParams {
    let system_message = request
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| native::message_content_text(m).chars().take(1000).collect());
    let user_message = request
        .messages
        .iter()
        .rfind(|m| m.role == "user")
        .map(|m| native::message_content_text(m).chars().take(1000).collect());
    GenAiRequestParams {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        system_message,
        user_message,
    }
}

async fn json_upstream(
    state: Arc<ServerState>,
    upstream_response: reqwest::Response,
    context: UpstreamRequestContext,
) -> Response {
    let UpstreamRequestContext {
        request_id,
        principal,
        model,
        upstream_model,
        upstream_timeout,
        external_provider,
        tool_audit,
        started,
        admission: _admission,
        gen_ai: _gen_ai,
    } = context;
    let status = upstream_response.status();
    let headers = response_headers(upstream_response.headers());
    let bytes = match timeout(upstream_timeout, upstream_response.bytes()).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(err)) => {
            record_upstream_failure(
                &state,
                request_id,
                &principal,
                &model,
                started,
                upstream_error_status(&err),
            )
            .await;
            return with_request_id(
                error_response(
                    StatusCode::BAD_GATEWAY,
                    "upstream_error",
                    "upstream request failed".to_string(),
                ),
                request_id,
            );
        }
        Err(_) => {
            record_upstream_failure(&state, request_id, &principal, &model, started, "timeout")
                .await;
            return with_request_id(
                error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    "timeout",
                    "upstream request timed out".to_string(),
                ),
                request_id,
            );
        }
    };

    let latency_ms = elapsed_ms(started);
    let (input_tokens, output_tokens) = usage_tokens(&bytes);
    let status_text = if status.is_success() {
        "ok"
    } else {
        "upstream_error"
    };
    record_upstream_telemetry(
        &model,
        &upstream_model,
        status.as_u16(),
        latency_ms,
        status_text,
    );
    record_usage(
        &state,
        UsageRecordInput {
            request_id,
            principal: &principal,
            model: &model,
            input_tokens,
            output_tokens,
            latency_ms,
            status: status_text,
            accounting_mode: "upstream",
            gen_ai_system: gen_ai_system_for_provider(external_provider.as_ref()),
        },
    )
    .await;
    record_audit(
        &state,
        Some(request_id),
        principal,
        "chat.completions",
        model.clone(),
        status_text,
        chat_route_audit_detail(
            &tool_audit,
            json!({ "status": status.as_u16() }),
            external_provider.as_ref(),
        ),
    )
    .await;

    let response = build_response(status, headers, Body::from(bytes), request_id);
    if status.is_success() {
        with_chat_metadata(response, &model, &upstream_model, "allowed")
    } else {
        response
    }
}

async fn stream_upstream(
    state: Arc<ServerState>,
    upstream_response: reqwest::Response,
    context: UpstreamRequestContext,
) -> Response {
    let UpstreamRequestContext {
        request_id,
        principal,
        model,
        upstream_model,
        upstream_timeout,
        external_provider,
        tool_audit,
        started,
        admission,
        gen_ai,
    } = context;
    let status = upstream_response.status();
    if !status.is_success() {
        record_usage(
            &state,
            UsageRecordInput {
                request_id,
                principal: &principal,
                model: &model,
                input_tokens: 0,
                output_tokens: 0,
                latency_ms: elapsed_ms(started),
                status: "upstream_error",
                accounting_mode: "none",
                gen_ai_system: gen_ai_system_for_provider(external_provider.as_ref()),
            },
        )
        .await;
        record_audit(
            &state,
            Some(request_id),
            principal,
            "chat.completions",
            model,
            "upstream_error",
            chat_route_audit_detail(
                &tool_audit,
                json!({ "status": status.as_u16(), "stream": true }),
                external_provider.as_ref(),
            ),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                "upstream request failed".to_string(),
            ),
            request_id,
        );
    }

    let mut headers = response_headers(upstream_response.headers());
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    let response_model = model.clone();
    let response_upstream_model = upstream_model.clone();
    let mut upstream_stream = upstream_response.bytes_stream();
    let idle_timeout = upstream_timeout;
    let capture_content = state.cfg.observability.gen_ai.capture_message_content;
    let stream = async_stream::stream! {
        let _admission = admission;
        let mut input_tokens = 0u64;
        let mut output_tokens = 0u64;
        let mut usage_parser = SseUsageParser::default();
        let mut content_parser = crate::observability::SseContentParser::default();
        let mut first_token_instant: Option<std::time::Instant> = None;
        let mut last_token_instant: Option<std::time::Instant> = None;
        loop {
            let chunk = match timeout(idle_timeout, upstream_stream.next()).await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(_) => {
                    record_usage(
                        &state,
                        UsageRecordInput {
                            request_id,
                            principal: &principal,
                            model: &model,
                            input_tokens,
                            output_tokens,
                            latency_ms: elapsed_ms(started),
                            status: "timeout",
                            accounting_mode: "upstream",
                            gen_ai_system: gen_ai_system_for_provider(external_provider.as_ref()),
                        },
                    )
                    .await;
                    record_upstream_telemetry(
                        &model,
                        &upstream_model,
                        status.as_u16(),
                        elapsed_ms(started),
                        "timeout",
                    );
                    record_audit(
                        &state,
                        Some(request_id),
                        principal.clone(),
                        "chat.completions",
                        model.clone(),
                        "timeout",
                        chat_route_audit_detail(
                            &tool_audit,
                            json!({ "status": status.as_u16(), "stream": true }),
                            external_provider.as_ref(),
                        ),
                    )
                    .await;
                    yield Err::<Bytes, std::io::Error>(std::io::Error::new(std::io::ErrorKind::TimedOut, "upstream stream timed out"));
                    return;
                }
            };
            match chunk {
                Ok(bytes) => {
                    content_parser.push(&bytes);
                    let total_content = content_parser.output_deltas()
                        .saturating_add(content_parser.thinking_deltas());
                    if first_token_instant.is_none() && total_content > 0 {
                        first_token_instant = Some(std::time::Instant::now());
                    }
                    if total_content > 0 {
                        last_token_instant = Some(std::time::Instant::now());
                    }
                    match usage_parser.push(&bytes) {
                        Ok((input, output)) => {
                            input_tokens = input_tokens.saturating_add(input);
                            output_tokens = output_tokens.saturating_add(output);
                            yield Ok::<Bytes, std::io::Error>(bytes)
                        }
                        Err(reason) => {
                            record_usage(
                                &state,
                                UsageRecordInput {
                                    request_id,
                                    principal: &principal,
                                    model: &model,
                                    input_tokens,
                                    output_tokens,
                                    latency_ms: elapsed_ms(started),
                                    status: "stream_error",
                                    accounting_mode: "upstream",
                                    gen_ai_system: gen_ai_system_for_provider(
                                        external_provider.as_ref(),
                                    ),
                                },
                            )
                            .await;
                            record_upstream_telemetry(
                                &model,
                                &upstream_model,
                                status.as_u16(),
                                elapsed_ms(started),
                                "stream_error",
                            );
                            record_audit(
                                &state,
                                Some(request_id),
                                principal.clone(),
                                "chat.completions",
                                model.clone(),
                                "stream_error",
                                chat_route_audit_detail(
                                    &tool_audit,
                                    json!({ "status": status.as_u16(), "stream": true, "reason": reason }),
                                    external_provider.as_ref(),
                                ),
                            )
                            .await;
                            yield Err::<Bytes, std::io::Error>(std::io::Error::other(reason));
                            return;
                        }
                    }
                }
                Err(err) => {
                    record_usage(
                        &state,
                        UsageRecordInput {
                            request_id,
                            principal: &principal,
                            model: &model,
                            input_tokens,
                            output_tokens,
                            latency_ms: elapsed_ms(started),
                            status: "stream_error",
                            accounting_mode: "upstream",
                            gen_ai_system: gen_ai_system_for_provider(external_provider.as_ref()),
                        },
                    )
                    .await;
                    record_upstream_telemetry(
                        &model,
                        &upstream_model,
                        status.as_u16(),
                        elapsed_ms(started),
                        "stream_error",
                    );
                    record_audit(
                        &state,
                        Some(request_id),
                        principal.clone(),
                        "chat.completions",
                        model.clone(),
                        "stream_error",
                        chat_route_audit_detail(
                            &tool_audit,
                            json!({ "status": status.as_u16(), "stream": true, "reason": upstream_error_status(&err) }),
                            external_provider.as_ref(),
                        ),
                    )
                    .await;
                    yield Err::<Bytes, std::io::Error>(std::io::Error::other("upstream stream failed"));
                    return;
                }
            }
        }
        emit_gen_ai_inference_span(
            &model,
            &gen_ai,
            input_tokens,
            output_tokens,
            content_parser.thinking_deltas(),
            content_parser.output_deltas(),
            started,
            first_token_instant,
            last_token_instant,
            stream_status(input_tokens, output_tokens),
            capture_content,
        );
        crate::observability::emit_gen_ai_thinking_metrics(
            &model,
            content_parser.thinking_deltas(),
            content_parser.output_deltas(),
        );
        record_usage(
            &state,
            UsageRecordInput {
                request_id,
                principal: &principal,
                model: &model,
                input_tokens,
                output_tokens,
                latency_ms: elapsed_ms(started),
                status: stream_status(input_tokens, output_tokens),
                accounting_mode: "upstream",
                gen_ai_system: gen_ai_system_for_provider(external_provider.as_ref()),
            },
        )
        .await;
        record_upstream_telemetry(
            &model,
            &upstream_model,
            status.as_u16(),
            elapsed_ms(started),
            stream_status(input_tokens, output_tokens),
        );
        record_audit(
            &state,
            Some(request_id),
            principal.clone(),
            "chat.completions",
            model.clone(),
            stream_status(input_tokens, output_tokens),
            chat_route_audit_detail(
                &tool_audit,
                json!({ "status": status.as_u16(), "stream": true, "metered": input_tokens > 0 || output_tokens > 0 }),
                external_provider.as_ref(),
            ),
        )
        .await;
    };
    let response = build_response(status, headers, Body::from_stream(stream), request_id);
    if status.is_success() {
        with_chat_metadata(
            response,
            &response_model,
            &response_upstream_model,
            "allowed",
        )
    } else {
        response
    }
}

fn model_upstream_timeout(state: &ServerState, alias: &str) -> Duration {
    state
        .cfg
        .server
        .model_upstream_timeout_seconds
        .get(alias)
        .copied()
        .map(Duration::from_secs)
        .unwrap_or_else(|| state.serving_limits.upstream_timeout())
        .max(Duration::from_millis(1))
}

fn upstream_request_error(
    err: &reqwest::Error,
) -> (StatusCode, &'static str, String, &'static str) {
    if err.is_timeout() {
        (
            StatusCode::GATEWAY_TIMEOUT,
            "timeout",
            "upstream request timed out".to_string(),
            "timeout",
        )
    } else {
        (
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            "upstream request failed".to_string(),
            "upstream_error",
        )
    }
}

fn upstream_error_status(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else {
        "upstream_error"
    }
}

async fn record_upstream_failure(
    state: &ServerState,
    request_id: Uuid,
    principal: &Principal,
    model: &str,
    started: Instant,
    status: &'static str,
) {
    record_usage(
        state,
        UsageRecordInput {
            request_id,
            principal,
            model,
            input_tokens: 0,
            output_tokens: 0,
            latency_ms: elapsed_ms(started),
            status,
            accounting_mode: "none",
            gen_ai_system: gen_ai_system_for_provider(None),
        },
    )
    .await;
    record_upstream_telemetry(model, "unknown", 0, elapsed_ms(started), status);
}

fn record_admission_busy_telemetry(model: &str, principal: &Principal, stream: bool) {
    let attributes = [
        KeyValue::new("llmctl.model", model.to_string()),
        KeyValue::new("llmctl.actor", principal.subject.clone()),
        KeyValue::new("llmctl.team", principal.team.clone()),
        KeyValue::new("llmctl.stream", stream),
        KeyValue::new("model", model.to_string()),
        KeyValue::new("team", principal.team.clone()),
        KeyValue::new("reason", "admission_limit_exceeded"),
    ];
    global::meter(crate::SERVICE_NAME)
        .u64_counter("llmctl_admission_rejections_total")
        .with_description("Requests rejected because global or scoped admission limits were full")
        .build()
        .add(1, &attributes);
}

fn record_upstream_telemetry(
    model: &str,
    upstream_model: &str,
    status_code: u16,
    latency_ms: u64,
    status: &str,
) {
    let attributes = [
        KeyValue::new("llmctl.model", model.to_string()),
        KeyValue::new("llmctl.upstream_model", upstream_model.to_string()),
        KeyValue::new("llmctl.status", status.to_string()),
        KeyValue::new("http.response.status_code", i64::from(status_code)),
        KeyValue::new("model", model.to_string()),
        KeyValue::new("upstream_model", upstream_model.to_string()),
        KeyValue::new("status", slo_status(status)),
    ];
    let meter = global::meter(crate::SERVICE_NAME);
    meter
        .u64_counter("llmctl_upstream_requests_total")
        .with_description("Upstream worker requests by routed model and status")
        .build()
        .add(1, &attributes);
    if status != "ok" {
        meter
            .u64_counter("llmctl_upstream_errors_total")
            .with_description("Failed upstream worker requests by routed model and status")
            .build()
            .add(1, &attributes);
    }
    meter
        .u64_histogram("llmctl_upstream_latency_ms")
        .with_description("Upstream worker round-trip or stream duration in milliseconds")
        .build()
        .record(latency_ms, &attributes);
}

fn record_circuit_breaker_state(upstream: &str, state: &str, consecutive_failures: u32) {
    let upstream_id = stable_upstream_id(upstream);
    let attributes = [
        KeyValue::new("llmctl.upstream.id", upstream_id.clone()),
        KeyValue::new("llmctl.circuit.state", state.to_string()),
        KeyValue::new("upstream_id", upstream_id.clone()),
        KeyValue::new("state", state.to_string()),
    ];
    let meter = global::meter(crate::SERVICE_NAME);
    meter
        .u64_counter("llmctl_upstream_circuit_state_total")
        .with_description("Circuit breaker state transitions and observations by upstream")
        .build()
        .add(1, &attributes);
    meter
        .u64_histogram("llmctl_upstream_circuit_consecutive_failures")
        .with_description("Observed consecutive upstream failures before circuit reset")
        .build()
        .record(u64::from(consecutive_failures), &attributes);
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Metric,
        TelemetryEventName::CircuitBreaker,
        Utc::now(),
        BTreeMap::from([
            ("llmctl.upstream.id".to_string(), json!(upstream_id)),
            ("llmctl.circuit.state".to_string(), json!(state)),
            (
                "llmctl.circuit.consecutive_failures".to_string(),
                json!(consecutive_failures),
            ),
        ]),
    ));
}

fn stable_upstream_id(upstream: &str) -> String {
    let digest = Sha256::digest(upstream.as_bytes());
    format!("upstream-{:x}", digest)[..25].to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedModelRoute {
    requested_alias: String,
    upstream_alias: String,
    fallback_aliases: Vec<String>,
    external_provider: Option<ResolvedExternalProvider>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedExternalProvider {
    id: String,
    kind: ExternalProviderKind,
    base_url: String,
    api_key_env: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelRouteError {
    UnknownAlias(String),
    NoConfiguredModels,
    ExternalProviderRoutingDisabled(String),
}

impl std::fmt::Display for ModelRouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownAlias(alias) => write!(f, "unknown model alias: {alias}"),
            Self::NoConfiguredModels => write!(f, "no models are configured"),
            Self::ExternalProviderRoutingDisabled(provider) => write!(
                f,
                "external provider routing is disabled for provider {provider}"
            ),
        }
    }
}

fn model_route_error_response(err: &ModelRouteError) -> Response {
    match err {
        ModelRouteError::UnknownAlias(_) | ModelRouteError::NoConfiguredModels => {
            error_response(StatusCode::NOT_FOUND, "model_not_found", err.to_string())
        }
        ModelRouteError::ExternalProviderRoutingDisabled(_) => {
            error_response(StatusCode::BAD_REQUEST, "bad_request", err.to_string())
        }
    }
}

fn routed_models(cfg: &Config) -> Vec<&ModelConfig> {
    let mut models: Vec<_> = cfg
        .models
        .iter()
        .filter(|model| model_is_routed_locally(cfg, model))
        .collect();
    models.sort_by(|left, right| left.alias.cmp(&right.alias));
    models
}

fn active_routed_models(cfg: &Config) -> Vec<&ModelConfig> {
    routed_models(cfg)
        .into_iter()
        .filter(|model| model.weight > 0)
        .collect()
}

fn model_is_routed_locally(cfg: &Config, model: &ModelConfig) -> bool {
    let placement = native::placement_plan_from_config(cfg);
    placement
        .nodes
        .iter()
        .find(|node| node.id == placement.local_node)
        .map(|node| node.model_aliases.iter().any(|alias| alias == &model.alias))
        .unwrap_or(false)
}

fn resolve_model_route(
    cfg: &Config,
    requested_alias: &str,
    request_id: Uuid,
) -> std::result::Result<ResolvedModelRoute, ModelRouteError> {
    if cfg.models.is_empty() {
        if requested_alias.trim().is_empty() {
            return Err(ModelRouteError::NoConfiguredModels);
        }
        return Ok(ResolvedModelRoute {
            requested_alias: requested_alias.to_string(),
            upstream_alias: requested_alias.to_string(),
            fallback_aliases: Vec::new(),
            external_provider: None,
        });
    }

    let requested = cfg
        .models
        .iter()
        .find(|model| model.alias == requested_alias && model_is_routed_locally(cfg, model))
        .ok_or_else(|| ModelRouteError::UnknownAlias(requested_alias.to_string()))?;

    let upstream = match cfg.mode {
        Mode::Single => cfg
            .models
            .iter()
            .find(|model| model_is_routed_locally(cfg, model))
            .ok_or(ModelRouteError::NoConfiguredModels)?,
        Mode::ColdSwap | Mode::HotSwap => requested,
        Mode::Weighted => weighted_model_for_request(cfg, request_id).unwrap_or(requested),
        Mode::Fallback => {
            if requested.weight > 0 {
                requested
            } else {
                weighted_model_for_request(cfg, request_id).unwrap_or(requested)
            }
        }
    };
    let fallback_aliases = if matches!(cfg.mode, Mode::Fallback) {
        fallback_aliases(cfg, &upstream.alias)
    } else {
        Vec::new()
    };
    if let Some(route) = cfg.external_providers.route_for_model(&upstream.alias) {
        return Err(ModelRouteError::ExternalProviderRoutingDisabled(
            route.provider.clone(),
        ));
    }

    Ok(ResolvedModelRoute {
        requested_alias: requested_alias.to_string(),
        upstream_alias: upstream.alias.clone(),
        fallback_aliases,
        external_provider: None,
    })
}

fn weighted_model_for_request(cfg: &Config, request_id: Uuid) -> Option<&ModelConfig> {
    let weighted = cfg
        .models
        .iter()
        .filter(|model| model.weight > 0 && model_is_routed_locally(cfg, model))
        .collect::<Vec<_>>();
    let total = weighted.iter().fold(0u64, |total, model| {
        total.saturating_add(u64::from(model.weight))
    });
    if total == 0 {
        return None;
    }

    let mut slot = request_id.as_u128() % u128::from(total);
    for model in weighted {
        let weight = u128::from(model.weight);
        if slot < weight {
            return Some(model);
        }
        slot -= weight;
    }

    None
}

fn fallback_aliases(cfg: &Config, selected_alias: &str) -> Vec<String> {
    let mut models = routed_models(cfg)
        .into_iter()
        .filter(|model| model.alias != selected_alias)
        .collect::<Vec<_>>();
    models.sort_by(|left, right| {
        right
            .weight
            .cmp(&left.weight)
            .then_with(|| left.alias.cmp(&right.alias))
    });
    models
        .into_iter()
        .map(|model| model.alias.clone())
        .collect()
}

fn rewrite_chat_model(
    body: &[u8],
    route: &ResolvedModelRoute,
) -> std::result::Result<Bytes, String> {
    if route.requested_alias == route.upstream_alias {
        return Ok(Bytes::copy_from_slice(body));
    }

    let mut value: Value =
        serde_json::from_slice(body).map_err(|_| "request body must be valid JSON".to_string())?;
    let Some(object) = value.as_object_mut() else {
        return Err("request body must be a JSON object".to_string());
    };
    object.insert(
        "model".to_string(),
        Value::String(route.upstream_alias.clone()),
    );
    serde_json::to_vec(&value)
        .map(Bytes::from)
        .map_err(|err| err.to_string())
}

async fn record_audit(
    state: &ServerState,
    request_id: Option<Uuid>,
    principal: Principal,
    action: impl Into<String>,
    resource: impl Into<String>,
    outcome: impl Into<String>,
    detail_json: Value,
) {
    let detail_json = if let Some(key_id) = principal.key_id.as_ref() {
        let mut detail = detail_json;
        match detail.as_object_mut() {
            Some(object) => {
                object
                    .entry("api_key_id".to_string())
                    .or_insert_with(|| json!(key_id));
                object
                    .entry("api_key_subject".to_string())
                    .or_insert_with(|| json!(principal.subject.as_str()));
                object
                    .entry("api_key_team".to_string())
                    .or_insert_with(|| json!(principal.team.as_str()));
                if let Some(owner) = principal.key_owner.as_deref() {
                    object
                        .entry("api_key_owner".to_string())
                        .or_insert_with(|| json!(owner));
                }
                if let Some(purpose) = principal.key_purpose.as_deref() {
                    object
                        .entry("api_key_purpose".to_string())
                        .or_insert_with(|| json!(purpose));
                }
                if let Some(status) = principal.key_status.as_deref() {
                    object
                        .entry("api_key_status".to_string())
                        .or_insert_with(|| json!(status));
                }
            }
            None => {
                detail = json!({
                    "detail": detail,
                    "api_key_id": key_id,
                    "api_key_subject": principal.subject.as_str(),
                    "api_key_team": principal.team.as_str(),
                    "api_key_owner": principal.key_owner.as_deref(),
                    "api_key_purpose": principal.key_purpose.as_deref(),
                    "api_key_status": principal.key_status.as_deref(),
                });
            }
        }
        detail
    } else {
        detail_json
    };
    let event = AuditEvent::new(
        request_id,
        principal.subject,
        principal.team,
        action,
        resource,
        outcome,
        detail_json,
    );
    if let Err(err) = state.storage.insert_audit_event(&event).await {
        tracing::warn!(error = %err, "failed to record audit event");
    }
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::RequestRouting,
        Utc::now(),
        BTreeMap::from([
            ("llmctl.request_id".to_string(), json_string(request_id)),
            (
                "llmctl.audit.action".to_string(),
                json!(event.action.as_str()),
            ),
            (
                "llmctl.audit.resource".to_string(),
                json!(event.resource.as_str()),
            ),
            (
                "llmctl.audit.outcome".to_string(),
                json!(event.outcome.as_str()),
            ),
            ("llmctl.actor".to_string(), json!(event.actor.as_str())),
            ("llmctl.team".to_string(), json!(event.team.as_str())),
        ]),
    ));
}

/// Records an audit event for a rejected/denied request and turns it into the
/// corresponding error response, in one call. This is the shared shape behind
/// the many `record_audit(...).await; return with_request_id(error_response(...))`
/// blocks in the chat and embeddings handlers.
///
/// The argument count is inherent to the data each audit entry + response
/// needs (actor, action/resource/outcome triple, and status/code/message);
/// splitting it into a config struct would just move the same fields one
/// level up without making call sites clearer.
#[allow(clippy::too_many_arguments)]
async fn audit_reject(
    state: &ServerState,
    request_id: Uuid,
    principal: Principal,
    action: impl Into<String>,
    resource: impl Into<String>,
    outcome: impl Into<String>,
    status: StatusCode,
    code: &str,
    message: String,
    detail: Value,
) -> Response {
    audit_reject_response(
        state,
        request_id,
        principal,
        action,
        resource,
        outcome,
        error_response(status, code, message),
        detail,
    )
    .await
}

/// Like `audit_reject`, but for the rejection paths whose response is built by
/// a helper other than `error_response` (e.g. `auth_error_response`,
/// `model_route_error_response`).
#[allow(clippy::too_many_arguments)]
async fn audit_reject_response(
    state: &ServerState,
    request_id: Uuid,
    principal: Principal,
    action: impl Into<String>,
    resource: impl Into<String>,
    outcome: impl Into<String>,
    response: Response,
    detail: Value,
) -> Response {
    record_audit(
        state,
        Some(request_id),
        principal,
        action,
        resource,
        outcome,
        detail,
    )
    .await;
    with_request_id(response, request_id)
}

async fn record_swap_execution(
    state: &ServerState,
    request_id: Option<Uuid>,
    principal: Principal,
    execution: &SwapExecution,
) {
    record_audit(
        state,
        request_id,
        principal,
        "admin.swap",
        execution.plan.replacement.as_str(),
        if execution.success {
            "allowed"
        } else {
            "failed"
        },
        json!({
            "mode": execution.mode,
            "active": execution.plan.active,
            "replacement": execution.plan.replacement,
            "steps": execution.plan.steps,
            "statuses": execution.statuses,
        }),
    )
    .await;
}

async fn record_quota_decision(
    state: &ServerState,
    request_id: Option<Uuid>,
    principal: &Principal,
    model: &str,
    decision: &crate::quota::QuotaDecision,
    policy_json: Value,
) {
    let record = QuotaDecisionRecord::new(
        request_id,
        principal,
        model.to_string(),
        decision,
        policy_json,
    );
    if let Err(err) = state.storage.insert_quota_decision(&record).await {
        tracing::warn!(error = %err, "failed to record quota decision");
    }
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::QuotaDecision,
        Utc::now(),
        BTreeMap::from([
            ("llmctl.request_id".to_string(), json_string(request_id)),
            ("llmctl.model".to_string(), json!(model)),
            (
                "llmctl.actor".to_string(),
                json!(principal.subject.as_str()),
            ),
            ("llmctl.team".to_string(), json!(principal.team.as_str())),
            ("llmctl.quota.allowed".to_string(), json!(decision.allowed)),
            (
                "llmctl.quota.reason".to_string(),
                json!(decision.reason.as_str()),
            ),
        ]),
    ));
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct RuntimeLineageMetadata {
    lineage_ids: Vec<String>,
    corpus: Option<String>,
}

async fn record_request_lineage_joins(
    state: &ServerState,
    request_id: Uuid,
    lineage: &RuntimeLineageMetadata,
    model: Option<&str>,
    source: &str,
) {
    for lineage_id in &lineage.lineage_ids {
        let lineage_id = sanitize_lineage_value(lineage_id);
        let corpus = lineage.corpus.as_deref().map(sanitize_lineage_value);
        let record = RequestLineageJoinRecord::new(
            request_id,
            lineage_id,
            model.map(str::to_string),
            corpus,
            source,
        );
        if let Err(err) = state.storage.insert_request_lineage_join(&record).await {
            tracing::warn!(error = %err, "failed to record request lineage join");
        }
    }
}

fn runtime_lineage_from_headers_and_metadata(
    headers: &HeaderMap,
    metadata: Option<&Value>,
) -> RuntimeLineageMetadata {
    let mut lineage = RuntimeLineageMetadata::default();
    extend_lineage_ids_from_header(headers, lineage_id_header_name(), &mut lineage.lineage_ids);
    extend_lineage_ids_from_header(headers, lineage_ids_header_name(), &mut lineage.lineage_ids);
    if let Some(corpus) = header_string(headers, corpus_header_name()) {
        lineage.corpus = Some(corpus);
    }

    if let Some(metadata) = metadata.and_then(|value| value.as_object()) {
        extend_lineage_ids_from_value(metadata.get("lineage_id"), &mut lineage.lineage_ids);
        extend_lineage_ids_from_value(metadata.get("lineage_ids"), &mut lineage.lineage_ids);
        if lineage.corpus.is_none() {
            lineage.corpus = metadata
                .get("corpus")
                .or_else(|| metadata.get("corpus_id"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
        }
    }

    let mut seen = BTreeSet::new();
    lineage
        .lineage_ids
        .retain(|lineage_id| seen.insert(lineage_id.clone()));
    lineage
}

fn extend_lineage_ids_from_header(
    headers: &HeaderMap,
    name: HeaderName,
    lineage_ids: &mut Vec<String>,
) {
    for value in headers.get_all(name) {
        if let Ok(value) = value.to_str() {
            extend_lineage_ids_from_str(value, lineage_ids);
        }
    }
}

fn extend_lineage_ids_from_value(value: Option<&Value>, lineage_ids: &mut Vec<String>) {
    match value {
        Some(Value::String(value)) => extend_lineage_ids_from_str(value, lineage_ids),
        Some(Value::Array(values)) => {
            for value in values {
                if let Some(value) = value.as_str() {
                    extend_lineage_ids_from_str(value, lineage_ids);
                }
            }
        }
        _ => {}
    }
}

fn extend_lineage_ids_from_str(raw: &str, lineage_ids: &mut Vec<String>) {
    lineage_ids.extend(
        raw.split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(sanitize_lineage_value),
    );
}

fn sanitize_lineage_value(raw: &str) -> String {
    let value = raw.trim();
    if value.len() <= 128
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-' | ':'))
        && !looks_sensitive_lineage_value(value)
    {
        return value.to_string();
    }
    let digest = Sha256::digest(value.as_bytes());
    format!("redacted:{:x}", digest)[..25].to_string()
}

fn looks_sensitive_lineage_value(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    value.contains('/')
        || value.contains('\\')
        || lower.contains("bearer ")
        || lower.contains("apikey")
        || lower.contains("api_key")
        || lower.contains("token")
        || lower.contains("secret")
        || lower.contains("password")
}

fn header_string(headers: &HeaderMap, name: HeaderName) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn json_string(value: Option<Uuid>) -> Value {
    value
        .map(|value| Value::String(value.to_string()))
        .unwrap_or(Value::Null)
}

struct UsageRecordInput<'a> {
    request_id: Uuid,
    principal: &'a Principal,
    model: &'a str,
    input_tokens: u64,
    output_tokens: u64,
    latency_ms: u64,
    status: &'a str,
    accounting_mode: &'a str,
    gen_ai_system: &'a str,
}

async fn record_usage(state: &ServerState, input: UsageRecordInput<'_>) {
    let event = UsageEvent {
        id: Uuid::new_v4(),
        request_id: input.request_id,
        at: Utc::now(),
        model: input.model.to_string(),
        actor: input.principal.subject.clone(),
        team: input.principal.team.clone(),
        input_tokens: input.input_tokens,
        output_tokens: input.output_tokens,
        latency_ms: input.latency_ms,
        status: input.status.to_string(),
    };
    if let Err(err) = state.storage.insert_usage_event(&event).await {
        tracing::warn!(error = %err, "failed to record usage event");
    }
    record_usage_telemetry(&event, input.accounting_mode, input.gen_ai_system);
    dispatch_usage_webhook(state, &event, input.accounting_mode);
}

fn gen_ai_system_for_provider(provider: Option<&ResolvedExternalProvider>) -> &'static str {
    match provider.map(|p| &p.kind) {
        None => "llmctl.native",
        Some(ExternalProviderKind::OpenAiCompatible) => "openai",
        Some(ExternalProviderKind::VertexAi) => "vertex_ai",
        Some(ExternalProviderKind::OpenRouter) => "openrouter",
    }
}

/// Build the attribute set for a usage span: the existing `llmctl.*` attributes
/// plus the OTel GenAI semantic-convention attributes
/// (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`,
/// `gen_ai.response.model`, `gen_ai.usage.*`) so traces align with the
/// conventions Langfuse and other GenAI-aware OTel consumers expect.
fn usage_span_attributes(
    event: &UsageEvent,
    accounting_mode: &str,
    gen_ai_system: &str,
) -> BTreeMap<String, Value> {
    BTreeMap::from([
        (
            "llmctl.request_id".to_string(),
            json!(event.request_id.to_string()),
        ),
        ("llmctl.model".to_string(), json!(event.model.as_str())),
        ("llmctl.actor".to_string(), json!(event.actor.as_str())),
        ("llmctl.team".to_string(), json!(event.team.as_str())),
        ("llmctl.latency_ms".to_string(), json!(event.latency_ms)),
        ("llmctl.status".to_string(), json!(event.status.as_str())),
        (
            "llmctl.token_accounting.mode".to_string(),
            json!(accounting_mode),
        ),
        ("gen_ai.system".to_string(), json!(gen_ai_system)),
        ("gen_ai.operation.name".to_string(), json!("chat")),
        (
            "gen_ai.request.model".to_string(),
            json!(event.model.as_str()),
        ),
        (
            "gen_ai.response.model".to_string(),
            json!(event.model.as_str()),
        ),
        (
            "gen_ai.usage.input_tokens".to_string(),
            json!(event.input_tokens),
        ),
        (
            "gen_ai.usage.output_tokens".to_string(),
            json!(event.output_tokens),
        ),
    ])
}

/// Emits a `gen_ai.chat` lifecycle span covering one complete inference request.
///
/// Span attributes follow the OTel GenAI semantic conventions.  Message content
/// is included only when `capture_content` is `true`; otherwise the body is
/// replaced with `[REDACTED]`.
#[allow(clippy::too_many_arguments)] // all parameters carry distinct domain meaning
fn emit_gen_ai_inference_span(
    model: &str,
    gen_ai: &GenAiRequestParams,
    input_tokens: u64,
    output_tokens: u64,
    thinking_deltas: u64,
    output_deltas: u64,
    started: std::time::Instant,
    first_token_instant: Option<std::time::Instant>,
    last_token_instant: Option<std::time::Instant>,
    status: &str,
    capture_content: bool,
) {
    let tracer = global::tracer(crate::SERVICE_NAME);
    let start_system_time = std::time::SystemTime::now()
        .checked_sub(started.elapsed())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    let mut span = tracer
        .span_builder("gen_ai.chat")
        .with_kind(SpanKind::Server)
        .with_start_time(start_system_time)
        .start(&tracer);

    span.set_attribute(KeyValue::new("gen_ai.system", "local"));
    span.set_attribute(KeyValue::new("gen_ai.operation.name", "chat"));
    span.set_attribute(KeyValue::new("gen_ai.request.model", model.to_string()));
    span.set_attribute(KeyValue::new("gen_ai.response.model", model.to_string()));
    span.set_attribute(KeyValue::new(
        "gen_ai.usage.input_tokens",
        i64::try_from(input_tokens).unwrap_or(i64::MAX),
    ));
    span.set_attribute(KeyValue::new(
        "gen_ai.usage.output_tokens",
        i64::try_from(output_tokens).unwrap_or(i64::MAX),
    ));
    span.set_attribute(KeyValue::new(
        "gen_ai.usage.thinking_tokens",
        i64::try_from(thinking_deltas).unwrap_or(i64::MAX),
    ));
    span.set_attribute(KeyValue::new(
        "gen_ai.usage.output_deltas",
        i64::try_from(output_deltas).unwrap_or(i64::MAX),
    ));
    span.set_attribute(KeyValue::new("llmctl.status", status.to_string()));

    if let Some(max_tokens) = gen_ai.max_tokens {
        span.set_attribute(KeyValue::new(
            "gen_ai.request.max_tokens",
            i64::from(max_tokens),
        ));
    }
    if let Some(temp) = gen_ai.temperature {
        span.set_attribute(KeyValue::new("gen_ai.request.temperature", f64::from(temp)));
    }

    if let (Some(first), Some(last)) = (first_token_instant, last_token_instant) {
        let ttft_secs = first.saturating_duration_since(started).as_secs_f64();
        let decode_secs = last.saturating_duration_since(first).as_secs_f64();
        span.set_attribute(KeyValue::new("gen_ai.ttft_seconds", ttft_secs));
        if decode_secs > 0.0 {
            let throughput = output_deltas as f64 / decode_secs;
            span.set_attribute(KeyValue::new(
                "gen_ai.decode_throughput_deltas_per_second",
                throughput,
            ));
        }
    }

    let redacted = crate::observability::REDACTED_ATTRIBUTE_VALUE;
    if let Some(sys) = &gen_ai.system_message {
        let body = if capture_content {
            sys.as_str()
        } else {
            redacted
        };
        span.add_event(
            "gen_ai.system.message",
            vec![KeyValue::new("body", body.to_string())],
        );
    }
    if let Some(user) = &gen_ai.user_message {
        let body = if capture_content {
            user.as_str()
        } else {
            redacted
        };
        span.add_event(
            "gen_ai.user.message",
            vec![KeyValue::new("body", body.to_string())],
        );
    }

    span.end();
}

/// JSON payload delivered to the configured usage webhook — the same
/// usage/lineage metadata recorded in the audit trail and emitted as OTel
/// attributes, shaped for ecosystems that consume callbacks rather than OTLP.
fn webhook_payload(event: &UsageEvent, accounting_mode: &str) -> Value {
    json!({
        "type": "llmctl.usage",
        "id": event.id.to_string(),
        "request_id": event.request_id.to_string(),
        "at": event.at.to_rfc3339(),
        "model": event.model,
        "actor": event.actor,
        "team": event.team,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "latency_ms": event.latency_ms,
        "status": event.status,
        "token_accounting_mode": accounting_mode,
    })
}

/// Fire-and-forget delivery of the usage webhook, if configured. Failures are
/// logged at `warn` and never affect the in-flight request — the webhook is a
/// best-effort sink, not part of the request's correctness contract.
fn dispatch_usage_webhook(state: &ServerState, event: &UsageEvent, accounting_mode: &str) {
    let webhook = &state.cfg.observability.webhook;
    if !webhook.enabled {
        return;
    }
    let Some(url) = webhook
        .url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(str::to_string)
    else {
        return;
    };

    if let Err(err) = crate::observability::validate_http_endpoint(&url) {
        tracing::warn!(url = %url, error = %err, "usage webhook URL rejected by SSRF guard");
        return;
    }

    let payload = webhook_payload(event, accounting_mode);
    let client = state.client.clone();
    let headers = webhook.headers.clone();
    let timeout = Duration::from_millis(webhook.timeout_ms.max(1));

    tokio::spawn(async move {
        let mut request = client.post(&url).json(&payload).timeout(timeout);
        for (name, value) in &headers {
            request = request.header(name.as_str(), value.as_str());
        }
        match request.send().await {
            Ok(response) if !response.status().is_success() => {
                tracing::warn!(
                    url = %url,
                    status = %response.status(),
                    "usage webhook delivery returned a non-success status"
                );
            }
            Err(err) => {
                tracing::warn!(url = %url, error = %err, "failed to deliver usage webhook");
            }
            Ok(_) => {}
        }
    });
}

fn record_usage_telemetry(event: &UsageEvent, accounting_mode: &str, gen_ai_system: &str) {
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::RequestRouting,
        Utc::now(),
        usage_span_attributes(event, accounting_mode, gen_ai_system),
    ));
    let attributes = [
        KeyValue::new("llmctl.model", event.model.clone()),
        KeyValue::new("llmctl.actor", event.actor.clone()),
        KeyValue::new("llmctl.team", event.team.clone()),
        KeyValue::new("llmctl.status", event.status.clone()),
        KeyValue::new("model", event.model.clone()),
        KeyValue::new("team", event.team.clone()),
        KeyValue::new("endpoint", "/v1/chat/completions"),
        KeyValue::new("status", slo_status(&event.status)),
        KeyValue::new("token_accounting_mode", accounting_mode.to_string()),
    ];
    let meter = global::meter(crate::SERVICE_NAME);
    meter
        .u64_counter("llmctl_requests_total")
        .with_description(
            "Total OpenAI-compatible model requests by endpoint, model, team, and status",
        )
        .build()
        .add(1, &attributes);
    if event.status != "ok" {
        meter
            .u64_counter("llmctl_request_errors_total")
            .with_description(
                "Failed OpenAI-compatible model requests by endpoint, model, team, and status",
            )
            .build()
            .add(1, &attributes);
    }
    meter
        .u64_counter("llmctl.tokens.input")
        .with_description("Input tokens reported by model workers")
        .build()
        .add(event.input_tokens, &attributes);
    meter
        .u64_counter("llmctl.tokens.output")
        .with_description("Output tokens reported by model workers")
        .build()
        .add(event.output_tokens, &attributes);
    let genai_attributes = [KeyValue::new("gen_ai.system", gen_ai_system.to_string())];
    crate::observability::genai_input_tokens_counter().add(event.input_tokens, &genai_attributes);
    crate::observability::genai_output_tokens_counter().add(event.output_tokens, &genai_attributes);
    meter
        .u64_histogram("llmctl_request_latency_ms")
        .with_description("Model request latency in milliseconds")
        .build()
        .record(event.latency_ms, &attributes);
    if event.status != "ok" || event.latency_ms > DEFAULT_SLO_LATENCY_MS {
        meter
            .u64_counter("llmctl_slo_violations_total")
            .with_description("Requests that violate the default latency or success SLO")
            .build()
            .add(1, &attributes);
    }
    tracing::info!(
        request_id = %event.request_id,
        model = %event.model,
        actor = %event.actor,
        team = %event.team,
        input_tokens = event.input_tokens,
        output_tokens = event.output_tokens,
        latency_ms = event.latency_ms,
        status = %event.status,
        "model usage recorded"
    );
}

fn slo_status(status: &str) -> &'static str {
    if status == "ok" {
        "ok"
    } else {
        "error"
    }
}

fn stream_status(input_tokens: u64, output_tokens: u64) -> &'static str {
    if input_tokens == 0 && output_tokens == 0 {
        "stream_unmetered"
    } else {
        "ok"
    }
}

fn response_headers(upstream_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in upstream_headers {
        if is_safe_upstream_response_header(name) {
            headers.insert(name.clone(), value.clone());
        }
    }
    headers
}

fn is_safe_upstream_response_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "content-type" | "cache-control" | "x-request-id"
    )
}

fn build_response(
    status: StatusCode,
    headers: HeaderMap,
    body: Body,
    request_id: Uuid,
) -> Response {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    with_request_id(response, request_id)
}

fn error_response(status: StatusCode, code: &str, message: String) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": code,
                "code": code,
                "status": status.as_u16()
            }
        })),
    )
        .into_response()
}

fn auth_error_response(message: String) -> Response {
    if message.contains("too many failed authentication attempts") {
        error_response(StatusCode::TOO_MANY_REQUESTS, "rate_limited", message)
    } else {
        error_response(StatusCode::UNAUTHORIZED, "unauthorized", message)
    }
}

fn request_id_from_headers(headers: &HeaderMap) -> Uuid {
    headers
        .get(request_id_header_name())
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
        .unwrap_or_else(Uuid::new_v4)
}

fn with_request_id(mut response: Response, request_id: Uuid) -> Response {
    response.headers_mut().insert(
        request_id_header_name(),
        HeaderValue::from_str(&request_id.to_string()).expect("uuid is a valid header value"),
    );
    response
}

fn with_model_count(mut response: Response, count: usize) -> Response {
    insert_header_value(
        response.headers_mut(),
        model_count_header_name(),
        &count.to_string(),
    );
    response
}

fn with_chat_metadata(
    mut response: Response,
    model: &str,
    upstream_model: &str,
    quota_decision: &str,
) -> Response {
    insert_header_value(response.headers_mut(), model_header_name(), model);
    insert_header_value(
        response.headers_mut(),
        upstream_model_header_name(),
        upstream_model,
    );
    insert_header_value(
        response.headers_mut(),
        quota_decision_header_name(),
        quota_decision,
    );
    response
}

fn insert_header_value(headers: &mut HeaderMap, name: HeaderName, value: &str) {
    match HeaderValue::from_str(value) {
        Ok(value) => {
            headers.insert(name, value);
        }
        Err(err) => {
            tracing::warn!(header = %name, error = %err, "skipping invalid response metadata header");
        }
    }
}

fn request_id_header_name() -> HeaderName {
    HeaderName::from_static("x-request-id")
}

fn lineage_id_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-lineage-id")
}

fn lineage_ids_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-lineage-ids")
}

fn corpus_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-corpus")
}

fn model_count_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-model-count")
}

fn model_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-model")
}

fn upstream_model_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-upstream-model")
}

fn quota_decision_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-quota-decision")
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

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    #[serde(default)]
    messages: Vec<native::NativeChatMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
}

#[derive(Debug, Clone)]
struct ToolAuditDetail {
    tool_schema_count: u64,
    tool_choice: Value,
    tool_call_count: u64,
}

impl ChatCompletionRequest {
    fn tool_audit_detail(&self) -> ToolAuditDetail {
        ToolAuditDetail {
            tool_schema_count: self
                .tools
                .as_ref()
                .and_then(Value::as_array)
                .map(|tools| tools.len() as u64)
                .unwrap_or(0),
            tool_choice: safe_tool_choice(self.tool_choice.as_ref()),
            tool_call_count: self
                .messages
                .iter()
                .filter_map(|message| message.tool_calls.as_ref())
                .map(tool_call_count)
                .sum(),
        }
    }
}

fn tool_call_count(value: &Value) -> u64 {
    value
        .as_array()
        .map(|calls| calls.len() as u64)
        .unwrap_or(1)
}

fn safe_tool_choice(value: Option<&Value>) -> Value {
    match value {
        None => Value::Null,
        Some(Value::String(choice)) => Value::String(choice.clone()),
        Some(Value::Object(object)) => {
            let mut safe = serde_json::Map::new();
            if let Some(choice_type) = object.get("type").and_then(Value::as_str) {
                safe.insert("type".to_string(), Value::String(choice_type.to_string()));
            }
            if let Some(function_name) = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
            {
                safe.insert(
                    "function_name".to_string(),
                    Value::String(function_name.to_string()),
                );
            }
            Value::Object(safe)
        }
        Some(other) => json!({ "type": other_type_name(other) }),
    }
}

fn other_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn chat_audit_detail(tool: &ToolAuditDetail, mut detail: Value) -> Value {
    if let Some(object) = detail.as_object_mut() {
        object.insert(
            "tool_schema_count".to_string(),
            Value::from(tool.tool_schema_count),
        );
        object.insert("tool_choice".to_string(), tool.tool_choice.clone());
        object.insert(
            "tool_call_count".to_string(),
            Value::from(tool.tool_call_count),
        );
    }
    detail
}

fn chat_route_audit_detail(
    tool: &ToolAuditDetail,
    detail: Value,
    provider: Option<&ResolvedExternalProvider>,
) -> Value {
    let mut detail = chat_audit_detail(tool, detail);
    if let Some(object) = detail.as_object_mut() {
        if let Some(provider) = provider {
            object.insert("provider_routing".to_string(), json!("external"));
            object.insert("provider_id".to_string(), json!(provider.id.as_str()));
            object.insert("provider_kind".to_string(), json!(provider.kind));
            object.insert("provider_api_key_source".to_string(), json!("env"));
        } else {
            object.insert("provider_routing".to_string(), json!("local"));
        }
    }
    detail
}

fn sanitize_native_chat_message(
    mut message: native::NativeChatMessage,
) -> native::NativeChatMessage {
    message.tool_calls = message.tool_calls.map(sanitize_tool_calls);
    message
}

fn sanitize_tool_calls(value: Value) -> Value {
    match value {
        Value::Array(calls) => Value::Array(calls.into_iter().map(sanitize_tool_call).collect()),
        other => sanitize_tool_call(other),
    }
}

fn sanitize_tool_call(value: Value) -> Value {
    let Value::Object(mut object) = value else {
        return value;
    };
    if let Some(Value::Object(function)) = object.get_mut("function") {
        function.remove("arguments");
    }
    Value::Object(object)
}

pub async fn serve(cfg: Config) -> Result<()> {
    // Log hardware tier and recommended model once at startup. Advisory only —
    // operator-selected models are honoured regardless of tier.
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    crate::tier::log_startup_recommendation();
    serve_with_shutdown(cfg, shutdown_signal()).await
}

pub async fn serve_with_shutdown<S>(cfg: Config, shutdown: S) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let storage = Storage::connect_config(&cfg.storage).await?;
    serve_with_storage_and_shutdown(cfg, storage, shutdown).await
}

pub async fn serve_with_storage(cfg: Config, storage: Storage) -> Result<()> {
    serve_with_storage_and_shutdown(cfg, storage, shutdown_signal()).await
}

pub async fn serve_with_storage_and_native_engine<S>(
    cfg: Config,
    storage: Storage,
    native_engine: Arc<dyn native::NativeEngine>,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let mut native_engines = NativeEngineRegistry::new();
    native_engines.insert(native_engine.model_alias().to_string(), native_engine);
    serve_with_storage_and_native_engines(cfg, storage, native_engines, shutdown).await
}

pub async fn serve_with_storage_and_native_engines<S>(
    cfg: Config,
    storage: Storage,
    native_engines: NativeEngineRegistry,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    serve_with_storage_worker_control_native_engine_and_shutdown(
        cfg,
        storage,
        None,
        native_engines,
        shutdown,
    )
    .await
}

pub async fn serve_with_storage_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    serve_with_storage_worker_control_native_engine_and_shutdown(
        cfg,
        storage,
        None,
        NativeEngineRegistry::new(),
        shutdown,
    )
    .await
}

pub async fn serve_with_storage_worker_control_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    serve_with_storage_worker_control_native_engine_and_shutdown(
        cfg,
        storage,
        worker_control,
        NativeEngineRegistry::new(),
        shutdown,
    )
    .await
}

async fn serve_with_storage_worker_control_native_engine_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    native_engines: NativeEngineRegistry,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    if cfg.server.tls.enabled {
        return serve_https_with_storage_worker_control_native_engine_and_shutdown(
            cfg,
            storage,
            worker_control,
            native_engines,
            shutdown,
        )
        .await;
    }

    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    let limits = ServingLimits::from_config(&cfg);
    let draining = Arc::new(AtomicBool::new(false));
    let shutdown = drain_before_shutdown(
        shutdown,
        draining.clone(),
        cfg.server.graceful_drain_seconds,
    );
    let heartbeat = spawn_runtime_heartbeat(&cfg, draining.clone());
    let result = axum::serve(
        listener,
        router_with_worker_control_native_engine_and_drain(
            cfg,
            storage,
            limits,
            worker_control,
            native_engines,
            draining,
        )
        .into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
    .context("serve HTTP API");
    if let Some(handle) = heartbeat {
        handle.abort();
    }
    result
}

async fn serve_https_with_storage_worker_control_native_engine_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    native_engines: NativeEngineRegistry,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let tls_acceptor = build_tls_acceptor(&cfg).await?;
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    let limits = ServingLimits::from_config(&cfg);
    let draining = Arc::new(AtomicBool::new(false));
    let shutdown = drain_before_shutdown(
        shutdown,
        draining.clone(),
        cfg.server.graceful_drain_seconds,
    );
    let app = router_with_worker_control_native_engine_and_drain(
        cfg.clone(),
        storage,
        limits,
        worker_control,
        native_engines,
        draining.clone(),
    );
    let heartbeat = spawn_runtime_heartbeat(&cfg, draining.clone());
    let result = serve_tls(listener, tls_acceptor, app, shutdown)
        .await
        .context("serve HTTPS API");
    if let Some(handle) = heartbeat {
        handle.abort();
    }
    result
}

async fn build_tls_acceptor(cfg: &Config) -> Result<TlsAcceptor> {
    let cert_path = cfg
        .server
        .tls
        .cert_path
        .as_ref()
        .context("server.tls.cert-path is required when server TLS is enabled")?;
    let key_path = cfg
        .server
        .tls
        .key_path
        .as_ref()
        .context("server.tls.key-path is required when server TLS is enabled")?;
    anyhow::ensure!(
        !cfg.server.tls.require_client_cert,
        "server.tls.require-client-cert is not supported without client CA configuration"
    );

    let cert_bytes = tokio::fs::read(cert_path)
        .await
        .with_context(|| format!("read TLS certificate {}", cert_path.display()))?;
    let key_bytes = tokio::fs::read(key_path)
        .await
        .with_context(|| format!("read TLS private key {}", key_path.display()))?;

    use rustls::pki_types::pem::PemObject;
    let certs = rustls::pki_types::CertificateDer::pem_slice_iter(&cert_bytes)
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("parse TLS certificate {}", cert_path.display()))?;
    anyhow::ensure!(
        !certs.is_empty(),
        "TLS certificate {} did not contain any certificates",
        cert_path.display()
    );
    let key = rustls::pki_types::PrivateKeyDer::from_pem_slice(&key_bytes)
        .with_context(|| format!("parse TLS private key {}", key_path.display()))?;
    let tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .context("build TLS server config")?;

    Ok(TlsAcceptor::from(Arc::new(tls_config)))
}

async fn serve_tls<S>(
    listener: tokio::net::TcpListener,
    tls_acceptor: TlsAcceptor,
    app: Router,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let (signal_tx, signal_rx) = tokio::sync::watch::channel(());
    let signal_tx = Arc::new(signal_tx);
    tokio::spawn(async move {
        shutdown.await;
        drop(signal_rx);
    });

    let (close_tx, close_rx) = tokio::sync::watch::channel(());
    loop {
        let (tcp_stream, remote_addr) = tokio::select! {
            result = listener.accept() => result.context("accept TLS connection")?,
            _ = signal_tx.closed() => break,
        };

        let tls_acceptor = tls_acceptor.clone();
        let app = app.clone();
        let signal_tx = Arc::clone(&signal_tx);
        let close_rx = close_rx.clone();

        tokio::spawn(async move {
            let tls_stream = tokio::select! {
                result = timeout(TLS_HANDSHAKE_TIMEOUT, tls_acceptor.accept(tcp_stream)) => {
                    match result {
                        Ok(Ok(stream)) => stream,
                        Ok(Err(err)) => {
                            tracing::debug!(%remote_addr, error = ?err, "TLS handshake failed");
                            drop(close_rx);
                            return;
                        }
                        Err(_) => {
                            tracing::debug!(%remote_addr, "TLS handshake timed out");
                            drop(close_rx);
                            return;
                        }
                    }
                }
                _ = signal_tx.closed() => {
                    drop(close_rx);
                    return;
                }
            };
            let io = TokioIo::new(tls_stream);
            let tower_service = app.map_request(move |mut req: axum::http::Request<Incoming>| {
                req.extensions_mut().insert(ConnectInfo(remote_addr));
                req.map(Body::new)
            });
            let hyper_service = TowerToHyperService::new(tower_service);
            let builder = HyperBuilder::new(TokioExecutor::new());
            let conn = builder.serve_connection_with_upgrades(io, hyper_service);
            pin_mut!(conn);

            let signal_closed = signal_tx.closed().fuse();
            pin_mut!(signal_closed);

            loop {
                tokio::select! {
                    result = conn.as_mut() => {
                        if let Err(err) = result {
                            tracing::debug!(%remote_addr, error = ?err, "failed to serve TLS connection");
                        }
                        break;
                    }
                    _ = &mut signal_closed => {
                        conn.as_mut().graceful_shutdown();
                    }
                }
            }

            drop(close_rx);
        });
    }

    drop(close_rx);
    drop(listener);
    close_tx.closed().await;

    Ok(())
}

async fn drain_before_shutdown<S>(shutdown: S, draining: Arc<AtomicBool>, drain_seconds: u64)
where
    S: Future<Output = ()>,
{
    shutdown.await;
    draining.store(true, Ordering::SeqCst);
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Metric,
        TelemetryEventName::RuntimeHeartbeat,
        Utc::now(),
        BTreeMap::from([
            ("llmctl.server.draining".to_string(), json!(true)),
            (
                "llmctl.server.graceful_drain_seconds".to_string(),
                json!(drain_seconds),
            ),
        ]),
    ));
    if drain_seconds > 0 {
        tokio::time::sleep(Duration::from_secs(drain_seconds)).await;
    }
}

fn spawn_runtime_heartbeat(cfg: &Config, draining: Arc<AtomicBool>) -> Option<JoinHandle<()>> {
    let interval_seconds = cfg.runtime.heartbeat_interval_seconds;
    if interval_seconds == 0 {
        return None;
    }

    let cfg = cfg.clone();
    Some(tokio::spawn(async move {
        emit_runtime_heartbeat(&cfg, draining.load(Ordering::SeqCst));
        loop {
            tokio::time::sleep(Duration::from_secs(interval_seconds)).await;
            emit_runtime_heartbeat(&cfg, draining.load(Ordering::SeqCst));
        }
    }))
}

fn emit_runtime_heartbeat(cfg: &Config, draining: bool) {
    let heartbeat = native::heartbeat_from_config(cfg);
    let mut attributes = heartbeat.safe_telemetry_attributes();
    attributes.insert("llmctl.server.draining".to_string(), json!(draining));
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Metric,
        TelemetryEventName::RuntimeHeartbeat,
        Utc::now(),
        attributes,
    ));
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!(%error, "failed to install Ctrl-C signal handler");
        }
    };

    #[cfg(unix)]
    {
        let terminate = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(error) => {
                    tracing::warn!(%error, "failed to install terminate signal handler");
                    std::future::pending::<()>().await;
                }
            }
        };

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    }

    #[cfg(not(unix))]
    ctrl_c.await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ApiKeyConfig, ClusterNodeConfig, SecurityConfig, ServerConfig};

    #[test]
    fn normalizes_upstream_urls() {
        assert_eq!(
            normalize_upstream("http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            normalize_upstream("127.0.0.1:8080"),
            "http://127.0.0.1:8080"
        );
    }

    #[test]
    fn model_capabilities_are_present_on_every_model_entry() {
        let mc = ModelConfig {
            alias: "qwen3-14b-q4_k_m".into(),
            path: std::path::PathBuf::from("/tmp/none.gguf"),
            role: "chat".into(),
            family: Some("qwen3".into()),
            weight: 1,
        };
        let snap = CapabilitySnapshot::current();
        let obj = build_model_object(&mc, snap);
        let json = serde_json::to_value(&obj).unwrap();

        // Standard OpenAI Models fields preserved.
        assert_eq!(json["id"], "qwen3-14b-q4_k_m");
        assert_eq!(json["object"], "model");
        assert_eq!(json["owned_by"], "rs-llmctl");

        // New capability fields all present.
        let caps = &json["capabilities"];
        assert!(caps.is_object(), "capabilities must be an object");
        assert!(caps["context_window"].is_number());
        assert!(caps["tool_protocol"].is_string());
        assert!(caps["model_size_b"].is_number());
        assert!(caps["gpu_backend"].is_string());
        assert!(caps["tier"].is_string());

        // Qwen3 family advertises its native tool protocol.
        assert_eq!(caps["tool_protocol"], "qwen3-native");
        // Alias contained "14b" → size parser extracts 14.0.
        assert_eq!(caps["model_size_b"], 14.0);
        // Qwen3 default context is 128k.
        assert_eq!(caps["context_window"], 131_072);
    }

    #[test]
    fn unknown_family_still_renders_capabilities_with_defaults() {
        let mc = ModelConfig {
            alias: "experimental-model".into(),
            path: std::path::PathBuf::from("/tmp/none.gguf"),
            role: "chat".into(),
            family: None,
            weight: 1,
        };
        let snap = CapabilitySnapshot::current();
        let obj = build_model_object(&mc, snap);
        let json = serde_json::to_value(&obj).unwrap();
        let caps = &json["capabilities"];
        // Unknown family → tool_protocol = "none", context_window = 0, size = 0.0.
        assert_eq!(caps["tool_protocol"], "none");
        assert_eq!(caps["context_window"], 0);
        assert_eq!(caps["model_size_b"], 0.0);
        // gpu_backend and tier are always populated, even for unknown families.
        assert!(caps["gpu_backend"].is_string());
        assert!(caps["tier"].is_string());
    }

    #[test]
    fn alias_size_parser_handles_common_shapes() {
        assert_eq!(parse_model_size_b_from_alias("qwen3-14b-q4_k_m"), 14.0);
        assert_eq!(parse_model_size_b_from_alias("Qwen3-Coder-30B-A3B"), 30.0);
        assert_eq!(parse_model_size_b_from_alias("llama-3.1-8B-instruct"), 8.0);
        assert_eq!(parse_model_size_b_from_alias("phi-3.5-mini-3.8b"), 3.8);
        // No size suffix.
        assert_eq!(parse_model_size_b_from_alias("custom"), 0.0);
        // Implausible numbers are rejected.
        assert_eq!(parse_model_size_b_from_alias("0b"), 0.0);
        assert_eq!(parse_model_size_b_from_alias("99999b"), 0.0);
    }

    #[test]
    fn backward_compat_legacy_openai_client_ignores_capabilities() {
        // A strict OpenAI SDK client deserialises into a struct that only knows
        // about id/object/owned_by. Confirm our payload still validates against
        // that shape (additive — capabilities is an extra field).
        #[derive(serde::Deserialize)]
        struct LegacyModelObject {
            id: String,
            object: String,
            owned_by: String,
        }
        let mc = ModelConfig {
            alias: "qwen3-8b".into(),
            path: std::path::PathBuf::from("/tmp/none.gguf"),
            role: "chat".into(),
            family: Some("qwen3".into()),
            weight: 1,
        };
        let json =
            serde_json::to_value(build_model_object(&mc, CapabilitySnapshot::current())).unwrap();
        let legacy: LegacyModelObject = serde_json::from_value(json).unwrap();
        assert_eq!(legacy.id, "qwen3-8b");
        assert_eq!(legacy.object, "model");
        assert_eq!(legacy.owned_by, "rs-llmctl");
    }

    #[test]
    fn tool_format_openai_for_qwen3_family() {
        let mc = ModelConfig {
            alias: "qwen3-14b".into(),
            path: std::path::PathBuf::from("/tmp/none.gguf"),
            role: "chat".into(),
            family: Some("qwen3".into()),
            weight: 1,
        };
        let json =
            serde_json::to_value(build_model_object(&mc, CapabilitySnapshot::current())).unwrap();
        assert_eq!(json["capabilities"]["tool_format"], "openai");
    }

    #[test]
    fn tool_format_xml_for_devstral_family() {
        let mc = ModelConfig {
            alias: "devstral-small-2505".into(),
            path: std::path::PathBuf::from("/tmp/none.gguf"),
            role: "chat".into(),
            family: Some("mistral".into()),
            weight: 1,
        };
        let json =
            serde_json::to_value(build_model_object(&mc, CapabilitySnapshot::current())).unwrap();
        assert_eq!(json["capabilities"]["tool_format"], "xml");
    }

    #[test]
    fn tool_format_openai_for_gemma4_family() {
        let mc = ModelConfig {
            alias: "gemma4-12b".into(),
            path: std::path::PathBuf::from("/tmp/none.gguf"),
            role: "chat".into(),
            family: Some("gemma4".into()),
            weight: 1,
        };
        let json =
            serde_json::to_value(build_model_object(&mc, CapabilitySnapshot::current())).unwrap();
        assert_eq!(json["capabilities"]["tool_format"], "openai");
    }

    #[test]
    fn readiness_status_reports_draining_as_not_ready() {
        let status = readiness_status_for(&Config::default(), true, true);
        assert_eq!(status["status"], "draining");
        assert_eq!(status["draining"], true);
    }

    #[test]
    fn usage_span_attributes_align_with_gen_ai_semantic_conventions() {
        let event = UsageEvent {
            id: Uuid::new_v4(),
            request_id: Uuid::new_v4(),
            at: Utc::now(),
            model: "llama".to_string(),
            actor: "alice".to_string(),
            team: "platform".to_string(),
            input_tokens: 11,
            output_tokens: 13,
            latency_ms: 42,
            status: "ok".to_string(),
        };

        let attrs = usage_span_attributes(&event, "estimated", "openai");

        assert_eq!(attrs["gen_ai.system"], json!("openai"));
        assert_eq!(attrs["gen_ai.operation.name"], json!("chat"));
        assert_eq!(attrs["gen_ai.request.model"], json!("llama"));
        assert_eq!(attrs["gen_ai.response.model"], json!("llama"));
        assert_eq!(attrs["gen_ai.usage.input_tokens"], json!(11));
        assert_eq!(attrs["gen_ai.usage.output_tokens"], json!(13));
        // Existing llmctl-prefixed attributes must be preserved alongside the
        // gen_ai.* alignment additions.
        assert_eq!(attrs["llmctl.model"], json!("llama"));
        assert_eq!(attrs["llmctl.token_accounting.mode"], json!("estimated"));
    }

    #[test]
    fn webhook_payload_carries_usage_and_accounting_metadata() {
        let event = UsageEvent {
            id: Uuid::new_v4(),
            request_id: Uuid::new_v4(),
            at: Utc::now(),
            model: "llama".to_string(),
            actor: "alice".to_string(),
            team: "platform".to_string(),
            input_tokens: 11,
            output_tokens: 13,
            latency_ms: 42,
            status: "ok".to_string(),
        };

        let payload = webhook_payload(&event, "estimated");

        assert_eq!(payload["type"], json!("llmctl.usage"));
        assert_eq!(payload["request_id"], json!(event.request_id.to_string()));
        assert_eq!(payload["model"], json!("llama"));
        assert_eq!(payload["actor"], json!("alice"));
        assert_eq!(payload["team"], json!("platform"));
        assert_eq!(payload["input_tokens"], json!(11));
        assert_eq!(payload["output_tokens"], json!(13));
        assert_eq!(payload["latency_ms"], json!(42));
        assert_eq!(payload["status"], json!("ok"));
        assert_eq!(payload["token_accounting_mode"], json!("estimated"));
    }

    #[test]
    fn playground_html_wires_models_and_chat_endpoints_with_api_key_field() {
        let html = playground_html();
        assert!(html.contains("<title>"));
        assert!(html.contains("/v1/models"));
        assert!(html.contains("/v1/chat/completions"));
        assert!(html.to_lowercase().contains("api key") || html.to_lowercase().contains("api-key"));
        assert!(html.contains("<script"));
    }

    #[test]
    fn auth_failure_limiter_blocks_after_configured_window_limit() {
        let limiter = AuthFailureLimiter::default();
        assert!(!limiter.is_limited("bad-token", 2));
        limiter.record_failure("bad-token", 2);
        assert!(!limiter.is_limited("bad-token", 2));
        limiter.record_failure("bad-token", 2);
        assert!(limiter.is_limited("bad-token", 2));
        limiter.record_success("bad-token");
        assert!(!limiter.is_limited("bad-token", 2));
    }

    #[test]
    fn trusted_proxy_forwarded_chain_uses_rightmost_untrusted_client() {
        let mut cfg = Config::default();
        cfg.security.trusted_proxies = vec!["10.0.0.0/8".to_string()];
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            HeaderValue::from_static("203.0.113.10, 198.51.100.20, 10.0.0.5"),
        );

        assert_eq!(
            forwarded_client_ip(&cfg, &headers).as_deref(),
            Some("198.51.100.20")
        );
    }

    #[test]
    fn trusted_proxy_forwarded_chain_reads_duplicate_headers() {
        let mut cfg = Config::default();
        cfg.security.trusted_proxies = vec!["10.0.0.0/8".to_string()];
        let mut headers = HeaderMap::new();
        headers.append("x-forwarded-for", HeaderValue::from_static("203.0.113.10"));
        headers.append(
            "x-forwarded-for",
            HeaderValue::from_static("198.51.100.20, 10.0.0.5"),
        );

        assert_eq!(
            forwarded_client_ip(&cfg, &headers).as_deref(),
            Some("198.51.100.20")
        );
    }

    #[test]
    fn trusted_proxy_matching_rejects_wildcard_runtime_entries() {
        let mut cfg = Config::default();
        cfg.security.trusted_proxies = vec!["*".to_string()];

        assert!(!is_trusted_proxy(
            &cfg,
            "203.0.113.1".parse::<IpAddr>().unwrap()
        ));
        cfg.security.trusted_proxies = vec!["0.0.0.0/0".to_string(), "::/0".to_string()];
        assert!(!is_trusted_proxy(
            &cfg,
            "203.0.113.1".parse::<IpAddr>().unwrap()
        ));
        assert!(!is_trusted_proxy(
            &cfg,
            "2001:db8::1".parse::<IpAddr>().unwrap()
        ));
    }

    #[test]
    fn trusted_proxy_matching_normalizes_ipv4_mapped_ipv6() {
        let mut cfg = Config::default();
        cfg.security.trusted_proxies = vec!["10.0.0.0/8".to_string()];

        // A dual-stack listener may report an IPv4 peer as ::ffff:10.x.x.x.
        assert!(is_trusted_proxy(
            &cfg,
            "::ffff:10.1.2.3".parse::<IpAddr>().unwrap()
        ));
        assert!(is_trusted_proxy(
            &cfg,
            "10.1.2.3".parse::<IpAddr>().unwrap()
        ));
        assert!(!is_trusted_proxy(
            &cfg,
            "::ffff:192.0.2.1".parse::<IpAddr>().unwrap()
        ));

        // Exact-match form should normalize too.
        cfg.security.trusted_proxies = vec!["10.1.2.3".to_string()];
        assert!(is_trusted_proxy(
            &cfg,
            "::ffff:10.1.2.3".parse::<IpAddr>().unwrap()
        ));
    }

    #[test]
    fn trusted_proxy_matching_rejects_out_of_range_prefix() {
        let mut cfg = Config::default();
        cfg.security.trusted_proxies = vec!["10.0.0.0/33".to_string()];
        assert!(!is_trusted_proxy(
            &cfg,
            "10.0.0.1".parse::<IpAddr>().unwrap()
        ));

        cfg.security.trusted_proxies = vec!["2001:db8::/129".to_string()];
        assert!(!is_trusted_proxy(
            &cfg,
            "2001:db8::1".parse::<IpAddr>().unwrap()
        ));
    }

    #[test]
    fn circuit_breaker_opens_after_threshold_and_half_opens_after_reset() {
        let breakers = CircuitBreakers::default();
        let upstream = "http://127.0.0.1:18765";
        assert!(breakers.allow_request(upstream, Duration::from_secs(30)));
        breakers.record_failure(upstream, 2);
        assert!(breakers.allow_request(upstream, Duration::from_secs(30)));
        breakers.record_failure(upstream, 2);
        assert!(!breakers.allow_request(upstream, Duration::from_secs(30)));
        assert!(breakers.allow_request(upstream, Duration::from_secs(0)));
        assert!(!breakers.allow_request(upstream, Duration::from_secs(0)));
        breakers.record_success(upstream);
        assert!(breakers.allow_request(upstream, Duration::from_secs(30)));
    }

    #[test]
    fn candle_native_routes_to_planned_worker_upstream_metadata() {
        let cfg = Config {
            models: vec![ModelConfig {
                alias: "llama".to_string(),
                path: "/models/llama.gguf".into(),
                role: "chat".to_string(),
                family: Some("qwen3".to_string()),
                weight: 1,
            }],
            ..Default::default()
        };

        assert_eq!(
            serving_upstreams(&cfg).get("llama").map(String::as_str),
            Some("http://127.0.0.1:18765")
        );
    }

    #[test]
    fn request_id_from_headers_accepts_valid_uuid() {
        let request_id = Uuid::new_v4();
        let mut headers = HeaderMap::new();
        headers.insert(
            request_id_header_name(),
            HeaderValue::from_str(&request_id.to_string()).unwrap(),
        );

        assert_eq!(request_id_from_headers(&headers), request_id);
    }

    #[test]
    fn request_id_from_headers_generates_uuid_when_missing_or_invalid() {
        let missing = request_id_from_headers(&HeaderMap::new());
        assert_ne!(missing, Uuid::nil());

        let mut headers = HeaderMap::new();
        headers.insert(
            request_id_header_name(),
            HeaderValue::from_static("not-a-uuid"),
        );
        let invalid = request_id_from_headers(&headers);
        assert_ne!(invalid, Uuid::nil());
        assert_ne!(invalid, missing);
    }

    #[test]
    fn serving_limits_default_to_internal_admission_limit_without_quota_config() {
        let cfg = Config::default();

        let limits = ServingLimits::from_config(&cfg);

        assert_eq!(limits.max_in_flight, DEFAULT_MAX_IN_FLIGHT);
        assert_eq!(limits.upstream_timeout(), DEFAULT_UPSTREAM_TIMEOUT);
    }

    #[test]
    fn serving_limits_use_configured_quota_concurrency_when_available() {
        let cfg = Config {
            quotas: vec![
                crate::config::QuotaConfig {
                    subject: "alice".to_string(),
                    team: "".to_string(),
                    requests_per_minute: 10,
                    tokens_per_day: 100,
                    max_concurrency: 2,
                    allowed_models: vec!["llama".to_string()],
                },
                crate::config::QuotaConfig {
                    subject: "bob".to_string(),
                    team: "".to_string(),
                    requests_per_minute: 10,
                    tokens_per_day: 100,
                    max_concurrency: 3,
                    allowed_models: vec!["llama".to_string()],
                },
            ],
            ..Default::default()
        };

        let limits = ServingLimits::from_config(&cfg);

        assert_eq!(limits.max_in_flight, 5);
    }

    #[test]
    fn quota_admission_scopes_include_subject_and_team_limits() {
        let cfg = Config {
            quotas: vec![
                crate::config::QuotaConfig {
                    subject: "alice".to_string(),
                    team: "platform".to_string(),
                    requests_per_minute: 10,
                    tokens_per_day: 100,
                    max_concurrency: 2,
                    allowed_models: vec!["llama".to_string()],
                },
                crate::config::QuotaConfig {
                    subject: "team-default".to_string(),
                    team: "platform".to_string(),
                    requests_per_minute: 10,
                    tokens_per_day: 100,
                    max_concurrency: 1,
                    allowed_models: vec!["llama".to_string()],
                },
            ],
            ..Default::default()
        };
        let principal = Principal {
            subject: "alice".to_string(),
            team: "platform".to_string(),
            scopes: vec!["chat".to_string()],
            key_id: Some("alice-key".to_string()),
            key_owner: None,
            key_purpose: None,
            key_status: Some("active".to_string()),
        };

        assert_eq!(
            quota_admission_scopes(&cfg, &principal),
            vec![
                ("subject:alice".to_string(), 2),
                ("team:platform".to_string(), 1)
            ]
        );
    }

    #[test]
    fn admission_controller_rejects_when_in_flight_limit_is_full() {
        let controller = AdmissionController::new(1);
        let first = controller.try_acquire_for(None).expect("first permit");

        assert_eq!(
            controller.try_acquire_for(None).unwrap_err(),
            AdmissionError::Busy
        );

        drop(first);
        assert!(controller.try_acquire_for(None).is_ok());
    }

    #[test]
    fn admission_controller_applies_all_scoped_limits() {
        let controller = AdmissionController::new(8);
        let first = controller
            .try_acquire_for_all(vec![
                ("subject:alice".to_string(), 2),
                ("team:platform".to_string(), 1),
            ])
            .expect("scoped permit");

        assert_eq!(
            controller
                .try_acquire_for_all(vec![
                    ("subject:alice".to_string(), 2),
                    ("team:platform".to_string(), 1),
                ])
                .unwrap_err(),
            AdmissionError::Busy
        );
        assert!(controller
            .try_acquire_for_all(vec![
                ("subject:alice".to_string(), 2),
                ("team:research".to_string(), 1),
            ])
            .is_ok());

        drop(first);
        assert!(controller
            .try_acquire_for_all(vec![
                ("subject:alice".to_string(), 2),
                ("team:platform".to_string(), 1),
            ])
            .is_ok());
    }

    #[test]
    fn admission_controller_applies_scoped_limits() {
        let controller = AdmissionController::new(8);
        let first = controller
            .try_acquire_for(Some(("subject:alice".to_string(), 1)))
            .expect("scoped permit");

        assert_eq!(
            controller
                .try_acquire_for(Some(("subject:alice".to_string(), 1)))
                .unwrap_err(),
            AdmissionError::Busy
        );
        assert!(controller
            .try_acquire_for(Some(("subject:bob".to_string(), 1)))
            .is_ok());

        drop(first);
        assert!(controller
            .try_acquire_for(Some(("subject:alice".to_string(), 1)))
            .is_ok());
    }

    #[test]
    fn bearer_auth_uses_configured_sha256_keys() {
        let token = "secret";
        let cfg = Config {
            server: ServerConfig::default(),
            security: SecurityConfig {
                production: false,
                require_auth: true,
                bind_external: false,
                api_keys: vec![ApiKeyConfig {
                    id: "dev".to_string(),
                    sha256: hex::encode(Sha256::digest(token.as_bytes())),
                    subject: "alice".to_string(),
                    team: "platform".to_string(),
                    scopes: vec!["chat".to_string()],
                    created_at: None,
                    expires_at: None,
                    rotated_at: None,
                    owner: None,
                    purpose: None,
                    last_four: None,
                    fingerprint: None,
                    status: "active".to_string(),
                }],
                ..SecurityConfig::default()
            },
            ..Default::default()
        };
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer secret"));

        let principal = authenticate(&cfg, &headers).expect("auth should pass");
        assert_eq!(principal.subject, "alice");
        assert_eq!(principal.team, "platform");
        assert!(principal.has_scope("chat"));
    }

    #[test]
    fn lists_configured_models_in_deterministic_order() {
        let cfg = config_with_models(
            Mode::HotSwap,
            vec![
                model("zeta", 1, "chat"),
                model("alpha", 1, "chat"),
                model("middle", 1, "chat"),
            ],
        );

        let aliases: Vec<_> = routed_models(&cfg)
            .iter()
            .map(|model| model.alias.as_str())
            .collect();

        assert_eq!(aliases, vec!["alpha", "middle", "zeta"]);
    }

    #[test]
    fn single_mode_routes_to_the_only_configured_model() {
        let cfg = config_with_models(Mode::Single, vec![model("llama", 0, "chat")]);

        let resolved = resolve_model_route(&cfg, "llama", Uuid::nil()).unwrap();

        assert_eq!(resolved.requested_alias, "llama");
        assert_eq!(resolved.upstream_alias, "llama");
    }

    #[test]
    fn swap_modes_validate_requested_aliases() {
        for mode in [Mode::ColdSwap, Mode::HotSwap] {
            let cfg = config_with_models(
                mode,
                vec![model("alpha", 0, "chat"), model("beta", 0, "chat")],
            );

            assert_eq!(
                resolve_model_route(&cfg, "beta", Uuid::nil())
                    .unwrap()
                    .upstream_alias,
                "beta"
            );
            assert!(matches!(
                resolve_model_route(&cfg, "missing", Uuid::nil()),
                Err(ModelRouteError::UnknownAlias(alias)) if alias == "missing"
            ));
        }
    }

    #[test]
    fn weighted_mode_selects_model_by_request_id_slot() {
        let cfg = config_with_models(
            Mode::Weighted,
            vec![
                model("light", 1, "chat"),
                model("heavy-b", 50, "chat"),
                model("heavy-a", 50, "chat"),
            ],
        );

        let resolved = resolve_model_route(&cfg, "light", Uuid::from_u128(50)).unwrap();

        assert_eq!(resolved.requested_alias, "light");
        assert_eq!(resolved.upstream_alias, "heavy-b");
    }

    #[test]
    fn cluster_node_routes_only_locally_placed_models() {
        let mut cfg = config_with_models(
            Mode::Weighted,
            vec![
                model("thinking", 100, "thinking"),
                model("coding", 100, "coding"),
            ],
        );
        cfg.cluster.node_id = "node-a".to_string();
        cfg.cluster.nodes = vec![
            ClusterNodeConfig {
                id: "node-a".to_string(),
                base_url: "http://node-a:8765".to_string(),
                roles: vec!["thinking".to_string()],
                model_aliases: Vec::new(),
            },
            ClusterNodeConfig {
                id: "node-b".to_string(),
                base_url: "http://node-b:8765".to_string(),
                roles: vec!["coding".to_string()],
                model_aliases: Vec::new(),
            },
        ];

        let aliases = routed_models(&cfg)
            .into_iter()
            .map(|model| model.alias.as_str())
            .collect::<Vec<_>>();
        assert_eq!(aliases, vec!["thinking"]);
        assert!(matches!(
            resolve_model_route(&cfg, "coding", Uuid::nil()),
            Err(ModelRouteError::UnknownAlias(alias)) if alias == "coding"
        ));
        assert_eq!(
            resolve_model_route(&cfg, "thinking", Uuid::from_u128(1))
                .unwrap()
                .upstream_alias,
            "thinking"
        );
    }

    #[test]
    fn readiness_counts_only_active_routed_models() {
        let cfg = config_with_models(
            Mode::Weighted,
            vec![model("active", 1, "chat"), model("inactive", 0, "chat")],
        );

        let status = readiness_status_for(&cfg, true, false);

        assert_eq!(status["status"], "ready");
        assert_eq!(status["models"]["configured"], 1);
        assert_eq!(status["models"]["aliases"], json!(["active"]));
    }

    #[tokio::test]
    async fn serve_with_storage_and_shutdown_exits_when_shutdown_future_completes() {
        let storage = Storage::in_memory().await.expect("storage");
        let mut cfg = Config::default();
        cfg.server.port = 0;
        cfg.server.graceful_drain_seconds = 0;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let server = tokio::spawn(serve_with_storage_and_shutdown(cfg, storage, async move {
            let _ = shutdown_rx.await;
        }));

        shutdown_tx.send(()).expect("send shutdown signal");
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), server)
            .await
            .expect("server exits after shutdown")
            .expect("server task joins");

        result.expect("server result");
    }

    #[tokio::test]
    async fn tls_enabled_without_cert_or_key_fails_before_serving() {
        let storage = Storage::in_memory().await.expect("storage");
        let mut cfg = Config::default();
        cfg.server.port = 0;
        cfg.server.tls.enabled = true;
        let (_shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let err = serve_with_storage_and_shutdown(cfg, storage, async move {
            let _ = shutdown_rx.await;
        })
        .await
        .expect_err("missing cert/key should fail before serving");

        assert!(
            err.to_string().contains("server.tls.cert-path"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fallback_mode_routes_zero_weight_models_to_first_positive_weight_model() {
        let cfg = config_with_models(
            Mode::Fallback,
            vec![
                model("primary", 100, "chat"),
                model("backup", 0, "chat"),
                model("tertiary", 10, "chat"),
            ],
        );

        assert_eq!(
            resolve_model_route(&cfg, "backup", Uuid::from_u128(1))
                .unwrap()
                .upstream_alias,
            "primary"
        );
        let tertiary = resolve_model_route(&cfg, "tertiary", Uuid::from_u128(120)).unwrap();
        assert_eq!(tertiary.upstream_alias, "tertiary");
        assert_eq!(tertiary.fallback_aliases, vec!["primary", "backup"]);
    }

    #[test]
    fn rewrites_chat_completion_model_for_upstream_route() {
        let body = br#"{"model":"light","messages":[]}"#;
        let route = ResolvedModelRoute {
            requested_alias: "light".to_string(),
            upstream_alias: "heavy".to_string(),
            fallback_aliases: Vec::new(),
            external_provider: None,
        };

        let rewritten = rewrite_chat_model(body, &route).unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(value["model"], "heavy");
        assert_eq!(value["messages"], json!([]));
    }

    fn config_with_models(mode: Mode, models: Vec<ModelConfig>) -> Config {
        Config {
            mode,
            models,
            ..Default::default()
        }
    }

    fn model(alias: &str, weight: u32, role: &str) -> ModelConfig {
        ModelConfig {
            alias: alias.to_string(),
            path: format!("/models/{alias}.gguf").into(),
            role: role.to_string(),
            family: Some("qwen3".to_string()),
            weight,
        }
    }

    fn make_chat_request(messages: Vec<native::NativeChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages,
            temperature: Some(0.7),
            max_tokens: Some(512),
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
        }
    }

    #[test]
    fn gen_ai_params_extracts_system_and_user_messages() {
        use serde_json::json;
        let request = make_chat_request(vec![
            native::NativeChatMessage {
                role: "system".to_string(),
                content: Some(json!("You are helpful.")),
                tool_calls: None,
                tool_call_id: None,
            },
            native::NativeChatMessage {
                role: "user".to_string(),
                content: Some(json!("Hello")),
                tool_calls: None,
                tool_call_id: None,
            },
        ]);
        let params = gen_ai_params_from_request(&request);
        assert_eq!(params.system_message.as_deref(), Some("You are helpful."));
        assert_eq!(params.user_message.as_deref(), Some("Hello"));
        assert_eq!(params.max_tokens, Some(512));
        assert!((params.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn gen_ai_params_truncates_long_messages_to_1000_chars() {
        use serde_json::json;
        let long_text: String = "x".repeat(2000);
        let request = make_chat_request(vec![native::NativeChatMessage {
            role: "user".to_string(),
            content: Some(json!(long_text)),
            tool_calls: None,
            tool_call_id: None,
        }]);
        let params = gen_ai_params_from_request(&request);
        assert_eq!(params.user_message.as_ref().map(|s| s.len()), Some(1000));
    }

    #[test]
    fn gen_ai_params_finds_last_user_message() {
        use serde_json::json;
        let request = make_chat_request(vec![
            native::NativeChatMessage {
                role: "user".to_string(),
                content: Some(json!("first")),
                tool_calls: None,
                tool_call_id: None,
            },
            native::NativeChatMessage {
                role: "assistant".to_string(),
                content: Some(json!("response")),
                tool_calls: None,
                tool_call_id: None,
            },
            native::NativeChatMessage {
                role: "user".to_string(),
                content: Some(json!("last")),
                tool_calls: None,
                tool_call_id: None,
            },
        ]);
        let params = gen_ai_params_from_request(&request);
        assert_eq!(params.user_message.as_deref(), Some("last"));
    }

    #[test]
    fn gen_ai_params_returns_none_for_missing_roles() {
        use serde_json::json;
        let request = make_chat_request(vec![native::NativeChatMessage {
            role: "assistant".to_string(),
            content: Some(json!("I can help.")),
            tool_calls: None,
            tool_call_id: None,
        }]);
        let params = gen_ai_params_from_request(&request);
        assert!(params.system_message.is_none());
        assert!(params.user_message.is_none());
    }
}

#[cfg(test)]
mod genai_semconv_tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_usage_event(input_tokens: u64, output_tokens: u64) -> UsageEvent {
        UsageEvent {
            id: Uuid::new_v4(),
            request_id: Uuid::new_v4(),
            at: Utc::now(),
            model: "test-model".to_string(),
            actor: "test-actor".to_string(),
            team: "test-team".to_string(),
            input_tokens,
            output_tokens,
            latency_ms: 42,
            status: "ok".to_string(),
        }
    }

    #[test]
    fn usage_span_attributes_include_genai_semconv_and_llmctl_attributes() {
        let event = make_usage_event(100, 50);
        let attrs = usage_span_attributes(&event, "exact", "vertex_ai");

        // GenAI SemConv attributes present
        assert_eq!(
            attrs.get("gen_ai.system").and_then(|v| v.as_str()),
            Some("vertex_ai")
        );
        assert_eq!(
            attrs.get("gen_ai.request.model").and_then(|v| v.as_str()),
            Some("test-model")
        );
        assert_eq!(
            attrs
                .get("gen_ai.usage.input_tokens")
                .and_then(|v| v.as_u64()),
            Some(100)
        );
        assert_eq!(
            attrs
                .get("gen_ai.usage.output_tokens")
                .and_then(|v| v.as_u64()),
            Some(50)
        );

        // Existing llmctl attributes preserved (no regression)
        assert!(attrs.contains_key("llmctl.model"));
        assert!(attrs.contains_key("llmctl.request_id"));
        assert!(attrs.contains_key("llmctl.latency_ms"));
    }

    #[test]
    fn usage_span_attributes_has_no_cost_usd_when_pricing_unknown() {
        let event = make_usage_event(100, 50);
        let attrs = usage_span_attributes(&event, "exact", "vertex_ai");
        // UsageEvent has no cost field; attribute must be absent, not zero
        assert!(!attrs.contains_key("gen_ai.usage.cost_usd"));
    }
}
