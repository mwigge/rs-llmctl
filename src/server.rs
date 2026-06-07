use crate::audit::{AuditEvent, UsageEvent};
use crate::config::{
    is_external_host, ApiKeyConfig, Config, ExternalProviderKind, Mode, ModelConfig,
    NativeEmbeddingMode,
};
use crate::guardrails;
use crate::native;
use crate::observability::{
    emit_runtime_telemetry, RuntimeTelemetryEvent, TelemetryEventName, TelemetrySignal,
};
use crate::quota::{check_quota, matching_quota_policies, quota_is_subject_scoped, Principal};
use crate::rag::{lexical_search, SearchDocument};
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
use opentelemetry::KeyValue;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use subtle::ConstantTimeEq;
use tokio::sync::{Mutex as AsyncMutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_rustls::TlsAcceptor;
use tower::ServiceExt;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::{DefaultOnResponse, TraceLayer};
use uuid::Uuid;

const DEFAULT_MAX_IN_FLIGHT: usize = 128;
#[cfg(test)]
const DEFAULT_UPSTREAM_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_SLO_LATENCY_MS: u64 = 10_000;
const MAX_SSE_USAGE_BUFFER_BYTES: usize = 1024 * 1024;
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
        .route("/v1/local/search", post(local_search))
        .route("/v1/local/recommendations", post(local_recommendations))
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

#[derive(Clone)]
struct AdmissionController {
    global: Arc<Semaphore>,
    scoped: Arc<Mutex<BTreeMap<String, Arc<Semaphore>>>>,
}

impl AdmissionController {
    fn new(max_in_flight: usize) -> Self {
        Self {
            global: Arc::new(Semaphore::new(max_in_flight.max(1))),
            scoped: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    #[cfg(test)]
    fn try_acquire_for(
        &self,
        scope: Option<(String, usize)>,
    ) -> std::result::Result<AdmissionPermit, AdmissionError> {
        self.try_acquire_for_all(scope.into_iter().collect())
    }

    fn try_acquire_for_all(
        &self,
        scopes: Vec<(String, usize)>,
    ) -> std::result::Result<AdmissionPermit, AdmissionError> {
        let global = self
            .global
            .clone()
            .try_acquire_owned()
            .map_err(|_| AdmissionError::Busy)?;

        let mut scoped_permits = Vec::with_capacity(scopes.len());
        for (scope, limit) in scopes {
            let permit = {
                let semaphore = {
                    let mut scoped = self.scoped.lock().map_err(|_| AdmissionError::Busy)?;
                    scoped
                        .entry(scope)
                        .or_insert_with(|| Arc::new(Semaphore::new(limit.max(1))))
                        .clone()
                };
                semaphore
                    .try_acquire_owned()
                    .map_err(|_| AdmissionError::Busy)?
            };
            scoped_permits.push(permit);
        }

        Ok(AdmissionPermit {
            _global: global,
            _scoped: scoped_permits,
        })
    }
}

struct AdmissionPermit {
    _global: OwnedSemaphorePermit,
    _scoped: Vec<OwnedSemaphorePermit>,
}

impl std::fmt::Debug for AdmissionPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmissionPermit").finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdmissionError {
    Busy,
}

#[derive(Debug, Clone, Default)]
struct CircuitBreakers {
    states: Arc<Mutex<BTreeMap<String, CircuitBreakerState>>>,
}

#[derive(Debug, Clone)]
struct CircuitBreakerState {
    consecutive_failures: u32,
    opened_at: Option<Instant>,
    half_open_probe_in_flight: bool,
}

impl CircuitBreakers {
    fn allow_request(&self, upstream: &str, reset_after: Duration) -> bool {
        let mut states = self.states.lock().expect("circuit breaker mutex poisoned");
        let Some(state) = states.get_mut(upstream) else {
            return true;
        };
        let Some(opened_at) = state.opened_at else {
            return true;
        };
        if opened_at.elapsed() >= reset_after {
            if state.half_open_probe_in_flight {
                record_circuit_breaker_state(
                    upstream,
                    "half_open_busy",
                    state.consecutive_failures,
                );
                false
            } else {
                state.half_open_probe_in_flight = true;
                record_circuit_breaker_state(upstream, "half_open", state.consecutive_failures);
                true
            }
        } else {
            record_circuit_breaker_state(upstream, "open", state.consecutive_failures);
            false
        }
    }

    fn record_success(&self, upstream: &str) {
        let mut states = self.states.lock().expect("circuit breaker mutex poisoned");
        let state = states
            .entry(upstream.to_string())
            .or_insert(CircuitBreakerState {
                consecutive_failures: 0,
                opened_at: None,
                half_open_probe_in_flight: false,
            });
        state.consecutive_failures = 0;
        state.opened_at = None;
        state.half_open_probe_in_flight = false;
        record_circuit_breaker_state(upstream, "closed", 0);
    }

    fn record_failure(&self, upstream: &str, threshold: u32) {
        let mut states = self.states.lock().expect("circuit breaker mutex poisoned");
        let state = states
            .entry(upstream.to_string())
            .or_insert(CircuitBreakerState {
                consecutive_failures: 0,
                opened_at: None,
                half_open_probe_in_flight: false,
            });
        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        state.half_open_probe_in_flight = false;
        if threshold > 0 && state.consecutive_failures >= threshold {
            state.opened_at = Some(Instant::now());
            record_circuit_breaker_state(upstream, "open", state.consecutive_failures);
        } else {
            record_circuit_breaker_state(upstream, "closed", state.consecutive_failures);
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AuthFailureLimiter {
    failures: Arc<Mutex<BTreeMap<String, AuthFailureWindow>>>,
}

#[derive(Debug, Clone)]
struct AuthFailureWindow {
    started: Instant,
    count: u32,
}

impl AuthFailureLimiter {
    fn is_limited(&self, key: &str, limit: u32) -> bool {
        if limit == 0 {
            return false;
        }
        let mut failures = self.failures.lock().expect("auth limiter mutex poisoned");
        let window = failures
            .entry(key.to_string())
            .or_insert(AuthFailureWindow {
                started: Instant::now(),
                count: 0,
            });
        if window.started.elapsed() >= Duration::from_secs(60) {
            window.started = Instant::now();
            window.count = 0;
        }
        window.count >= limit
    }

    fn record_failure(&self, key: &str, limit: u32) {
        if limit == 0 {
            return;
        }
        let mut failures = self.failures.lock().expect("auth limiter mutex poisoned");
        let window = failures
            .entry(key.to_string())
            .or_insert(AuthFailureWindow {
                started: Instant::now(),
                count: 0,
            });
        if window.started.elapsed() >= Duration::from_secs(60) {
            window.started = Instant::now();
            window.count = 0;
        }
        window.count = window.count.saturating_add(1);
        let meter = global::meter(crate::SERVICE_NAME);
        meter
            .u64_counter("llmctl_auth_failures_total")
            .with_description("Failed bearer authentication attempts by throttle state")
            .build()
            .add(
                1,
                &[
                    KeyValue::new("limited", window.count >= limit),
                    KeyValue::new(
                        "status",
                        if window.count >= limit {
                            "limited"
                        } else {
                            "failed"
                        },
                    ),
                ],
            );
    }

    fn record_success(&self, key: &str) {
        let mut failures = self.failures.lock().expect("auth limiter mutex poisoned");
        failures.remove(key);
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

    let response = with_request_id(
        Json(ModelList {
            object: "list",
            data: routed_models(&state.cfg)
                .into_iter()
                .map(ModelObject::from)
                .collect(),
        })
        .into_response(),
        request_id,
    );
    with_model_count(response, state.cfg.models.len())
}

#[derive(Debug, Deserialize)]
struct LocalSearchRequest {
    query: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default)]
    metadata: Option<Value>,
    documents: Vec<SearchDocument>,
}

fn default_search_limit() -> usize {
    10
}

async fn local_search(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<LocalSearchRequest>,
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
            return with_request_id(auth_error_response(err), request_id);
        }
    };

    if !principal.has_scope("chat") {
        return with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing chat scope".to_string(),
            ),
            request_id,
        );
    }

    let hits = lexical_search(&request.query, &request.documents, request.limit.min(50));
    let lineage = runtime_lineage_from_headers_and_metadata(&headers, request.metadata.as_ref());
    record_request_lineage_joins(&state, request_id, &lineage, None, "local.search").await;
    record_audit(
        &state,
        Some(request_id),
        principal,
        "local.search",
        "documents",
        "allowed",
        json!({ "documents": request.documents.len(), "hits": hits.len() }),
    )
    .await;
    with_request_id(
        Json(json!({
            "object": "search.results",
            "query": request.query,
            "data": hits
        }))
        .into_response(),
        request_id,
    )
}

#[derive(Debug, Deserialize)]
struct LocalRecommendationRequest {
    task: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default)]
    metadata: Option<Value>,
    documents: Vec<SearchDocument>,
}

async fn local_recommendations(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<LocalRecommendationRequest>,
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
            return with_request_id(auth_error_response(err), request_id);
        }
    };

    if !principal.has_scope("chat") {
        return with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing chat scope".to_string(),
            ),
            request_id,
        );
    }

    let hits = lexical_search(&request.task, &request.documents, request.limit.min(50));
    let recommendations = local_recommendation_items(&request.task, &hits);
    let lineage = runtime_lineage_from_headers_and_metadata(&headers, request.metadata.as_ref());
    record_request_lineage_joins(&state, request_id, &lineage, None, "local.recommendations").await;
    record_audit(
        &state,
        Some(request_id),
        principal,
        "local.recommendations",
        "documents",
        "allowed",
        json!({
            "documents": request.documents.len(),
            "hits": hits.len(),
            "recommendations": recommendations.len()
        }),
    )
    .await;
    with_request_id(
        Json(json!({
            "object": "recommendation.results",
            "task": request.task,
            "data": hits,
            "recommendations": recommendations
        }))
        .into_response(),
        request_id,
    )
}

fn local_recommendation_items(
    task: &str,
    hits: &[crate::rag::SearchHit],
) -> Vec<BTreeMap<&'static str, String>> {
    hits.iter()
        .take(5)
        .enumerate()
        .map(|(index, hit)| {
            let title = hit.title.clone().unwrap_or_else(|| hit.id.clone());
            BTreeMap::from([
                ("rank", (index + 1).to_string()),
                ("document_id", hit.id.clone()),
                ("title", title.clone()),
                ("reason", recommendation_reason(task, &title)),
            ])
        })
        .collect()
}

fn recommendation_reason(task: &str, title: &str) -> String {
    let task = task.trim();
    if task.is_empty() {
        return format!("Use `{title}` as supporting local context.");
    }
    format!("Use `{title}` because it matches local context for `{task}`.")
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
                "embeddings",
                "unknown",
                "denied",
                json!({ "reason": err }),
            )
            .await;
            return with_request_id(auth_error_response(err), request_id);
        }
    };

    if !principal.has_scope("chat") {
        record_audit(
            &state,
            Some(request_id),
            principal,
            "embeddings",
            "unknown",
            "denied",
            json!({ "reason": "missing chat scope" }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing chat scope".to_string(),
            ),
            request_id,
        );
    }

    let request: EmbeddingRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            record_audit(
                &state,
                Some(request_id),
                principal,
                "embeddings",
                "unknown",
                "rejected",
                json!({ "reason": err.to_string() }),
            )
            .await;
            return with_request_id(
                error_response(
                    StatusCode::BAD_REQUEST,
                    "bad_request",
                    "request body must be valid JSON".to_string(),
                ),
                request_id,
            );
        }
    };

    if request
        .encoding_format
        .as_deref()
        .is_some_and(|format| format != "float")
    {
        record_audit(
            &state,
            Some(request_id),
            principal,
            "embeddings",
            request.model,
            "rejected",
            json!({ "reason": "native embeddings support only float encoding_format" }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::BAD_REQUEST,
                "unsupported_encoding_format",
                "native embeddings support only float encoding_format".to_string(),
            ),
            request_id,
        );
    }

    let route = match resolve_model_route(&state.cfg, &request.model, request_id) {
        Ok(route) => route,
        Err(err) => {
            record_audit(
                &state,
                Some(request_id),
                principal,
                "embeddings",
                request.model,
                "rejected",
                json!({ "reason": err.to_string() }),
            )
            .await;
            return with_request_id(model_route_error_response(&err), request_id);
        }
    };

    let embedding_selection = match native_embedding_selection(&state.cfg, &route) {
        Ok(selection) => selection,
        Err(err) => {
            record_audit(
                &state,
                Some(request_id),
                principal,
                "embeddings",
                route.requested_alias,
                "rejected",
                json!({ "reason": err }),
            )
            .await;
            return with_request_id(
                error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "native_embedding_model_unavailable",
                    err,
                ),
                request_id,
            );
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
                record_audit(
                    &state,
                    Some(request_id),
                    principal,
                    "embeddings",
                    route.requested_alias,
                    "error",
                    json!({
                        "reason": "native_embedding_model_unavailable",
                        "runtime_backend": "candle-native",
                        "embedding_mode": embedding_selection.mode.as_str(),
                        "embedding_model_alias": embedding_selection.model_alias
                    }),
                )
                .await;
                return with_request_id(
                    error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "native_embedding_model_unavailable",
                        "semantic native embedding model is not loaded".to_string(),
                    ),
                    request_id,
                );
            };

            match engine.embeddings(native_request).await {
                Ok(response) => response,
                Err(err) => {
                    tracing::warn!(error = %err, "native embedding runtime failed");
                    record_audit(
                        &state,
                        Some(request_id),
                        principal,
                        "embeddings",
                        route.requested_alias,
                        "error",
                        json!({
                            "reason": "native_embedding_runtime_error",
                            "runtime_backend": "candle-native",
                            "embedding_mode": embedding_selection.mode.as_str(),
                            "embedding_model_alias": embedding_selection.model_alias
                        }),
                    )
                    .await;
                    return with_request_id(
                        error_response(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "native_embedding_runtime_error",
                            "native runtime failed to serve semantic embeddings".to_string(),
                        ),
                        request_id,
                    );
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
                "chat.completions",
                "unknown",
                "denied",
                json!({ "reason": err }),
            )
            .await;
            return with_request_id(auth_error_response(err), request_id);
        }
    };

    if !principal.has_scope("chat") {
        record_audit(
            &state,
            Some(request_id),
            principal,
            "chat.completions",
            "unknown",
            "denied",
            json!({ "reason": "missing chat scope" }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing chat scope".to_string(),
            ),
            request_id,
        );
    }

    let mut request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                "unknown",
                "rejected",
                json!({ "reason": err.to_string() }),
            )
            .await;
            return with_request_id(
                error_response(
                    StatusCode::BAD_REQUEST,
                    "bad_request",
                    "request body must be valid JSON".to_string(),
                ),
                request_id,
            );
        }
    };
    let lineage = runtime_lineage_from_headers_and_metadata(&headers, request.metadata.as_ref());

    let route = match resolve_model_route(&state.cfg, &request.model, request_id) {
        Ok(route) => route,
        Err(err) => {
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                request.model,
                "rejected",
                json!({ "reason": err.to_string() }),
            )
            .await;
            return with_request_id(model_route_error_response(&err), request_id);
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
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                model,
                "denied",
                json!({
                    "reason": "guardrail_violation",
                    "guardrails": verdict.block_reasons,
                    "findings": verdict.findings.audit_detail(),
                }),
            )
            .await;
            return with_request_id(
                error_response(
                    StatusCode::BAD_REQUEST,
                    "guardrail_blocked",
                    format!(
                        "request blocked by guardrails: {}",
                        verdict.block_reasons.join(", ")
                    ),
                ),
                request_id,
            );
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
    let quota_guard = match state.storage.quota_admission_guard().await {
        Ok(guard) => guard,
        Err(err) => {
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                model,
                "rejected",
                json!({ "reason": err.to_string() }),
            )
            .await;
            return with_request_id(
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "quota_error",
                    "quota admission is unavailable".to_string(),
                ),
                request_id,
            );
        }
    };
    let quota = match check_quota(&state.storage, &state.cfg.quotas, &principal, &model).await {
        Ok(decision) => decision,
        Err(err) => {
            let _ = quota_guard.release().await;
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                model,
                "rejected",
                json!({ "reason": err.to_string() }),
            )
            .await;
            return with_request_id(
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "quota_error",
                    "quota admission is unavailable".to_string(),
                ),
                request_id,
            );
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
    if let Err(err) = quota_guard.release().await {
        tracing::warn!(error = %err, "failed to release quota admission lock");
    }

    if !quota.allowed {
        record_audit(
            &state,
            Some(request_id),
            principal,
            "chat.completions",
            model,
            "denied",
            json!({ "reason": quota.reason }),
        )
        .await;
        return with_request_id(
            error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "quota_exceeded",
                quota.reason,
            ),
            request_id,
        );
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
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                model,
                "denied",
                json!({ "reason": "admission_limit_exceeded" }),
            )
            .await;
            return with_request_id(
                error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    "rate_limit_exceeded",
                    "server is busy; retry later".to_string(),
                ),
                request_id,
            );
        }
    };

    if route.external_provider.is_some() {
        let tool_audit = request.tool_audit_detail();
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
        let mut request_builder = state
            .client
            .post(upstream)
            .header(CONTENT_TYPE, "application/json")
            .body(body);
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
    let stream = async_stream::stream! {
        let _admission = admission;
        let mut input_tokens = 0u64;
        let mut output_tokens = 0u64;
        let mut usage_parser = SseUsageParser::default();
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

fn authenticate(cfg: &Config, headers: &HeaderMap) -> std::result::Result<Principal, String> {
    if !cfg.security.require_auth {
        return Ok(Principal::anonymous());
    }

    let Some(header) = headers.get(AUTHORIZATION).and_then(|h| h.to_str().ok()) else {
        return Err("missing bearer token".to_string());
    };
    let Some(token) = header.strip_prefix("Bearer ") else {
        return Err("authorization must use Bearer token".to_string());
    };

    let digest = hex::encode(Sha256::digest(token.as_bytes()));
    cfg.security
        .api_keys
        .iter()
        .filter(|key| api_key_can_authenticate(key))
        .find(|key| constant_time_eq_case_insensitive(&key.sha256, &digest))
        .map(|key| Principal {
            subject: key.subject.clone(),
            team: key.team.clone(),
            scopes: key.scopes.clone(),
            key_id: Some(key.id.clone()),
            key_owner: key.owner.clone(),
            key_purpose: key.purpose.clone(),
            key_status: Some(key.status.clone()),
        })
        .ok_or_else(|| "invalid bearer token".to_string())
}

fn api_key_can_authenticate(key: &ApiKeyConfig) -> bool {
    if !matches!(key.status.as_str(), "active" | "retiring") {
        return false;
    }
    key.expires_at
        .is_none_or(|expires_at| expires_at > Utc::now())
}

fn authenticate_request(
    state: &ServerState,
    headers: &HeaderMap,
    source_key: String,
) -> std::result::Result<Principal, String> {
    let key = auth_failure_key(&source_key);
    if state
        .auth_failures
        .is_limited(&key, state.cfg.security.auth_failure_limit_per_minute)
    {
        return Err("too many failed authentication attempts; retry later".to_string());
    }

    match authenticate(&state.cfg, headers) {
        Ok(principal) => {
            state.auth_failures.record_success(&key);
            Ok(principal)
        }
        Err(err) => {
            state
                .auth_failures
                .record_failure(&key, state.cfg.security.auth_failure_limit_per_minute);
            Err(err)
        }
    }
}

fn auth_source_key(
    cfg: &Config,
    headers: &HeaderMap,
    connect_info: Option<ConnectInfo<SocketAddr>>,
) -> String {
    let peer = connect_info.map(|ConnectInfo(addr)| addr.ip());
    if peer.is_some_and(|ip| is_trusted_proxy(cfg, ip)) {
        if let Some(forwarded) = forwarded_client_ip(cfg, headers) {
            return forwarded;
        }
    }
    peer.map(|ip| ip.to_string())
        .unwrap_or_else(|| "unknown-source".to_string())
}

fn is_trusted_proxy(cfg: &Config, ip: IpAddr) -> bool {
    cfg.security
        .trusted_proxies
        .iter()
        .any(|trusted| trusted_proxy_matches(trusted, ip))
}

fn trusted_proxy_matches(trusted: &str, ip: IpAddr) -> bool {
    let trusted = trusted.trim();
    if let Ok(exact) = trusted.parse::<IpAddr>() {
        return exact == ip;
    }
    let Some((network, prefix)) = trusted.split_once('/') else {
        return false;
    };
    let Ok(network) = network.parse::<IpAddr>() else {
        return false;
    };
    let Ok(prefix) = prefix.parse::<u8>() else {
        return false;
    };
    match (network, ip) {
        (IpAddr::V4(network), IpAddr::V4(ip)) if prefix > 0 && prefix <= 32 => {
            let mask = u32::MAX << (32 - prefix);
            u32::from(network) & mask == u32::from(ip) & mask
        }
        (IpAddr::V6(network), IpAddr::V6(ip)) if prefix > 0 && prefix <= 128 => {
            let mask = u128::MAX << (128 - prefix);
            u128::from(network) & mask == u128::from(ip) & mask
        }
        _ => false,
    }
}

fn forwarded_client_ip(cfg: &Config, headers: &HeaderMap) -> Option<String> {
    let forwarded_ips = headers
        .get_all("forwarded")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| {
            value
                .split(',')
                .flat_map(|element| element.split(';'))
                .filter_map(|part| {
                    let part = part.trim();
                    let for_value = part.strip_prefix("for=")?;
                    parse_forwarded_ip(for_value)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if !forwarded_ips.is_empty() {
        if let Some(ip) = first_untrusted_forwarded_ip(cfg, forwarded_ips) {
            return Some(ip);
        }
    }
    let forwarded_ips = headers
        .get_all("x-forwarded-for")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| {
            value
                .split(',')
                .filter_map(parse_forwarded_ip)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if !forwarded_ips.is_empty() {
        if let Some(ip) = first_untrusted_forwarded_ip(cfg, forwarded_ips) {
            return Some(ip);
        }
    }
    None
}

fn parse_forwarded_ip(value: &str) -> Option<IpAddr> {
    let value = value
        .trim()
        .trim_matches('"')
        .trim_matches('[')
        .trim_matches(']');
    let candidate = if let Ok(ip) = value.parse::<IpAddr>() {
        ip
    } else if let Some((host, _port)) = value.rsplit_once(':') {
        host.trim_matches('[').trim_matches(']').parse().ok()?
    } else {
        return None;
    };
    Some(candidate)
}

fn first_untrusted_forwarded_ip(cfg: &Config, forwarded_ips: Vec<IpAddr>) -> Option<String> {
    for ip in forwarded_ips.into_iter().rev() {
        if !is_trusted_proxy(cfg, ip) {
            return Some(ip.to_string());
        }
    }
    None
}

fn auth_failure_key(source_key: &str) -> String {
    let digest = Sha256::digest(source_key.as_bytes());
    format!("{:x}", digest)[..16].to_string()
}

fn constant_time_eq_case_insensitive(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let left = left.to_ascii_lowercase();
    let right = right.to_ascii_lowercase();
    left.as_bytes().ct_eq(right.as_bytes()).into()
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
    record_usage_telemetry(&event, input.accounting_mode);
    dispatch_usage_webhook(state, &event, input.accounting_mode);
}

/// GenAI semantic-convention name for this service — surfaced as `gen_ai.system`
/// on every usage span so downstream OTel consumers (Langfuse, generic GenAI
/// dashboards) can group spans emitted by rs-llmctl regardless of which model
/// served the request.
const GEN_AI_SYSTEM: &str = "rs-llmctl";

/// Build the attribute set for a usage span: the existing `llmctl.*` attributes
/// plus the OTel GenAI semantic-convention attributes
/// (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`,
/// `gen_ai.response.model`, `gen_ai.usage.*`) so traces align with the
/// conventions Langfuse and other GenAI-aware OTel consumers expect.
fn usage_span_attributes(event: &UsageEvent, accounting_mode: &str) -> BTreeMap<String, Value> {
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
        ("gen_ai.system".to_string(), json!(GEN_AI_SYSTEM)),
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

fn record_usage_telemetry(event: &UsageEvent, accounting_mode: &str) {
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::RequestRouting,
        Utc::now(),
        usage_span_attributes(event, accounting_mode),
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

fn usage_tokens(bytes: &[u8]) -> (u64, u64) {
    let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
        return (0, 0);
    };
    usage_tokens_from_value(&value)
}

fn usage_tokens_from_value(value: &Value) -> (u64, u64) {
    let Some(usage) = value.get("usage") else {
        return (0, 0);
    };
    let input = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    (input, output)
}

#[derive(Debug, Default)]
struct SseUsageParser {
    buffer: String,
}

impl SseUsageParser {
    fn push(&mut self, bytes: &[u8]) -> std::result::Result<(u64, u64), &'static str> {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);
        if self.buffer.len() > MAX_SSE_USAGE_BUFFER_BYTES && !self.buffer.contains("\n\n") {
            return Err("SSE usage parser buffer exceeded maximum frame size");
        }
        let mut input_tokens = 0u64;
        let mut output_tokens = 0u64;

        while let Some(frame_end) = self.buffer.find("\n\n") {
            if frame_end > MAX_SSE_USAGE_BUFFER_BYTES {
                return Err("SSE usage parser buffer exceeded maximum frame size");
            }
            let frame = self.buffer[..frame_end].to_string();
            self.buffer.drain(..frame_end + 2);
            let (input, output) = sse_frame_usage_tokens(&frame);
            input_tokens = input_tokens.saturating_add(input);
            output_tokens = output_tokens.saturating_add(output);
        }

        Ok((input_tokens, output_tokens))
    }
}

#[cfg(test)]
fn sse_usage_tokens(bytes: &[u8]) -> (u64, u64) {
    SseUsageParser::default().push(bytes).expect("valid SSE")
}

fn sse_frame_usage_tokens(frame: &str) -> (u64, u64) {
    frame
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("data:"))
        .map(str::trim)
        .filter(|data| !data.is_empty() && *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .map(|value| usage_tokens_from_value(&value))
        .fold(
            (0u64, 0u64),
            |(total_input, total_output), (input, output)| {
                (
                    total_input.saturating_add(input),
                    total_output.saturating_add(output),
                )
            },
        )
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

#[derive(Debug, Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

impl From<&ModelConfig> for ModelObject {
    fn from(model: &ModelConfig) -> Self {
        Self {
            id: model.alias.clone(),
            object: "model",
            owned_by: "rs-llmctl",
        }
    }
}

pub async fn serve(cfg: Config) -> Result<()> {
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
    fn extracts_openai_usage_tokens() {
        let body = br#"{"usage":{"prompt_tokens":11,"completion_tokens":13}}"#;
        assert_eq!(usage_tokens(body), (11, 13));
    }

    #[test]
    fn extracts_streaming_sse_usage_tokens() {
        let chunk = br#"event: completion
data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":9}}

data: [DONE]
"#;

        assert_eq!(sse_usage_tokens(chunk), (7, 9));
        assert_eq!(stream_status(0, 0), "stream_unmetered");
        assert_eq!(stream_status(7, 9), "ok");
    }

    #[test]
    fn extracts_split_streaming_sse_usage_tokens() {
        let mut parser = SseUsageParser::default();

        assert_eq!(
            parser.push(br#"data: {"choices":[],"usage":{"prompt_tokens":7"#),
            Ok((0, 0))
        );
        assert_eq!(
            parser.push(
                br#","completion_tokens":9}}

"#
            ),
            Ok((7, 9))
        );
    }

    #[test]
    fn sse_usage_parser_rejects_unbounded_partial_frames() {
        let mut parser = SseUsageParser::default();
        let oversized = vec![b'a'; MAX_SSE_USAGE_BUFFER_BYTES + 1];
        assert!(parser.push(&oversized).is_err());
    }

    #[test]
    fn sse_usage_parser_rejects_oversized_complete_frames() {
        let mut parser = SseUsageParser::default();
        let mut oversized = vec![b'a'; MAX_SSE_USAGE_BUFFER_BYTES + 1];
        oversized.extend_from_slice(b"\n\n");
        assert!(parser.push(&oversized).is_err());
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

        let attrs = usage_span_attributes(&event, "estimated");

        assert_eq!(attrs["gen_ai.system"], json!("rs-llmctl"));
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
}
