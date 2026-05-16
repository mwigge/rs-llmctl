use crate::audit::{AuditEvent, UsageEvent};
use crate::config::{Config, Mode, ModelConfig};
use crate::quota::{check_quota, Principal};
use crate::storage::{QuotaDecisionRecord, Storage};
use crate::worker::StartupPlan;
use anyhow::{Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::timeout;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

const DEFAULT_MAX_IN_FLIGHT: usize = 128;
const DEFAULT_UPSTREAM_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone)]
pub struct ServerState {
    cfg: Arc<Config>,
    storage: Storage,
    client: reqwest::Client,
    upstream: String,
    admission: AdmissionController,
    serving_limits: ServingLimits,
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
    let upstream = normalize_upstream(&cfg.server.llama_server);
    let admission = AdmissionController::new(serving_limits.max_in_flight);
    let state = ServerState {
        cfg: Arc::new(cfg),
        storage,
        client: reqwest::Client::new(),
        upstream,
        admission,
        serving_limits,
    };

    Router::new()
        .route("/healthz", get(healthz))
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([AUTHORIZATION, CONTENT_TYPE, request_id_header_name()])
                .expose_headers([
                    request_id_header_name(),
                    model_count_header_name(),
                    model_header_name(),
                    upstream_model_header_name(),
                    quota_decision_header_name(),
                ]),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(Arc::new(state))
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

        Self::new(max_in_flight, DEFAULT_UPSTREAM_TIMEOUT)
    }

    fn upstream_timeout(&self) -> Duration {
        self.upstream_timeout
    }
}

#[derive(Clone)]
struct AdmissionController {
    permits: Arc<Semaphore>,
}

impl AdmissionController {
    fn new(max_in_flight: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(max_in_flight.max(1))),
        }
    }

    fn try_acquire(&self) -> std::result::Result<AdmissionPermit, AdmissionError> {
        self.permits
            .clone()
            .try_acquire_owned()
            .map(|permit| AdmissionPermit { _permit: permit })
            .map_err(|_| AdmissionError::Busy)
    }
}

struct AdmissionPermit {
    _permit: OwnedSemaphorePermit,
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

async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn livez() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn readyz(State(state): State<Arc<ServerState>>) -> Response {
    let storage_ready = storage_ready(&state.storage).await;
    let http_status = if storage_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        http_status,
        Json(readiness_status_for(&state.cfg, storage_ready)),
    )
        .into_response()
}

pub async fn readiness_status(cfg: &Config, storage: &Storage) -> Value {
    readiness_status_for(cfg, storage_ready(storage).await)
}

fn readiness_status_for(cfg: &Config, storage_ready: bool) -> Value {
    let aliases: Vec<_> = routed_models(cfg)
        .into_iter()
        .map(|model| model.alias.as_str())
        .collect();
    let worker_plan = StartupPlan::from_config(cfg);

    json!({
        "status": if storage_ready { "ready" } else { "unavailable" },
        "mode": cfg.mode,
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

fn is_external_host(host: &str) -> bool {
    !matches!(host, "127.0.0.1" | "localhost" | "::1")
}

async fn list_models(State(state): State<Arc<ServerState>>, headers: HeaderMap) -> Response {
    let request_id = request_id_from_headers(&headers);
    let principal = match authenticate(&state.cfg, &headers) {
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
            return with_request_id(
                error_response(StatusCode::UNAUTHORIZED, "unauthorized", err),
                request_id,
            );
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

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    let started = Instant::now();
    let principal = match authenticate(&state.cfg, &headers) {
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
            return with_request_id(
                error_response(StatusCode::UNAUTHORIZED, "unauthorized", err),
                request_id,
            );
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

    let request: ChatCompletionRequest = match serde_json::from_slice(&body) {
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

    let route = match resolve_model_route(&state.cfg, &request.model) {
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
            return with_request_id(
                error_response(StatusCode::BAD_REQUEST, "unknown_model", err.to_string()),
                request_id,
            );
        }
    };
    let body = match rewrite_chat_model(&body, &route) {
        Ok(body) => body,
        Err(err) => {
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                route.requested_alias,
                "rejected",
                json!({ "reason": err }),
            )
            .await;
            return with_request_id(
                error_response(StatusCode::BAD_REQUEST, "bad_request", err),
                request_id,
            );
        }
    };
    let model = route.requested_alias.clone();
    let quota = match check_quota(&state.storage, &state.cfg.quotas, &principal, &model).await {
        Ok(decision) => decision,
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
                    err.to_string(),
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
        json!({ "stream": request.stream, "upstream_model": route.upstream_alias }),
    )
    .await;

    let admission = match state.admission.try_acquire() {
        Ok(permit) => permit,
        Err(AdmissionError::Busy) => {
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

    let upstream = format!("{}/v1/chat/completions", state.upstream);
    let upstream_response = match timeout(
        upstream_timeout_budget(&state),
        state
            .client
            .post(upstream)
            .header(CONTENT_TYPE, "application/json")
            .body(body)
            .send(),
    )
    .await
    {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => {
            let (status, code, message, usage_status) = upstream_request_error(&err);
            record_upstream_failure(
                &state,
                request_id,
                &principal,
                &model,
                started,
                usage_status,
            )
            .await;
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                model,
                "error",
                json!({ "reason": usage_status }),
            )
            .await;
            return with_request_id(error_response(status, code, message), request_id);
        }
        Err(_) => {
            record_upstream_failure(&state, request_id, &principal, &model, started, "timeout")
                .await;
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                model,
                "error",
                json!({ "reason": "timeout" }),
            )
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

    if request.stream {
        let upstream_context = UpstreamRequestContext {
            request_id,
            principal,
            model,
            upstream_model: route.upstream_alias,
            started,
            admission,
        };
        stream_upstream(state, upstream_response, upstream_context).await
    } else {
        let upstream_context = UpstreamRequestContext {
            request_id,
            principal,
            model,
            upstream_model: route.upstream_alias,
            started,
            admission,
        };
        json_upstream(state, upstream_response, upstream_context).await
    }
}

struct UpstreamRequestContext {
    request_id: Uuid,
    principal: Principal,
    model: String,
    upstream_model: String,
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
        started,
        admission: _admission,
    } = context;
    let status = upstream_response.status();
    let headers = response_headers(upstream_response.headers());
    let bytes = match timeout(upstream_timeout_budget(&state), upstream_response.bytes()).await {
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
        json!({ "status": status.as_u16() }),
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
        started,
        admission,
    } = context;
    let status = upstream_response.status();
    let mut headers = response_headers(upstream_response.headers());
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    let stream_status = if status.is_success() {
        "ok"
    } else {
        "upstream_error"
    };
    let response_model = model.clone();
    let response_upstream_model = upstream_model.clone();
    let mut upstream_stream = upstream_response.bytes_stream();
    let stream = async_stream::stream! {
        let _admission = admission;
        while let Some(chunk) = upstream_stream.next().await {
            match chunk {
                Ok(bytes) => yield Ok::<Bytes, std::io::Error>(bytes),
                Err(err) => {
                    record_usage(
                        &state,
                        UsageRecordInput {
                            request_id,
                            principal: &principal,
                            model: &model,
                            input_tokens: 0,
                            output_tokens: 0,
                            latency_ms: elapsed_ms(started),
                            status: "stream_error",
                        },
                    )
                    .await;
                    record_audit(
                        &state,
                        Some(request_id),
                        principal.clone(),
                        "chat.completions",
                        model.clone(),
                        "stream_error",
                        json!({ "status": status.as_u16(), "stream": true, "reason": upstream_error_status(&err) }),
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
                input_tokens: 0,
                output_tokens: 0,
                latency_ms: elapsed_ms(started),
                status: stream_status,
            },
        )
        .await;
        record_audit(
            &state,
            Some(request_id),
            principal.clone(),
            "chat.completions",
            model.clone(),
            stream_status,
            json!({ "status": status.as_u16(), "stream": true }),
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

fn upstream_timeout_budget(state: &ServerState) -> Duration {
    state.serving_limits.upstream_timeout()
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
        },
    )
    .await;
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedModelRoute {
    requested_alias: String,
    upstream_alias: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelRouteError {
    UnknownAlias(String),
    NoConfiguredModels,
}

impl std::fmt::Display for ModelRouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownAlias(alias) => write!(f, "unknown model alias: {alias}"),
            Self::NoConfiguredModels => write!(f, "no models are configured"),
        }
    }
}

fn routed_models(cfg: &Config) -> Vec<&ModelConfig> {
    let mut models: Vec<_> = cfg.models.iter().collect();
    models.sort_by(|left, right| left.alias.cmp(&right.alias));
    models
}

fn resolve_model_route(
    cfg: &Config,
    requested_alias: &str,
) -> std::result::Result<ResolvedModelRoute, ModelRouteError> {
    if cfg.models.is_empty() {
        if requested_alias.trim().is_empty() {
            return Err(ModelRouteError::NoConfiguredModels);
        }
        return Ok(ResolvedModelRoute {
            requested_alias: requested_alias.to_string(),
            upstream_alias: requested_alias.to_string(),
        });
    }

    let requested = cfg
        .models
        .iter()
        .find(|model| model.alias == requested_alias)
        .ok_or_else(|| ModelRouteError::UnknownAlias(requested_alias.to_string()))?;

    let upstream = match cfg.mode {
        Mode::Single => routed_models(cfg)
            .into_iter()
            .next()
            .ok_or(ModelRouteError::NoConfiguredModels)?,
        Mode::ColdSwap | Mode::HotSwap => requested,
        Mode::Weighted => preferred_weighted_model(cfg).unwrap_or(requested),
        Mode::Fallback => {
            if requested.weight > 0 {
                requested
            } else {
                preferred_weighted_model(cfg).unwrap_or(requested)
            }
        }
    };

    Ok(ResolvedModelRoute {
        requested_alias: requested_alias.to_string(),
        upstream_alias: upstream.alias.clone(),
    })
}

fn preferred_weighted_model(cfg: &Config) -> Option<&ModelConfig> {
    cfg.models
        .iter()
        .filter(|model| model.weight > 0)
        .max_by(|left, right| {
            left.weight
                .cmp(&right.weight)
                .then_with(|| right.alias.cmp(&left.alias))
        })
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
        .find(|key| key.sha256.eq_ignore_ascii_case(&digest))
        .map(|key| Principal {
            subject: key.subject.clone(),
            team: key.team.clone(),
            scopes: key.scopes.clone(),
        })
        .ok_or_else(|| "invalid bearer token".to_string())
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
}

struct UsageRecordInput<'a> {
    request_id: Uuid,
    principal: &'a Principal,
    model: &'a str,
    input_tokens: u64,
    output_tokens: u64,
    latency_ms: u64,
    status: &'a str,
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
}

fn usage_tokens(bytes: &[u8]) -> (u64, u64) {
    let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
        return (0, 0);
    };
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

fn response_headers(upstream_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in upstream_headers {
        if !is_hop_by_hop(name) {
            headers.insert(name.clone(), value.clone());
        }
    }
    headers
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
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
                "code": code
            }
        })),
    )
        .into_response()
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

fn normalize_upstream(raw: &str) -> String {
    let raw = raw.trim().trim_end_matches('/');
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    #[serde(default)]
    stream: bool,
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
    let storage = Storage::connect(&cfg.storage.db_path).await?;
    serve_with_storage_and_shutdown(cfg, storage, shutdown).await
}

pub async fn serve_with_storage(cfg: Config, storage: Storage) -> Result<()> {
    serve_with_storage_and_shutdown(cfg, storage, shutdown_signal()).await
}

pub async fn serve_with_storage_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    axum::serve(listener, router(cfg, storage))
        .with_graceful_shutdown(shutdown)
        .await
        .context("serve HTTP API")
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
    use crate::config::{ApiKeyConfig, SecurityConfig, ServerConfig};

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
    fn admission_controller_rejects_when_in_flight_limit_is_full() {
        let controller = AdmissionController::new(1);
        let first = controller.try_acquire().expect("first permit");

        assert_eq!(controller.try_acquire().unwrap_err(), AdmissionError::Busy);

        drop(first);
        assert!(controller.try_acquire().is_ok());
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
                }],
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

        let resolved = resolve_model_route(&cfg, "llama").unwrap();

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
                resolve_model_route(&cfg, "beta").unwrap().upstream_alias,
                "beta"
            );
            assert!(matches!(
                resolve_model_route(&cfg, "missing"),
                Err(ModelRouteError::UnknownAlias(alias)) if alias == "missing"
            ));
        }
    }

    #[test]
    fn weighted_mode_selects_highest_weight_deterministically() {
        let cfg = config_with_models(
            Mode::Weighted,
            vec![
                model("light", 1, "chat"),
                model("heavy-b", 50, "chat"),
                model("heavy-a", 50, "chat"),
            ],
        );

        let resolved = resolve_model_route(&cfg, "light").unwrap();

        assert_eq!(resolved.requested_alias, "light");
        assert_eq!(resolved.upstream_alias, "heavy-a");
    }

    #[tokio::test]
    async fn serve_with_storage_and_shutdown_exits_when_shutdown_future_completes() {
        let storage = Storage::in_memory().await.expect("storage");
        let mut cfg = Config::default();
        cfg.server.port = 0;
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
            resolve_model_route(&cfg, "backup").unwrap().upstream_alias,
            "primary"
        );
        assert_eq!(
            resolve_model_route(&cfg, "tertiary")
                .unwrap()
                .upstream_alias,
            "tertiary"
        );
    }

    #[test]
    fn rewrites_chat_completion_model_for_upstream_route() {
        let body = br#"{"model":"light","messages":[]}"#;
        let route = ResolvedModelRoute {
            requested_alias: "light".to_string(),
            upstream_alias: "heavy".to_string(),
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
            weight,
        }
    }
}
