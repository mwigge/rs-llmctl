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
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

#[derive(Clone)]
pub struct ServerState {
    cfg: Arc<Config>,
    storage: Storage,
    client: reqwest::Client,
    upstream: String,
}

pub fn router(cfg: Config, storage: Storage) -> Router {
    let upstream = normalize_upstream(&cfg.server.llama_server);
    let state = ServerState {
        cfg: Arc::new(cfg),
        storage,
        client: reqwest::Client::new(),
        upstream,
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
                .allow_headers([AUTHORIZATION, CONTENT_TYPE]),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(Arc::new(state))
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
    let request_id = Uuid::new_v4();
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
            return error_response(StatusCode::UNAUTHORIZED, "unauthorized", err);
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
        return error_response(
            StatusCode::FORBIDDEN,
            "forbidden",
            "missing models.read scope".to_string(),
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

    Json(ModelList {
        object: "list",
        data: routed_models(&state.cfg)
            .into_iter()
            .map(ModelObject::from)
            .collect(),
    })
    .into_response()
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request_id = Uuid::new_v4();
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
            return error_response(StatusCode::UNAUTHORIZED, "unauthorized", err);
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
        return error_response(
            StatusCode::FORBIDDEN,
            "forbidden",
            "missing chat scope".to_string(),
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
            return error_response(
                StatusCode::BAD_REQUEST,
                "bad_request",
                "request body must be valid JSON".to_string(),
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
            return error_response(StatusCode::BAD_REQUEST, "unknown_model", err.to_string());
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
            return error_response(StatusCode::BAD_REQUEST, "bad_request", err);
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
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "quota_error",
                err.to_string(),
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
        return error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "quota_exceeded",
            quota.reason,
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

    let upstream = format!("{}/v1/chat/completions", state.upstream);
    let upstream_response = match state
        .client
        .post(upstream)
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(err) => {
            let latency_ms = elapsed_ms(started);
            record_usage(
                &state,
                UsageRecordInput {
                    request_id,
                    principal: &principal,
                    model: &model,
                    input_tokens: 0,
                    output_tokens: 0,
                    latency_ms,
                    status: "upstream_error",
                },
            )
            .await;
            record_audit(
                &state,
                Some(request_id),
                principal,
                "chat.completions",
                model,
                "error",
                json!({ "reason": err.to_string() }),
            )
            .await;
            return error_response(StatusCode::BAD_GATEWAY, "upstream_error", err.to_string());
        }
    };

    if request.stream {
        stream_upstream(
            state,
            request_id,
            principal,
            model,
            started,
            upstream_response,
        )
        .await
    } else {
        json_upstream(
            state,
            request_id,
            principal,
            model,
            started,
            upstream_response,
        )
        .await
    }
}

async fn json_upstream(
    state: Arc<ServerState>,
    request_id: Uuid,
    principal: Principal,
    model: String,
    started: Instant,
    upstream_response: reqwest::Response,
) -> Response {
    let status = upstream_response.status();
    let headers = response_headers(upstream_response.headers());
    let bytes = match upstream_response.bytes().await {
        Ok(bytes) => bytes,
        Err(err) => {
            let latency_ms = elapsed_ms(started);
            record_usage(
                &state,
                UsageRecordInput {
                    request_id,
                    principal: &principal,
                    model: &model,
                    input_tokens: 0,
                    output_tokens: 0,
                    latency_ms,
                    status: "upstream_error",
                },
            )
            .await;
            return error_response(StatusCode::BAD_GATEWAY, "upstream_error", err.to_string());
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
        model,
        status_text,
        json!({ "status": status.as_u16() }),
    )
    .await;

    build_response(status, headers, Body::from(bytes))
}

async fn stream_upstream(
    state: Arc<ServerState>,
    request_id: Uuid,
    principal: Principal,
    model: String,
    started: Instant,
    upstream_response: reqwest::Response,
) -> Response {
    let status = upstream_response.status();
    let mut headers = response_headers(upstream_response.headers());
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    let stream_status = if status.is_success() {
        "ok"
    } else {
        "upstream_error"
    };
    let mut upstream_stream = upstream_response.bytes_stream();
    let stream = async_stream::stream! {
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
                        json!({ "status": status.as_u16(), "stream": true, "reason": err.to_string() }),
                    )
                    .await;
                    yield Err::<Bytes, std::io::Error>(std::io::Error::other(err));
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
    build_response(status, headers, Body::from_stream(stream))
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

fn build_response(status: StatusCode, headers: HeaderMap, body: Body) -> Response {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
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
