use crate::audit::{AuditEvent, UsageEvent};
use crate::config::{Config, ModelConfig};
use crate::quota::{check_quota, Principal};
use crate::storage::{QuotaDecisionRecord, Storage};
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
        data: state.cfg.models.iter().map(ModelObject::from).collect(),
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

    let model = request.model.clone();
    let quota = match check_quota(&state.cfg.quotas, &principal, &model) {
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
        json!({ "stream": request.stream }),
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
    let storage = Storage::connect(&cfg.storage.db_path).await?;
    serve_with_storage(cfg, storage).await
}

pub async fn serve_with_storage(cfg: Config, storage: Storage) -> Result<()> {
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    axum::serve(listener, router(cfg, storage))
        .await
        .context("serve HTTP API")
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
}
