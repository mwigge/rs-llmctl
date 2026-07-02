use super::{
    audit_reject, audit_reject_response, authenticate_with_chat_scope, draining_response,
    elapsed_ms, error_response, gen_ai_system_for_provider, model_route_error_response,
    record_audit, record_usage, request_id_from_headers, resolve_model_route,
    token_accounting_label, with_chat_metadata, with_request_id, ResolvedModelRoute, ServerState,
    UsageRecordInput,
};
use crate::config::{Config, NativeEmbeddingMode};
use crate::native;
use crate::quota::Principal;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

pub(super) async fn proxy_embeddings(
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
