use super::{
    build_response, chat_audit_detail, elapsed_ms, error_response, gen_ai_system_for_provider,
    record_audit, record_usage, sanitize_native_chat_message, with_chat_metadata, with_request_id,
    AdmissionPermit, ChatCompletionRequest, ServerState, ToolAuditDetail, UsageRecordInput,
};
use crate::native;
use crate::quota::Principal;
use axum::body::{Body, Bytes};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

pub(super) async fn dispatch_native_chat(
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

pub(super) struct NativeChatContext {
    pub(super) request_id: Uuid,
    pub(super) principal: Principal,
    pub(super) model: String,
    pub(super) upstream_model: String,
    pub(super) tool_audit: ToolAuditDetail,
    pub(super) started: Instant,
    pub(super) _admission: AdmissionPermit,
}

pub(super) fn token_accounting_label(mode: &native::TokenAccountingMode) -> &'static str {
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
