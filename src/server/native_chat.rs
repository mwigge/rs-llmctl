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

    // The native runtime serves a single completion per request. `n` is now
    // declared (so it is no longer silently dropped by serde); flag the
    // unsupported case rather than pretending to honor it.
    if request.n.is_some_and(|n| n > 1) {
        tracing::warn!(
            requested_n = request.n,
            "native runtime serves a single completion; n > 1 is not supported and is ignored"
        );
    }

    let stop = request.stop_sequences();
    let native_request = native::NativeChatRequest {
        model: context.upstream_model.clone(),
        messages: request
            .messages
            .into_iter()
            .map(sanitize_native_chat_message)
            .collect(),
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        top_k: request.top_k,
        seed: request.seed,
        stop,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        tools: request.tools,
        tool_choice: request.tool_choice,
        metadata: native_chat_metadata(request.metadata, &context, request.stream),
    };

    if request.stream {
        return native_chat_stream_response(state, context, engine, native_request).await;
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

/// Terminal outcome of a native streaming generation, sent from the detached
/// generation task to the SSE stream once decoding finishes.
#[derive(Debug)]
enum NativeStreamOutcome {
    /// Generation completed; carries the final `finish_reason`.
    Done { finish_reason: String },
    /// Generation failed after the SSE response was already committed.
    Failed,
}

/// Streams a native chat completion incrementally (Bug 10).
///
/// Generation runs in a detached task that forwards each decoded token delta
/// over a channel and records terminal usage/audit itself. Recording lives in
/// that task — not in the SSE generator — so a mid-stream client disconnect
/// (which drops the response stream) cannot skip usage/audit accounting. The
/// task also holds the admission permit until decoding actually finishes.
async fn native_chat_stream_response(
    state: Arc<ServerState>,
    context: NativeChatContext,
    engine: Arc<dyn native::NativeEngine>,
    native_request: native::NativeChatRequest,
) -> Response {
    let NativeChatContext {
        request_id,
        principal,
        model,
        upstream_model,
        tool_audit,
        started,
        _admission,
    } = context;

    let sse_model = model.clone();
    let metadata_model = model.clone();
    let sse_upstream_model = upstream_model.clone();

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (final_tx, final_rx) = tokio::sync::oneshot::channel::<NativeStreamOutcome>();

    tokio::spawn(async move {
        // Keep the admission permit alive for the whole decode.
        let _admission = _admission;
        let result = engine.chat_stream_tokens(native_request, token_tx).await;
        // `token_tx` is dropped here as the future returns, closing `token_rx`.
        match result {
            Ok(response) => {
                record_usage(
                    &state,
                    UsageRecordInput {
                        request_id,
                        principal: &principal,
                        model: &model,
                        input_tokens: response.usage.input_tokens,
                        output_tokens: response.usage.output_tokens,
                        latency_ms: elapsed_ms(started),
                        status: "ok",
                        accounting_mode: token_accounting_label(&response.usage.accounting_mode),
                        gen_ai_system: gen_ai_system_for_provider(None),
                    },
                )
                .await;
                let detail = chat_audit_detail(
                    &tool_audit,
                    json!({
                        "runtime_backend": "candle-native",
                        "stream": true,
                        "token_accounting": response.usage.accounting_mode.clone()
                    }),
                );
                record_audit(
                    &state,
                    Some(request_id),
                    principal,
                    "chat.completions",
                    model,
                    "ok",
                    detail,
                )
                .await;
                let _ = final_tx.send(NativeStreamOutcome::Done {
                    finish_reason: response.finish_reason,
                });
            }
            Err(err) => {
                tracing::warn!(error = %err, "native streaming generation failed");
                record_usage(
                    &state,
                    UsageRecordInput {
                        request_id,
                        principal: &principal,
                        model: &model,
                        input_tokens: 0,
                        output_tokens: 0,
                        latency_ms: elapsed_ms(started),
                        status: "native_runtime_error",
                        accounting_mode: "none",
                        gen_ai_system: gen_ai_system_for_provider(None),
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
                    json!({
                        "reason": "native_runtime_error",
                        "runtime_backend": "candle-native",
                        "stream": true
                    }),
                )
                .await;
                let _ = final_tx.send(NativeStreamOutcome::Failed);
            }
        }
    });

    let stream = native_completion_sse_stream(request_id, sse_model, token_rx, final_rx);

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    let response = build_response(
        StatusCode::OK,
        headers,
        Body::from_stream(stream),
        request_id,
    );
    with_chat_metadata(response, &metadata_model, &sse_upstream_model, "allowed")
}

/// Builds the SSE body for a native streaming completion: one
/// `chat.completion.chunk` per decoded token delta (Bug 10), then a terminal
/// chunk carrying the `finish_reason` and the `[DONE]` sentinel. Token deltas
/// arrive over `token_rx`; the terminal outcome arrives over `final_rx` once
/// generation finishes.
fn native_completion_sse_stream(
    request_id: Uuid,
    model: String,
    mut token_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    final_rx: tokio::sync::oneshot::Receiver<NativeStreamOutcome>,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> {
    async_stream::stream! {
        while let Some(delta) = token_rx.recv().await {
            if delta.is_empty() {
                continue;
            }
            let chunk = json!({
                "id": format!("chatcmpl-{request_id}"),
                "object": "chat.completion.chunk",
                "created": Utc::now().timestamp(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": { "content": delta },
                    "finish_reason": Value::Null
                }]
            });
            yield Ok::<Bytes, std::io::Error>(Bytes::from(format!("data: {chunk}\n\n")));
        }

        // Generation finished (channel closed); await the terminal outcome.
        let finish_reason = match final_rx.await {
            Ok(NativeStreamOutcome::Done { finish_reason }) => finish_reason,
            // The response was already committed as 200 OK; surface the failure
            // as an SSE error object so the client can distinguish it.
            Ok(NativeStreamOutcome::Failed) | Err(_) => {
                let error_chunk = json!({
                    "id": format!("chatcmpl-{request_id}"),
                    "object": "chat.completion.chunk",
                    "error": {
                        "type": "native_runtime_error",
                        "message": "native runtime failed while streaming"
                    }
                });
                yield Ok::<Bytes, std::io::Error>(Bytes::from(format!("data: {error_chunk}\n\n")));
                "error".to_string()
            }
        };

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
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    /// Bug 10 regression: the native streaming path must emit one
    /// `chat.completion.chunk` per decoded token, not a single buffered chunk.
    ///
    /// Before the fix, the entire response was generated and emitted as one
    /// content chunk (so `content` chunks == 1 regardless of token count). Here
    /// we drive the SSE builder with a stubbed short generation of three token
    /// deltas and assert it yields more than one content chunk.
    #[tokio::test]
    async fn native_stream_emits_one_chunk_per_token() {
        let request_id = Uuid::new_v4();
        let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (final_tx, final_rx) = tokio::sync::oneshot::channel::<NativeStreamOutcome>();

        // Stubbed generation: three incremental token deltas, then completion.
        for piece in ["Hel", "lo ", "world"] {
            token_tx.send(piece.to_string()).expect("send token");
        }
        final_tx
            .send(NativeStreamOutcome::Done {
                finish_reason: "stop".to_string(),
            })
            .expect("send terminal outcome");
        drop(token_tx);

        let mut stream = Box::pin(native_completion_sse_stream(
            request_id,
            "qwen".to_string(),
            token_rx,
            final_rx,
        ));

        let mut frames: Vec<String> = Vec::new();
        while let Some(item) = stream.next().await {
            let bytes = item.expect("stream frame");
            frames.push(String::from_utf8(bytes.to_vec()).expect("utf8 frame"));
        }

        let content_chunks = frames
            .iter()
            .filter(|frame| {
                frame.contains("chat.completion.chunk") && frame.contains("\"content\"")
            })
            .count();
        assert!(
            content_chunks > 1,
            "streaming must yield multiple content chunks, got {content_chunks}: {frames:?}"
        );

        // The three deltas must each appear as their own chunk and reassemble.
        let joined: String = frames.concat();
        assert!(joined.contains("Hel") && joined.contains("lo ") && joined.contains("world"));
        // Terminal framing: a finish_reason chunk and the [DONE] sentinel.
        assert!(joined.contains("\"finish_reason\":\"stop\""));
        assert!(joined.contains("data: [DONE]"));
    }

    /// A generation failure after the response is committed surfaces an SSE
    /// error object and still terminates the stream cleanly.
    #[tokio::test]
    async fn native_stream_failure_emits_error_object_and_done() {
        let request_id = Uuid::new_v4();
        let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (final_tx, final_rx) = tokio::sync::oneshot::channel::<NativeStreamOutcome>();
        final_tx
            .send(NativeStreamOutcome::Failed)
            .expect("send fail");
        drop(token_tx);

        let mut stream = Box::pin(native_completion_sse_stream(
            request_id,
            "qwen".to_string(),
            token_rx,
            final_rx,
        ));
        let mut joined = String::new();
        while let Some(item) = stream.next().await {
            joined.push_str(&String::from_utf8(item.expect("frame").to_vec()).expect("utf8"));
        }
        assert!(joined.contains("native_runtime_error"));
        assert!(joined.contains("data: [DONE]"));
    }
}
