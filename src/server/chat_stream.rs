use super::{
    build_response, chat_route_audit_detail, elapsed_ms, emit_gen_ai_inference_span,
    error_response, gen_ai_system_for_provider, record_audit, record_upstream_telemetry,
    record_usage, response_headers, upstream_error_status, with_chat_metadata, with_request_id,
    ServerState, SseUsageParser, UpstreamRequestContext, UsageRecordInput,
};
use axum::body::{Body, Bytes};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use tokio::time::timeout;

pub(super) async fn stream_upstream(
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

pub(super) fn stream_status(input_tokens: u64, output_tokens: u64) -> &'static str {
    if input_tokens == 0 && output_tokens == 0 {
        "stream_unmetered"
    } else {
        "ok"
    }
}
