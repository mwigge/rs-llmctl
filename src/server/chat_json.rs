use super::{
    build_response, chat_route_audit_detail, elapsed_ms, error_response,
    gen_ai_system_for_provider, record_audit, record_upstream_failure, record_upstream_telemetry,
    record_usage, response_headers, upstream_error_status, usage_tokens, with_chat_metadata,
    with_request_id, ServerState, UpstreamRequestContext, UsageRecordInput,
};
use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use serde_json::json;
use std::sync::Arc;
use tokio::time::timeout;

pub(super) async fn json_upstream(
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
