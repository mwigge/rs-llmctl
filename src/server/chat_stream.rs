use super::{
    build_response, chat_route_audit_detail, elapsed_ms, emit_gen_ai_inference_span,
    error_response, gen_ai_system_for_provider, record_audit, record_upstream_telemetry,
    record_usage, response_headers, upstream_error_status, with_chat_metadata, with_request_id,
    ResolvedExternalProvider, ServerState, SseUsageParser, ToolAuditDetail, UpstreamRequestContext,
    UsageRecordInput,
};
use crate::quota::Principal;
use axum::body::{Body, Bytes};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::time::timeout;
use uuid::Uuid;

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
        // Bug 9: terminal usage/audit for the normal/timeout/error branches is
        // recorded inline (and the guard disarmed). If the client disconnects
        // first, this guard's Drop records it instead so accounting is not lost.
        let mut disconnect_guard = StreamDisconnectGuard::new(StreamDisconnectData {
            state: state.clone(),
            request_id,
            principal: principal.clone(),
            model: model.clone(),
            upstream_model: upstream_model.clone(),
            external_provider: external_provider.clone(),
            tool_audit: tool_audit.clone(),
            status_code: status.as_u16(),
            started,
            input_tokens: 0,
            output_tokens: 0,
        });
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
                    disconnect_guard.disarm();
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
                            disconnect_guard.update_counts(input_tokens, output_tokens);
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
                            disconnect_guard.disarm();
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
                    disconnect_guard.disarm();
                    yield Err::<Bytes, std::io::Error>(std::io::Error::other("upstream stream failed"));
                    return;
                }
            }
        }
        // Reached the end of the upstream stream while still connected: the
        // terminal usage/audit below runs inline, so disarm the disconnect guard.
        disconnect_guard.disarm();
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

/// Status recorded when a streaming client disconnects before the upstream
/// stream finishes and the terminal accounting inside the generator is skipped.
pub(super) const STREAM_DISCONNECT_STATUS: &str = "client_disconnected";

/// Terminal accounting inputs for an in-flight upstream stream.
///
/// The normal-completion, timeout, and error branches of the streaming
/// generator already record usage/audit inline and then [`disarm`] this guard.
/// If instead the client disconnects mid-stream, the generator future is
/// dropped without reaching any of those branches — so [`Drop`] records the
/// terminal usage and audit from a detached task, closing the billing/audit
/// gap a client could otherwise trigger at will (Bug 9).
///
/// [`disarm`]: StreamDisconnectGuard::disarm
pub(super) struct StreamDisconnectData {
    pub(super) state: Arc<ServerState>,
    pub(super) request_id: Uuid,
    pub(super) principal: Principal,
    pub(super) model: String,
    pub(super) upstream_model: String,
    pub(super) external_provider: Option<ResolvedExternalProvider>,
    pub(super) tool_audit: ToolAuditDetail,
    pub(super) status_code: u16,
    pub(super) started: Instant,
    pub(super) input_tokens: u64,
    pub(super) output_tokens: u64,
}

pub(super) struct StreamDisconnectGuard {
    data: Option<StreamDisconnectData>,
}

impl StreamDisconnectGuard {
    pub(super) fn new(data: StreamDisconnectData) -> Self {
        Self { data: Some(data) }
    }

    /// Updates the running token counts so a later disconnect records the most
    /// recent metered totals.
    pub(super) fn update_counts(&mut self, input_tokens: u64, output_tokens: u64) {
        if let Some(data) = self.data.as_mut() {
            data.input_tokens = input_tokens;
            data.output_tokens = output_tokens;
        }
    }

    /// Marks the terminal accounting as already recorded (normal completion,
    /// timeout, or upstream error), so [`Drop`] does not record it again.
    pub(super) fn disarm(&mut self) {
        self.data = None;
    }

    /// Records terminal usage + audit for `data`. Shared by the disconnect
    /// [`Drop`] path; the status reflects why the stream ended.
    pub(super) async fn record_terminal(data: StreamDisconnectData, status: &str) {
        let metered = data.input_tokens > 0 || data.output_tokens > 0;
        record_usage(
            &data.state,
            UsageRecordInput {
                request_id: data.request_id,
                principal: &data.principal,
                model: &data.model,
                input_tokens: data.input_tokens,
                output_tokens: data.output_tokens,
                latency_ms: elapsed_ms(data.started),
                status,
                accounting_mode: "upstream",
                gen_ai_system: gen_ai_system_for_provider(data.external_provider.as_ref()),
            },
        )
        .await;
        record_upstream_telemetry(
            &data.model,
            &data.upstream_model,
            data.status_code,
            elapsed_ms(data.started),
            status,
        );
        record_audit(
            &data.state,
            Some(data.request_id),
            data.principal.clone(),
            "chat.completions",
            data.model.clone(),
            status,
            chat_route_audit_detail(
                &data.tool_audit,
                json!({
                    "status": data.status_code,
                    "stream": true,
                    "disconnected": true,
                    "metered": metered
                }),
                data.external_provider.as_ref(),
            ),
        )
        .await;
    }
}

impl Drop for StreamDisconnectGuard {
    fn drop(&mut self) {
        let Some(data) = self.data.take() else {
            return;
        };
        // Detach the recording: Drop cannot await. Only spawn when a runtime is
        // present (it always is on the request path); otherwise skip rather than
        // panic inside Drop.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                StreamDisconnectGuard::record_terminal(data, STREAM_DISCONNECT_STATUS).await;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::storage::Storage;
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    fn test_state(storage: Storage) -> Arc<ServerState> {
        Arc::new(ServerState {
            cfg: Arc::new(Config::default()),
            storage,
            started_at: std::time::Instant::now(),
            client: reqwest::Client::new(),
            upstreams: BTreeMap::new(),
            admission: crate::server::AdmissionController::new(1),
            serving_limits: crate::server::ServingLimits::new(1, Duration::from_secs(30)),
            native_engines: BTreeMap::new(),
            worker_control: None,
            worker_admissions: None,
            draining: Arc::new(AtomicBool::new(false)),
            circuit_breakers: crate::server::CircuitBreakers::default(),
            auth_failures: crate::server::AuthFailureLimiter::default(),
        })
    }

    fn test_principal() -> Principal {
        Principal {
            subject: "alice".to_string(),
            team: "platform".to_string(),
            scopes: vec!["chat".to_string()],
            key_id: Some("alice-key".to_string()),
            key_owner: None,
            key_purpose: None,
            key_status: Some("active".to_string()),
        }
    }

    /// Bug 9 regression: a mid-stream client disconnect drops the streaming
    /// generator before any terminal branch runs. The disconnect guard must
    /// still record terminal usage and audit so a client cannot skip
    /// billing/audit by hanging up early.
    ///
    /// Before the fix the terminal `record_usage`/`record_audit` lived inside
    /// the generator, so dropping it recorded nothing — this test would find
    /// zero usage and zero audit rows.
    #[tokio::test]
    async fn dropped_stream_still_records_usage_and_audit() {
        let storage = Storage::in_memory().await.expect("storage");
        let state = test_state(storage.clone());
        let request_id = Uuid::new_v4();

        let guard = StreamDisconnectGuard::new(StreamDisconnectData {
            state: state.clone(),
            request_id,
            principal: test_principal(),
            model: "gpt-proxy".to_string(),
            upstream_model: "upstream-model".to_string(),
            external_provider: None,
            tool_audit: ToolAuditDetail {
                tool_schema_count: 0,
                tool_choice: serde_json::Value::Null,
                tool_call_count: 0,
            },
            status_code: 200,
            started: Instant::now(),
            input_tokens: 7,
            output_tokens: 11,
        });
        // Simulate the disconnect: drop without disarming (no terminal branch ran).
        drop(guard);

        // Drop detaches the recording; poll storage until it lands.
        let mut usage = Vec::new();
        for _ in 0..100 {
            usage = storage
                .usage_events_for_request(request_id)
                .await
                .expect("usage query");
            if !usage.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(usage.len(), 1, "dropped stream must still record usage");
        assert_eq!(usage[0].input_tokens, 7);
        assert_eq!(usage[0].output_tokens, 11);
        assert_eq!(usage[0].status, STREAM_DISCONNECT_STATUS);

        let audit = storage
            .audit_events_for_request(request_id)
            .await
            .expect("audit query");
        assert_eq!(audit.len(), 1, "dropped stream must still record audit");
        assert_eq!(audit[0].outcome.as_str(), STREAM_DISCONNECT_STATUS);
    }

    /// A disarmed guard (normal completion / timeout / error already recorded)
    /// must not double-record on drop.
    #[tokio::test]
    async fn disarmed_guard_records_nothing_on_drop() {
        let storage = Storage::in_memory().await.expect("storage");
        let state = test_state(storage.clone());
        let request_id = Uuid::new_v4();

        let mut guard = StreamDisconnectGuard::new(StreamDisconnectData {
            state: state.clone(),
            request_id,
            principal: test_principal(),
            model: "gpt-proxy".to_string(),
            upstream_model: "upstream-model".to_string(),
            external_provider: None,
            tool_audit: ToolAuditDetail {
                tool_schema_count: 0,
                tool_choice: serde_json::Value::Null,
                tool_call_count: 0,
            },
            status_code: 200,
            started: Instant::now(),
            input_tokens: 3,
            output_tokens: 4,
        });
        guard.disarm();
        drop(guard);

        // Give any (erroneously) spawned task a chance to run.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let usage = storage
            .usage_events_for_request(request_id)
            .await
            .expect("usage query");
        assert!(usage.is_empty(), "disarmed guard must not record usage");
    }
}
