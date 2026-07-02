use super::{
    elapsed_ms, gen_ai_system_for_provider, record_usage, slo_status, ServerState, UsageRecordInput,
};
use crate::observability::{
    emit_runtime_telemetry, RuntimeTelemetryEvent, TelemetryEventName, TelemetrySignal,
};
use crate::quota::Principal;
use axum::http::StatusCode;
use chrono::Utc;
use opentelemetry::global;
use opentelemetry::KeyValue;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

pub(super) fn model_upstream_timeout(state: &ServerState, alias: &str) -> Duration {
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

pub(super) fn upstream_request_error(
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

pub(super) fn upstream_error_status(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else {
        "upstream_error"
    }
}

pub(super) async fn record_upstream_failure(
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
            gen_ai_system: gen_ai_system_for_provider(None),
        },
    )
    .await;
    record_upstream_telemetry(model, "unknown", 0, elapsed_ms(started), status);
}

pub(super) fn record_admission_busy_telemetry(model: &str, principal: &Principal, stream: bool) {
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

pub(super) fn record_upstream_telemetry(
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

pub(super) fn record_circuit_breaker_state(upstream: &str, state: &str, consecutive_failures: u32) {
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
