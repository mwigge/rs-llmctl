use super::{
    error_response, with_request_id, ResolvedExternalProvider, ServerState, DEFAULT_SLO_LATENCY_MS,
};
use crate::audit::{AuditEvent, UsageEvent};
use crate::config::ExternalProviderKind;
use crate::observability::{
    emit_runtime_telemetry, RuntimeTelemetryEvent, TelemetryEventName, TelemetrySignal,
};
use crate::quota::Principal;
use crate::worker::SwapExecution;
use axum::http::StatusCode;
use axum::response::Response;
use chrono::Utc;
use opentelemetry::global;
use opentelemetry::KeyValue;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Duration;
use uuid::Uuid;

pub(super) async fn record_audit(
    state: &ServerState,
    request_id: Option<Uuid>,
    principal: Principal,
    action: impl Into<String>,
    resource: impl Into<String>,
    outcome: impl Into<String>,
    detail_json: Value,
) {
    let detail_json = if let Some(key_id) = principal.key_id.as_ref() {
        let mut detail = detail_json;
        match detail.as_object_mut() {
            Some(object) => {
                object
                    .entry("api_key_id".to_string())
                    .or_insert_with(|| json!(key_id));
                object
                    .entry("api_key_subject".to_string())
                    .or_insert_with(|| json!(principal.subject.as_str()));
                object
                    .entry("api_key_team".to_string())
                    .or_insert_with(|| json!(principal.team.as_str()));
                if let Some(owner) = principal.key_owner.as_deref() {
                    object
                        .entry("api_key_owner".to_string())
                        .or_insert_with(|| json!(owner));
                }
                if let Some(purpose) = principal.key_purpose.as_deref() {
                    object
                        .entry("api_key_purpose".to_string())
                        .or_insert_with(|| json!(purpose));
                }
                if let Some(status) = principal.key_status.as_deref() {
                    object
                        .entry("api_key_status".to_string())
                        .or_insert_with(|| json!(status));
                }
            }
            None => {
                detail = json!({
                    "detail": detail,
                    "api_key_id": key_id,
                    "api_key_subject": principal.subject.as_str(),
                    "api_key_team": principal.team.as_str(),
                    "api_key_owner": principal.key_owner.as_deref(),
                    "api_key_purpose": principal.key_purpose.as_deref(),
                    "api_key_status": principal.key_status.as_deref(),
                });
            }
        }
        detail
    } else {
        detail_json
    };
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
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::RequestRouting,
        Utc::now(),
        BTreeMap::from([
            ("llmctl.request_id".to_string(), json_string(request_id)),
            (
                "llmctl.audit.action".to_string(),
                json!(event.action.as_str()),
            ),
            (
                "llmctl.audit.resource".to_string(),
                json!(event.resource.as_str()),
            ),
            (
                "llmctl.audit.outcome".to_string(),
                json!(event.outcome.as_str()),
            ),
            ("llmctl.actor".to_string(), json!(event.actor.as_str())),
            ("llmctl.team".to_string(), json!(event.team.as_str())),
        ]),
    ));
}

/// Records an audit event for a rejected/denied request and turns it into the
/// corresponding error response, in one call. This is the shared shape behind
/// the many `record_audit(...).await; return with_request_id(error_response(...))`
/// blocks in the chat and embeddings handlers.
///
/// The argument count is inherent to the data each audit entry + response
/// needs (actor, action/resource/outcome triple, and status/code/message);
/// splitting it into a config struct would just move the same fields one
/// level up without making call sites clearer.
#[allow(clippy::too_many_arguments)]
pub(super) async fn audit_reject(
    state: &ServerState,
    request_id: Uuid,
    principal: Principal,
    action: impl Into<String>,
    resource: impl Into<String>,
    outcome: impl Into<String>,
    status: StatusCode,
    code: &str,
    message: String,
    detail: Value,
) -> Response {
    audit_reject_response(
        state,
        request_id,
        principal,
        action,
        resource,
        outcome,
        error_response(status, code, message),
        detail,
    )
    .await
}

/// Like `audit_reject`, but for the rejection paths whose response is built by
/// a helper other than `error_response` (e.g. `auth_error_response`,
/// `model_route_error_response`).
#[allow(clippy::too_many_arguments)]
pub(super) async fn audit_reject_response(
    state: &ServerState,
    request_id: Uuid,
    principal: Principal,
    action: impl Into<String>,
    resource: impl Into<String>,
    outcome: impl Into<String>,
    response: Response,
    detail: Value,
) -> Response {
    record_audit(
        state,
        Some(request_id),
        principal,
        action,
        resource,
        outcome,
        detail,
    )
    .await;
    with_request_id(response, request_id)
}

pub(super) async fn record_swap_execution(
    state: &ServerState,
    request_id: Option<Uuid>,
    principal: Principal,
    execution: &SwapExecution,
) {
    record_audit(
        state,
        request_id,
        principal,
        "admin.swap",
        execution.plan.replacement.as_str(),
        if execution.success {
            "allowed"
        } else {
            "failed"
        },
        json!({
            "mode": execution.mode,
            "active": execution.plan.active,
            "replacement": execution.plan.replacement,
            "steps": execution.plan.steps,
            "statuses": execution.statuses,
        }),
    )
    .await;
}

/// Emits quota-decision telemetry. The counting row itself is persisted
/// atomically under the admission lock by [`crate::quota::admit_request`]
/// (fail-closed), so this is telemetry only — it must not write the counting
/// row, or the requests-per-minute count would be double-incremented.
pub(super) fn record_quota_decision(
    request_id: Option<Uuid>,
    principal: &Principal,
    model: &str,
    decision: &crate::quota::QuotaDecision,
) {
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::QuotaDecision,
        Utc::now(),
        BTreeMap::from([
            ("llmctl.request_id".to_string(), json_string(request_id)),
            ("llmctl.model".to_string(), json!(model)),
            (
                "llmctl.actor".to_string(),
                json!(principal.subject.as_str()),
            ),
            ("llmctl.team".to_string(), json!(principal.team.as_str())),
            ("llmctl.quota.allowed".to_string(), json!(decision.allowed)),
            (
                "llmctl.quota.reason".to_string(),
                json!(decision.reason.as_str()),
            ),
        ]),
    ));
}

fn json_string(value: Option<Uuid>) -> Value {
    value
        .map(|value| Value::String(value.to_string()))
        .unwrap_or(Value::Null)
}

pub(super) struct UsageRecordInput<'a> {
    pub(super) request_id: Uuid,
    pub(super) principal: &'a Principal,
    pub(super) model: &'a str,
    pub(super) input_tokens: u64,
    pub(super) output_tokens: u64,
    pub(super) latency_ms: u64,
    pub(super) status: &'a str,
    pub(super) accounting_mode: &'a str,
    pub(super) gen_ai_system: &'a str,
}

pub(super) async fn record_usage(state: &ServerState, input: UsageRecordInput<'_>) {
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
    record_usage_telemetry(&event, input.accounting_mode, input.gen_ai_system);
    dispatch_usage_webhook(state, &event, input.accounting_mode);
}

pub(super) fn gen_ai_system_for_provider(
    provider: Option<&ResolvedExternalProvider>,
) -> &'static str {
    match provider.map(|p| &p.kind) {
        None => "llmctl.native",
        Some(ExternalProviderKind::OpenAiCompatible) => "openai",
        Some(ExternalProviderKind::VertexAi) => "vertex_ai",
        Some(ExternalProviderKind::OpenRouter) => "openrouter",
    }
}

/// Build the attribute set for a usage span: the existing `llmctl.*` attributes
/// plus the OTel GenAI semantic-convention attributes
/// (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`,
/// `gen_ai.response.model`, `gen_ai.usage.*`) so traces align with the
/// conventions Langfuse and other GenAI-aware OTel consumers expect.
pub(super) fn usage_span_attributes(
    event: &UsageEvent,
    accounting_mode: &str,
    gen_ai_system: &str,
) -> BTreeMap<String, Value> {
    BTreeMap::from([
        (
            "llmctl.request_id".to_string(),
            json!(event.request_id.to_string()),
        ),
        ("llmctl.model".to_string(), json!(event.model.as_str())),
        ("llmctl.actor".to_string(), json!(event.actor.as_str())),
        ("llmctl.team".to_string(), json!(event.team.as_str())),
        ("llmctl.latency_ms".to_string(), json!(event.latency_ms)),
        ("llmctl.status".to_string(), json!(event.status.as_str())),
        (
            "llmctl.token_accounting.mode".to_string(),
            json!(accounting_mode),
        ),
        ("gen_ai.system".to_string(), json!(gen_ai_system)),
        ("gen_ai.operation.name".to_string(), json!("chat")),
        (
            "gen_ai.request.model".to_string(),
            json!(event.model.as_str()),
        ),
        (
            "gen_ai.response.model".to_string(),
            json!(event.model.as_str()),
        ),
        (
            "gen_ai.usage.input_tokens".to_string(),
            json!(event.input_tokens),
        ),
        (
            "gen_ai.usage.output_tokens".to_string(),
            json!(event.output_tokens),
        ),
    ])
}

/// JSON payload delivered to the configured usage webhook — the same
/// usage/lineage metadata recorded in the audit trail and emitted as OTel
/// attributes, shaped for ecosystems that consume callbacks rather than OTLP.
pub(super) fn webhook_payload(event: &UsageEvent, accounting_mode: &str) -> Value {
    json!({
        "type": "llmctl.usage",
        "id": event.id.to_string(),
        "request_id": event.request_id.to_string(),
        "at": event.at.to_rfc3339(),
        "model": event.model,
        "actor": event.actor,
        "team": event.team,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "latency_ms": event.latency_ms,
        "status": event.status,
        "token_accounting_mode": accounting_mode,
    })
}

/// Fire-and-forget delivery of the usage webhook, if configured. Failures are
/// logged at `warn` and never affect the in-flight request — the webhook is a
/// best-effort sink, not part of the request's correctness contract.
fn dispatch_usage_webhook(state: &ServerState, event: &UsageEvent, accounting_mode: &str) {
    let webhook = &state.cfg.observability.webhook;
    if !webhook.enabled {
        return;
    }
    let Some(url) = webhook
        .url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(str::to_string)
    else {
        return;
    };

    if let Err(err) = crate::observability::validate_http_endpoint(&url) {
        tracing::warn!(url = %url, error = %err, "usage webhook URL rejected by SSRF guard");
        return;
    }

    let payload = webhook_payload(event, accounting_mode);
    let client = state.client.clone();
    let headers = webhook.headers.clone();
    let timeout = Duration::from_millis(webhook.timeout_ms.max(1));

    tokio::spawn(async move {
        let mut request = client.post(&url).json(&payload).timeout(timeout);
        for (name, value) in &headers {
            request = request.header(name.as_str(), value.as_str());
        }
        match request.send().await {
            Ok(response) if !response.status().is_success() => {
                tracing::warn!(
                    url = %url,
                    status = %response.status(),
                    "usage webhook delivery returned a non-success status"
                );
            }
            Err(err) => {
                tracing::warn!(url = %url, error = %err, "failed to deliver usage webhook");
            }
            Ok(_) => {}
        }
    });
}

fn record_usage_telemetry(event: &UsageEvent, accounting_mode: &str, gen_ai_system: &str) {
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::RequestRouting,
        Utc::now(),
        usage_span_attributes(event, accounting_mode, gen_ai_system),
    ));
    let attributes = [
        KeyValue::new("llmctl.model", event.model.clone()),
        KeyValue::new("llmctl.actor", event.actor.clone()),
        KeyValue::new("llmctl.team", event.team.clone()),
        KeyValue::new("llmctl.status", event.status.clone()),
        KeyValue::new("model", event.model.clone()),
        KeyValue::new("team", event.team.clone()),
        KeyValue::new("endpoint", "/v1/chat/completions"),
        KeyValue::new("status", slo_status(&event.status)),
        KeyValue::new("token_accounting_mode", accounting_mode.to_string()),
    ];
    let meter = global::meter(crate::SERVICE_NAME);
    meter
        .u64_counter("llmctl_requests_total")
        .with_description(
            "Total OpenAI-compatible model requests by endpoint, model, team, and status",
        )
        .build()
        .add(1, &attributes);
    if event.status != "ok" {
        meter
            .u64_counter("llmctl_request_errors_total")
            .with_description(
                "Failed OpenAI-compatible model requests by endpoint, model, team, and status",
            )
            .build()
            .add(1, &attributes);
    }
    meter
        .u64_counter("llmctl.tokens.input")
        .with_description("Input tokens reported by model workers")
        .build()
        .add(event.input_tokens, &attributes);
    meter
        .u64_counter("llmctl.tokens.output")
        .with_description("Output tokens reported by model workers")
        .build()
        .add(event.output_tokens, &attributes);
    let genai_attributes = [KeyValue::new("gen_ai.system", gen_ai_system.to_string())];
    crate::observability::genai_input_tokens_counter().add(event.input_tokens, &genai_attributes);
    crate::observability::genai_output_tokens_counter().add(event.output_tokens, &genai_attributes);
    meter
        .u64_histogram("llmctl_request_latency_ms")
        .with_description("Model request latency in milliseconds")
        .build()
        .record(event.latency_ms, &attributes);
    if event.status != "ok" || event.latency_ms > DEFAULT_SLO_LATENCY_MS {
        meter
            .u64_counter("llmctl_slo_violations_total")
            .with_description("Requests that violate the default latency or success SLO")
            .build()
            .add(1, &attributes);
    }
    tracing::info!(
        request_id = %event.request_id,
        model = %event.model,
        actor = %event.actor,
        team = %event.team,
        input_tokens = event.input_tokens,
        output_tokens = event.output_tokens,
        latency_ms = event.latency_ms,
        status = %event.status,
        "model usage recorded"
    );
}

pub(super) fn slo_status(status: &str) -> &'static str {
    if status == "ok" {
        "ok"
    } else {
        "error"
    }
}
