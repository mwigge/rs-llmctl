use super::{error_response, with_request_id, ServerState};
use crate::audit::AuditEvent;
use crate::observability::{
    emit_runtime_telemetry, RuntimeTelemetryEvent, TelemetryEventName, TelemetrySignal,
};
use crate::quota::Principal;
use crate::storage::QuotaDecisionRecord;
use crate::worker::SwapExecution;
use axum::http::StatusCode;
use axum::response::Response;
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::BTreeMap;
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

pub(super) async fn record_quota_decision(
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
