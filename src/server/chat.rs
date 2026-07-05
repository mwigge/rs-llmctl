use super::{
    apply_message_redactions, audit_reject, audit_reject_response, authenticate_with_chat_scope,
    chat_route_audit_detail, dispatch_chat_request, dispatch_native_chat, draining_response,
    elapsed_ms, error_response, gen_ai_params_from_request, gen_ai_system_for_provider,
    json_upstream, model_route_error_response, model_upstream_timeout,
    record_admission_busy_telemetry, record_audit, record_quota_decision,
    record_request_lineage_joins, record_usage, request_id_from_headers, resolve_model_route,
    runtime_lineage_from_headers_and_metadata, stream_upstream, with_request_id, AdmissionError,
    ChatCompletionRequest, DispatchFailure, NativeChatContext, ServerState, UpstreamRequestContext,
    UsageRecordInput,
};
use crate::config::Config;
use crate::guardrails;
use crate::native;
use crate::quota::{matching_quota_policies, quota_is_subject_scoped, Principal};
use axum::body::Bytes;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

pub(super) async fn chat_completions(
    State(state): State<Arc<ServerState>>,
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
        "chat.completions",
    )
    .await
    {
        Ok(principal) => principal,
        Err(response) => return response,
    };

    let mut request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            return audit_reject(
                &state,
                request_id,
                principal,
                "chat.completions",
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
    let lineage = runtime_lineage_from_headers_and_metadata(&headers, request.metadata.as_ref());

    let route = match resolve_model_route(&state.cfg, &request.model, request_id) {
        Ok(route) => route,
        Err(err) => {
            let response = model_route_error_response(&err);
            return audit_reject_response(
                &state,
                request_id,
                principal,
                "chat.completions",
                request.model,
                "rejected",
                response,
                json!({ "reason": err.to_string() }),
            )
            .await;
        }
    };
    let model = route.requested_alias.clone();

    // Body forwarded on the proxy path. Kept in sync with any guardrail
    // redactions applied below so the worker/upstream never receives content
    // the audit trail reports as redacted.
    let mut outbound_body = body.clone();

    if state.cfg.guardrails.is_active() {
        let message_texts: Vec<(usize, String)> = request
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| (index, native::message_content_text(message)))
            .collect();
        let verdict = guardrails::evaluate(&message_texts, &state.cfg.guardrails);

        if verdict.is_blocked() {
            let message = format!(
                "request blocked by guardrails: {}",
                verdict.block_reasons.join(", ")
            );
            return audit_reject(
                &state,
                request_id,
                principal,
                "chat.completions",
                model,
                "denied",
                StatusCode::BAD_REQUEST,
                "guardrail_blocked",
                message,
                json!({
                    "reason": "guardrail_violation",
                    "guardrails": verdict.block_reasons,
                    "findings": verdict.findings.audit_detail(),
                }),
            )
            .await;
        }

        if verdict.has_findings() {
            record_audit(
                &state,
                Some(request_id),
                principal.clone(),
                "chat.completions",
                model.clone(),
                "flagged",
                json!({
                    "reason": "guardrail_match",
                    "findings": verdict.findings.audit_detail(),
                    "redacted": !verdict.redactions.is_empty(),
                }),
            )
            .await;
        }

        if !verdict.redactions.is_empty() {
            // Apply the same redactions to the raw body that gets proxied
            // upstream, not just the parsed request struct — otherwise the
            // worker/upstream path would forward the original secret while the
            // audit records `redacted: true`.
            match apply_message_redactions(&body, &verdict.redactions) {
                Ok(redacted) => outbound_body = redacted,
                Err(err) => {
                    return audit_reject(
                        &state,
                        request_id,
                        principal,
                        "chat.completions",
                        model,
                        "rejected",
                        StatusCode::BAD_REQUEST,
                        "bad_request",
                        "request body must be valid JSON".to_string(),
                        json!({ "reason": err }),
                    )
                    .await;
                }
            }
            for (index, redacted_text) in &verdict.redactions {
                if let Some(message) = request.messages.get_mut(*index) {
                    message.content = Some(Value::String(redacted_text.clone()));
                }
            }
        }
    }

    record_request_lineage_joins(
        &state,
        request_id,
        &lineage,
        Some(model.as_str()),
        "chat.completions",
    )
    .await;
    let quota = match crate::quota::admit_request(
        &state.storage,
        &state.cfg.quotas,
        &principal,
        &model,
        Some(request_id),
        json!({ "configured_quotas": state.cfg.quotas.len() }),
    )
    .await
    {
        Ok(decision) => decision,
        Err(err) => {
            return audit_reject(
                &state,
                request_id,
                principal,
                "chat.completions",
                model,
                "rejected",
                StatusCode::INTERNAL_SERVER_ERROR,
                "quota_error",
                "quota admission is unavailable".to_string(),
                json!({ "reason": err.to_string() }),
            )
            .await;
        }
    };
    record_quota_decision(Some(request_id), &principal, &model, &quota);

    if !quota.allowed {
        let reason = quota.reason.clone();
        return audit_reject(
            &state,
            request_id,
            principal,
            "chat.completions",
            model,
            "denied",
            StatusCode::TOO_MANY_REQUESTS,
            "quota_exceeded",
            quota.reason,
            json!({ "reason": reason }),
        )
        .await;
    }

    record_audit(
        &state,
        Some(request_id),
        principal.clone(),
        "chat.completions",
        model.clone(),
        "allowed",
        chat_route_audit_detail(
            &request.tool_audit_detail(),
            json!({ "stream": request.stream, "upstream_model": route.upstream_alias }),
            route.external_provider.as_ref(),
        ),
    )
    .await;

    let admission = match state
        .admission
        .try_acquire_for_all(quota_admission_scopes(&state.cfg, &principal))
    {
        Ok(permit) => permit,
        Err(AdmissionError::Busy) => {
            record_admission_busy_telemetry(&model, &principal, request.stream);
            return audit_reject(
                &state,
                request_id,
                principal,
                "chat.completions",
                model,
                "denied",
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
                "server is busy; retry later".to_string(),
                json!({ "reason": "admission_limit_exceeded" }),
            )
            .await;
        }
    };

    let has_subprocess_upstream =
        state.upstreams.contains_key(&route.upstream_alias) || state.upstreams.contains_key("*");

    if route.external_provider.is_some() || has_subprocess_upstream {
        // Gate on live worker state before proxying: a supervised worker that is
        // draining/stopping/stopped/crashed must not receive the request, even
        // though its static upstream entry still exists. The guard keeps the
        // worker's in-flight count raised so drain can wait for it to finish.
        let _worker_guard = if route.external_provider.is_none() {
            match admit_to_worker(&state, request_id, &route.upstream_alias).await {
                Ok(guard) => guard,
                Err(response) => return response,
            }
        } else {
            None
        };
        let tool_audit = request.tool_audit_detail();
        let gen_ai = gen_ai_params_from_request(&request);
        let upstream_timeout = model_upstream_timeout(&state, &model);
        return match dispatch_chat_request(
            &state,
            &route,
            &outbound_body,
            request_id,
            &principal,
            &model,
            started,
        )
        .await
        {
            Ok((upstream_response, upstream_model)) if request.stream => {
                stream_upstream(
                    state,
                    upstream_response,
                    UpstreamRequestContext {
                        request_id,
                        principal,
                        model,
                        upstream_model,
                        upstream_timeout,
                        external_provider: route.external_provider,
                        tool_audit,
                        started,
                        admission,
                        gen_ai,
                    },
                )
                .await
            }
            Ok((upstream_response, upstream_model)) => {
                json_upstream(
                    state,
                    upstream_response,
                    UpstreamRequestContext {
                        request_id,
                        principal,
                        model,
                        upstream_model,
                        upstream_timeout,
                        external_provider: route.external_provider,
                        tool_audit,
                        started,
                        admission,
                        gen_ai,
                    },
                )
                .await
            }
            Err(DispatchFailure::BadRequest(err)) => with_request_id(
                error_response(StatusCode::BAD_REQUEST, "bad_request", err),
                request_id,
            ),
            Err(DispatchFailure::NoUpstream(message)) => with_request_id(
                error_response(StatusCode::BAD_GATEWAY, "upstream_unavailable", message),
                request_id,
            ),
            Err(DispatchFailure::Request {
                status,
                code,
                message,
                usage_status,
            }) => {
                record_usage(
                    &state,
                    UsageRecordInput {
                        request_id,
                        principal: &principal,
                        model: &model,
                        input_tokens: 0,
                        output_tokens: 0,
                        latency_ms: elapsed_ms(started),
                        status: usage_status,
                        accounting_mode: "none",
                        gen_ai_system: gen_ai_system_for_provider(None),
                    },
                )
                .await;
                with_request_id(error_response(status, code, message), request_id)
            }
        };
    }

    dispatch_native_chat(
        state,
        NativeChatContext {
            request_id,
            principal,
            model,
            upstream_model: route.upstream_alias,
            tool_audit: request.tool_audit_detail(),
            started,
            _admission: admission,
        },
        request,
    )
    .await
}

/// Gates a request on the live admission state of the supervised worker serving
/// `alias`. Returns:
/// - `Ok(None)` when there is no external worker supervisor (native in-process
///   serving), leaving routing to the native engine path unchanged.
/// - `Ok(Some(guard))` when the worker is admitting; the guard holds the
///   worker's in-flight count raised until dropped so drain can wait for it.
/// - `Err(response)` — a 503 `model_unavailable` — when the worker is unknown,
///   not yet ready, or draining/stopping/stopped/crashed.
async fn admit_to_worker(
    state: &ServerState,
    request_id: uuid::Uuid,
    alias: &str,
) -> Result<Option<crate::worker::InFlightGuard>, Response> {
    let worker_id = crate::worker::WorkerId::new(alias);
    // Fast path: read the shared admission registry without locking the
    // supervisor (which a swap/drain may hold across a long model load).
    let admission = if let Some(registry) = state.worker_admissions.as_ref() {
        registry
            .read()
            .ok()
            .and_then(|registry| registry.get(&worker_id).cloned())
    } else if let Some(worker_control) = state.worker_control.as_ref() {
        // Fallback for callers that wired a supervisor without exposing its
        // registry: read under the supervisor lock.
        worker_control.lock().await.worker_admission(&worker_id)
    } else {
        // No external worker supervisor (native in-process serving): leave
        // routing to the native engine path unchanged.
        return Ok(None);
    };
    match admission.and_then(|admission| admission.try_enter()) {
        Some(guard) => Ok(Some(guard)),
        None => Err(with_request_id(
            error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_unavailable",
                format!("model {alias} is not currently serving; retry shortly"),
            ),
            request_id,
        )),
    }
}

pub(super) fn quota_admission_scopes(cfg: &Config, principal: &Principal) -> Vec<(String, usize)> {
    let mut scopes = BTreeMap::<String, usize>::new();
    for quota in matching_quota_policies(&cfg.quotas, principal) {
        let Ok(limit) = usize::try_from(quota.max_concurrency) else {
            continue;
        };
        if limit == 0 {
            continue;
        }
        let scope = if quota_is_subject_scoped(quota, principal) {
            format!("subject:{}", principal.subject)
        } else {
            format!("team:{}", principal.team)
        };
        scopes
            .entry(scope)
            .and_modify(|existing| *existing = (*existing).min(limit))
            .or_insert(limit);
    }
    scopes.into_iter().collect()
}
