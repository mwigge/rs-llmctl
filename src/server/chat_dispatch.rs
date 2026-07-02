use super::{
    model_upstream_timeout, record_upstream_failure, rewrite_chat_model, upstream_request_error,
    AdmissionPermit, GenAiRequestParams, ResolvedExternalProvider, ResolvedModelRoute, ServerState,
    ToolAuditDetail,
};
use crate::observability::inject_trace_context;
use crate::quota::Principal;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use std::env;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use uuid::Uuid;

fn upstream_for_alias(
    state: &ServerState,
    upstream_alias: &str,
) -> std::result::Result<String, String> {
    state
        .upstreams
        .get(upstream_alias)
        .or_else(|| state.upstreams.get("*"))
        .cloned()
        .ok_or_else(|| format!("no upstream configured for model {upstream_alias}"))
}

#[derive(Debug)]
pub(super) enum DispatchFailure {
    NoUpstream(String),
    BadRequest(String),
    Request {
        status: StatusCode,
        code: &'static str,
        message: String,
        usage_status: &'static str,
    },
}

pub(super) async fn dispatch_chat_request(
    state: &ServerState,
    route: &ResolvedModelRoute,
    original_body: &[u8],
    request_id: Uuid,
    principal: &Principal,
    model: &str,
    started: Instant,
) -> std::result::Result<(reqwest::Response, String), DispatchFailure> {
    let mut aliases = vec![route.upstream_alias.clone()];
    aliases.extend(route.fallback_aliases.clone());
    let mut last_failure = None;

    for alias in aliases {
        let external_provider = if alias == route.upstream_alias {
            route.external_provider.clone()
        } else {
            None
        };
        let attempt_route = ResolvedModelRoute {
            requested_alias: route.requested_alias.clone(),
            upstream_alias: alias.clone(),
            fallback_aliases: Vec::new(),
            external_provider,
        };
        let body = rewrite_chat_model(original_body, &attempt_route)
            .map_err(DispatchFailure::BadRequest)?;
        let target = dispatch_target(state, &attempt_route)?;
        let upstream_base = target.base_url.clone();
        if !state.circuit_breakers.allow_request(
            &upstream_base,
            Duration::from_secs(state.cfg.server.circuit_breaker_reset_seconds),
        ) {
            last_failure = Some(DispatchFailure::Request {
                status: StatusCode::BAD_GATEWAY,
                code: "upstream_circuit_open",
                message: "upstream circuit breaker is open".to_string(),
                usage_status: "upstream_circuit_open",
            });
            continue;
        }
        let upstream = format!("{upstream_base}/v1/chat/completions");
        let mut request_builder = inject_trace_context(
            state
                .client
                .post(upstream)
                .header(CONTENT_TYPE, "application/json")
                .body(body),
        );
        if let Some(api_key) = target.api_key {
            request_builder = request_builder.bearer_auth(api_key);
        }
        match timeout(
            model_upstream_timeout(state, &route.requested_alias),
            request_builder.send(),
        )
        .await
        {
            Ok(Ok(response)) if should_retry_upstream_status(response.status()) => {
                state
                    .circuit_breakers
                    .record_failure(&upstream_base, state.cfg.server.circuit_breaker_failures);
                retry_after_delay(&response).await;
                last_failure = Some(DispatchFailure::Request {
                    status: StatusCode::BAD_GATEWAY,
                    code: "upstream_error",
                    message: "upstream request failed".to_string(),
                    usage_status: "upstream_error",
                });
                continue;
            }
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    state.circuit_breakers.record_success(&upstream_base);
                }
                return Ok((response, alias));
            }
            Ok(Err(err)) => {
                state
                    .circuit_breakers
                    .record_failure(&upstream_base, state.cfg.server.circuit_breaker_failures);
                let (status, code, message, usage_status) = upstream_request_error(&err);
                record_upstream_failure(state, request_id, principal, model, started, usage_status)
                    .await;
                last_failure = Some(DispatchFailure::Request {
                    status,
                    code,
                    message,
                    usage_status,
                });
            }
            Err(_) => {
                state
                    .circuit_breakers
                    .record_failure(&upstream_base, state.cfg.server.circuit_breaker_failures);
                record_upstream_failure(state, request_id, principal, model, started, "timeout")
                    .await;
                last_failure = Some(DispatchFailure::Request {
                    status: StatusCode::GATEWAY_TIMEOUT,
                    code: "timeout",
                    message: "upstream request timed out".to_string(),
                    usage_status: "timeout",
                });
            }
        }
    }

    Err(last_failure.unwrap_or_else(|| {
        DispatchFailure::NoUpstream(format!(
            "no upstream configured for model {}",
            route.upstream_alias
        ))
    }))
}

#[derive(Debug)]
struct DispatchTarget {
    base_url: String,
    api_key: Option<String>,
}

fn dispatch_target(
    state: &ServerState,
    route: &ResolvedModelRoute,
) -> std::result::Result<DispatchTarget, DispatchFailure> {
    if let Some(provider) = route.external_provider.as_ref() {
        let api_key = env::var(&provider.api_key_env).map_err(|_| DispatchFailure::Request {
            status: StatusCode::BAD_GATEWAY,
            code: "provider_api_key_unavailable",
            message: format!(
                "external provider {} API key is not available from configured environment reference",
                provider.id
            ),
            usage_status: "provider_api_key_unavailable",
        })?;
        if api_key.trim().is_empty() {
            return Err(DispatchFailure::Request {
                status: StatusCode::BAD_GATEWAY,
                code: "provider_api_key_unavailable",
                message: format!(
                    "external provider {} API key is empty in configured environment reference",
                    provider.id
                ),
                usage_status: "provider_api_key_unavailable",
            });
        }
        return Ok(DispatchTarget {
            base_url: provider.base_url.clone(),
            api_key: Some(api_key.trim().to_string()),
        });
    }

    Ok(DispatchTarget {
        base_url: upstream_for_alias(state, &route.upstream_alias)
            .map_err(DispatchFailure::NoUpstream)?,
        api_key: None,
    })
}

fn should_retry_upstream_status(status: StatusCode) -> bool {
    status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS
}

async fn retry_after_delay(response: &reqwest::Response) {
    let delay = response
        .headers()
        .get(axum::http::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|seconds| Duration::from_secs(seconds.min(2)))
        .unwrap_or_else(|| Duration::from_millis(100));
    tokio::time::sleep(delay).await;
}

pub(super) struct UpstreamRequestContext {
    pub(super) request_id: Uuid,
    pub(super) principal: Principal,
    pub(super) model: String,
    pub(super) upstream_model: String,
    pub(super) upstream_timeout: Duration,
    pub(super) external_provider: Option<ResolvedExternalProvider>,
    pub(super) tool_audit: ToolAuditDetail,
    pub(super) started: Instant,
    pub(super) admission: AdmissionPermit,
    pub(super) gen_ai: GenAiRequestParams,
}
