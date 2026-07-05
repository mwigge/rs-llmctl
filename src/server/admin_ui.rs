//! Read-only admin/observability endpoints backing the embedded web UI at
//! `/ui`. Every `/v1/admin/*` data endpoint requires the `admin` scope and is
//! a thin wrapper over existing storage/config/runtime functions — no endpoint
//! here mutates state (all writes still flow through the `llmctl` CLI). Handlers
//! return typed JSON and fall back to empty/default shapes on missing data
//! rather than surfacing a 500 to the page.

use super::{
    auth_error_response, auth_source_key, authenticate_request, error_response, record_audit,
    request_id_from_headers, with_request_id, ServerState,
};
use crate::quota::Principal;
use crate::reporting;
use crate::storage::ModelInventoryRecord;
use crate::storage::QuotaDecisionRecord;
use crate::worker::StartupPlan;
use axum::extract::{ConnectInfo, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use chrono::{Duration as ChronoDuration, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use uuid::Uuid;

/// Cap on rows returned by the windowed usage/audit endpoints so the page
/// cannot pull an unbounded history into the browser.
const MAX_WINDOW_ROWS: usize = 500;
/// Default look-back window when the `window` query parameter is absent or
/// unparseable.
const DEFAULT_WINDOW: ChronoDuration = ChronoDuration::hours(24);
/// Longest look-back window honoured; larger requests are clamped to this.
const MAX_WINDOW_DAYS: i64 = 90;

fn admin_ui_html() -> &'static str {
    include_str!("../../assets/admin.html")
}

/// Serves the embedded single-file admin SPA. Public HTML — every data call the
/// page makes is authenticated with the `admin`-scoped bearer key entered
/// in-page, mirroring the `/playground` pattern.
pub(super) async fn admin_ui() -> impl IntoResponse {
    Html(admin_ui_html())
}

#[derive(Debug, Deserialize)]
pub(super) struct WindowQuery {
    window: Option<String>,
}

/// Parses a look-back window like `15m`, `1h`, `24h`, or `7d`. Falls back to
/// [`DEFAULT_WINDOW`] on absent/invalid input and clamps to [`MAX_WINDOW_DAYS`].
fn parse_window(raw: Option<&str>) -> ChronoDuration {
    let parsed = raw.and_then(|value| {
        let value = value.trim();
        let (digits, unit) = value.split_at(value.find(|c: char| !c.is_ascii_digit())?);
        let amount: i64 = digits.parse().ok()?;
        if amount <= 0 {
            return None;
        }
        match unit {
            "m" => Some(ChronoDuration::minutes(amount)),
            "h" => Some(ChronoDuration::hours(amount)),
            "d" => Some(ChronoDuration::days(amount)),
            _ => None,
        }
    });
    let window = parsed.unwrap_or(DEFAULT_WINDOW);
    window.min(ChronoDuration::days(MAX_WINDOW_DAYS))
}

/// Shared prologue for the read-only admin data endpoints: authenticates the
/// request, requires the `admin` scope, and records an audit entry for both the
/// allowed and denied paths. Returns `Err(response)` so callers can `return` it
/// directly.
async fn authorize_admin(
    state: &ServerState,
    headers: &HeaderMap,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    request_id: Uuid,
    action: &str,
) -> Result<Principal, Response> {
    let principal = match authenticate_request(
        state,
        headers,
        auth_source_key(&state.cfg, headers, connect_info),
    ) {
        Ok(principal) => principal,
        Err(err) => {
            record_audit(
                state,
                Some(request_id),
                Principal::anonymous(),
                action,
                "admin_ui",
                "denied",
                json!({ "reason": err }),
            )
            .await;
            return Err(with_request_id(auth_error_response(err), request_id));
        }
    };

    if !principal.has_scope("admin") {
        record_audit(
            state,
            Some(request_id),
            principal,
            action,
            "admin_ui",
            "denied",
            json!({ "reason": "missing admin scope" }),
        )
        .await;
        return Err(with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing admin scope".to_string(),
            ),
            request_id,
        ));
    }

    record_audit(
        state,
        Some(request_id),
        principal.clone(),
        action,
        "admin_ui",
        "allowed",
        json!({}),
    )
    .await;
    Ok(principal)
}

/// `GET /v1/admin/status` — richer sibling of `/readyz`: serving mode, readiness,
/// model aliases, worker plan + live worker readiness, admission occupancy vs
/// limits, uptime, and SLO status.
pub(super) async fn admin_status(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Err(response) =
        authorize_admin(&state, &headers, connect_info, request_id, "admin.status").await
    {
        return response;
    }

    let aliases: Vec<String> = super::active_routed_models(&state.cfg)
        .into_iter()
        .map(|model| model.alias.clone())
        .collect();
    let plan = StartupPlan::from_config(&state.cfg);

    // Live worker readiness from the lock-free admission registry (Wave-2
    // supervisor state on `ServerState`). Absent in native in-process mode.
    let (serving_backend, worker_rows, live_ready_workers) = match state.worker_admissions.as_ref()
    {
        Some(registry) => {
            let rows = registry
                .read()
                .map(|registry| {
                    registry
                        .iter()
                        .map(|(id, admission)| {
                            json!({
                                "worker": id.as_str(),
                                "admitting": admission.is_admitting(),
                                "in_flight": admission.in_flight(),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let ready = rows
                .iter()
                .filter(|row| row["admitting"].as_bool().unwrap_or(false))
                .count();
            ("worker", rows, Some(ready))
        }
        None => {
            let rows = state
                .native_engines
                .keys()
                .map(|alias| json!({ "engine": alias, "resident": true }))
                .collect::<Vec<_>>();
            ("native", rows, None)
        }
    };

    let draining = state.draining.load(std::sync::atomic::Ordering::SeqCst);
    let storage_ready = storage_ready(&state).await;
    let ready = storage_ready
        && !draining
        && !aliases.is_empty()
        && live_ready_workers.map(|count| count > 0).unwrap_or(true);

    let max_in_flight = state.serving_limits.max_in_flight;
    let available = state.admission.available_permits().min(max_in_flight);
    let in_flight = max_in_flight.saturating_sub(available);

    let body = json!({
        "status": if ready { "ready" } else if draining { "draining" } else { "unavailable" },
        "ready": ready,
        "draining": draining,
        "serving_backend": serving_backend,
        "mode": state.cfg.mode,
        "uptime_seconds": state.started_at.elapsed().as_secs(),
        "storage": { "ready": storage_ready },
        "auth": { "required": state.cfg.security.require_auth },
        "models": {
            "configured": state.cfg.models.len(),
            "routed_active": aliases.len(),
            "aliases": aliases,
        },
        "workers": {
            "planned": plan.workers.len(),
            "live_ready": live_ready_workers,
            "detail": worker_rows,
        },
        "admission": {
            "in_flight": in_flight,
            "available": available,
            "max_in_flight": max_in_flight,
        },
        "slo_status": slo_status_label(ready, draining),
    });

    with_request_id(Json(body).into_response(), request_id)
}

/// SLO roll-up for the status card: `ok` when the node is ready, `draining`
/// while shedding load, `error` otherwise. Uses the same success/error framing
/// as [`super::slo_status`].
fn slo_status_label(ready: bool, draining: bool) -> &'static str {
    if ready {
        "ok"
    } else if draining {
        "draining"
    } else {
        "error"
    }
}

/// `GET /v1/admin/models` — per-model inventory: configured model config
/// (alias/role/family/weight/state) joined to the storage `list_models` record
/// and Gemma-4 readiness, mirroring the CLI's `model inventory`.
pub(super) async fn admin_models(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Err(response) =
        authorize_admin(&state, &headers, connect_info, request_id, "admin.models").await
    {
        return response;
    }

    let persisted: BTreeMap<String, ModelInventoryRecord> = state
        .storage
        .list_models()
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|record| (record.alias.clone(), record))
        .collect();
    let routed: std::collections::BTreeSet<String> = super::active_routed_models(&state.cfg)
        .into_iter()
        .map(|model| model.alias.clone())
        .collect();

    let models: Vec<Value> = state
        .cfg
        .models
        .iter()
        .map(|model| {
            let record = persisted.get(&model.alias);
            let readiness = model
                .family
                .as_deref()
                .filter(|family| family.eq_ignore_ascii_case("gemma4"))
                .map(|_| {
                    crate::readiness::read_state(&crate::readiness::evidence_path(
                        &state.cfg.storage.model_dir,
                        &model.alias,
                    ))
                });
            json!({
                "alias": model.alias,
                "role": model.role,
                "family": model.family,
                "weight": model.weight,
                "state": if model.weight > 0 { "active" } else { "disabled" },
                "path": model.path.to_string_lossy(),
                "routed": routed.contains(&model.alias),
                "readiness": readiness,
                "inventory": record.map(|record| {
                    json!({
                        "path": record.path,
                        "role": record.role,
                        "weight": record.weight,
                        "updated_at": record.updated_at.to_rfc3339(),
                    })
                }),
            })
        })
        .collect();

    let body = json!({
        "configured": state.cfg.models.len(),
        "models": models,
    });
    with_request_id(Json(body).into_response(), request_id)
}

/// `GET /v1/admin/quotas` — configured quota policies joined to live usage
/// (`usage_tokens_total`, `allowed_quota_decision_count`) over the trailing 24h,
/// team-scoped.
pub(super) async fn admin_quotas(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Err(response) =
        authorize_admin(&state, &headers, connect_info, request_id, "admin.quotas").await
    {
        return response;
    }

    let to = Utc::now();
    let from = to - ChronoDuration::days(1);
    let mut quotas = Vec::with_capacity(state.cfg.quotas.len());
    for quota in &state.cfg.quotas {
        let principal = Principal {
            subject: quota.subject.clone(),
            team: quota.team.clone(),
            scopes: Vec::new(),
            key_id: None,
            key_owner: None,
            key_purpose: None,
            key_status: None,
        };
        let tokens_used = state
            .storage
            .usage_tokens_total(&principal, false, from, to)
            .await
            .unwrap_or(0);
        let allowed_requests = state
            .storage
            .allowed_quota_decision_count(&principal, false, from, to)
            .await
            .unwrap_or(0);
        quotas.push(json!({
            "subject": quota.subject,
            "team": quota.team,
            "requests_per_minute": quota.requests_per_minute,
            "tokens_per_day": quota.tokens_per_day,
            "max_concurrency": quota.max_concurrency,
            "allowed_models": quota.allowed_models,
            "usage": {
                "window": "24h",
                "scope": "team",
                "tokens_used": tokens_used,
                "allowed_requests": allowed_requests,
            },
        }));
    }

    with_request_id(
        Json(json!({ "quotas": quotas })).into_response(),
        request_id,
    )
}

/// `GET /v1/admin/usage?window=…` — windowed usage summary
/// (`reporting::usage_summary`) plus the quota decisions recorded in the same
/// window (`quota_decisions_between_limited`).
pub(super) async fn admin_usage(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    Query(query): Query<WindowQuery>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Err(response) =
        authorize_admin(&state, &headers, connect_info, request_id, "admin.usage").await
    {
        return response;
    }

    let window = parse_window(query.window.as_deref());
    let to = Utc::now();
    let from = to - window;

    let summary = reporting::usage_summary(&state.storage, from, to)
        .await
        .ok()
        .and_then(|summary| serde_json::to_value(summary).ok())
        .unwrap_or_else(|| json!({}));
    let decisions: Vec<QuotaDecisionRecord> = state
        .storage
        .quota_decisions_between_limited(from, to, Some(MAX_WINDOW_ROWS))
        .await
        .unwrap_or_default();

    let body = json!({
        "window": window_label(window),
        "from": from.to_rfc3339(),
        "to": to.to_rfc3339(),
        "usage_summary": summary,
        "quota_decisions": decisions,
        "quota_decision_count": decisions.len(),
    });
    with_request_id(Json(body).into_response(), request_id)
}

/// `GET /v1/admin/audit?window=…` — audit events in the window
/// (`audit_events_between_limited`).
pub(super) async fn admin_audit(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    Query(query): Query<WindowQuery>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Err(response) =
        authorize_admin(&state, &headers, connect_info, request_id, "admin.audit").await
    {
        return response;
    }

    let window = parse_window(query.window.as_deref());
    let to = Utc::now();
    let from = to - window;
    let events = state
        .storage
        .audit_events_between_limited(from, to, Some(MAX_WINDOW_ROWS))
        .await
        .unwrap_or_default();

    let body = json!({
        "window": window_label(window),
        "from": from.to_rfc3339(),
        "to": to.to_rfc3339(),
        "events": events,
        "count": events.len(),
    });
    with_request_id(Json(body).into_response(), request_id)
}

/// `GET /v1/admin/keys` — API key metadata with the `sha256` hash STRIPPED.
/// The hash (and any fingerprint) is never serialized; only identity/policy
/// metadata is exposed.
pub(super) async fn admin_keys(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Err(response) =
        authorize_admin(&state, &headers, connect_info, request_id, "admin.keys").await
    {
        return response;
    }

    let keys: Vec<Value> = state
        .cfg
        .security
        .api_keys
        .iter()
        .map(redacted_key)
        .collect();
    with_request_id(
        Json(json!({ "keys": keys, "require_auth": state.cfg.security.require_auth }))
            .into_response(),
        request_id,
    )
}

/// Projects an [`crate::config::ApiKeyConfig`] to the fields safe to surface in
/// the UI. The `sha256` secret hash and `fingerprint` are deliberately omitted.
fn redacted_key(key: &crate::config::ApiKeyConfig) -> Value {
    json!({
        "id": key.id,
        "subject": key.subject,
        "team": key.team,
        "scopes": key.scopes,
        "owner": key.owner,
        "purpose": key.purpose,
        "last_four": key.last_four,
        "status": key.status,
        "created_at": key.created_at.map(|at| at.to_rfc3339()),
        "expires_at": key.expires_at.map(|at| at.to_rfc3339()),
        "rotated_at": key.rotated_at.map(|at| at.to_rfc3339()),
    })
}

fn window_label(window: ChronoDuration) -> String {
    let minutes = window.num_minutes();
    if minutes % (60 * 24) == 0 {
        format!("{}d", minutes / (60 * 24))
    } else if minutes % 60 == 0 {
        format!("{}h", minutes / 60)
    } else {
        format!("{}m", minutes)
    }
}

async fn storage_ready(state: &ServerState) -> bool {
    sqlx::query_scalar::<_, i64>("SELECT 1")
        .fetch_one(state.storage.pool())
        .await
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_parsing_handles_units_and_defaults() {
        assert_eq!(parse_window(Some("15m")), ChronoDuration::minutes(15));
        assert_eq!(parse_window(Some("2h")), ChronoDuration::hours(2));
        assert_eq!(parse_window(Some("7d")), ChronoDuration::days(7));
        // Absent / invalid → default.
        assert_eq!(parse_window(None), DEFAULT_WINDOW);
        assert_eq!(parse_window(Some("garbage")), DEFAULT_WINDOW);
        assert_eq!(parse_window(Some("0h")), DEFAULT_WINDOW);
        // Clamped to the maximum look-back.
        assert_eq!(
            parse_window(Some("999d")),
            ChronoDuration::days(MAX_WINDOW_DAYS)
        );
    }

    #[test]
    fn window_label_roundtrips_common_units() {
        assert_eq!(window_label(ChronoDuration::minutes(15)), "15m");
        assert_eq!(window_label(ChronoDuration::hours(6)), "6h");
        assert_eq!(window_label(ChronoDuration::days(7)), "7d");
    }

    #[test]
    fn redacted_key_strips_secret_hash_and_fingerprint() {
        let key = crate::config::ApiKeyConfig {
            id: "key-1".into(),
            sha256: "DEADBEEFsecrethash".into(),
            subject: "svc".into(),
            team: "platform".into(),
            scopes: vec!["admin".into()],
            owner: Some("morgan".into()),
            purpose: Some("ops".into()),
            last_four: Some("ab12".into()),
            fingerprint: Some("fp-secret".into()),
            status: "active".into(),
            ..Default::default()
        };
        let value = redacted_key(&key);
        let serialized = serde_json::to_string(&value).unwrap();
        assert!(!serialized.contains("DEADBEEFsecrethash"));
        assert!(!serialized.to_lowercase().contains("sha256"));
        assert!(!serialized.contains("fp-secret"));
        assert!(!serialized.contains("fingerprint"));
        // Metadata is preserved.
        assert_eq!(value["id"], "key-1");
        assert_eq!(value["subject"], "svc");
        assert_eq!(value["team"], "platform");
        assert_eq!(value["last_four"], "ab12");
        assert_eq!(value["status"], "active");
    }

    #[test]
    fn slo_label_reflects_ready_and_draining() {
        assert_eq!(slo_status_label(true, false), "ok");
        assert_eq!(slo_status_label(false, true), "draining");
        assert_eq!(slo_status_label(false, false), "error");
    }

    #[test]
    fn admin_ui_html_wires_admin_endpoints_and_key_field() {
        let html = admin_ui_html();
        assert!(html.contains("<title>"));
        assert!(html.contains("/v1/admin/status"));
        assert!(html.contains("/v1/admin/models"));
        assert!(html.contains("/v1/admin/keys"));
        assert!(html.contains("/v1/chat/completions"));
        assert!(html.to_lowercase().contains("api key") || html.contains("api-key"));
    }
}
