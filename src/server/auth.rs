use super::{audit_reject, audit_reject_response, auth_error_response, ServerState};
use crate::config::{ApiKeyConfig, Config};
use crate::quota::Principal;
use axum::extract::ConnectInfo;
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use chrono::Utc;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::net::{IpAddr, SocketAddr};
use subtle::ConstantTimeEq;
use uuid::Uuid;

pub(super) fn authenticate(
    cfg: &Config,
    headers: &HeaderMap,
) -> std::result::Result<Principal, String> {
    if !cfg.security.require_auth {
        return Ok(Principal::anonymous());
    }

    let Some(header) = headers.get(AUTHORIZATION).and_then(|h| h.to_str().ok()) else {
        return Err("missing bearer token".to_string());
    };
    let Some(token) = header.strip_prefix("Bearer ") else {
        return Err("authorization must use Bearer token".to_string());
    };

    let digest = hex::encode(Sha256::digest(token.as_bytes()));
    cfg.security
        .api_keys
        .iter()
        .filter(|key| api_key_can_authenticate(key))
        .find(|key| constant_time_eq_case_insensitive(&key.sha256, &digest))
        .map(|key| Principal {
            subject: key.subject.clone(),
            team: key.team.clone(),
            scopes: key.scopes.clone(),
            key_id: Some(key.id.clone()),
            key_owner: key.owner.clone(),
            key_purpose: key.purpose.clone(),
            key_status: Some(key.status.clone()),
        })
        .ok_or_else(|| "invalid bearer token".to_string())
}

fn api_key_can_authenticate(key: &ApiKeyConfig) -> bool {
    if !matches!(key.status.as_str(), "active" | "retiring") {
        return false;
    }
    key.expires_at
        .is_none_or(|expires_at| expires_at > Utc::now())
}

pub(super) fn authenticate_request(
    state: &ServerState,
    headers: &HeaderMap,
    source_key: String,
) -> std::result::Result<Principal, String> {
    let key = auth_failure_key(&source_key);
    if state
        .auth_failures
        .is_limited(&key, state.cfg.security.auth_failure_limit_per_minute)
    {
        return Err("too many failed authentication attempts; retry later".to_string());
    }

    match authenticate(&state.cfg, headers) {
        Ok(principal) => {
            state.auth_failures.record_success(&key);
            Ok(principal)
        }
        Err(err) => {
            state
                .auth_failures
                .record_failure(&key, state.cfg.security.auth_failure_limit_per_minute);
            Err(err)
        }
    }
}

pub(super) fn auth_source_key(
    cfg: &Config,
    headers: &HeaderMap,
    connect_info: Option<ConnectInfo<SocketAddr>>,
) -> String {
    let peer = connect_info.map(|ConnectInfo(addr)| addr.ip());
    if peer.is_some_and(|ip| is_trusted_proxy(cfg, ip)) {
        if let Some(forwarded) = forwarded_client_ip(cfg, headers) {
            return forwarded;
        }
    }
    peer.map(|ip| ip.to_string())
        .unwrap_or_else(|| "unknown-source".to_string())
}

/// Shared prologue for `chat_completions` and `native_embeddings`:
/// authenticates the request and checks for the `chat` scope, auditing and
/// building the appropriate error response on either failure. Returns
/// `Err(response)` so callers can `return` it directly.
pub(super) async fn authenticate_with_chat_scope(
    state: &ServerState,
    headers: &HeaderMap,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    request_id: Uuid,
    action: &str,
) -> std::result::Result<Principal, Response> {
    let principal = match authenticate_request(
        state,
        headers,
        auth_source_key(&state.cfg, headers, connect_info),
    ) {
        Ok(principal) => principal,
        Err(err) => {
            return Err(audit_reject_response(
                state,
                request_id,
                Principal::anonymous(),
                action,
                "unknown",
                "denied",
                auth_error_response(err.clone()),
                json!({ "reason": err }),
            )
            .await);
        }
    };

    if !principal.has_scope("chat") {
        return Err(audit_reject(
            state,
            request_id,
            principal,
            action,
            "unknown",
            "denied",
            StatusCode::FORBIDDEN,
            "forbidden",
            "missing chat scope".to_string(),
            json!({ "reason": "missing chat scope" }),
        )
        .await);
    }

    Ok(principal)
}

pub(super) fn is_trusted_proxy(cfg: &Config, ip: IpAddr) -> bool {
    cfg.security
        .trusted_proxies
        .iter()
        .any(|trusted| trusted_proxy_matches(trusted, ip))
}

fn trusted_proxy_matches(trusted: &str, ip: IpAddr) -> bool {
    // Normalize IPv4-mapped IPv6 addresses (e.g. ::ffff:10.0.0.1) to their
    // IPv4 form so a trusted proxy entry written as an IPv4 address/CIDR
    // still matches connections reported via a dual-stack listener.
    let ip = ip.to_canonical();
    let trusted = trusted.trim();
    if let Ok(exact) = trusted.parse::<IpAddr>() {
        return exact.to_canonical() == ip;
    }
    let Some((network, prefix)) = trusted.split_once('/') else {
        return false;
    };
    let Ok(network) = network.parse::<IpAddr>() else {
        return false;
    };
    let network = network.to_canonical();
    let Ok(prefix) = prefix.parse::<u8>() else {
        return false;
    };
    match (network, ip) {
        (IpAddr::V4(network), IpAddr::V4(ip)) if prefix > 0 && prefix <= 32 => {
            let mask = u32::MAX << (32 - prefix);
            u32::from(network) & mask == u32::from(ip) & mask
        }
        (IpAddr::V6(network), IpAddr::V6(ip)) if prefix > 0 && prefix <= 128 => {
            let mask = u128::MAX << (128 - prefix);
            u128::from(network) & mask == u128::from(ip) & mask
        }
        _ => false,
    }
}

pub(super) fn forwarded_client_ip(cfg: &Config, headers: &HeaderMap) -> Option<String> {
    let forwarded_ips = headers
        .get_all("forwarded")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| {
            value
                .split(',')
                .flat_map(|element| element.split(';'))
                .filter_map(|part| {
                    let part = part.trim();
                    let for_value = part.strip_prefix("for=")?;
                    parse_forwarded_ip(for_value)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if !forwarded_ips.is_empty() {
        if let Some(ip) = first_untrusted_forwarded_ip(cfg, forwarded_ips) {
            return Some(ip);
        }
    }
    let forwarded_ips = headers
        .get_all("x-forwarded-for")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| {
            value
                .split(',')
                .filter_map(parse_forwarded_ip)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if !forwarded_ips.is_empty() {
        if let Some(ip) = first_untrusted_forwarded_ip(cfg, forwarded_ips) {
            return Some(ip);
        }
    }
    None
}

fn parse_forwarded_ip(value: &str) -> Option<IpAddr> {
    let value = value
        .trim()
        .trim_matches('"')
        .trim_matches('[')
        .trim_matches(']');
    let candidate = if let Ok(ip) = value.parse::<IpAddr>() {
        ip
    } else if let Some((host, _port)) = value.rsplit_once(':') {
        host.trim_matches('[').trim_matches(']').parse().ok()?
    } else {
        return None;
    };
    Some(candidate)
}

fn first_untrusted_forwarded_ip(cfg: &Config, forwarded_ips: Vec<IpAddr>) -> Option<String> {
    for ip in forwarded_ips.into_iter().rev() {
        if !is_trusted_proxy(cfg, ip) {
            return Some(ip.to_string());
        }
    }
    None
}

fn auth_failure_key(source_key: &str) -> String {
    let digest = Sha256::digest(source_key.as_bytes());
    format!("{:x}", digest)[..16].to_string()
}

fn constant_time_eq_case_insensitive(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let left = left.to_ascii_lowercase();
    let right = right.to_ascii_lowercase();
    left.as_bytes().ct_eq(right.as_bytes()).into()
}
