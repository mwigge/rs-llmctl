use crate::config::{is_external_host, ApiKeyConfig, Config};
use anyhow::Result;
use chrono::Utc;
use std::collections::BTreeSet;

/// Enforce the production security posture for an `rs-llmctl` configuration.
///
/// The validator is intentionally fail-closed for externally reachable service
/// configs. It rejects plaintext API-key material, unknown or empty scopes,
/// invalid key lifecycle metadata, plaintext observability secrets, malformed
/// native TLS settings, unsafe external provider settings, missing auth, missing
/// TLS evidence, missing audit retention/monthly reports, and incomplete OTLP
/// telemetry exporters.
pub fn validate_production_security(cfg: &Config) -> Result<()> {
    validate_api_keys_are_hashes(&cfg.security.api_keys)?;
    validate_api_key_scopes(&cfg.security.api_keys)?;
    validate_api_key_metadata(&cfg.security.api_keys)?;
    validate_no_plaintext_observability_secrets(cfg)?;
    validate_native_tls_config(cfg)?;
    validate_external_provider_security(cfg)?;

    if cfg.security.production || cfg.security.bind_external || is_external_host(&cfg.server.host) {
        anyhow::ensure!(
            cfg.security.require_auth && !cfg.security.api_keys.is_empty(),
            "external/production serving requires authentication"
        );
        for proxy in &cfg.security.trusted_proxies {
            validate_trusted_proxy(proxy)?;
        }
        validate_api_keys_for_active_serving(&cfg.security.api_keys)?;
        validate_external_tls_posture(cfg)?;
        anyhow::ensure!(
            cfg.audit.retention_days > 0,
            "CRA Article 14 active control requires audit retention"
        );
        anyhow::ensure!(
            cfg.audit.monthly_reports,
            "CRA Article 14 active control requires monthly audit reports"
        );
        anyhow::ensure!(
            cfg.audit
                .report_directory
                .as_ref()
                .is_some_and(|path| !path.as_os_str().is_empty()),
            "CRA Article 14 active control requires audit.report-directory for monthly evidence"
        );
        anyhow::ensure!(
            cfg.observability.traces_enabled
                && cfg.observability.metrics_enabled
                && cfg.observability.logs_enabled,
            "CRA Article 14 active control requires OTel traces, metrics, and logs"
        );
        anyhow::ensure!(
            cfg.observability
                .exporter
                .endpoint
                .as_deref()
                .or(cfg.observability.otlp_endpoint.as_deref())
                .is_some_and(|endpoint| !endpoint.trim().is_empty()),
            "CRA Article 14 active control requires an OTel exporter endpoint"
        );
    }

    Ok(())
}

fn validate_external_provider_security(cfg: &Config) -> Result<()> {
    if !cfg.external_providers.enabled {
        return Ok(());
    }

    anyhow::bail!(
        "external provider routing is disabled for this native-only release; route traffic through local rs-llmctl models"
    )
}

fn validate_external_tls_posture(cfg: &Config) -> Result<()> {
    if native_tls_has_cert_and_key(cfg) {
        return Ok(());
    }

    anyhow::ensure!(
        cfg.security.tls_termination.enabled,
        "external/production serving requires native TLS with cert/key or documented TLS termination or mTLS"
    );
    anyhow::ensure!(
        cfg.security
            .tls_termination
            .provider
            .as_deref()
            .is_some_and(|provider| !provider.trim().is_empty()),
        "TLS termination must declare a provider"
    );
    anyhow::ensure!(
        cfg.security
            .tls_termination
            .evidence
            .as_deref()
            .is_some_and(|evidence| !evidence.trim().is_empty()),
        "TLS termination must declare evidence"
    );
    anyhow::ensure!(
        !cfg.security.trusted_proxies.is_empty(),
        "TLS termination must declare trusted-proxies for proxy-aware auth throttling"
    );
    Ok(())
}

fn validate_trusted_proxy(value: &str) -> Result<()> {
    let value = value.trim();
    anyhow::ensure!(
        !value.is_empty() && value != "*",
        "trusted-proxies must list explicit IP addresses or CIDR ranges; wildcard is not allowed"
    );
    if let Some((addr, prefix)) = value.split_once('/') {
        let ip: std::net::IpAddr = addr
            .parse()
            .map_err(|_| anyhow::anyhow!("trusted proxy `{value}` has invalid CIDR address"))?;
        let prefix: u8 = prefix
            .parse()
            .map_err(|_| anyhow::anyhow!("trusted proxy `{value}` has invalid CIDR prefix"))?;
        let max = if ip.is_ipv4() { 32 } else { 128 };
        anyhow::ensure!(
            prefix > 0 && prefix <= max,
            "trusted proxy `{value}` has invalid CIDR prefix"
        );
        return Ok(());
    }
    let _ip: std::net::IpAddr = value
        .parse()
        .map_err(|_| anyhow::anyhow!("trusted proxy `{value}` must be an IP address or CIDR"))?;
    Ok(())
}

fn validate_native_tls_config(cfg: &Config) -> Result<()> {
    if !cfg.server.tls.enabled {
        return Ok(());
    }

    anyhow::ensure!(
        cfg.server
            .tls
            .cert_path
            .as_ref()
            .is_some_and(|path| !path.as_os_str().is_empty()),
        "server.tls.cert-path is required when server TLS is enabled"
    );
    anyhow::ensure!(
        cfg.server
            .tls
            .key_path
            .as_ref()
            .is_some_and(|path| !path.as_os_str().is_empty()),
        "server.tls.key-path is required when server TLS is enabled"
    );
    anyhow::ensure!(
        !cfg.server.tls.require_client_cert,
        "server.tls.require-client-cert is not supported without client CA configuration"
    );
    Ok(())
}

fn native_tls_has_cert_and_key(cfg: &Config) -> bool {
    cfg.server.tls.enabled
        && cfg
            .server
            .tls
            .cert_path
            .as_ref()
            .is_some_and(|path| !path.as_os_str().is_empty())
        && cfg
            .server
            .tls
            .key_path
            .as_ref()
            .is_some_and(|path| !path.as_os_str().is_empty())
        && !cfg.server.tls.require_client_cert
}

pub fn validate_api_keys_for_active_serving(keys: &[ApiKeyConfig]) -> Result<()> {
    for key in keys {
        anyhow::ensure!(
            key.status != "revoked",
            "api key `{}` is revoked and must not be present in active serving config",
            key.id
        );
        if let Some(expires_at) = key.expires_at.as_ref() {
            anyhow::ensure!(
                *expires_at > Utc::now(),
                "api key `{}` is expired and must be rotated or removed",
                key.id
            );
        }
    }
    Ok(())
}

pub fn validate_api_secret_material(secret: &str) -> Result<()> {
    anyhow::ensure!(
        secret.len() >= 32,
        "api key secret must be at least 32 bytes"
    );
    anyhow::ensure!(
        secret.chars().any(|ch| ch.is_ascii_alphabetic())
            && secret.chars().any(|ch| ch.is_ascii_digit()),
        "api key secret must include both letters and digits"
    );
    Ok(())
}

fn validate_no_plaintext_observability_secrets(cfg: &Config) -> Result<()> {
    for (name, value) in &cfg.observability.exporter.headers {
        if is_sensitive_name(name) {
            anyhow::ensure!(
                value.starts_with("env:"),
                "observability header `{name}` must not contain a plaintext secret; use env:NAME"
            );
        }
    }
    Ok(())
}

fn validate_api_keys_are_hashes(keys: &[ApiKeyConfig]) -> Result<()> {
    let mut ids = BTreeSet::new();
    for key in keys {
        anyhow::ensure!(!key.id.trim().is_empty(), "api key id must not be empty");
        anyhow::ensure!(
            ids.insert(key.id.as_str()),
            "api key `{}` is declared more than once",
            key.id
        );
        anyhow::ensure!(
            !key.subject.trim().is_empty(),
            "api key `{}` must declare a subject",
            key.id
        );
        anyhow::ensure!(
            !key.team.trim().is_empty(),
            "api key `{}` must declare a team",
            key.id
        );
        anyhow::ensure!(
            is_sha256_hex(&key.sha256),
            "api key `{}` must be stored as a sha256 hex digest",
            key.id
        );
    }
    Ok(())
}

fn validate_api_key_metadata(keys: &[ApiKeyConfig]) -> Result<()> {
    for key in keys {
        anyhow::ensure!(
            matches!(key.status.as_str(), "active" | "retiring" | "revoked"),
            "api key `{}` has invalid status `{}`",
            key.id,
            key.status
        );
        if let Some(last_four) = key.last_four.as_ref() {
            anyhow::ensure!(
                last_four.len() == 4
                    && last_four
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_')),
                "api key `{}` last_four must contain four safe characters",
                key.id
            );
        }
        if let Some(fingerprint) = key.fingerprint.as_ref() {
            anyhow::ensure!(
                !fingerprint.trim().is_empty(),
                "api key `{}` fingerprint must not be empty",
                key.id
            );
        }
    }
    Ok(())
}

fn validate_api_key_scopes(keys: &[ApiKeyConfig]) -> Result<()> {
    for key in keys {
        anyhow::ensure!(
            !key.scopes.is_empty(),
            "api key `{}` must declare at least one scope",
            key.id
        );
        for scope in &key.scopes {
            anyhow::ensure!(
                !scope.trim().is_empty(),
                "api key `{}` has an empty scope",
                key.id
            );
            anyhow::ensure!(
                is_allowed_api_key_scope(scope),
                "api key `{}` has unknown scope `{scope}`; allowed scopes are chat, models.read, models, admin",
                key.id
            );
        }
    }
    Ok(())
}

fn is_allowed_api_key_scope(scope: &str) -> bool {
    matches!(scope, "chat" | "models.read" | "models" | "admin")
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_sensitive_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("authorization")
        || name.contains("api-key")
        || name.contains("apikey")
        || name.contains("token")
        || name.contains("secret")
}

#[cfg(test)]
mod tests {
    use super::{
        is_allowed_api_key_scope, is_sensitive_name, is_sha256_hex,
        validate_api_keys_for_active_serving, validate_production_security,
    };
    use crate::config::{ApiKeyConfig, Config};
    use chrono::{Duration, Utc};

    #[test]
    fn recognizes_sha256_hex_digests() {
        assert!(is_sha256_hex(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ));
        assert!(!is_sha256_hex("plain-secret"));
        assert!(!is_sha256_hex(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg"
        ));
    }

    #[test]
    fn recognizes_sensitive_config_names() {
        assert!(is_sensitive_name("authorization"));
        assert!(is_sensitive_name("x-api-key"));
        assert!(is_sensitive_name("collector-token"));
        assert!(!is_sensitive_name("x-tenant"));
    }

    #[test]
    fn recognizes_allowed_api_key_scopes() {
        assert!(is_allowed_api_key_scope("chat"));
        assert!(is_allowed_api_key_scope("models.read"));
        assert!(is_allowed_api_key_scope("models"));
        assert!(is_allowed_api_key_scope("admin"));
        assert!(!is_allowed_api_key_scope(""));
        assert!(!is_allowed_api_key_scope("models:read"));
    }

    fn hashed_key() -> ApiKeyConfig {
        ApiKeyConfig {
            id: "platform-chat".to_string(),
            sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            subject: "platform".to_string(),
            team: "infra".to_string(),
            scopes: vec!["chat".to_string()],
            created_at: None,
            expires_at: None,
            rotated_at: None,
            owner: None,
            purpose: None,
            last_four: None,
            fingerprint: None,
            status: "active".to_string(),
        }
    }

    #[test]
    fn active_serving_rejects_revoked_key() {
        let mut key = hashed_key();
        key.status = "revoked".to_string();

        let err = validate_api_keys_for_active_serving(&[key]).expect_err("revoked key rejected");

        assert!(err.to_string().contains("revoked"));
    }

    #[test]
    fn active_serving_rejects_expired_key() {
        let mut key = hashed_key();
        key.expires_at = Some(Utc::now() - Duration::minutes(1));

        let err = validate_api_keys_for_active_serving(&[key]).expect_err("expired key rejected");

        assert!(err.to_string().contains("expired"));
    }

    #[test]
    fn production_validation_rejects_invalid_key_metadata() {
        let mut cfg = Config::default();
        let mut key = hashed_key();
        key.status = "disabled".to_string();
        cfg.security.api_keys = vec![key];

        let err = validate_production_security(&cfg).expect_err("invalid status rejected");

        assert!(err.to_string().contains("invalid status"));
    }
}
