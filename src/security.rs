use crate::config::{ApiKeyConfig, Config};
use anyhow::Result;
use std::collections::BTreeSet;

pub fn validate_production_security(cfg: &Config) -> Result<()> {
    validate_api_keys_are_hashes(&cfg.security.api_keys)?;
    validate_api_key_scopes(&cfg.security.api_keys)?;
    validate_no_plaintext_observability_secrets(cfg)?;

    if cfg.security.production || cfg.security.bind_external || cfg.server.host == "0.0.0.0" {
        anyhow::ensure!(
            cfg.security.require_auth && !cfg.security.api_keys.is_empty(),
            "external/production serving requires authentication"
        );
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
    use super::{is_allowed_api_key_scope, is_sensitive_name, is_sha256_hex};

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
}
