use crate::config::{ApiKeyConfig, Config};
use anyhow::Result;

pub fn validate_production_security(cfg: &Config) -> Result<()> {
    validate_api_keys_are_hashes(&cfg.security.api_keys)?;
    validate_no_plaintext_observability_secrets(cfg)?;

    if cfg.security.production || cfg.security.bind_external || cfg.server.host == "0.0.0.0" {
        anyhow::ensure!(
            cfg.security.require_auth && !cfg.security.api_keys.is_empty(),
            "external/production serving requires authentication"
        );
    }

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
    for key in keys {
        anyhow::ensure!(
            is_sha256_hex(&key.sha256),
            "api key `{}` must be stored as a sha256 hex digest",
            key.id
        );
    }
    Ok(())
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
    use super::{is_sensitive_name, is_sha256_hex};

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
}
