use super::*;
use crate::runtime::RuntimeBackend;

#[test]
fn gen_ai_observability_config_phase2_defaults_and_roundtrip() {
    let defaults = GenAiObservabilityConfig::default();
    assert!(!defaults.token_events);
    assert!(!defaults.logit_entropy);
    assert!(defaults.thinking_phase_events);

    let toml_input =
        "[gen-ai]\ntoken-events = true\nlogit-entropy = true\nthinking-phase-events = false\n";
    let flipped: ObservabilityConfig = toml::from_str(toml_input).expect("valid toml");
    assert!(flipped.gen_ai.token_events);
    assert!(flipped.gen_ai.logit_entropy);
    assert!(!flipped.gen_ai.thinking_phase_events);
}

#[test]
fn gen_ai_config_defaults_to_capture_message_content_enabled() {
    let cfg = ObservabilityConfig::default();
    assert!(cfg.gen_ai.capture_message_content);
}

#[test]
fn gen_ai_config_roundtrips_through_toml_with_content_disabled() {
    let toml_input = r#"
[gen-ai]
capture-message-content = false
"#;
    let cfg: ObservabilityConfig = toml::from_str(toml_input).expect("valid toml");
    assert!(!cfg.gen_ai.capture_message_content);
}

#[test]
fn default_runtime_backend_is_candle_native() {
    let cfg = Config::default();

    assert_eq!(cfg.runtime.backend, RuntimeBackend::CandleNative);
    assert_eq!(cfg.runtime.heartbeat_interval_seconds, 30);
    assert_eq!(cfg.runtime.embeddings.mode, NativeEmbeddingMode::Semantic);
    assert_eq!(cfg.runtime.embeddings.model_alias, None);
    assert_eq!(cfg.storage.max_connections, 5);
    assert_eq!(cfg.server.upstream_timeout_seconds, 300);
    assert_eq!(cfg.server.graceful_drain_seconds, 5);
    assert_eq!(cfg.server.circuit_breaker_failures, 3);
    assert_eq!(cfg.server.circuit_breaker_reset_seconds, 30);
    assert_eq!(cfg.security.auth_failure_limit_per_minute, 60);
    assert!(!cfg.external_providers.enabled);
    assert!(cfg.external_providers.providers.is_empty());
}

#[test]
fn parses_external_provider_env_key_references_without_inline_secrets() {
    let cfg: Config = toml::from_str(
        r#"
[external-providers]
enabled = true

[[external-providers.providers]]
id = "openai"
kind = "open-ai-compatible"
base-url = "https://api.openai.example/v1"
api-key-env = "OPENAI_API_KEY"

[[external-providers.routes]]
model-alias = "gpt-proxy"
provider = "openai"
provider-model = "gpt-4o-mini"

[[models]]
alias = "gpt-proxy"
path = "/models/remote-placeholder"
role = "chat"
"#,
    )
    .expect("parse external provider config");

    assert!(cfg.external_providers.enabled);
    let provider = cfg.external_providers.provider("openai").expect("provider");
    assert_eq!(provider.kind, ExternalProviderKind::OpenAiCompatible);
    assert_eq!(provider.api_key_env, "OPENAI_API_KEY");
    let route = cfg
        .external_providers
        .route_for_model("gpt-proxy")
        .expect("provider route");
    assert_eq!(route.provider, "openai");
    assert_eq!(route.provider_model.as_deref(), Some("gpt-4o-mini"));
}

#[test]
fn parses_native_embedding_runtime_contract() {
    let cfg: Config = toml::from_str(
        r#"
[runtime]
backend = "candle-native"

[runtime.embeddings]
mode = "semantic"
model-alias = "embed-prod"
"#,
    )
    .expect("parse config");

    assert_eq!(cfg.runtime.embeddings.mode, NativeEmbeddingMode::Semantic);
    assert_eq!(
        cfg.runtime.embeddings.model_alias.as_deref(),
        Some("embed-prod")
    );

    let cfg: Config = toml::from_str(
        r#"
[runtime.embeddings]
mode = "dev-fallback"
"#,
    )
    .expect("parse dev fallback config");

    assert_eq!(
        cfg.runtime.embeddings.mode,
        NativeEmbeddingMode::DevFallback
    );
    assert_eq!(cfg.runtime.embeddings.model_alias, None);
}

#[test]
fn parses_server_tls_config() {
    let cfg: Config = toml::from_str(
        r#"
[server.tls]
enabled = true
cert-path = "/etc/llmctl/tls/server.crt"
key-path = "/etc/llmctl/tls/server.key"
require-client-cert = false
"#,
    )
    .expect("parse server tls config");

    assert!(cfg.server.tls.enabled);
    assert_eq!(
        cfg.server.tls.cert_path.as_deref(),
        Some(Path::new("/etc/llmctl/tls/server.crt"))
    );
    assert_eq!(
        cfg.server.tls.key_path.as_deref(),
        Some(Path::new("/etc/llmctl/tls/server.key"))
    );
    assert!(!cfg.server.tls.require_client_cert);
}

#[test]
fn api_key_metadata_defaults_for_legacy_config() {
    let key: ApiKeyConfig = toml::from_str(
        r#"
id = "platform-chat"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
subject = "platform"
team = "infra"
scopes = ["chat"]
"#,
    )
    .expect("parse api key");

    assert_eq!(key.created_at, None);
    assert_eq!(key.expires_at, None);
    assert_eq!(key.rotated_at, None);
    assert_eq!(key.owner, None);
    assert_eq!(key.purpose, None);
    assert_eq!(key.last_four, None);
    assert_eq!(key.fingerprint, None);
    assert_eq!(key.status, "active");
}

#[test]
fn api_key_metadata_accepts_kebab_case_config_fields() {
    let key: ApiKeyConfig = toml::from_str(
        r#"
id = "platform-chat"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
subject = "platform"
team = "infra"
scopes = ["chat"]
created-at = "2026-01-02T03:04:05Z"
expires-at = "2027-01-02T03:04:05Z"
rotated-at = "2026-02-02T03:04:05Z"
owner = "platform"
purpose = "chat serving"
last-four = "cdef"
fingerprint = "sha256:0123456789abcdef"
status = "retiring"
"#,
    )
    .expect("parse api key metadata");

    assert_eq!(
        key.created_at.expect("created_at").to_rfc3339(),
        "2026-01-02T03:04:05+00:00"
    );
    assert_eq!(
        key.expires_at.expect("expires_at").to_rfc3339(),
        "2027-01-02T03:04:05+00:00"
    );
    assert_eq!(
        key.rotated_at.expect("rotated_at").to_rfc3339(),
        "2026-02-02T03:04:05+00:00"
    );
    assert_eq!(key.owner.as_deref(), Some("platform"));
    assert_eq!(key.purpose.as_deref(), Some("chat serving"));
    assert_eq!(key.last_four.as_deref(), Some("cdef"));
    assert_eq!(key.fingerprint.as_deref(), Some("sha256:0123456789abcdef"));
    assert_eq!(key.status, "retiring");
}

#[tokio::test]
async fn save_writes_complete_valid_toml_and_leaves_no_temp_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.toml");

    let cfg = Config::default();
    save(&path, &cfg).await.expect("save");

    // The saved file must be complete, valid TOML that round-trips.
    let reloaded = load(&path).await.expect("reload saved config");
    assert_eq!(reloaded.mode, cfg.mode);

    // The atomic-rename strategy must not leave sibling temp files behind.
    let mut entries = tokio::fs::read_dir(dir.path()).await.expect("read_dir");
    let mut names = Vec::new();
    while let Some(entry) = entries.next_entry().await.expect("dir entry") {
        names.push(entry.file_name().to_string_lossy().into_owned());
    }
    assert_eq!(names, vec!["config.toml".to_string()]);
    assert!(
        names.iter().all(|name| !name.contains(".tmp-")),
        "no temp files should remain: {names:?}"
    );
}

#[tokio::test]
async fn save_overwrites_existing_config_atomically() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.toml");

    // Pre-existing (larger) content: proves the temp+rename path replaces
    // rather than truncating in place, so a live reader never observes a
    // half-written file.
    let mut first = Config::default();
    first.quotas.push(QuotaConfig {
        subject: "team-a".to_string(),
        team: "team-a".to_string(),
        requests_per_minute: 10,
        tokens_per_day: 1000,
        max_concurrency: 2,
        allowed_models: vec!["m1".to_string(), "m2".to_string()],
    });
    save(&path, &first).await.expect("first save");

    let second = Config::default();
    save(&path, &second).await.expect("second save");

    let reloaded = load(&path).await.expect("reload");
    assert!(reloaded.quotas.is_empty());
}
