use rs_llmctl::config::{
    self, ApiKeyConfig, Config, ExternalProviderConfig, ExternalProviderKind,
    ObservabilityExporterConfig, OtlpProtocol, ServerConfig,
};
use rs_llmctl::observability::{sanitize_otel_attributes, Exporter, ObservabilityPlan};
use serde_json::json;
use std::collections::BTreeMap;

fn hashed_key() -> ApiKeyConfig {
    ApiKeyConfig {
        id: "ops".to_string(),
        sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        subject: "operator".to_string(),
        team: "platform".to_string(),
        scopes: vec!["models.read".to_string()],
        ..Default::default()
    }
}

fn enable_tls_termination(cfg: &mut Config) {
    cfg.security.tls_termination.enabled = true;
    cfg.security.tls_termination.provider = Some("envoy-edge".to_string());
    cfg.security.tls_termination.evidence = Some("change-record-123".to_string());
    cfg.security.tls_termination.m_tls = true;
    cfg.audit.monthly_reports = true;
    cfg.observability.exporter.endpoint = Some("https://otel.example.test/v1/traces".to_string());
}

fn enable_native_tls(cfg: &mut Config) {
    cfg.server.tls.enabled = true;
    cfg.server.tls.cert_path = Some("/etc/llmctl/tls/server.crt".into());
    cfg.server.tls.key_path = Some("/etc/llmctl/tls/server.key".into());
    cfg.audit.monthly_reports = true;
    cfg.observability.exporter.endpoint = Some("https://otel.example.test/v1/traces".to_string());
}

#[test]
fn external_bind_requires_auth_and_hashed_api_keys() {
    let mut cfg = Config::default();
    cfg.server.host = "0.0.0.0".to_string();

    let err = config::validate_production_security(&cfg).expect_err("auth is required");
    assert!(
        err.to_string().contains("requires authentication"),
        "unexpected error: {err}"
    );

    cfg.security.require_auth = true;
    cfg.security.api_keys = vec![ApiKeyConfig {
        sha256: "not-a-sha256".to_string(),
        ..hashed_key()
    }];

    let err = config::validate_production_security(&cfg).expect_err("hash format is required");
    assert!(
        err.to_string().contains("sha256"),
        "unexpected error: {err}"
    );

    cfg.security.api_keys = vec![hashed_key()];
    enable_tls_termination(&mut cfg);
    config::validate_production_security(&cfg).expect("valid production posture");
}

#[test]
fn external_bind_requires_documented_tls_termination() {
    let mut cfg = Config::default();
    cfg.server.host = "0.0.0.0".to_string();
    cfg.security.require_auth = true;
    cfg.security.api_keys = vec![hashed_key()];

    let err = config::validate_production_security(&cfg).expect_err("TLS termination is required");
    assert!(
        err.to_string().contains("TLS termination"),
        "unexpected error: {err}"
    );

    enable_tls_termination(&mut cfg);
    config::validate_production_security(&cfg).expect("documented TLS termination is accepted");
}

#[test]
fn external_bind_accepts_native_tls_cert_and_key() {
    let mut cfg = Config::default();
    cfg.server.host = "0.0.0.0".to_string();
    cfg.security.require_auth = true;
    cfg.security.api_keys = vec![hashed_key()];
    enable_native_tls(&mut cfg);

    config::validate_production_security(&cfg).expect("native TLS cert/key is accepted");
}

#[test]
fn production_external_provider_requires_https_base_url() {
    let mut cfg = Config::default();
    cfg.security.production = true;
    cfg.security.require_auth = true;
    cfg.security.api_keys = vec![hashed_key()];
    enable_tls_termination(&mut cfg);
    cfg.external_providers.enabled = true;
    cfg.external_providers.providers = vec![ExternalProviderConfig {
        id: "openai".to_string(),
        kind: ExternalProviderKind::OpenAiCompatible,
        base_url: "http://api.openai.example/v1".to_string(),
        api_key_env: "OPENAI_API_KEY".to_string(),
    }];

    let err = config::validate_production_security(&cfg).expect_err("https is required");
    assert!(
        err.to_string().contains("must use https"),
        "unexpected error: {err}"
    );

    cfg.external_providers.providers[0].base_url = "https://api.openai.example/v1".to_string();
    config::validate_production_security(&cfg).expect("https provider is accepted");
}

#[test]
fn native_tls_validation_rejects_missing_cert_or_key() {
    let mut cfg = Config::default();
    cfg.server.tls.enabled = true;
    cfg.server.tls.key_path = Some("/etc/llmctl/tls/server.key".into());

    let err = config::validate_production_security(&cfg).expect_err("missing cert rejected");
    assert!(
        err.to_string().contains("server.tls.cert-path"),
        "unexpected error: {err}"
    );

    cfg.server.tls.cert_path = Some("/etc/llmctl/tls/server.crt".into());
    cfg.server.tls.key_path = None;
    let err = config::validate_production_security(&cfg).expect_err("missing key rejected");
    assert!(
        err.to_string().contains("server.tls.key-path"),
        "unexpected error: {err}"
    );
}

#[test]
fn external_bind_requires_cra_active_audit_and_otel_controls() {
    let mut cfg = Config::default();
    cfg.server.host = "0.0.0.0".to_string();
    cfg.security.require_auth = true;
    cfg.security.api_keys = vec![hashed_key()];
    cfg.security.tls_termination.enabled = true;
    cfg.security.tls_termination.provider = Some("envoy-edge".to_string());
    cfg.security.tls_termination.evidence = Some("change-record-123".to_string());

    let err = config::validate_production_security(&cfg).expect_err("monthly reports required");
    assert!(
        err.to_string().contains("monthly audit reports"),
        "unexpected error: {err}"
    );

    cfg.audit.monthly_reports = true;
    let err = config::validate_production_security(&cfg).expect_err("OTel endpoint required");
    assert!(
        err.to_string().contains("OTel exporter endpoint"),
        "unexpected error: {err}"
    );

    cfg.observability.exporter.endpoint = Some("https://otel.example.test/v1/traces".to_string());
    config::validate_production_security(&cfg).expect("CRA active controls accepted");
}

#[test]
fn production_security_accepts_known_api_key_scopes() {
    let mut cfg = Config::default();
    cfg.security.production = true;
    cfg.security.require_auth = true;
    cfg.security.api_keys = vec![ApiKeyConfig {
        scopes: vec![
            "chat".to_string(),
            "models.read".to_string(),
            "models".to_string(),
            "admin".to_string(),
        ],
        ..hashed_key()
    }];
    enable_tls_termination(&mut cfg);

    config::validate_production_security(&cfg).expect("known scopes are allowed");
}

#[test]
fn production_security_rejects_unknown_api_key_scopes() {
    let mut cfg = Config::default();
    cfg.security.production = true;
    cfg.security.require_auth = true;
    enable_tls_termination(&mut cfg);
    cfg.security.api_keys = vec![ApiKeyConfig {
        scopes: vec!["chat".to_string(), "models:read".to_string()],
        ..hashed_key()
    }];

    let err = config::validate_production_security(&cfg).expect_err("unknown scope rejected");
    assert!(
        err.to_string().contains("unknown scope `models:read`"),
        "unexpected error: {err}"
    );
}

#[test]
fn production_security_rejects_empty_api_key_scopes() {
    let mut cfg = Config::default();
    cfg.security.production = true;
    cfg.security.require_auth = true;
    enable_tls_termination(&mut cfg);
    cfg.security.api_keys = vec![ApiKeyConfig {
        scopes: vec!["chat".to_string(), "".to_string()],
        ..hashed_key()
    }];

    let err = config::validate_production_security(&cfg).expect_err("empty scope rejected");
    assert!(
        err.to_string().contains("empty scope"),
        "unexpected error: {err}"
    );

    cfg.security.api_keys = vec![ApiKeyConfig {
        scopes: vec![],
        ..hashed_key()
    }];

    let err = config::validate_production_security(&cfg).expect_err("missing scope rejected");
    assert!(
        err.to_string().contains("at least one scope"),
        "unexpected error: {err}"
    );
}

#[test]
fn production_security_rejects_ambiguous_api_key_identity() {
    let mut cfg = Config::default();
    cfg.security.production = true;
    cfg.security.require_auth = true;
    enable_tls_termination(&mut cfg);
    cfg.security.api_keys = vec![ApiKeyConfig {
        id: " ".to_string(),
        ..hashed_key()
    }];

    let err = config::validate_production_security(&cfg).expect_err("blank key id rejected");
    assert!(
        err.to_string().contains("api key id must not be empty"),
        "unexpected error: {err}"
    );

    cfg.security.api_keys = vec![
        hashed_key(),
        ApiKeyConfig {
            subject: "other".to_string(),
            ..hashed_key()
        },
    ];
    let err = config::validate_production_security(&cfg).expect_err("duplicate key id rejected");
    assert!(
        err.to_string().contains("declared more than once"),
        "unexpected error: {err}"
    );

    cfg.security.api_keys = vec![ApiKeyConfig {
        subject: "".to_string(),
        ..hashed_key()
    }];
    let err = config::validate_production_security(&cfg).expect_err("blank subject rejected");
    assert!(
        err.to_string().contains("must declare a subject"),
        "unexpected error: {err}"
    );

    cfg.security.api_keys = vec![ApiKeyConfig {
        team: " ".to_string(),
        ..hashed_key()
    }];
    let err = config::validate_production_security(&cfg).expect_err("blank team rejected");
    assert!(
        err.to_string().contains("must declare a team"),
        "unexpected error: {err}"
    );
}

#[test]
fn config_rejects_plaintext_secret_fields() {
    let body = r#"
[security]
require-auth = true
api-key = "plain-secret"
"#;

    let err = toml::from_str::<Config>(body).expect_err("plaintext secrets are rejected");
    assert!(
        err.to_string().contains("api-key"),
        "unexpected error: {err}"
    );
}

#[test]
fn production_security_rejects_plaintext_observability_secrets() {
    let mut cfg = Config::default();
    cfg.security.production = true;
    cfg.security.require_auth = true;
    cfg.security.api_keys = vec![hashed_key()];
    enable_tls_termination(&mut cfg);
    cfg.observability.exporter.headers.insert(
        "authorization".to_string(),
        "Bearer plain-collector-token".to_string(),
    );

    let err = config::validate_production_security(&cfg).expect_err("plaintext header rejected");
    assert!(
        err.to_string().contains("plaintext secret"),
        "unexpected error: {err}"
    );

    cfg.observability.exporter.headers.insert(
        "authorization".to_string(),
        "env:OTEL_EXPORTER_OTLP_HEADERS".to_string(),
    );
    config::validate_production_security(&cfg).expect("env secret reference is allowed");
}

#[test]
fn observability_attribute_sanitizer_redacts_exporter_and_request_secrets() {
    let attrs: BTreeMap<_, _> = [
        (
            "collector.header.authorization".to_string(),
            json!("Bearer collector-secret"),
        ),
        ("request.prompt".to_string(), json!("private prompt text")),
        ("message.content".to_string(), json!("private message text")),
        (
            "cache.path".to_string(),
            json!("/home/operator/.cache/model.gguf"),
        ),
        ("quota.allowed".to_string(), json!(true)),
    ]
    .into();

    let sanitized = sanitize_otel_attributes(attrs);

    assert_eq!(
        sanitized.get("collector.header.authorization"),
        Some(&json!("[REDACTED]"))
    );
    assert_eq!(sanitized.get("request.prompt"), Some(&json!("[REDACTED]")));
    assert_eq!(sanitized.get("message.content"), Some(&json!("[REDACTED]")));
    assert_eq!(sanitized.get("cache.path"), Some(&json!("[REDACTED]")));
    assert_eq!(sanitized.get("quota.allowed"), Some(&json!(true)));
}

#[test]
fn audit_retention_and_report_settings_are_configurable() {
    let body = r#"
[audit]
retention-days = 90
report-directory = "/var/lib/llmctl/reports"
report-formats = ["json", "csv"]
monthly-reports = true
"#;

    let cfg: Config = toml::from_str(body).expect("audit config parses");
    assert_eq!(cfg.audit.retention_days, 90);
    assert_eq!(
        cfg.audit
            .report_directory
            .as_deref()
            .unwrap()
            .to_str()
            .unwrap(),
        "/var/lib/llmctl/reports"
    );
    assert_eq!(cfg.audit.report_formats, vec!["json", "csv"]);
    assert!(cfg.audit.monthly_reports);
}

#[test]
fn observability_plan_resolves_otlp_exporter() {
    let cfg = Config {
        server: ServerConfig::default(),
        observability: rs_llmctl::config::ObservabilityConfig {
            service_name: Some("llmctl-prod".to_string()),
            exporter: ObservabilityExporterConfig {
                endpoint: Some("https://otel-collector.example:4317".to_string()),
                protocol: OtlpProtocol::Grpc,
                headers: [("x-tenant".to_string(), "platform".to_string())].into(),
                timeout_ms: 10_000,
            },
            traces_enabled: true,
            metrics_enabled: true,
            logs_enabled: false,
            ..Default::default()
        },
        ..Config::default()
    };

    let plan = ObservabilityPlan::from_config(&cfg).expect("observability plan");
    assert_eq!(plan.service_name, "llmctl-prod");
    assert!(plan.traces_enabled);
    assert!(plan.metrics_enabled);
    assert!(!plan.logs_enabled);
    assert_eq!(
        plan.exporter,
        Exporter::Otlp {
            endpoint: "https://otel-collector.example:4317".to_string(),
            protocol: OtlpProtocol::Grpc,
            headers: [("x-tenant".to_string(), "platform".to_string())].into(),
            timeout_ms: 10_000,
        }
    );
}
