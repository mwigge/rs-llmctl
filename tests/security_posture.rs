use rs_llmctl::config::{
    self, ApiKeyConfig, Config, ObservabilityExporterConfig, OtlpProtocol, ServerConfig,
};
use rs_llmctl::observability::{Exporter, ObservabilityPlan};

fn hashed_key() -> ApiKeyConfig {
    ApiKeyConfig {
        id: "ops".to_string(),
        sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        subject: "operator".to_string(),
        team: "platform".to_string(),
        scopes: vec!["models:read".to_string()],
    }
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
    config::validate_production_security(&cfg).expect("valid production posture");
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
