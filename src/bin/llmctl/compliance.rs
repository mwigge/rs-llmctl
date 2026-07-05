use crate::*;

pub(crate) async fn compliance_command(
    path: &Path,
    command: ComplianceCommand,
    as_json: bool,
) -> Result<()> {
    let cfg = load_config(path).await?;
    let evidence = compliance_evidence(&cfg);
    match command {
        ComplianceCommand::Evidence => emit(as_json, &evidence),
        ComplianceCommand::CraArticle14 => emit(as_json, &evidence["cra_article_14"]),
        ComplianceCommand::PciDss => emit(as_json, &evidence["pci_dss"]),
        ComplianceCommand::ReleaseChecklist => emit(as_json, &evidence["release_integrity"]),
    }
}

fn compliance_evidence(cfg: &Config) -> serde_json::Value {
    json!({
        "generated_at": Utc::now(),
        "status": "operator_evidence_ready",
        "security_posture": {
            "production": cfg.security.production,
            "require_auth": cfg.security.require_auth,
            "bind_external": cfg.security.bind_external || config::is_external_host(&cfg.server.host),
            "hashed_api_keys": cfg.security.api_keys.iter().all(|key| key.sha256.len() == 64),
            "api_key_count": cfg.security.api_keys.len(),
            "tls_termination": {
                "enabled": cfg.security.tls_termination.enabled,
                "provider": cfg.security.tls_termination.provider.as_deref(),
                "evidence": cfg.security.tls_termination.evidence.as_deref(),
                "m_tls": cfg.security.tls_termination.m_tls
            },
            "audit_retention_days": cfg.audit.retention_days,
            "monthly_reports": cfg.audit.monthly_reports
        },
        "evidence_completeness": {
            "production_security_validation": cfg.security.production || cfg.security.bind_external || config::is_external_host(&cfg.server.host),
            "hashed_api_keys": cfg.security.api_keys.iter().all(|key| key.sha256.len() == 64),
            "tls_termination_documented": cfg.security.tls_termination.enabled
                && cfg.security.tls_termination.provider.as_deref().is_some_and(|value| !value.trim().is_empty())
                && cfg.security.tls_termination.evidence.as_deref().is_some_and(|value| !value.trim().is_empty()),
            "audit_reports_enabled": cfg.audit.monthly_reports && cfg.audit.retention_days > 0,
            "otel_enabled": cfg.observability.traces_enabled && cfg.observability.metrics_enabled && cfg.observability.logs_enabled,
            "otel_exporter_configured": cfg.observability.exporter.endpoint.is_some() || cfg.observability.otlp_endpoint.is_some(),
            "cra_article_14_active_control": cfg.audit.monthly_reports
                && cfg.audit.retention_days > 0
                && cfg.observability.traces_enabled
                && cfg.observability.metrics_enabled
                && cfg.observability.logs_enabled
                && (cfg.observability.exporter.endpoint.is_some() || cfg.observability.otlp_endpoint.is_some()),
            "release_integrity_scripts": [
                "packaging/generate-sbom.sh",
                "packaging/generate-checksums.sh",
                "packaging/sign-release.sh"
            ]
        },
        "cra_article_14": {
            "regulation": "Regulation (EU) 2024/2847",
            "operational_status": "active_control",
            "control_assumption": "treat CRA Article 14 obligations as live for all production operations",
            "early_warning_due": "within_24_hours",
            "vulnerability_notification_due": "within_72_hours",
            "final_vulnerability_report_due": "within_14_days_after_mitigation",
            "severe_incident_notification_due": "without_undue_delay",
            "evidence_commands": [
                "llmctl security audit-config",
                "llmctl audit report monthly --envelope",
                "llmctl data export --envelope",
                "llmctl compliance evidence"
            ],
            "workflow": [
                "classify vulnerability or severe incident",
                "open incident record with impacted versions and mitigations",
                "submit regulatory notification in the required window",
                "attach signed release artifacts, SBOM, audit envelope, and data export",
                "close with final report after mitigation verification"
            ]
        },
        "pci_dss": {
            "baseline": "pci_dss_v4_0_1_aligned_controls",
            "controls": [
                { "area": "access_control", "evidence": "hashed API keys, scopes, auth-required production validation" },
                { "area": "audit_logging", "evidence": "audit events, request IDs, report envelopes, retention plan/apply" },
                { "area": "vulnerability_management", "evidence": "cargo audit, SBOM generation, signed release checksums" },
                { "area": "secure_configuration", "evidence": "security check, audit-config, documented TLS termination or mTLS, least privilege systemd unit" },
                { "area": "monitoring", "evidence": "OTel traces, metrics, logs, resource snapshots, drift observations" }
            ],
            "regular_reports": [
                "monthly audit report",
                "per-request audit report",
                "data export envelope",
                "usage chargeback report",
                "quota report"
            ]
        },
        "release_integrity": {
            "sbom": "packaging/generate-sbom.sh",
            "checksums": "packaging/generate-checksums.sh",
            "signing": "packaging/sign-release.sh",
            "provenance": "CI run URL, git commit, signed tag, SBOM, checksums, signature",
            "required_before_release": [
                "cargo fmt --check",
                "cargo clippy --all-targets --all-features -- -D warnings",
                "cargo test --all-targets --all-features",
                "cargo audit",
                "packaging/generate-sbom.sh",
                "packaging/generate-checksums.sh",
                "packaging/sign-release.sh dist"
            ]
        }
    })
}
