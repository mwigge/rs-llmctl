use crate::*;

pub(crate) async fn audit_command(path: &Path, command: AuditCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let storage = init_storage(&cfg.storage).await?;
    match command {
        AuditCommand::Report { command } => match command {
            AuditReportCommand::Monthly(args) => {
                let now = Utc::now();
                let year = args.year.unwrap_or_else(|| now.year());
                let month = args.month.unwrap_or_else(|| now.month());
                if args.envelope {
                    let report =
                        reporting::monthly_audit_report_envelope(&storage, year, month).await?;
                    if args.write {
                        let path =
                            write_audit_report(&cfg, year, month, "envelope", &report).await?;
                        return emit(
                            as_json,
                            &json!({"status": "written", "path_redacted": redact_display_path(&path)}),
                        );
                    }
                    emit(as_json, &report)
                } else {
                    let report = reporting::monthly_audit_report(&storage, year, month).await?;
                    if args.write {
                        let path = write_audit_report(&cfg, year, month, "report", &report).await?;
                        return emit(
                            as_json,
                            &json!({"status": "written", "path_redacted": redact_display_path(&path)}),
                        );
                    }
                    emit(as_json, &report)
                }
            }
            AuditReportCommand::Request(args) => {
                if args.envelope {
                    let report =
                        reporting::per_request_audit_report_envelope(&storage, args.request_id)
                            .await?;
                    emit(as_json, &report)
                } else {
                    let report =
                        reporting::per_request_audit_report(&storage, args.request_id).await?;
                    emit(as_json, &report)
                }
            }
        },
        AuditCommand::Retention { command } => match command {
            AuditRetentionCommand::Plan(args) => {
                let report = audit_retention_plan(&cfg, &storage).await?;
                if args.envelope {
                    let envelope =
                        reporting::report_envelope(reporting::ReportKind::RetentionPlan, report)?;
                    emit(as_json, &envelope)
                } else {
                    emit(as_json, &report)
                }
            }
            AuditRetentionCommand::Apply(args) => {
                anyhow::ensure!(
                    args.yes,
                    "audit retention apply requires --yes; run audit retention plan first"
                );
                let report = audit_retention_apply(&cfg, &storage).await?;
                if args.envelope {
                    let envelope =
                        reporting::report_envelope(reporting::ReportKind::RetentionPlan, report)?;
                    emit(as_json, &envelope)
                } else {
                    emit(as_json, &report)
                }
            }
        },
        AuditCommand::Request(args) => {
            let event = AuditEvent::new(
                None,
                args.actor,
                args.team,
                args.action,
                args.resource,
                args.outcome,
                json!({ "source": "llmctl audit request" }),
            );
            storage.insert_audit_event(&event).await?;
            emit(as_json, &event)
        }
    }
}

async fn audit_retention_plan(cfg: &Config, storage: &Storage) -> Result<serde_json::Value> {
    let generated_at = Utc::now();
    let cutoff = generated_at - Duration::days(i64::from(cfg.audit.retention_days));
    let counts = storage.audit_retention_counts(cutoff).await?;

    Ok(json!({
        "status": "planned",
        "operation": "audit_retention",
        "dry_run": true,
        "deletes": false,
        "generated_at": generated_at,
        "retention": {
            "days": cfg.audit.retention_days,
            "cutoff": cutoff,
            "report_directory": cfg.audit.report_directory,
            "report_formats": cfg.audit.report_formats,
            "monthly_reports": cfg.audit.monthly_reports
        },
        "counts": counts
    }))
}

async fn audit_retention_apply(cfg: &Config, storage: &Storage) -> Result<serde_json::Value> {
    let generated_at = Utc::now();
    let cutoff = generated_at - Duration::days(i64::from(cfg.audit.retention_days));
    let before = storage.audit_retention_counts(cutoff).await?;
    let deleted = storage.delete_audit_events_before(cutoff).await?;
    let after = storage.audit_retention_counts(cutoff).await?;

    Ok(json!({
        "status": "applied",
        "operation": "audit_retention",
        "dry_run": false,
        "deletes": true,
        "generated_at": generated_at,
        "retention": {
            "days": cfg.audit.retention_days,
            "cutoff": cutoff,
            "report_directory": cfg.audit.report_directory,
            "report_formats": cfg.audit.report_formats,
            "monthly_reports": cfg.audit.monthly_reports
        },
        "deleted": deleted,
        "counts_before": before,
        "counts_after": after
    }))
}

async fn write_audit_report<T: Serialize>(
    cfg: &Config,
    year: i32,
    month: u32,
    suffix: &str,
    report: &T,
) -> Result<PathBuf> {
    let dir = cfg
        .audit
        .report_directory
        .as_ref()
        .context("audit report monthly --write requires audit.report-directory")?;
    fs::create_dir_all(dir)
        .await
        .with_context(|| format!("create audit report directory {}", dir.display()))?;
    #[cfg(unix)]
    fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
        .await
        .with_context(|| format!("secure audit report directory {}", dir.display()))?;
    let path = dir.join(format!("monthly-audit-{year:04}-{month:02}-{suffix}.json"));
    let body = serde_json::to_vec_pretty(report)?;
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(&path)
        .await
        .with_context(|| format!("write audit report {}", path.display()))?;
    file.write_all(&body)
        .await
        .with_context(|| format!("write audit report {}", path.display()))?;
    Ok(path)
}

pub(crate) async fn audit_config_report(
    path: &Path,
    cfg: &Config,
    systemd_unit: Option<&Path>,
) -> Result<serde_json::Value> {
    let external_bind = cfg.security.bind_external || config::is_external_host(&cfg.server.host);
    let key_reports: Vec<_> = cfg
        .security
        .api_keys
        .iter()
        .map(|key| {
            json!({
                "id": key.id,
                "subject": key.subject,
                "team": key.team,
                "scopes": key.scopes,
                "sha256_present": !key.sha256.is_empty(),
                "sha256_valid": is_sha256_hex(&key.sha256)
            })
        })
        .collect();
    let hashed_api_keys = cfg
        .security
        .api_keys
        .iter()
        .all(|key| is_sha256_hex(&key.sha256));
    let secret_headers: Vec<_> = cfg
        .observability
        .exporter
        .headers
        .iter()
        .filter(|(name, _)| is_sensitive_name(name))
        .map(|(name, value)| {
            let value_source = if value.starts_with("env:") {
                "env"
            } else {
                "plaintext"
            };
            json!({
                "name": name,
                "value_source": value_source,
                "reference": if value.starts_with("env:") { Some(value.as_str()) } else { None }
            })
        })
        .collect();
    let systemd = systemd_audit(systemd_unit).await?;
    let trusted_proxy_reports: Vec<_> = cfg
        .security
        .trusted_proxies
        .iter()
        .map(|proxy| {
            let valid = trusted_proxy_is_explicit(proxy);
            json!({
                "value": proxy,
                "valid": valid,
                "wildcard": proxy.trim() == "*"
            })
        })
        .collect();
    let trusted_proxies_valid = trusted_proxy_reports
        .iter()
        .all(|proxy| proxy["valid"].as_bool().unwrap_or(false));

    let mut findings = Vec::new();
    if !hashed_api_keys {
        findings.push("api keys must be stored as sha256 hex digests".to_string());
    }
    if (cfg.security.production || external_bind)
        && (!cfg.security.require_auth || cfg.security.api_keys.is_empty())
    {
        findings.push("external/production serving requires authentication".to_string());
    }
    if cfg.security.production || external_bind {
        let native_tls = cfg.server.tls.enabled
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
                .is_some_and(|path| !path.as_os_str().is_empty());
        if !native_tls && !cfg.security.tls_termination.enabled {
            findings.push(
                "external/production serving requires native TLS or documented TLS termination or mTLS"
                    .to_string(),
            );
        }
        if !native_tls
            && cfg
                .security
                .tls_termination
                .provider
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            findings.push("TLS termination must declare a provider".to_string());
        }
        if !native_tls
            && cfg
                .security
                .tls_termination
                .evidence
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            findings.push("TLS termination must declare evidence".to_string());
        }
        if !native_tls && cfg.security.trusted_proxies.is_empty() {
            findings.push("TLS termination must declare trusted-proxies".to_string());
        }
        if !native_tls && !trusted_proxies_valid {
            findings.push(
                "trusted-proxies must list explicit IP addresses or CIDR ranges; wildcard is not allowed"
                    .to_string(),
            );
        }
    }
    if cfg
        .observability
        .exporter
        .headers
        .iter()
        .any(|(name, value)| is_sensitive_name(name) && !value.starts_with("env:"))
    {
        findings.push("observability secret headers must use environment references".to_string());
    }
    if cfg.audit.retention_days == 0 {
        findings.push("audit retention must be greater than zero days".to_string());
    }
    if cfg.security.production || external_bind {
        if !cfg.audit.monthly_reports {
            findings
                .push("CRA Article 14 active control requires monthly audit reports".to_string());
        }
        if cfg
            .audit
            .report_directory
            .as_ref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            findings
                .push("CRA Article 14 active control requires audit.report-directory".to_string());
        }
        if !(cfg.observability.traces_enabled
            && cfg.observability.metrics_enabled
            && cfg.observability.logs_enabled)
        {
            findings.push(
                "CRA Article 14 active control requires OTel traces, metrics, and logs".to_string(),
            );
        }
        if cfg
            .observability
            .exporter
            .endpoint
            .as_deref()
            .or(cfg.observability.otlp_endpoint.as_deref())
            .is_none_or(|endpoint| endpoint.trim().is_empty())
        {
            findings.push(
                "CRA Article 14 active control requires an OTel exporter endpoint".to_string(),
            );
        }
    }
    if systemd_unit.is_some()
        && (!systemd["present"].as_bool().unwrap_or(false)
            || !systemd["has_exec_start"].as_bool().unwrap_or(false))
    {
        findings.push("systemd unit template is missing or incomplete".to_string());
    }

    Ok(json!({
        "status": if findings.is_empty() { "ok" } else { "warning" },
        "config": path.file_name().and_then(|name| name.to_str()).unwrap_or("config.toml"),
        "config_path_redacted": redact_evidence_path(path),
        "external_bind": {
            "enabled": external_bind,
            "host": cfg.server.host,
            "port": cfg.server.port,
            "declared": cfg.security.bind_external
        },
        "auth": {
            "production": cfg.security.production,
            "require_auth": cfg.security.require_auth,
            "api_key_count": cfg.security.api_keys.len(),
            "hashed_api_keys": hashed_api_keys,
            "keys": key_reports
        },
        "tls_termination": {
            "enabled": cfg.security.tls_termination.enabled,
            "provider": cfg.security.tls_termination.provider.as_deref(),
            "evidence": cfg.security.tls_termination.evidence.as_deref(),
            "m_tls": cfg.security.tls_termination.m_tls,
            "trusted_proxies": {
                "count": cfg.security.trusted_proxies.len(),
                "valid": trusted_proxies_valid,
                "entries": trusted_proxy_reports
            }
        },
        "observability": {
            "endpoint_configured": cfg.observability.exporter.endpoint.is_some() || cfg.observability.otlp_endpoint.is_some(),
            "secret_header_count": secret_headers.len(),
            "secret_headers": secret_headers
        },
        "audit": {
            "retention_days": cfg.audit.retention_days,
            "report_directory": cfg.audit.report_directory.as_ref().and_then(|path| path.file_name()).and_then(|name| name.to_str()),
            "report_directory_redacted": cfg.audit.report_directory.as_ref().map(|path| redact_evidence_path(path)),
            "report_formats": cfg.audit.report_formats,
            "monthly_reports": cfg.audit.monthly_reports
        },
        "cra_article_14": {
            "regulation": "Regulation (EU) 2024/2847",
            "operational_status": "active_control",
            "monthly_reports": cfg.audit.monthly_reports,
            "otel_exporter_configured": cfg.observability.exporter.endpoint.is_some() || cfg.observability.otlp_endpoint.is_some(),
            "required_evidence": [
                "security audit-config",
                "audit report monthly --envelope",
                "audit report request --envelope",
                "data export --envelope",
                "compliance evidence"
            ]
        },
        "systemd": systemd,
        "findings": findings
    }))
}

async fn systemd_audit(systemd_unit: Option<&Path>) -> Result<serde_json::Value> {
    let Some(path) = systemd_unit else {
        return Ok(json!({
            "checked": false,
            "present": false,
            "artifact": null,
            "path_redacted": null,
            "has_service_section": false,
            "has_exec_start": false,
            "mentions_llmctld": false
        }));
    };

    match fs::read_to_string(path).await {
        Ok(body) => Ok(json!({
            "checked": true,
            "present": true,
            "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("systemd-unit"),
            "path_redacted": redact_evidence_path(path),
            "has_service_section": body.contains("[Service]"),
            "has_exec_start": body.lines().any(|line| line.trim_start().starts_with("ExecStart=")),
            "mentions_llmctld": body.contains("llmctld")
        })),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(json!({
            "checked": true,
            "present": false,
            "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("systemd-unit"),
            "path_redacted": redact_evidence_path(path),
            "has_service_section": false,
            "has_exec_start": false,
            "mentions_llmctld": false
        })),
        Err(err) => Err(err).with_context(|| format!("read systemd unit {}", path.display())),
    }
}
