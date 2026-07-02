use crate::cli::{AiopsCommand, AiopsIncidentTemplateArgs, AiopsSloPlanArgs, AiopsSloPlanFormat};
use crate::emit;
use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::json;
use std::fs as stdfs;

pub(crate) async fn aiops_command(command: AiopsCommand, as_json: bool) -> Result<()> {
    match command {
        AiopsCommand::Gaps => emit(as_json, &aiops_gaps_report()),
        AiopsCommand::SloPlan(args) => emit_slo_plan(args),
        AiopsCommand::IncidentTemplate(args) => emit(as_json, &incident_template(args)),
    }
}

fn aiops_gaps_report() -> serde_json::Value {
    json!({
        "status": "tracked",
        "delivered": [
            "typed production/local config profiles",
            "SSE, log, event, OTel, and data-fabric config fields",
            "schema-versioned contracts for security, observability, usage, user, finops, model, drift, and audit datasets",
            "domain-filtered JSON, JSONL, Arrow-schema JSON, Arrow IPC, and Parquet exports",
            "CRA Article 14 active-control evidence and PCI DSS aligned reporting commands",
            "OpenAI-compatible model and chat serving, local search, recommendations, quotas, and worker lifecycle controls",
            "manifest-driven eval suites that execute golden prompts against OpenAI-compatible endpoints",
            "runtime request-to-lineage joins for chat, local search, and recommendations",
            "Prometheus/Alertmanager rules and Grafana dashboard renderers for SLOs",
            "HMAC policy bundles plus Ed25519 policy signatures and hash-chained transparency logs",
            "Candle-native greedy autoregressive decoding for Qwen3, Gemma-family, and Mistral safetensors paths where Candle exposes model support"
        ],
        "gaps": [
            {
                "area": "native-inference",
                "gap": "DeepSeek, Kimi, and MiniMax remain tracked native backend targets; DeepSeek metadata exists in Candle but is not wired and verified, while Kimi and MiniMax do not expose reviewed Candle architecture modules to instantiate",
                "next_control": "wire DeepSeek first if Candle deepseek2 maps cleanly to the target artifacts, then upgrade Candle or vendor reviewed Kimi and MiniMax model implementations behind the NativeCandleDecoder"
            },
            {
                "area": "observability",
                "gap": "RED metrics, upstream circuit-breaker state metrics, heartbeat, admission rejection metrics, and worker lifecycle telemetry are emitted; deeper burn-rate deployment sync remains operator-managed",
                "next_control": "add optional push/apply helpers for Prometheus and Grafana provisioning"
            },
            {
                "area": "model-quality",
                "gap": "eval suites execute configured prompts, but advanced judges and rubric scoring are not bundled",
                "next_control": "add optional LLM-as-judge and rubric evaluators with deterministic evidence output"
            },
            {
                "area": "lineage",
                "gap": "runtime joins are recorded when clients provide lineage IDs; automatic corpus/model lineage inference is not complete",
                "next_control": "derive lineage IDs from configured model manifests and managed RAG indexes"
            },
            {
                "area": "operations",
                "gap": "SLO plans include Prometheus/Alertmanager rules and Grafana dashboards; live deployment sync is operator-managed",
                "next_control": "add optional push/apply helpers for Prometheus rule files and Grafana dashboard provisioning"
            },
            {
                "area": "governance",
                "gap": "Ed25519 signing and a local transparency log exist; Sigstore/Rekor publication is not bundled",
                "next_control": "add optional Sigstore/Rekor publication for organizations that want public transparency"
            }
        ]
    })
}

fn emit_slo_plan(args: AiopsSloPlanArgs) -> Result<()> {
    let rendered = match args.format {
        AiopsSloPlanFormat::Plan => serde_json::to_string_pretty(&slo_plan(&args))?,
        AiopsSloPlanFormat::Prometheus => prometheus_slo_rules(&args),
        AiopsSloPlanFormat::Grafana => serde_json::to_string_pretty(&grafana_slo_dashboard(&args))?,
    };

    if let Some(output) = args.output {
        stdfs::write(&output, rendered).with_context(|| format!("write {}", output.display()))?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn slo_plan(args: &AiopsSloPlanArgs) -> serde_json::Value {
    json!({
        "schema_version": 1,
        "kind": "slo-plan",
        "generated_at": Utc::now(),
        "slos": {
            "availability_percent": args.availability_percent,
            "latency_p95_ms": args.latency_p95_ms,
            "error_rate_percent": args.error_rate_percent
        },
        "alert_rules": [
            {
                "name": "llmctl_availability_below_slo",
                "expr": format!("100 * (1 - (sum(rate(llmctl_request_errors_total[5m])) / sum(rate(llmctl_requests_total[5m])))) < {}", args.availability_percent),
                "for": "10m",
                "severity": "page"
            },
            {
                "name": "llmctl_high_error_rate",
                "expr": format!("sum(rate(llmctl_request_errors_total[5m])) / sum(rate(llmctl_requests_total[5m])) > {}", args.error_rate_percent / 100.0),
                "for": "10m",
                "severity": "page"
            },
            {
                "name": "llmctl_fast_burn_error_budget",
                "expr": format!("(sum(rate(llmctl_slo_violations_total[5m])) / sum(rate(llmctl_requests_total[5m])) > {fast_burn}) and (sum(rate(llmctl_slo_violations_total[1h])) / sum(rate(llmctl_requests_total[1h])) > {fast_burn})", fast_burn = (100.0 - args.availability_percent) / 100.0 * 14.4),
                "for": "2m",
                "severity": "page"
            },
            {
                "name": "llmctl_slow_burn_error_budget",
                "expr": format!("(sum(rate(llmctl_slo_violations_total[30m])) / sum(rate(llmctl_requests_total[30m])) > {slow_burn}) and (sum(rate(llmctl_slo_violations_total[6h])) / sum(rate(llmctl_requests_total[6h])) > {slow_burn})", slow_burn = (100.0 - args.availability_percent) / 100.0 * 6.0),
                "for": "15m",
                "severity": "ticket"
            },
            {
                "name": "llmctl_high_latency_p95",
                "expr": format!("histogram_quantile(0.95, rate(llmctl_request_latency_ms_bucket[5m])) > {}", args.latency_p95_ms),
                "for": "15m",
                "severity": "ticket"
            }
        ],
        "evidence_commands": [
            "llmctl observe plan",
            "llmctl usage report --hours 24",
            "llmctl compliance evidence"
        ]
    })
}

fn prometheus_slo_rules(args: &AiopsSloPlanArgs) -> String {
    format!(
        r#"groups:
  - name: llmctl_slo_alerts
    rules:
      - alert: LlmctlAvailabilityBelowSlo
        expr: 100 * (1 - (sum(rate(llmctl_request_errors_total[5m])) / sum(rate(llmctl_requests_total[5m])))) < {availability_percent}
        for: 10m
        labels:
          severity: page
          service: llmctl
        annotations:
          summary: rs-llmctl availability is below SLO
          description: Availability over 5m is below {availability_percent}%.
      - alert: LlmctlHighErrorRate
        expr: sum(rate(llmctl_request_errors_total[5m])) / sum(rate(llmctl_requests_total[5m])) > {error_rate}
        for: 10m
        labels:
          severity: page
          service: llmctl
        annotations:
          summary: rs-llmctl error rate exceeds SLO
          description: Error rate over 5m is above {error_rate_percent}%.
      - alert: LlmctlFastBurnErrorBudget
        expr: (sum(rate(llmctl_slo_violations_total[5m])) / sum(rate(llmctl_requests_total[5m])) > {fast_burn}) and (sum(rate(llmctl_slo_violations_total[1h])) / sum(rate(llmctl_requests_total[1h])) > {fast_burn})
        for: 2m
        labels:
          severity: page
          service: llmctl
        annotations:
          summary: rs-llmctl is burning error budget quickly
          description: 5m and 1h burn-rate windows both exceed the fast-burn threshold.
      - alert: LlmctlSlowBurnErrorBudget
        expr: (sum(rate(llmctl_slo_violations_total[30m])) / sum(rate(llmctl_requests_total[30m])) > {slow_burn}) and (sum(rate(llmctl_slo_violations_total[6h])) / sum(rate(llmctl_requests_total[6h])) > {slow_burn})
        for: 15m
        labels:
          severity: ticket
          service: llmctl
        annotations:
          summary: rs-llmctl is steadily burning error budget
          description: 30m and 6h burn-rate windows both exceed the slow-burn threshold.
      - alert: LlmctlHighLatencyP95
        expr: histogram_quantile(0.95, sum(rate(llmctl_request_latency_ms_bucket[5m])) by (le)) > {latency_p95_ms}
        for: 15m
        labels:
          severity: ticket
          service: llmctl
        annotations:
          summary: rs-llmctl p95 latency exceeds SLO
          description: Request latency p95 is above {latency_p95_ms}ms.
"#,
        availability_percent = args.availability_percent,
        error_rate = args.error_rate_percent / 100.0,
        error_rate_percent = args.error_rate_percent,
        fast_burn = (100.0 - args.availability_percent) / 100.0 * 14.4,
        slow_burn = (100.0 - args.availability_percent) / 100.0 * 6.0,
        latency_p95_ms = args.latency_p95_ms,
    )
}

fn grafana_slo_dashboard(args: &AiopsSloPlanArgs) -> serde_json::Value {
    json!({
        "uid": "llmctl-slos",
        "title": "rs-llmctl SLOs",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "tags": ["llmctl", "slo", "aiops"],
        "time": {
            "from": "now-6h",
            "to": "now"
        },
        "templating": {
            "list": [
                {
                    "name": "datasource",
                    "type": "datasource",
                    "query": "prometheus",
                    "current": {
                        "text": "Prometheus",
                        "value": "Prometheus"
                    }
                }
            ]
        },
        "panels": [
            grafana_timeseries_panel(
                1,
                "Availability",
                0,
                0,
                "percent",
                "100 * sum(rate(llmctl_requests_total{status!=\"error\"}[5m])) / sum(rate(llmctl_requests_total[5m]))".to_string(),
                Some(args.availability_percent),
            ),
            grafana_timeseries_panel(
                2,
                "Error Rate",
                12,
                0,
                "percentunit",
                "sum(rate(llmctl_requests_total{status=\"error\"}[5m])) / sum(rate(llmctl_requests_total[5m]))".to_string(),
                Some(args.error_rate_percent / 100.0),
            ),
            grafana_timeseries_panel(
                3,
                "Latency p95",
                0,
                8,
                "ms",
                "histogram_quantile(0.95, sum(rate(llmctl_request_latency_ms_bucket[5m])) by (le))".to_string(),
                Some(args.latency_p95_ms as f64),
            ),
        ]
    })
}

fn grafana_timeseries_panel(
    id: u64,
    title: &str,
    x: u64,
    y: u64,
    unit: &str,
    expr: String,
    threshold: Option<f64>,
) -> serde_json::Value {
    json!({
        "id": id,
        "type": "timeseries",
        "title": title,
        "datasource": {
            "type": "prometheus",
            "uid": "${datasource}"
        },
        "gridPos": {
            "h": 8,
            "w": 12,
            "x": x,
            "y": y
        },
        "targets": [
            {
                "refId": "A",
                "expr": expr,
                "legendFormat": title
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {
                            "color": "green",
                            "value": null
                        },
                        {
                            "color": "red",
                            "value": threshold
                        }
                    ]
                }
            },
            "overrides": []
        },
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": "bottom"
            },
            "tooltip": {
                "mode": "single",
                "sort": "none"
            }
        }
    })
}

fn incident_template(args: AiopsIncidentTemplateArgs) -> serde_json::Value {
    json!({
        "schema_version": 1,
        "kind": "incident-evidence-template",
        "generated_at": Utc::now(),
        "severity": args.severity,
        "team": args.team,
        "cra_article_14": {
            "operational_status": "active_control",
            "early_warning_due": "within_24_hours",
            "vulnerability_notification_due": "within_72_hours",
            "final_vulnerability_report_due": "within_14_days_after_mitigation"
        },
        "sections": [
            "summary",
            "timeline",
            "affected_models",
            "affected_users_or_teams",
            "security_impact",
            "data_impact",
            "mitigation",
            "evidence"
        ],
        "evidence_commands": [
            "llmctl security audit-config",
            "llmctl audit report monthly --envelope",
            "llmctl data export --envelope",
            "llmctl lineage list",
            "llmctl eval report"
        ]
    })
}
