use crate::*;

pub(crate) async fn data_command(path: &Path, command: DataCommand, as_json: bool) -> Result<()> {
    match command {
        DataCommand::Export(args) => {
            let cfg = load_config(path).await?;
            let storage = init_storage(&cfg.storage).await?;
            let (from, to) = window(args.hours);
            if args.envelope {
                anyhow::ensure!(
                    matches!(args.dataset, DataDataset::All)
                        && matches!(args.format, DataExportFormat::Json),
                    "data export --envelope currently wraps the canonical all/json export"
                );
                let report = reporting::data_export_envelope_limited(
                    &storage,
                    from,
                    to,
                    Some(args.max_rows),
                )
                .await?;
                emit(as_json, &report)
            } else {
                let report = if matches!(args.dataset, DataDataset::All) {
                    reporting::data_export_limited(&storage, from, to, Some(args.max_rows)).await?
                } else {
                    let dataset = args
                        .dataset
                        .contract_kind()
                        .context("data export requires a concrete dataset")?;
                    reporting::data_export_dataset_limited(
                        &storage,
                        from,
                        to,
                        dataset,
                        Some(args.max_rows),
                    )
                    .await?
                };
                let output = format_data_export(
                    report,
                    args.dataset,
                    args.format,
                    args.output.as_deref(),
                    args.max_rows,
                )?;
                emit(as_json, &output)
            }
        }
        DataCommand::Contracts(args) => {
            let contracts = if let Some(dataset) = args.dataset {
                vec![contracts::contract_for(dataset.into())]
            } else {
                contracts::all_contracts()
            };
            emit(
                as_json,
                &json!({
                    "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
                    "contracts": contracts
                }),
            )
        }
        DataCommand::VerifyEnvelope(args) => {
            let envelope_bytes = fs::read(&args.path)
                .await
                .with_context(|| format!("read {}", args.path.display()))?;
            let envelope: serde_json::Value = serde_json::from_slice(&envelope_bytes)
                .with_context(|| format!("parse {}", args.path.display()))?;
            let verification = reporting::verify_envelope_value(&envelope)?;
            let mut output = serde_json::to_value(verification)?;
            if let Some(object) = output.as_object_mut() {
                object.insert(
                    "artifact".to_string(),
                    serde_json::Value::String(
                        args.path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("envelope.json")
                            .to_string(),
                    ),
                );
                object.insert(
                    "path_redacted".to_string(),
                    serde_json::Value::String(redact_display_path(&args.path)),
                );
            }
            emit(as_json, &output)
        }
    }
}

fn format_data_export(
    report: reporting::DataExport,
    dataset: DataDataset,
    format: DataExportFormat,
    output: Option<&Path>,
    max_rows: usize,
) -> Result<serde_json::Value> {
    let rows = dataset_rows(&report, dataset)?;
    if rows.len() > max_rows {
        bail!(
            "data export for dataset `{}` produced {} rows, exceeding --max-rows {}; narrow --hours or raise --max-rows",
            dataset.as_str(),
            rows.len(),
            max_rows
        );
    }
    let dataset_name = dataset.as_str();
    let contract = dataset.contract_kind().map(contracts::contract_for);

    match format {
        DataExportFormat::Json if matches!(dataset, DataDataset::All) => {
            Ok(serde_json::to_value(report)?)
        }
        DataExportFormat::Json => Ok(json!({
            "format": "json",
            "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
            "dataset": dataset_name,
            "from": report.from,
            "to": report.to,
            "report_summary": report.report_summary,
            "rows": rows
        })),
        DataExportFormat::Jsonl => Ok(json!({
            "format": "jsonl",
            "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
            "dataset": dataset_name,
            "from": report.from,
            "to": report.to,
            "lines": rows.into_iter().map(|row| serde_json::to_string(&row)).collect::<Result<Vec<_>, _>>()?
        })),
        DataExportFormat::ArrowJson => Ok(json!({
            "format": "arrow-json",
            "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
            "dataset": dataset_name,
            "from": report.from,
            "to": report.to,
            "arrow_schema": contract.map(|contract| contract.arrow_schema).unwrap_or_else(|| json!({
                "format": "arrow-json-schema",
                "name": "rs_llmctl_all_v1",
                "fields": []
            })),
            "rows": rows
        })),
        DataExportFormat::ArrowIpc => {
            let path = output.ok_or_else(|| {
                anyhow::anyhow!("data export --format arrow-ipc requires --output")
            })?;
            let contract = contract.ok_or_else(|| {
                anyhow::anyhow!("data export --format arrow-ipc requires a concrete --dataset")
            })?;
            let row_count = rs_llmctl::data_fabric::write_arrow_ipc(path, &contract, &rows)?;
            let output_path = redact_display_path(path);
            Ok(json!({
                "format": "arrow-ipc",
                "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
                "dataset": dataset_name,
                "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("data.arrow"),
                "output_path_redacted": output_path,
                "rows": row_count,
                "arrow_schema": contract.arrow_schema
            }))
        }
        DataExportFormat::Parquet => {
            let path = output
                .ok_or_else(|| anyhow::anyhow!("data export --format parquet requires --output"))?;
            let contract = contract.ok_or_else(|| {
                anyhow::anyhow!("data export --format parquet requires a concrete --dataset")
            })?;
            let row_count = rs_llmctl::data_fabric::write_parquet(path, &contract, &rows)?;
            let output_path = redact_display_path(path);
            Ok(json!({
                "format": "parquet",
                "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
                "dataset": dataset_name,
                "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("data.parquet"),
                "output_path_redacted": output_path,
                "rows": row_count,
                "arrow_schema": contract.arrow_schema
            }))
        }
    }
}

fn dataset_rows(
    report: &reporting::DataExport,
    dataset: DataDataset,
) -> Result<Vec<serde_json::Value>> {
    match dataset {
        DataDataset::All => Ok(vec![serde_json::to_value(report)?]),
        DataDataset::Security => Ok(report
            .audit_events
            .iter()
            .map(|event| {
                json!({
                    "at": event.at,
                    "kind": "audit",
                    "actor": event.actor,
                    "team": event.team,
                    "resource": event.resource,
                    "outcome": event.outcome,
                    "request_id": event.request_id
                })
            })
            .chain(report.quota_decisions.iter().map(|decision| {
                json!({
                    "at": decision.at,
                    "kind": "quota-decision",
                    "actor": decision.actor,
                    "team": decision.team,
                    "resource": decision.model,
                    "outcome": if decision.allowed { "allowed" } else { "denied" },
                    "request_id": decision.request_id
                })
            }))
            .collect()),
        DataDataset::Observability => Ok(report
            .observation_events
            .iter()
            .map(|event| {
                json!({
                    "at": event.at,
                    "kind": event.kind,
                    "source": event.source,
                    "model": event.model,
                    "value": event.value,
                    "unit": event.unit,
                    "request_id": event.request_id
                })
            })
            .collect()),
        DataDataset::Usage => Ok(report
            .usage_events
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?),
        DataDataset::User => Ok(report
            .usage_summary
            .by_actor
            .iter()
            .map(|actor| {
                let team = report
                    .usage_events
                    .iter()
                    .find(|event| event.actor == actor.key)
                    .map(|event| event.team.as_str())
                    .unwrap_or("unknown");
                json!({
                    "actor": actor.key,
                    "team": team,
                    "request_count": actor.request_count,
                    "input_tokens": actor.input_tokens,
                    "output_tokens": actor.output_tokens,
                    "total_tokens": actor.total_tokens
                })
            })
            .collect()),
        DataDataset::Finops => {
            let mut rows = Vec::new();
            rows.extend(report.usage_summary.by_team.iter().map(|team| {
                json!({
                    "team": team.key,
                    "actor": null,
                    "model": null,
                    "request_count": team.request_count,
                    "total_tokens": team.total_tokens,
                    "total_latency_ms": team.total_latency_ms
                })
            }));
            rows.extend(report.usage_summary.by_actor.iter().map(|actor| {
                json!({
                    "team": null,
                    "actor": actor.key,
                    "model": null,
                    "request_count": actor.request_count,
                    "total_tokens": actor.total_tokens,
                    "total_latency_ms": actor.total_latency_ms
                })
            }));
            rows.extend(report.usage_summary.by_model.iter().map(|model| {
                json!({
                    "team": null,
                    "actor": null,
                    "model": model.key,
                    "request_count": model.request_count,
                    "total_tokens": model.total_tokens,
                    "total_latency_ms": model.total_latency_ms
                })
            }));
            Ok(rows)
        }
        DataDataset::Models => Ok(report
            .models
            .iter()
            .map(|model| {
                json!({
                    "alias": model.alias,
                    "role": model.role,
                    "weight": model.weight,
                    "updated_at": model.updated_at
                })
            })
            .collect()),
        DataDataset::Drift => Ok(report
            .observation_events
            .iter()
            .filter(|event| event.kind.contains("drift"))
            .map(|event| {
                json!({
                    "at": event.at,
                    "kind": event.kind,
                    "model": event.model,
                    "value": event.value,
                    "unit": event.unit,
                    "request_id": event.request_id
                })
            })
            .collect()),
        DataDataset::Lineage => Ok(report
            .lineage
            .iter()
            .map(|join| {
                json!({
                    "at": join.at,
                    "request_id": join.request_id,
                    "lineage_id": join.lineage_id,
                    "model": join.model,
                    "corpus": join.corpus,
                    "source": join.source
                })
            })
            .collect()),
        DataDataset::Audit => Ok(report
            .audit_events
            .iter()
            .map(|event| {
                json!({
                    "at": event.at,
                    "action": event.action,
                    "actor": event.actor,
                    "team": event.team,
                    "resource": event.resource,
                    "outcome": event.outcome,
                    "request_id": event.request_id
                })
            })
            .collect()),
    }
}
