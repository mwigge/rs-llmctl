use crate::*;

pub(crate) async fn runtime_command(
    path: &Path,
    command: RuntimeCommand,
    as_json: bool,
) -> Result<()> {
    let cfg = load_config(path).await?;
    match command {
        RuntimeCommand::Status => {
            let status = runtime::status_from_config(&cfg);
            let mut attributes = BTreeMap::new();
            attributes.insert("runtime.backend".to_string(), json!(status.backend));
            attributes.insert("runtime.engine".to_string(), json!(status.engine));
            attributes.insert("runtime.primary".to_string(), json!(status.primary));
            attributes.insert("runtime.implemented".to_string(), json!(status.implemented));
            attributes.insert(
                "runtime.resource.budget_fraction".to_string(),
                json!(status.resource_policy.budget_fraction),
            );
            emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
                TelemetrySignal::Log,
                TelemetryEventName::NativeRuntimeStatus,
                Utc::now(),
                attributes,
            ));
            emit(as_json, &status)
        }
        RuntimeCommand::Heartbeat => {
            let heartbeat = native::heartbeat_from_config(&cfg);
            emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
                TelemetrySignal::Metric,
                TelemetryEventName::RuntimeHeartbeat,
                Utc::now(),
                heartbeat.safe_telemetry_attributes(),
            ));
            emit(as_json, &heartbeat)
        }
        RuntimeCommand::Placement => emit(as_json, &native::placement_plan_from_config(&cfg)),
        RuntimeCommand::Route(args) => {
            let placement = native::placement_plan_from_config(&cfg);
            native::validate_placement_plan(&placement)?;
            let selection = match (args.model.as_deref(), args.role.as_deref()) {
                (Some(model), None) => native::route_selection_for_model(&placement, model)?,
                (None, Some(role)) => native::route_selection_for_role(&placement, role)?,
                (None, None) => bail!("runtime route requires --model or --role"),
                (Some(_), Some(_)) => unreachable!("clap enforces conflicts_with"),
            };
            emit(as_json, &selection)
        }
        RuntimeCommand::AmdQualification(args) => emit(
            as_json,
            &rs_llmctl::amd::qualification_report_with_evidence(
                args.preview,
                args.arch_opt_in,
                args.evidence.as_deref(),
            ),
        ),
        RuntimeCommand::Gemma4Readiness(args) => {
            let evidence =
                rs_llmctl::readiness::run_gemma4_readiness(&args.model_path, &args.alias).await?;
            let evidence_path = args.evidence_output.unwrap_or_else(|| {
                rs_llmctl::readiness::evidence_path(&cfg.storage.model_dir, &args.alias)
            });
            rs_llmctl::readiness::write_evidence(&evidence_path, &evidence)?;
            let result = rs_llmctl::readiness::ensure_qualified(&evidence);
            emit(as_json, &evidence)?;
            result
        }
        RuntimeCommand::ValidationPlan(args) => emit(
            as_json,
            &runtime::native_validation_plan(
                &cfg,
                runtime::NativeRuntimeValidationOptions {
                    soak_minutes: args.soak_minutes,
                    streaming_concurrency: args.streaming_concurrency,
                    rotation_keys: args.rotation_keys,
                    quota_concurrency: args.quota_concurrency,
                },
            ),
        ),
        RuntimeCommand::ValidationRun(args) => {
            let evidence = runtime_validation_run(&cfg).await;
            if let Some(path) = args.evidence_output.as_ref() {
                write_json_file(path, &evidence).await?;
            }
            let failed = evidence["checks"]
                .as_array()
                .map(|checks| checks.iter().any(|check| check["status"] != "ok"))
                .unwrap_or(true);
            emit(as_json, &evidence)?;
            if failed {
                bail!("native runtime validation failed; inspect validation-run evidence");
            }
            Ok(())
        }
        RuntimeCommand::Validate => {
            let placement = native::placement_plan_from_config(&cfg);
            native::validate_placement_plan(&placement)?;
            emit(
                as_json,
                &json!({
                    "status": "ok",
                    "routing_mode": placement.routing_mode,
                    "nodes": placement.nodes.len(),
                    "unassigned_models": placement.unassigned_models,
                }),
            )
        }
    }
}

async fn runtime_validation_run(cfg: &Config) -> Value {
    let mut checks = Vec::new();
    let placement = native::placement_plan_from_config(cfg);
    checks.push(validation_check(
        "placement",
        native::validate_placement_plan(&placement).map(|_| ()),
    ));

    let runnable_models = cfg.models.iter().filter(|model| model.weight > 0);
    let mut runnable_count = 0usize;
    for model in runnable_models {
        runnable_count += 1;
        let result = configured_candle_family(model)
            .and_then(|family| native::validate_candle_model_artifacts(family, model).map(|_| ()));
        checks.push(validation_check(
            &format!("artifact:{}", model.alias),
            result,
        ));
    }
    if runnable_count == 0 {
        checks.push(json!({
            "name": "artifacts",
            "status": "failed",
            "error": "no positive-weight native models are configured",
        }));
    }

    let failed = checks.iter().any(|check| check["status"] != "ok");
    json!({
        "status": if failed { "failed" } else { "ok" },
        "runtime_backend": rs_llmctl::runtime::RuntimeBackend::CandleNative,
        "executable": true,
        "models_checked": runnable_count,
        "checks": checks,
    })
}

fn validation_check(name: &str, result: Result<()>) -> Value {
    match result {
        Ok(()) => json!({ "name": name, "status": "ok" }),
        Err(err) => json!({ "name": name, "status": "failed", "error": err.to_string() }),
    }
}
