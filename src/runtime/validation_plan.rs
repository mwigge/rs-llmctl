//! Native runtime validation-plan construction.
use super::*;

pub fn native_validation_plan(
    cfg: &Config,
    options: NativeRuntimeValidationOptions,
) -> NativeRuntimeValidationPlan {
    NativeRuntimeValidationPlan {
        status: "planned".to_string(),
        mode: "deterministic-offline".to_string(),
        network: false,
        runtime_backend: cfg.runtime.backend,
        real_artifact_smoke_tests: real_artifact_smoke_tests(cfg),
        hardware_matrix: hardware_matrix(),
        soak_tests: soak_tests(&options),
        graceful_drain: graceful_drain_plan(),
        circuit_breaker_and_heartbeat: circuit_breaker_heartbeat_plan(cfg),
        api_key_rotation_and_quota: api_key_quota_plan(&options),
        benchmark: benchmark_plan(),
        json_commands: vec![
            "llmctl --json runtime status".to_string(),
            "llmctl --json runtime heartbeat".to_string(),
            "llmctl --json runtime validation-plan".to_string(),
            "llmctl --json server plan".to_string(),
        ],
    }
}

pub(crate) fn real_artifact_smoke_tests(cfg: &Config) -> Vec<NativeArtifactSmokePlan> {
    [
        CandleModelFamily::Qwen3,
        CandleModelFamily::Gemma4,
        CandleModelFamily::Mistral,
        CandleModelFamily::DeepSeek,
    ]
    .into_iter()
    .map(|family| artifact_smoke_test(cfg, family))
    .collect()
}

fn artifact_smoke_test(cfg: &Config, family: CandleModelFamily) -> NativeArtifactSmokePlan {
    let required_format = NativeModelFormat::Safetensors;
    let configured_model = configured_model_for_family(&cfg.models, family);
    let artifact_validation =
        artifact_validation_plan(family, required_format, configured_model.as_ref());

    NativeArtifactSmokePlan {
        family,
        engine: family.engine_name().to_string(),
        required_format,
        configured_alias: configured_model.as_ref().map(|model| model.alias.clone()),
        configured_artifact: configured_model
            .as_ref()
            .map(|model| model.path.exists())
            .unwrap_or(false),
        artifact_validation,
    }
}

fn configured_model_for_family(
    models: &[ModelConfig],
    family: CandleModelFamily,
) -> Option<ModelConfig> {
    models
        .iter()
        .find(|model| model_matches_family(model, family))
        .cloned()
}

fn model_matches_family(model: &ModelConfig, family: CandleModelFamily) -> bool {
    let needle = family.as_str().replace(['3', '4'], "");
    let haystack = format!(
        "{} {} {}",
        model.alias,
        model.role,
        path_file_name(&model.path)
    )
    .to_ascii_lowercase();
    haystack.contains(family.as_str()) || haystack.contains(&needle)
}

fn path_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string()
}

fn artifact_validation_plan(
    family: CandleModelFamily,
    required_format: NativeModelFormat,
    model: Option<&ModelConfig>,
) -> NativeArtifactValidationPlan {
    let required_files = CandleArtifactLayout::for_format(required_format)
        .requirements
        .into_iter()
        .filter(|requirement| requirement.required)
        .map(|requirement| requirement.filename)
        .collect::<Vec<_>>();
    let smoke_prompt = "Reply with exactly: llmctl native smoke ok".to_string();

    let Some(model) = model else {
        return NativeArtifactValidationPlan {
            status: "planned-missing-local-artifact".to_string(),
            required_files,
            evidence: vec![
                "configure a local safetensors model directory or weight file".to_string(),
                "no network access is required or attempted by this plan".to_string(),
            ],
            smoke_prompt,
        };
    };

    match native::validate_candle_model_artifacts(family, model) {
        Ok(validation) if validation.model_format == required_format => {
            let mut evidence = validation
                .weight_files
                .into_iter()
                .map(|weight| format!("weight: {weight}"))
                .collect::<Vec<_>>();
            if let Some(tokenizer) = validation.tokenizer_file {
                evidence.push(format!("tokenizer: {tokenizer}"));
            }
            if let Some(config) = validation.config_file {
                evidence.push(format!("config: {config}"));
            }
            NativeArtifactValidationPlan {
                status: "ready".to_string(),
                required_files,
                evidence,
                smoke_prompt,
            }
        }
        Ok(validation) => NativeArtifactValidationPlan {
            status: "blocked-wrong-format".to_string(),
            required_files,
            evidence: vec![format!(
                "configured artifact format is {}; safetensors is required for this smoke track",
                validation.model_format.as_str()
            )],
            smoke_prompt,
        },
        Err(err) => NativeArtifactValidationPlan {
            status: "blocked-missing-artifacts".to_string(),
            required_files,
            evidence: vec![err.to_string()],
            smoke_prompt,
        },
    }
}

pub(crate) fn hardware_matrix() -> Vec<NativeHardwareValidationTarget> {
    // "amd-vulkan" is a resource-planning target only — no candle-native
    // execution backend implements it yet. See docs/adr/0001-amd-gpu-acceleration.md.
    [
        ("cpu", NativeAcceleration::Cpu, true),
        ("nvidia-cuda", NativeAcceleration::NvidiaCuda, false),
        ("amd-vulkan", NativeAcceleration::AmdRocm, false),
        ("apple-metal", NativeAcceleration::AppleMetal, false),
    ]
    .into_iter()
    .map(
        |(target, acceleration, required)| NativeHardwareValidationTarget {
            target: target.to_string(),
            acceleration,
            required,
            available_in_plan: "probe-at-runtime-or-mark-skipped".to_string(),
            benchmark_dimensions: vec![
                "latency_ms".to_string(),
                "tokens_per_second".to_string(),
                "rss_bytes".to_string(),
                "vram_bytes".to_string(),
            ],
        },
    )
    .collect()
}

pub(crate) fn soak_tests(options: &NativeRuntimeValidationOptions) -> Vec<NativeSoakTestPlan> {
    vec![
        NativeSoakTestPlan {
            name: "long-streaming-chat".to_string(),
            duration_minutes: options.soak_minutes,
            concurrency: options.streaming_concurrency,
            stream: true,
            assertions: vec![
                "all streams terminate with stop or explicit cancellation".to_string(),
                "token accounting remains monotonic".to_string(),
                "latency and tokens/sec are emitted per interval".to_string(),
            ],
        },
        NativeSoakTestPlan {
            name: "scheduler-under-load".to_string(),
            duration_minutes: options.soak_minutes,
            concurrency: options.streaming_concurrency.saturating_mul(2).max(1),
            stream: false,
            assertions: vec![
                "FIFO admission metadata is present".to_string(),
                "queue-full rejections are bounded and explicit".to_string(),
                "heartbeat remains available while requests are active".to_string(),
            ],
        },
    ]
}

fn graceful_drain_plan() -> NativeDrainValidationPlan {
    NativeDrainValidationPlan {
        scenario: "start streaming requests, trigger hot-swap drain, reject new work, let active streams finish or cancel cleanly".to_string(),
        assertions: vec![
            "no stream is truncated without a terminal event".to_string(),
            "drain state is visible in JSON status".to_string(),
            "audit records active stream count and drain result".to_string(),
        ],
    }
}

fn circuit_breaker_heartbeat_plan(cfg: &Config) -> NativeCircuitBreakerHeartbeatPlan {
    NativeCircuitBreakerHeartbeatPlan {
        scenarios: vec![
            "force repeated native load failures".to_string(),
            "saturate scheduler queue while polling heartbeat".to_string(),
            "recover by lowering concurrency and verify closed breaker resets".to_string(),
        ],
        heartbeat_interval_seconds: cfg.runtime.heartbeat_interval_seconds,
        assertions: vec![
            "breaker opens before unbounded retries".to_string(),
            "heartbeat JSON is emitted under load".to_string(),
            "safe telemetry excludes prompts, keys, and local paths".to_string(),
        ],
    }
}

fn api_key_quota_plan(options: &NativeRuntimeValidationOptions) -> NativeApiKeyQuotaValidationPlan {
    NativeApiKeyQuotaValidationPlan {
        rotation_keys: options.rotation_keys,
        quota_concurrency: options.quota_concurrency,
        assertions: vec![
            "old and new keys are accepted only inside the configured overlap".to_string(),
            "quota counters stay isolated by principal and key id".to_string(),
            "429 responses are deterministic under concurrent quota pressure".to_string(),
        ],
    }
}

fn benchmark_plan() -> NativeBenchmarkPlan {
    NativeBenchmarkPlan {
        output_format: "jsonl".to_string(),
        metrics: vec![
            "latency_ms".to_string(),
            "first_token_latency_ms".to_string(),
            "tokens_per_second".to_string(),
            "input_tokens".to_string(),
            "output_tokens".to_string(),
        ],
        memory_fields: vec![
            "rss_bytes".to_string(),
            "peak_rss_bytes".to_string(),
            "vram_bytes".to_string(),
            "peak_vram_bytes".to_string(),
        ],
        sample_json: serde_json::json!({
            "model": "qwen-safetensors",
            "hardware": "cpu",
            "latency_ms": 0,
            "first_token_latency_ms": 0,
            "tokens_per_second": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "rss_bytes": 0,
            "peak_rss_bytes": 0,
            "vram_bytes": null,
            "peak_vram_bytes": null
        }),
    }
}
