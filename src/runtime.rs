use crate::config::{Config, ModelConfig, NativeEmbeddingMode};
use crate::native::{
    self, CandleArtifactLayout, CandleModelFamily, NativeAcceleration, NativeModelFormat,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeBackend {
    #[default]
    CandleNative,
}

impl RuntimeBackend {
    pub fn is_in_process(self) -> bool {
        matches!(self, Self::CandleNative)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub backend: RuntimeBackend,
    pub primary: bool,
    pub implemented: bool,
    pub engine: String,
    pub execution_model: String,
    pub starter_roles: Vec<RuntimeStarterRole>,
    pub resource_policy: RuntimeResourcePolicy,
    pub token_accounting: String,
    pub embeddings: RuntimeEmbeddingContract,
    pub observability: Vec<String>,
    pub security: Vec<String>,
    pub compatibility: Vec<String>,
    pub next_step: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEmbeddingContract {
    pub mode: NativeEmbeddingMode,
    pub semantic_model_alias: Option<String>,
    pub semantic_backend: String,
    pub fallback_backend: String,
    pub fallback_status: String,
    pub fallback_dev_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStarterRole {
    pub role: String,
    pub default_family: String,
    pub alternative_families: Vec<String>,
    pub eu_friendly_family: Option<String>,
    pub formats: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeResourcePolicy {
    pub budget_fraction: f64,
    pub cpu_fraction: f64,
    pub ram_fraction: f64,
    pub gpu_vram_fraction: f64,
    pub gpu_detection: Vec<String>,
    pub enforcement: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRuntimeValidationOptions {
    pub soak_minutes: u64,
    pub streaming_concurrency: u32,
    pub rotation_keys: u32,
    pub quota_concurrency: u32,
}

impl Default for NativeRuntimeValidationOptions {
    fn default() -> Self {
        Self {
            soak_minutes: 240,
            streaming_concurrency: 8,
            rotation_keys: 3,
            quota_concurrency: 16,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRuntimeValidationPlan {
    pub status: String,
    pub mode: String,
    pub network: bool,
    pub runtime_backend: RuntimeBackend,
    pub real_artifact_smoke_tests: Vec<NativeArtifactSmokePlan>,
    pub hardware_matrix: Vec<NativeHardwareValidationTarget>,
    pub soak_tests: Vec<NativeSoakTestPlan>,
    pub graceful_drain: NativeDrainValidationPlan,
    pub circuit_breaker_and_heartbeat: NativeCircuitBreakerHeartbeatPlan,
    pub api_key_rotation_and_quota: NativeApiKeyQuotaValidationPlan,
    pub benchmark: NativeBenchmarkPlan,
    pub json_commands: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeArtifactSmokePlan {
    pub family: CandleModelFamily,
    pub engine: String,
    pub required_format: NativeModelFormat,
    pub configured_alias: Option<String>,
    pub configured_artifact: bool,
    pub artifact_validation: NativeArtifactValidationPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeArtifactValidationPlan {
    pub status: String,
    pub required_files: Vec<String>,
    pub evidence: Vec<String>,
    pub smoke_prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeHardwareValidationTarget {
    pub target: String,
    pub acceleration: NativeAcceleration,
    pub required: bool,
    pub available_in_plan: String,
    pub benchmark_dimensions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeSoakTestPlan {
    pub name: String,
    pub duration_minutes: u64,
    pub concurrency: u32,
    pub stream: bool,
    pub assertions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeDrainValidationPlan {
    pub scenario: String,
    pub assertions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCircuitBreakerHeartbeatPlan {
    pub scenarios: Vec<String>,
    pub heartbeat_interval_seconds: u64,
    pub assertions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeApiKeyQuotaValidationPlan {
    pub rotation_keys: u32,
    pub quota_concurrency: u32,
    pub assertions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeBenchmarkPlan {
    pub output_format: String,
    pub metrics: Vec<String>,
    pub memory_fields: Vec<String>,
    pub sample_json: serde_json::Value,
}

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

fn real_artifact_smoke_tests(cfg: &Config) -> Vec<NativeArtifactSmokePlan> {
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

fn hardware_matrix() -> Vec<NativeHardwareValidationTarget> {
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

fn soak_tests(options: &NativeRuntimeValidationOptions) -> Vec<NativeSoakTestPlan> {
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

pub fn status_from_config(cfg: &Config) -> RuntimeStatus {
    candle_native_status(
        cfg.resources.budget,
        cfg.runtime.embeddings.mode,
        cfg.runtime.embeddings.model_alias.clone(),
    )
}

fn candle_native_status(
    budget_fraction: f64,
    embedding_mode: NativeEmbeddingMode,
    embedding_model_alias: Option<String>,
) -> RuntimeStatus {
    RuntimeStatus {
        backend: RuntimeBackend::CandleNative,
        primary: true,
        implemented: true,
        engine: "candle-native".to_string(),
        execution_model: "in-process-rust-engine".to_string(),
        starter_roles: starter_roles(),
        resource_policy: resource_policy(budget_fraction, "systemd-cpu-ram-and-vram-metadata"),
        token_accounting: "native-tokenizer-exact-when-model-tokenizer-loads".to_string(),
        embeddings: embedding_contract(embedding_mode, embedding_model_alias),
        observability: vec![
            "runtime telemetry event: llmctl.runtime.native.status".to_string(),
            "OpenTelemetry-safe attributes only".to_string(),
            "no prompts, bearer tokens, API keys, or local model paths in status".to_string(),
        ],
        security: vec![
            "single process keeps API key auth, scopes, quotas, audit, and redaction".to_string(),
            "runtime status intentionally reports capability metadata, not secrets".to_string(),
        ],
        compatibility: vec![
            "OpenAI-compatible HTTP contract remains the serving surface".to_string(),
            "Candle-native is the only runtime backend".to_string(),
        ],
        next_step: "wire and verify DeepSeek first, then add reviewed Kimi and MiniMax architecture implementations when Candle-compatible decoders are available"
            .to_string(),
    }
}

fn embedding_contract(
    mode: NativeEmbeddingMode,
    semantic_model_alias: Option<String>,
) -> RuntimeEmbeddingContract {
    RuntimeEmbeddingContract {
        mode,
        semantic_model_alias,
        semantic_backend: "candle-bert-embeddings".to_string(),
        fallback_backend: "deterministic-local-fallback".to_string(),
        fallback_status: "non-semantic-dev-fallback".to_string(),
        fallback_dev_only: true,
    }
}

fn starter_roles() -> Vec<RuntimeStarterRole> {
    ["query", "recommendation", "thinking", "coding"]
        .into_iter()
        .map(|role| RuntimeStarterRole {
            role: role.to_string(),
            default_family: "qwen3".to_string(),
            alternative_families: vec![
                "gemma4".to_string(),
                "kimi".to_string(),
                "mistral".to_string(),
            ],
            eu_friendly_family: Some("mistral".to_string()),
            formats: vec!["gguf".to_string(), "safetensors".to_string()],
            status: if role == "coding" {
                "mvp-target-kimi-blocked-until-candle-architecture-lands".to_string()
            } else {
                "mvp-target".to_string()
            },
        })
        .collect()
}

fn resource_policy(budget_fraction: f64, enforcement: &str) -> RuntimeResourcePolicy {
    RuntimeResourcePolicy {
        budget_fraction,
        cpu_fraction: budget_fraction,
        ram_fraction: budget_fraction,
        gpu_vram_fraction: budget_fraction,
        gpu_detection: vec![
            "nvidia/cuda via nvidia-smi, including small Turing/Tesla cards".to_string(),
            "amd/rocm via sysfs or rocm-smi".to_string(),
            "apple/metal target backend".to_string(),
            "cpu fallback when no supported accelerator is available".to_string(),
        ],
        enforcement: enforcement.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn native_validation_plan_covers_artifacts_hardware_soak_and_benchmarks() {
        let cfg = Config::default();
        let plan = native_validation_plan(
            &cfg,
            NativeRuntimeValidationOptions {
                soak_minutes: 30,
                streaming_concurrency: 5,
                rotation_keys: 2,
                quota_concurrency: 7,
            },
        );

        assert_eq!(plan.status, "planned");
        assert_eq!(plan.mode, "deterministic-offline");
        assert!(!plan.network);
        assert_eq!(plan.real_artifact_smoke_tests.len(), 4);
        assert!(plan
            .real_artifact_smoke_tests
            .iter()
            .all(|test| test.required_format == NativeModelFormat::Safetensors));
        assert!(plan
            .real_artifact_smoke_tests
            .iter()
            .all(|test| test.artifact_validation.status == "planned-missing-local-artifact"));
        assert!(plan
            .hardware_matrix
            .iter()
            .any(|target| target.target == "cpu"));
        assert!(plan
            .hardware_matrix
            .iter()
            .any(|target| target.target == "nvidia-cuda"));
        assert!(plan
            .hardware_matrix
            .iter()
            .any(|target| target.target == "amd-vulkan"));
        assert!(plan
            .hardware_matrix
            .iter()
            .any(|target| target.target == "apple-metal"));
        assert_eq!(plan.soak_tests[0].duration_minutes, 30);
        assert_eq!(plan.soak_tests[0].concurrency, 5);
        assert_eq!(plan.api_key_rotation_and_quota.rotation_keys, 2);
        assert_eq!(plan.api_key_rotation_and_quota.quota_concurrency, 7);
        assert!(plan
            .benchmark
            .metrics
            .contains(&"tokens_per_second".to_string()));
        assert!(plan
            .benchmark
            .memory_fields
            .contains(&"peak_vram_bytes".to_string()));
    }

    #[test]
    fn candle_native_status_declares_starter_roles_and_default_budget() {
        let cfg = Config::default();
        let status = status_from_config(&cfg);

        assert_eq!(status.backend, RuntimeBackend::CandleNative);
        assert!(status.primary);
        assert!(status.implemented);
        assert_eq!(status.resource_policy.budget_fraction, 0.80);
        assert_eq!(status.resource_policy.cpu_fraction, 0.80);
        assert_eq!(status.resource_policy.ram_fraction, 0.80);
        assert_eq!(status.resource_policy.gpu_vram_fraction, 0.80);
        assert_eq!(status.embeddings.mode, NativeEmbeddingMode::Semantic);
        assert_eq!(status.embeddings.semantic_model_alias, None);
        assert_eq!(status.embeddings.semantic_backend, "candle-bert-embeddings");
        assert_eq!(
            status.embeddings.fallback_backend,
            "deterministic-local-fallback"
        );
        assert_eq!(
            status.embeddings.fallback_status,
            "non-semantic-dev-fallback"
        );
        assert!(status.embeddings.fallback_dev_only);
        assert_eq!(
            status
                .starter_roles
                .iter()
                .map(|role| role.role.as_str())
                .collect::<Vec<_>>(),
            vec!["query", "recommendation", "thinking", "coding"]
        );
        assert!(status
            .starter_roles
            .iter()
            .all(|role| role.default_family == "qwen3"));
        assert!(status
            .starter_roles
            .iter()
            .all(|role| role.alternative_families == ["gemma4", "kimi", "mistral"]));
        assert!(status
            .starter_roles
            .iter()
            .all(|role| role.eu_friendly_family.as_deref() == Some("mistral")));
        assert!(status
            .resource_policy
            .gpu_detection
            .iter()
            .any(|entry| entry.contains("nvidia")));
        assert!(status
            .resource_policy
            .gpu_detection
            .iter()
            .any(|entry| entry.contains("amd")));
        assert!(status
            .resource_policy
            .gpu_detection
            .iter()
            .any(|entry| entry.contains("apple")));
    }
}
