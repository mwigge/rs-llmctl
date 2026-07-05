//! Runtime status derivation from configuration.
use super::*;

pub fn status_from_config(cfg: &Config) -> RuntimeStatus {
    let mut status = candle_native_status(
        cfg.resources.budget,
        cfg.runtime.embeddings.mode,
        cfg.runtime.embeddings.model_alias.clone(),
    );
    status.model_readiness = cfg
        .models
        .iter()
        .filter(|model| {
            model
                .family
                .as_deref()
                .is_some_and(|family| family.eq_ignore_ascii_case("gemma4"))
        })
        .map(|model| {
            let evidence_path = readiness::evidence_path(&cfg.storage.model_dir, &model.alias);
            ModelReadinessStatus {
                alias: model.alias.clone(),
                family: "gemma4".to_string(),
                state: readiness::read_state(&evidence_path),
                evidence_present: evidence_path.exists(),
            }
        })
        .collect();
    status
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
        model_readiness: Vec::new(),
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

pub(crate) fn starter_roles() -> Vec<RuntimeStarterRole> {
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

pub(crate) fn resource_policy(budget_fraction: f64, enforcement: &str) -> RuntimeResourcePolicy {
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
