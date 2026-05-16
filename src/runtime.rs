use crate::config::Config;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeBackend {
    #[default]
    CandleNative,
    LlamaServer,
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
    pub observability: Vec<String>,
    pub security: Vec<String>,
    pub compatibility: Vec<String>,
    pub next_step: String,
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

pub fn status_from_config(cfg: &Config) -> RuntimeStatus {
    match cfg.runtime.backend {
        RuntimeBackend::CandleNative => candle_native_status(cfg.resources.budget),
        RuntimeBackend::LlamaServer => llama_server_status(cfg.resources.budget),
    }
}

fn candle_native_status(budget_fraction: f64) -> RuntimeStatus {
    RuntimeStatus {
        backend: RuntimeBackend::CandleNative,
        primary: true,
        implemented: true,
        engine: "candle-native".to_string(),
        execution_model: "in-process-rust-engine".to_string(),
        starter_roles: starter_roles(),
        resource_policy: resource_policy(budget_fraction, "systemd-cpu-ram-and-vram-metadata"),
        token_accounting: "native-tokenizer-exact-when-model-tokenizer-loads".to_string(),
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
            "llama-server stays available only as compatibility backend".to_string(),
        ],
        next_step:
            "add a Kimi architecture implementation or upgrade Candle when Kimi lands upstream"
                .to_string(),
    }
}

fn llama_server_status(budget_fraction: f64) -> RuntimeStatus {
    RuntimeStatus {
        backend: RuntimeBackend::LlamaServer,
        primary: false,
        implemented: true,
        engine: "llama-server-compatibility".to_string(),
        execution_model: "managed-external-process".to_string(),
        starter_roles: starter_roles(),
        resource_policy: resource_policy(budget_fraction, "planned-and-reported"),
        token_accounting: "upstream-usage-metadata".to_string(),
        observability: vec![
            "runtime telemetry event: llmctl.runtime.native.status".to_string(),
            "proxy routing, quota, worker lifecycle, and usage events".to_string(),
        ],
        security: vec![
            "external worker process receives scrubbed environment/status output".to_string(),
            "API key auth, scopes, audit, and telemetry redaction remain enforced".to_string(),
        ],
        compatibility: vec![
            "compatibility backend for existing llama.cpp deployments".to_string(),
            "not the target backend for the MVP native runtime".to_string(),
        ],
        next_step: "migrate models to candle-native as role coverage lands".to_string(),
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
