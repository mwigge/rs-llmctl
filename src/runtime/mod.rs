use crate::config::{Config, ModelConfig, NativeEmbeddingMode};
use crate::native::{
    self, CandleArtifactLayout, CandleModelFamily, NativeAcceleration, NativeModelFormat,
};
use crate::readiness::{self, ReadinessState};
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
    pub model_readiness: Vec<ModelReadinessStatus>,
    pub next_step: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelReadinessStatus {
    pub alias: String,
    pub family: String,
    pub state: ReadinessState,
    pub evidence_present: bool,
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

mod validation_plan;
pub use validation_plan::*;
mod status;
pub use status::*;

#[cfg(test)]
mod tests;
