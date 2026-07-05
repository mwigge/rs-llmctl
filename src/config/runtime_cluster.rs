//! Runtime, scheduler, embedding, and cluster configuration.
use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct RuntimeConfig {
    pub backend: RuntimeBackend,
    pub heartbeat_interval_seconds: u64,
    pub embeddings: NativeEmbeddingRuntimeConfig,
    pub scheduler: NativeSchedulerRuntimeConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            backend: RuntimeBackend::CandleNative,
            heartbeat_interval_seconds: 30,
            embeddings: NativeEmbeddingRuntimeConfig::default(),
            scheduler: NativeSchedulerRuntimeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct NativeSchedulerRuntimeConfig {
    pub max_concurrent_requests: usize,
    pub max_queued_requests: usize,
    pub max_batch_size: usize,
    pub max_batch_wait_ms: u64,
    pub kv_cache_budget_bytes: u64,
}

impl Default for NativeSchedulerRuntimeConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 1,
            max_queued_requests: 127,
            max_batch_size: 1,
            max_batch_wait_ms: 0,
            kv_cache_budget_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct NativeEmbeddingRuntimeConfig {
    pub mode: NativeEmbeddingMode,
    #[serde(alias = "model_alias")]
    pub model_alias: Option<String>,
}

impl Default for NativeEmbeddingRuntimeConfig {
    fn default() -> Self {
        Self {
            mode: NativeEmbeddingMode::Semantic,
            model_alias: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum NativeEmbeddingMode {
    #[default]
    Semantic,
    DevFallback,
}

impl NativeEmbeddingMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::DevFallback => "dev-fallback",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct ClusterConfig {
    pub node_id: String,
    pub nodes: Vec<ClusterNodeConfig>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            node_id: "local".to_string(),
            nodes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct ClusterNodeConfig {
    pub id: String,
    pub base_url: String,
    pub roles: Vec<String>,
    pub model_aliases: Vec<String>,
}

impl Default for ClusterNodeConfig {
    fn default() -> Self {
        Self {
            id: "local".to_string(),
            base_url: "http://127.0.0.1:8765/v1".to_string(),
            roles: Vec::new(),
            model_aliases: Vec::new(),
        }
    }
}
