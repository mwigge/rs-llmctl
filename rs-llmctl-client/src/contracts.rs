use crate::{LlmctlError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    #[default]
    LocalLlmctl,
    OpenAiCompatible,
    VertexAi,
    OpenRouter,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderRouting {
    LocalOnly,
    ExternalReserved,
    ExternalOpenAiCompatible,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderStatus {
    Implemented,
    ContractOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderContract {
    pub kind: ProviderKind,
    pub routing: ProviderRouting,
    pub status: ProviderStatus,
    pub local_first: bool,
    pub routes_external_provider_traffic: bool,
    pub base_url_env: Vec<String>,
    pub api_key_env: Vec<String>,
}

impl ProviderContract {
    pub fn local_llmctl() -> Self {
        Self {
            kind: ProviderKind::LocalLlmctl,
            routing: ProviderRouting::LocalOnly,
            status: ProviderStatus::Implemented,
            local_first: true,
            routes_external_provider_traffic: false,
            base_url_env: vec![
                "LLMCTL_BASE_URL".to_string(),
                "RS_LLMCTL_BASE_URL".to_string(),
            ],
            api_key_env: vec![
                "LLMCTL_API_KEY".to_string(),
                "RS_LLMCTL_API_KEY".to_string(),
            ],
        }
    }

    pub fn reserved(kind: ProviderKind) -> Self {
        Self {
            kind,
            routing: ProviderRouting::ExternalReserved,
            status: ProviderStatus::ContractOnly,
            local_first: true,
            routes_external_provider_traffic: false,
            base_url_env: Vec::new(),
            api_key_env: Vec::new(),
        }
    }

    pub fn for_kind(kind: ProviderKind) -> Self {
        match kind {
            ProviderKind::LocalLlmctl => Self::local_llmctl(),
            provider => Self::reserved(provider),
        }
    }

    pub fn validate_routable(&self) -> Result<()> {
        if self.status != ProviderStatus::Implemented {
            return Err(LlmctlError::BadRequest {
                message: format!(
                    "provider {:?} is contract-only metadata and cannot route traffic",
                    self.kind
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueDiscipline {
    Fifo,
    WeightedFair,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueContract {
    pub discipline: QueueDiscipline,
    pub admission_backpressure: bool,
    pub priority_metadata_keys: Vec<String>,
    pub implemented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchingContract {
    pub continuous_batching: bool,
    pub max_batch_size_metadata_key: String,
    pub max_wait_ms_metadata_key: String,
    pub implemented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheContract {
    pub cache_scope: String,
    pub cache_budget_metadata_key: String,
    pub eviction_policy: String,
    pub implemented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancellationContract {
    pub cancellation_token_metadata_key: String,
    pub drain_on_cancel: bool,
    pub implemented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerContract {
    pub queue: QueueContract,
    pub batching: BatchingContract,
    pub kv_cache: KvCacheContract,
    pub cancellation: CancellationContract,
    pub contract_only: bool,
}

impl SchedulerContract {
    pub fn fifo_runtime() -> Self {
        Self {
            queue: QueueContract {
                discipline: QueueDiscipline::Fifo,
                admission_backpressure: true,
                priority_metadata_keys: vec![
                    "llmctl.scheduler.priority".to_string(),
                    "llmctl.scheduler.tenant".to_string(),
                ],
                implemented: true,
            },
            batching: BatchingContract {
                continuous_batching: false,
                max_batch_size_metadata_key: "llmctl.scheduler.max_batch_size".to_string(),
                max_wait_ms_metadata_key: "llmctl.scheduler.max_wait_ms".to_string(),
                implemented: false,
            },
            kv_cache: KvCacheContract {
                cache_scope: "model-worker".to_string(),
                cache_budget_metadata_key: "llmctl.scheduler.kv_cache_budget_bytes".to_string(),
                eviction_policy: "metadata-only-lru-target".to_string(),
                implemented: false,
            },
            cancellation: CancellationContract {
                cancellation_token_metadata_key: "llmctl.scheduler.cancel_token".to_string(),
                drain_on_cancel: true,
                implemented: false,
            },
            contract_only: false,
        }
    }

    pub fn metadata_only() -> Self {
        Self {
            queue: QueueContract {
                discipline: QueueDiscipline::Fifo,
                admission_backpressure: true,
                priority_metadata_keys: vec![
                    "llmctl.scheduler.priority".to_string(),
                    "llmctl.scheduler.tenant".to_string(),
                ],
                implemented: false,
            },
            batching: BatchingContract {
                continuous_batching: true,
                max_batch_size_metadata_key: "llmctl.scheduler.max_batch_size".to_string(),
                max_wait_ms_metadata_key: "llmctl.scheduler.max_wait_ms".to_string(),
                implemented: false,
            },
            kv_cache: KvCacheContract {
                cache_scope: "model-worker".to_string(),
                cache_budget_metadata_key: "llmctl.scheduler.kv_cache_budget_bytes".to_string(),
                eviction_policy: "metadata-only-lru-target".to_string(),
                implemented: false,
            },
            cancellation: CancellationContract {
                cancellation_token_metadata_key: "llmctl.scheduler.cancel_token".to_string(),
                drain_on_cancel: true,
                implemented: false,
            },
            contract_only: true,
        }
    }

    pub fn validate_runtime_contract(&self) -> Result<()> {
        if self.contract_only {
            return Err(LlmctlError::BadRequest {
                message: "scheduler contract must report implemented FIFO queue runtime"
                    .to_string(),
            });
        }
        if self.queue.discipline != QueueDiscipline::Fifo
            || !self.queue.implemented
            || self.batching.implemented
            || self.kv_cache.implemented
            || self.cancellation.implemented
        {
            return Err(LlmctlError::BadRequest {
                message: "scheduler must implement FIFO queue while batching, KV cache, and cancellation remain metadata-only"
                    .to_string(),
            });
        }
        Ok(())
    }

    pub fn validate_contract_only(&self) -> Result<()> {
        if !self.contract_only {
            return Err(LlmctlError::BadRequest {
                message: "scheduler contract is not metadata-only".to_string(),
            });
        }
        if self.queue.implemented
            || self.batching.implemented
            || self.kv_cache.implemented
            || self.cancellation.implemented
        {
            return Err(LlmctlError::BadRequest {
                message: "scheduler queue, batching, KV cache, and cancellation are contract-only"
                    .to_string(),
            });
        }
        Ok(())
    }
}
