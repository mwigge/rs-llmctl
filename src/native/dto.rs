//! Request/response DTOs and scheduler contract data types for the native runtime.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeChatRequest {
    pub model: String,
    pub messages: Vec<NativeChatMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TokenAccountingMode {
    /// Counts came from a model-compatible native tokenizer.
    NativeExact,
    /// Counts came from the deterministic fallback estimator and are not exact model tokens.
    Estimated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeTokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub accounting_mode: TokenAccountingMode,
}

impl NativeTokenUsage {
    pub const fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            accounting_mode: TokenAccountingMode::Estimated,
        }
    }

    pub const fn with_mode(
        input_tokens: u64,
        output_tokens: u64,
        accounting_mode: TokenAccountingMode,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            accounting_mode,
        }
    }

    pub const fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeChatResponse {
    pub model: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    pub finish_reason: String,
    pub usage: NativeTokenUsage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeEmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeEmbeddingResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub usage: NativeTokenUsage,
    pub backend: String,
    pub status: String,
    pub semantic: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeQueueDiscipline {
    Fifo,
    WeightedFair,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeQueueContract {
    pub discipline: NativeQueueDiscipline,
    pub admission_backpressure: bool,
    pub priority_metadata_keys: Vec<String>,
    pub implemented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeBatchingContract {
    pub continuous_batching: bool,
    pub prefill_decode_phase_scheduling: bool,
    pub max_batch_size_metadata_key: String,
    pub max_wait_ms_metadata_key: String,
    pub unsupported_reason: String,
    pub implemented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeKvCacheContract {
    pub cache_scope: String,
    pub cache_budget_metadata_key: String,
    pub cache_key_metadata_key: String,
    pub eviction_policy: String,
    pub reuse_implemented: bool,
    pub unsupported_reason: String,
    pub implemented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCancellationContract {
    pub cancellation_token_metadata_key: String,
    pub cancelled_metadata_key: String,
    pub drain_on_cancel: bool,
    pub admission_check_implemented: bool,
    pub decode_loop_check_implemented: bool,
    pub unsupported_reason: String,
    pub implemented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeSchedulerContract {
    pub queue: NativeQueueContract,
    pub batching: NativeBatchingContract,
    pub kv_cache: NativeKvCacheContract,
    pub cancellation: NativeCancellationContract,
    pub contract_only: bool,
}

impl NativeSchedulerContract {
    pub fn fifo_runtime() -> Self {
        Self {
            queue: NativeQueueContract {
                discipline: NativeQueueDiscipline::Fifo,
                admission_backpressure: true,
                priority_metadata_keys: vec![
                    "llmctl.scheduler.priority".to_string(),
                    "llmctl.scheduler.tenant".to_string(),
                ],
                implemented: true,
            },
            batching: NativeBatchingContract {
                continuous_batching: false,
                prefill_decode_phase_scheduling: true,
                max_batch_size_metadata_key: "llmctl.scheduler.max_batch_size".to_string(),
                max_wait_ms_metadata_key: "llmctl.scheduler.max_wait_ms".to_string(),
                unsupported_reason:
                    "continuous batching is not active until native decode can interleave batch members"
                        .to_string(),
                implemented: false,
            },
            kv_cache: NativeKvCacheContract {
                cache_scope: "model-worker".to_string(),
                cache_budget_metadata_key: "llmctl.scheduler.kv_cache_budget_bytes".to_string(),
                cache_key_metadata_key: "llmctl.scheduler.kv_cache_key".to_string(),
                eviction_policy: "request-local-reset".to_string(),
                reuse_implemented: false,
                unsupported_reason:
                    "cross-request KV-cache reuse is disabled until cache ownership and invalidation are verified"
                        .to_string(),
                implemented: false,
            },
            cancellation: NativeCancellationContract {
                cancellation_token_metadata_key: "llmctl.scheduler.cancel_token".to_string(),
                cancelled_metadata_key: "llmctl.scheduler.cancelled".to_string(),
                drain_on_cancel: true,
                admission_check_implemented: true,
                decode_loop_check_implemented: false,
                unsupported_reason:
                    "HTTP disconnect and token-level decode cancellation are not yet wired through Candle generation"
                        .to_string(),
                implemented: false,
            },
            contract_only: false,
        }
    }

    pub fn planned_metadata_only() -> Self {
        Self::fifo_runtime()
    }
}
