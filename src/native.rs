use crate::config::{ClusterNodeConfig, Config, ModelConfig, ResourceConfig};
use crate::resources::GpuVendor;
use crate::runtime::RuntimeBackend;
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
use anyhow::Context;
use anyhow::{bail, Result};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
use std::sync::Mutex;
use std::time::Instant;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

const STARTER_ROLES: &[&str] = &["query", "recommendation", "thinking", "coding"];

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

pub trait NativeEngine: Send + Sync {
    fn model_alias(&self) -> &str;
    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>>;

    fn chat_stream(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        self.chat(request)
    }

    fn embeddings(
        &self,
        request: NativeEmbeddingRequest,
    ) -> BoxFuture<'_, Result<NativeEmbeddingResponse>> {
        Box::pin(async move {
            bail!(
                "native engine for model '{}' does not implement semantic embeddings",
                request.model
            )
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeSchedulerConfig {
    pub max_concurrent_requests: usize,
    pub max_queued_requests: usize,
    pub max_batch_size: usize,
    pub max_batch_wait_ms: u64,
    pub kv_cache_budget_bytes: u64,
}

impl Default for NativeSchedulerConfig {
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

#[derive(Clone)]
pub struct NativeSchedulerEngine {
    inner: Arc<dyn NativeEngine>,
    config: NativeSchedulerConfig,
    permits: Arc<Semaphore>,
    waiting: Arc<AtomicUsize>,
}

impl std::fmt::Debug for NativeSchedulerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeSchedulerEngine")
            .field("model_alias", &self.model_alias())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl NativeSchedulerEngine {
    pub fn new(inner: Arc<dyn NativeEngine>, config: NativeSchedulerConfig) -> Self {
        let config = NativeSchedulerConfig {
            max_concurrent_requests: config.max_concurrent_requests.max(1),
            max_queued_requests: config.max_queued_requests,
            max_batch_size: config.max_batch_size.max(1),
            max_batch_wait_ms: config.max_batch_wait_ms,
            kv_cache_budget_bytes: config.kv_cache_budget_bytes,
        };
        Self {
            inner,
            permits: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            waiting: Arc::new(AtomicUsize::new(0)),
            config,
        }
    }
}

impl NativeEngine for NativeSchedulerEngine {
    fn model_alias(&self) -> &str {
        self.inner.model_alias()
    }

    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        scheduled_native_chat(
            self.inner.clone(),
            self.permits.clone(),
            self.waiting.clone(),
            self.config,
            request,
            NativeScheduledOperation::Chat,
        )
    }

    fn chat_stream(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        scheduled_native_chat(
            self.inner.clone(),
            self.permits.clone(),
            self.waiting.clone(),
            self.config,
            request,
            NativeScheduledOperation::Stream,
        )
    }

    fn embeddings(
        &self,
        request: NativeEmbeddingRequest,
    ) -> BoxFuture<'_, Result<NativeEmbeddingResponse>> {
        self.inner.embeddings(request)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeScheduledOperation {
    Chat,
    Stream,
}

struct NativeSchedulerWaitGuard {
    waiting: Arc<AtomicUsize>,
    queued_before: usize,
}

impl NativeSchedulerWaitGuard {
    fn enter(waiting: Arc<AtomicUsize>, max_queued_requests: usize) -> Result<Self> {
        let queued_before = waiting.fetch_add(1, Ordering::AcqRel);
        if queued_before >= max_queued_requests {
            waiting.fetch_sub(1, Ordering::AcqRel);
            bail!("native scheduler queue is full");
        }
        Ok(Self {
            waiting,
            queued_before,
        })
    }
}

impl Drop for NativeSchedulerWaitGuard {
    fn drop(&mut self) {
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }
}

fn scheduled_native_chat(
    inner: Arc<dyn NativeEngine>,
    permits: Arc<Semaphore>,
    waiting: Arc<AtomicUsize>,
    config: NativeSchedulerConfig,
    mut request: NativeChatRequest,
    operation: NativeScheduledOperation,
) -> BoxFuture<'static, Result<NativeChatResponse>> {
    Box::pin(async move {
        reject_cancelled_request(&request.metadata)?;
        let queued_at = Instant::now();
        let mut queued_before_admit = 0usize;
        let permit = match permits.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => {
                let wait_guard =
                    NativeSchedulerWaitGuard::enter(waiting, config.max_queued_requests)?;
                queued_before_admit = wait_guard.queued_before;
                let permit = permits
                    .acquire_owned()
                    .await
                    .map_err(|_| anyhow::anyhow!("native scheduler is closed"))?;
                drop(wait_guard);
                permit
            }
            Err(TryAcquireError::Closed) => bail!("native scheduler is closed"),
        };
        reject_cancelled_request(&request.metadata)?;
        stamp_scheduler_metadata(
            &mut request.metadata,
            queued_at,
            queued_before_admit,
            config,
        );
        run_scheduled_native_chat(inner, request, operation, permit).await
    })
}

fn reject_cancelled_request(metadata: &BTreeMap<String, Value>) -> Result<()> {
    if metadata
        .get("llmctl.scheduler.cancelled")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        bail!("native scheduler request was cancelled before decode");
    }
    Ok(())
}

async fn run_scheduled_native_chat(
    inner: Arc<dyn NativeEngine>,
    request: NativeChatRequest,
    operation: NativeScheduledOperation,
    _permit: OwnedSemaphorePermit,
) -> Result<NativeChatResponse> {
    match operation {
        NativeScheduledOperation::Chat => inner.chat(request).await,
        NativeScheduledOperation::Stream => inner.chat_stream(request).await,
    }
}

fn stamp_scheduler_metadata(
    metadata: &mut BTreeMap<String, Value>,
    queued_at: Instant,
    queued_before_admit: usize,
    config: NativeSchedulerConfig,
) {
    let wait_ms = queued_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    metadata.insert(
        "llmctl.scheduler.discipline".to_string(),
        Value::String("fifo".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.queue.implemented".to_string(),
        Value::Bool(true),
    );
    metadata.insert(
        "llmctl.scheduler.queue_wait_ms".to_string(),
        Value::from(wait_ms),
    );
    metadata.insert(
        "llmctl.scheduler.admission_wait_ms".to_string(),
        Value::from(wait_ms),
    );
    metadata.insert(
        "llmctl.scheduler.queued_requests_before_admit".to_string(),
        Value::from(queued_before_admit as u64),
    );
    metadata.insert(
        "llmctl.scheduler.max_concurrent_requests".to_string(),
        Value::from(config.max_concurrent_requests as u64),
    );
    metadata.insert(
        "llmctl.scheduler.max_queued_requests".to_string(),
        Value::from(config.max_queued_requests as u64),
    );
    metadata.insert(
        "llmctl.scheduler.batching.continuous.implemented".to_string(),
        Value::Bool(false),
    );
    metadata.insert(
        "llmctl.scheduler.batching.phase_scheduling.implemented".to_string(),
        Value::Bool(true),
    );
    metadata.insert(
        "llmctl.scheduler.phase".to_string(),
        Value::String("prefill-then-decode".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.prefill.phase".to_string(),
        Value::String("scheduled".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.decode.phase".to_string(),
        Value::String("scheduled".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.max_batch_size".to_string(),
        Value::from(config.max_batch_size as u64),
    );
    metadata.insert(
        "llmctl.scheduler.max_wait_ms".to_string(),
        Value::from(config.max_batch_wait_ms),
    );
    metadata.insert(
        "llmctl.scheduler.kv_cache_budget_bytes".to_string(),
        Value::from(config.kv_cache_budget_bytes),
    );
    metadata.insert(
        "llmctl.scheduler.kv_cache.reuse_implemented".to_string(),
        Value::Bool(false),
    );
    metadata.insert(
        "llmctl.scheduler.kv_cache.policy".to_string(),
        Value::String("request-local-reset".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.cancellation.admission_check_implemented".to_string(),
        Value::Bool(true),
    );
    metadata.insert(
        "llmctl.scheduler.cancellation.decode_loop_check_implemented".to_string(),
        Value::Bool(false),
    );
}

pub trait NativeTokenCounter: Send + Sync {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::NativeExact
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64>;
    fn count_text(&self, text: &str) -> Result<u64>;
}

pub trait NativeTokenAccountingAdapter: NativeTokenCounter {}

impl<T> NativeTokenAccountingAdapter for T where T: NativeTokenCounter + ?Sized {}

pub fn canonical_native_chat_input(messages: &[NativeChatMessage]) -> String {
    let mut input = String::new();
    for message in messages {
        input.push_str("<|");
        input.push_str(&message.role);
        input.push_str("|>\n");
        input.push_str(&message_content_text(message));
        if message.tool_calls.is_some() {
            input.push_str("\n<|assistant_tool_calls|>");
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            input.push_str("\n<|tool_call_id|>");
            input.push_str(tool_call_id);
        }
        input.push('\n');
    }
    input
}

/// Renders `messages` using Gemma 4's chat template turn format —
/// `<|turn>{role}\n{content}<turn|>\n` — as embedded in this model's GGUF
/// `tokenizer.chat_template`. The older Gemma 2/3
/// `<start_of_turn>{role}\n{content}<end_of_turn>\n` format is NOT used here:
/// those tokens are absent from this model's vocabulary and would otherwise
/// be split into garbage sub-tokens, corrupting the prompt.
///
/// `assistant` maps to the `model` role; `system` is passed through
/// unchanged; all other roles (`user`, `tool`, etc.) map to `user`. The
/// rendered prompt always ends with the generation cue
/// `<|turn>model\n<|channel>thought\n<channel|>` (no closing `<turn|>`).
#[must_use]
pub fn gemma_chat_input(messages: &[NativeChatMessage]) -> String {
    let mut input = String::new();

    for message in messages {
        let role = match message.role.as_str() {
            "assistant" => "model",
            "system" => "system",
            _ => "user",
        };

        input.push_str("<|turn>");
        input.push_str(role);
        input.push('\n');
        input.push_str(message_content_text(message).trim());
        input.push_str("<turn|>\n");
    }

    input.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    input
}

pub fn message_content_text(message: &NativeChatMessage) -> String {
    match &message.content {
        Some(Value::String(content)) => content.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EstimatedNativeTokenCounter;

impl EstimatedNativeTokenCounter {
    const CHARS_PER_TOKEN: u64 = 4;
    const MESSAGE_OVERHEAD_TOKENS: u64 = 4;

    fn estimate_text_tokens(text: &str) -> u64 {
        let normalized_chars = text.chars().filter(|ch| !ch.is_control()).count() as u64;
        if normalized_chars == 0 {
            return 0;
        }
        normalized_chars
            .saturating_add(Self::CHARS_PER_TOKEN - 1)
            .saturating_div(Self::CHARS_PER_TOKEN)
            .max(1)
    }
}

impl NativeTokenCounter for EstimatedNativeTokenCounter {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::Estimated
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64> {
        Ok(messages
            .iter()
            .map(|message| {
                Self::MESSAGE_OVERHEAD_TOKENS
                    .saturating_add(Self::estimate_text_tokens(&message.role))
                    .saturating_add(Self::estimate_text_tokens(&message_content_text(message)))
                    .saturating_add(if message.tool_calls.is_some() { 1 } else { 0 })
                    .saturating_add(
                        message
                            .tool_call_id
                            .as_deref()
                            .map(Self::estimate_text_tokens)
                            .unwrap_or(0),
                    )
            })
            .sum())
    }

    fn count_text(&self, text: &str) -> Result<u64> {
        Ok(Self::estimate_text_tokens(text))
    }
}

#[cfg(feature = "native-tokenizers")]
#[derive(Debug, Clone)]
pub struct TokenizersNativeTokenCounter {
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "native-tokenizers")]
impl TokenizersNativeTokenCounter {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|err| anyhow::anyhow!("failed to load tokenizer json: {err}"))?;
        Ok(Self::from_tokenizer(tokenizer))
    }

    pub const fn from_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    fn count_serialized_input(&self, input: &str) -> Result<u64> {
        let encoding = self
            .tokenizer
            .encode(input, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize native input: {err}"))?;
        Ok(encoding.len() as u64)
    }
}

#[cfg(feature = "native-tokenizers")]
impl NativeTokenCounter for TokenizersNativeTokenCounter {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::NativeExact
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64> {
        self.count_serialized_input(&canonical_native_chat_input(messages))
    }

    fn count_text(&self, text: &str) -> Result<u64> {
        self.count_serialized_input(text)
    }
}

pub fn usage_from_native_tokens(
    counter: &dyn NativeTokenAccountingAdapter,
    request: &NativeChatRequest,
    response_text: &str,
) -> Result<NativeTokenUsage> {
    Ok(NativeTokenUsage::with_mode(
        counter.count_chat_input(&request.messages)?,
        counter.count_text(response_text)?,
        counter.accounting_mode(),
    ))
}

pub const DETERMINISTIC_EMBEDDING_DIMENSIONS: usize = 64;

#[derive(Debug)]
pub struct NativeBertEmbeddingEngine {
    alias: String,
    encoder: NativeBertEmbeddingEncoder,
}

impl NativeBertEmbeddingEngine {
    pub fn load(alias: impl Into<String>, model_path: impl AsRef<Path>) -> Result<Self> {
        let alias = alias.into();
        let encoder = NativeBertEmbeddingEncoder::load(model_path.as_ref())?;
        Ok(Self { alias, encoder })
    }
}

impl NativeEngine for NativeBertEmbeddingEngine {
    fn model_alias(&self) -> &str {
        &self.alias
    }

    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        Box::pin(async move {
            bail!(
                "native BERT embedding model '{}' does not serve chat completions",
                request.model
            )
        })
    }

    fn embeddings(
        &self,
        request: NativeEmbeddingRequest,
    ) -> BoxFuture<'_, Result<NativeEmbeddingResponse>> {
        Box::pin(async move {
            let (embeddings, input_tokens) = self.encoder.embed(&request.input)?;
            Ok(NativeEmbeddingResponse {
                model: request.model,
                embeddings,
                usage: NativeTokenUsage::with_mode(
                    input_tokens,
                    0,
                    TokenAccountingMode::NativeExact,
                ),
                backend: "candle-bert-embeddings".to_string(),
                status: "semantic-native".to_string(),
                semantic: true,
            })
        })
    }
}

#[derive(Debug)]
enum NativeBertEmbeddingEncoder {
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    Real(RealBertEmbeddingEncoder),
    #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
    Unavailable,
}

impl NativeBertEmbeddingEncoder {
    fn load(model_path: &Path) -> Result<Self> {
        load_real_bert_embedding_encoder(model_path)
    }

    fn embed(&self, input: &[String]) -> Result<(Vec<Vec<f32>>, u64)> {
        match self {
            #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
            Self::Real(encoder) => encoder.embed(input),
            #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
            Self::Unavailable => {
                let _ = input;
                bail!(
                    "semantic native embeddings require the native-candle and native-tokenizers features"
                )
            }
        }
    }
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
fn load_real_bert_embedding_encoder(_model_path: &Path) -> Result<NativeBertEmbeddingEncoder> {
    Ok(NativeBertEmbeddingEncoder::Unavailable)
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
struct RealBertEmbeddingEncoder {
    tokenizer: tokenizers::tokenizer::Tokenizer,
    model: Mutex<candle_transformers::models::bert::BertModel>,
    pad_token_id: u32,
    max_position_embeddings: usize,
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl std::fmt::Debug for RealBertEmbeddingEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealBertEmbeddingEncoder")
            .field("pad_token_id", &self.pad_token_id)
            .field("max_position_embeddings", &self.max_position_embeddings)
            .finish_non_exhaustive()
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn load_real_bert_embedding_encoder(model_path: &Path) -> Result<NativeBertEmbeddingEncoder> {
    let artifact_dir = safetensors_artifact_dir(model_path);
    let model = ModelConfig {
        alias: "semantic-embedding".to_string(),
        path: model_path.to_path_buf(),
        role: "embedding".to_string(),
        family: Some("qwen3".to_string()),
        weight: 1,
    };
    let artifacts = validate_semantic_embedding_artifacts(&model)?;
    let paths = artifacts
        .weight_files
        .iter()
        .map(|name| artifact_dir.join(name))
        .collect::<Vec<_>>();
    let config_path = artifact_dir.join("config.json");
    let tokenizer_path = artifact_dir.join("tokenizer.json");
    let config: candle_transformers::models::bert::Config = read_json_config(&config_path)?;
    let tokenizer = tokenizers::tokenizer::Tokenizer::from_file(&tokenizer_path)
        .map_err(|err| anyhow::anyhow!("failed to load tokenizer.json: {err}"))?;
    let device = candle_core::Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::F32, &device)
    }
    .with_context(|| "failed to mmap BERT embedding safetensors with Candle")?;
    let bert = candle_transformers::models::bert::BertModel::load(vb, &config)
        .with_context(|| "failed to construct BERT embedding model")?;
    Ok(NativeBertEmbeddingEncoder::Real(RealBertEmbeddingEncoder {
        tokenizer,
        model: Mutex::new(bert),
        pad_token_id: config.pad_token_id as u32,
        max_position_embeddings: config.max_position_embeddings,
    }))
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl RealBertEmbeddingEncoder {
    fn embed(&self, input: &[String]) -> Result<(Vec<Vec<f32>>, u64)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let mut encoded = Vec::with_capacity(input.len());
        let mut max_len = 0usize;
        let mut input_tokens = 0u64;
        for text in input {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|err| anyhow::anyhow!("failed to tokenize embedding input: {err}"))?;
            let mut ids = encoding.get_ids().to_vec();
            if ids.is_empty() {
                ids.push(self.pad_token_id);
            }
            ids.truncate(self.max_position_embeddings.max(1));
            input_tokens = input_tokens.saturating_add(ids.len() as u64);
            max_len = max_len.max(ids.len());
            encoded.push(ids);
        }

        let mut input_ids = Vec::with_capacity(encoded.len());
        let mut token_type_ids = Vec::with_capacity(encoded.len());
        let mut attention_mask = Vec::with_capacity(encoded.len());
        for ids in &encoded {
            let mut padded = ids.clone();
            let mut mask = vec![1u32; ids.len()];
            padded.resize(max_len, self.pad_token_id);
            mask.resize(max_len, 0);
            input_ids.push(padded);
            token_type_ids.push(vec![0u32; max_len]);
            attention_mask.push(mask);
        }

        let device = candle_core::Device::Cpu;
        let input_ids = candle_core::Tensor::new(input_ids, &device)
            .with_context(|| "failed to build BERT input_ids tensor")?;
        let token_type_ids = candle_core::Tensor::new(token_type_ids, &device)
            .with_context(|| "failed to build BERT token_type_ids tensor")?;
        let attention = candle_core::Tensor::new(attention_mask.clone(), &device)
            .with_context(|| "failed to build BERT attention_mask tensor")?;
        let model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("native BERT embedding model lock is poisoned"))?;
        let sequence = model
            .forward(&input_ids, &token_type_ids, Some(&attention))
            .with_context(|| "native BERT embedding forward pass failed")?;
        let hidden = sequence
            .to_vec3::<f32>()
            .with_context(|| "failed to read BERT embedding tensor")?;
        let embeddings = mean_pool_hidden(hidden, &attention_mask);
        Ok((embeddings, input_tokens))
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn mean_pool_hidden(hidden: Vec<Vec<Vec<f32>>>, masks: &[Vec<u32>]) -> Vec<Vec<f32>> {
    hidden
        .into_iter()
        .zip(masks.iter())
        .map(|(tokens, mask)| {
            let dimensions = tokens.first().map(Vec::len).unwrap_or(0);
            let mut pooled = vec![0.0f32; dimensions];
            let mut count = 0f32;
            for (token, active) in tokens.into_iter().zip(mask.iter()) {
                if *active == 0 {
                    continue;
                }
                for (slot, value) in pooled.iter_mut().zip(token) {
                    *slot += value;
                }
                count += 1.0;
            }
            if count > 0.0 {
                for value in &mut pooled {
                    *value /= count;
                }
            }
            normalize_vector(pooled)
        })
        .collect()
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn validate_semantic_embedding_artifacts(model: &ModelConfig) -> Result<CandleArtifactValidation> {
    let format = infer_native_artifact_format(&model.path);
    if format != NativeModelFormat::Safetensors {
        bail!(
            "candle-bert-embeddings cannot load model alias '{}' because semantic embeddings require safetensors weights with tokenizer.json and config.json",
            model.alias
        );
    }
    let layout = CandleArtifactLayout::for_format(NativeModelFormat::Safetensors);
    validate_safetensors_artifacts(CandleModelFamily::Qwen3, model, layout).map_err(|err| {
        anyhow::anyhow!(
            "{}",
            err.to_string()
                .replace("candle-native-qwen3", "candle-bert-embeddings")
        )
    })
}

pub fn deterministic_native_embeddings(
    request: NativeEmbeddingRequest,
) -> Result<NativeEmbeddingResponse> {
    let input_tokens = request
        .input
        .iter()
        .map(|input| EstimatedNativeTokenCounter::estimate_text_tokens(input))
        .sum();
    let embeddings = request
        .input
        .iter()
        .map(|input| deterministic_embedding_vector(&request.model, input))
        .collect();
    Ok(NativeEmbeddingResponse {
        model: request.model,
        embeddings,
        usage: NativeTokenUsage::with_mode(input_tokens, 0, TokenAccountingMode::Estimated),
        backend: "deterministic-local-fallback".to_string(),
        status: "non-semantic-dev-fallback".to_string(),
        semantic: false,
    })
}

fn deterministic_embedding_vector(model: &str, input: &str) -> Vec<f32> {
    let mut vector = Vec::with_capacity(DETERMINISTIC_EMBEDDING_DIMENSIONS);
    let mut counter = 0u64;
    while vector.len() < DETERMINISTIC_EMBEDDING_DIMENSIONS {
        let mut hasher = Sha256::new();
        hasher.update(model.as_bytes());
        hasher.update(b"\0");
        hasher.update(input.as_bytes());
        hasher.update(b"\0");
        hasher.update(counter.to_le_bytes());
        let digest = hasher.finalize();
        for chunk in digest.chunks_exact(4) {
            let raw = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let unit = (raw as f32) / (u32::MAX as f32);
            vector.push(unit.mul_add(2.0, -1.0));
            if vector.len() == DETERMINISTIC_EMBEDDING_DIMENSIONS {
                break;
            }
        }
        counter = counter.saturating_add(1);
    }
    normalize_vector(vector)
}

fn normalize_vector(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeModelFormat {
    Gguf,
    Safetensors,
    Unknown,
}

impl NativeModelFormat {
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some(extension) if extension.eq_ignore_ascii_case("gguf") => Self::Gguf,
            Some(extension) if extension.eq_ignore_ascii_case("safetensors") => Self::Safetensors,
            _ => Self::Unknown,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeAcceleration {
    Cpu,
    NvidiaCuda,
    /// Resource-planning hook only — no candle-native execution backend
    /// implements AMD GPU execution yet (candle 0.10.2 has no ROCm/HIP/
    /// Vulkan device backend). Selecting this still fails closed to CPU
    /// via `NativeCandleEngineLoader::load`. See `docs/adr/0001-amd-gpu-acceleration.md`.
    AmdRocm,
    AppleMetal,
    Auto,
}

impl NativeAcceleration {
    pub fn from_resources(resources: &ResourceConfig) -> Self {
        if resources.cpu_only {
            return Self::Cpu;
        }

        match resources.gpu_vendor.trim().to_ascii_lowercase().as_str() {
            "nvidia" | "cuda" => Self::NvidiaCuda,
            "amd" | "rocm" | "hip" => Self::AmdRocm,
            "apple" | "metal" => Self::AppleMetal,
            "auto" | "" => Self::Auto,
            _ => Self::Cpu,
        }
    }

    pub fn compatible_gpu_vendor(&self) -> Option<GpuVendor> {
        match self {
            Self::NvidiaCuda => Some(GpuVendor::Nvidia),
            Self::AmdRocm => Some(GpuVendor::Amd),
            Self::AppleMetal => Some(GpuVendor::Apple),
            Self::Cpu | Self::Auto => None,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::NvidiaCuda => "nvidia-cuda",
            Self::AmdRocm => "amd-rocm",
            Self::AppleMetal => "apple-metal",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleModelFamily {
    Qwen3,
    Gemma4,
    DeepSeek,
    Kimi,
    Mistral,
    MiniMax,
}

impl CandleModelFamily {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Gemma4 => "gemma4",
            Self::DeepSeek => "deepseek",
            Self::Kimi => "kimi",
            Self::Mistral => "mistral",
            Self::MiniMax => "minimax",
        }
    }

    pub const fn engine_name(&self) -> &'static str {
        match self {
            Self::Qwen3 => "candle-native-qwen3",
            Self::Gemma4 => "candle-native-gemma4",
            Self::DeepSeek => "candle-native-deepseek",
            Self::Kimi => "candle-native-kimi",
            Self::Mistral => "candle-native-mistral",
            Self::MiniMax => "candle-native-minimax",
        }
    }

    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3",
            Self::Gemma4 => "Gemma 4",
            Self::DeepSeek => "DeepSeek",
            Self::Kimi => "Kimi",
            Self::Mistral => "Mistral",
            Self::MiniMax => "MiniMax",
        }
    }

    pub const fn all() -> &'static [Self] {
        &[
            Self::Qwen3,
            Self::Gemma4,
            Self::DeepSeek,
            Self::Kimi,
            Self::Mistral,
            Self::MiniMax,
        ]
    }

    pub const fn has_native_decoder(&self) -> bool {
        matches!(
            self,
            Self::Qwen3 | Self::Gemma4 | Self::DeepSeek | Self::Mistral
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleTokenizerRequirement {
    GgufMetadata,
    TokenizerJson,
    UnsupportedFormat,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleSupportedOperation {
    ChatCompletion,
    ChatTokenCounting,
    CompletionTokenCounting,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleFamilySupportMetadata {
    pub model_family: CandleModelFamily,
    pub display_name: String,
    pub engine: String,
    pub supported_formats: Vec<NativeModelFormat>,
    pub supported_accelerators: Vec<NativeAcceleration>,
    pub supported_operations: Vec<CandleSupportedOperation>,
    pub candle_crates_required: Vec<String>,
    pub tokenizer_contracts: Vec<CandleTokenizerContract>,
    pub generation_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleTokenizerContract {
    pub model_format: NativeModelFormat,
    pub requirement: CandleTokenizerRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleArtifactKind {
    GgufWeights,
    SafetensorsWeights,
    TokenizerJson,
    ConfigJson,
}

impl CandleArtifactKind {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::GgufWeights => "GGUF weights",
            Self::SafetensorsWeights => "safetensors weights",
            Self::TokenizerJson => "tokenizer.json",
            Self::ConfigJson => "config.json",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleArtifactRequirement {
    pub kind: CandleArtifactKind,
    pub filename: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleArtifactLayout {
    pub model_format: NativeModelFormat,
    pub requirements: Vec<CandleArtifactRequirement>,
}

impl CandleArtifactLayout {
    pub fn for_format(format: NativeModelFormat) -> Self {
        let requirements = match format {
            NativeModelFormat::Gguf => vec![CandleArtifactRequirement {
                kind: CandleArtifactKind::GgufWeights,
                filename: "*.gguf".to_string(),
                required: true,
            }],
            NativeModelFormat::Safetensors => vec![
                CandleArtifactRequirement {
                    kind: CandleArtifactKind::SafetensorsWeights,
                    filename: "*.safetensors".to_string(),
                    required: true,
                },
                CandleArtifactRequirement {
                    kind: CandleArtifactKind::TokenizerJson,
                    filename: "tokenizer.json".to_string(),
                    required: true,
                },
                CandleArtifactRequirement {
                    kind: CandleArtifactKind::ConfigJson,
                    filename: "config.json".to_string(),
                    required: true,
                },
            ],
            NativeModelFormat::Unknown => Vec::new(),
        };

        Self {
            model_format: format,
            requirements,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleArtifactValidation {
    pub model_family: CandleModelFamily,
    pub model_format: NativeModelFormat,
    pub layout: CandleArtifactLayout,
    pub weight_files: Vec<String>,
    pub tokenizer_file: Option<String>,
    pub config_file: Option<String>,
}

impl CandleFamilySupportMetadata {
    pub fn for_family(family: CandleModelFamily) -> Self {
        let supported_formats = supported_candle_formats_for_family(family);
        let supported_operations = if family.has_native_decoder() {
            vec![
                CandleSupportedOperation::ChatCompletion,
                CandleSupportedOperation::ChatTokenCounting,
                CandleSupportedOperation::CompletionTokenCounting,
            ]
        } else {
            Vec::new()
        };
        let tokenizer_contracts = supported_formats
            .iter()
            .copied()
            .map(|model_format| CandleTokenizerContract {
                model_format,
                requirement: tokenizer_requirement_for_supported_format(model_format),
            })
            .collect();

        Self {
            model_family: family,
            display_name: family.display_name().to_string(),
            engine: family.engine_name().to_string(),
            supported_formats,
            supported_accelerators: vec![
                NativeAcceleration::Cpu,
                NativeAcceleration::NvidiaCuda,
                NativeAcceleration::AmdRocm,
                NativeAcceleration::AppleMetal,
                NativeAcceleration::Auto,
            ],
            supported_operations,
            candle_crates_required: vec![
                "candle-core".to_string(),
                "candle-nn".to_string(),
                "candle-transformers".to_string(),
                "tokenizers".to_string(),
            ],
            tokenizer_contracts,
            generation_status: candle_family_generation_status(family),
        }
    }

    pub fn tokenizer_requirement(&self, format: NativeModelFormat) -> CandleTokenizerRequirement {
        self.tokenizer_contracts
            .iter()
            .find(|contract| contract.model_format == format)
            .map(|contract| contract.requirement.clone())
            .unwrap_or(CandleTokenizerRequirement::UnsupportedFormat)
    }
}

fn supported_candle_formats_for_family(family: CandleModelFamily) -> Vec<NativeModelFormat> {
    match family {
        CandleModelFamily::Qwen3 | CandleModelFamily::Gemma4 => {
            vec![NativeModelFormat::Gguf, NativeModelFormat::Safetensors]
        }
        CandleModelFamily::DeepSeek | CandleModelFamily::Mistral => {
            vec![NativeModelFormat::Safetensors]
        }
        CandleModelFamily::Kimi | CandleModelFamily::MiniMax => Vec::new(),
    }
}

fn tokenizer_requirement_for_supported_format(
    format: NativeModelFormat,
) -> CandleTokenizerRequirement {
    match format {
        NativeModelFormat::Gguf => CandleTokenizerRequirement::GgufMetadata,
        NativeModelFormat::Safetensors => CandleTokenizerRequirement::TokenizerJson,
        NativeModelFormat::Unknown => CandleTokenizerRequirement::UnsupportedFormat,
    }
}

fn candle_family_generation_status(family: CandleModelFamily) -> String {
    match family {
        CandleModelFamily::Qwen3 | CandleModelFamily::Gemma4 | CandleModelFamily::Mistral => {
            format!(
                "Candle {} artifact loading and greedy autoregressive decoding are wired where Candle exposes the required architecture and artifact format",
                family.as_str()
            )
        }
        CandleModelFamily::DeepSeek => {
            "Candle deepseek2 safetensors artifact loading and greedy autoregressive decoding are wired through DeepSeekV2; GGUF/quantized DeepSeek remains fail-closed because Candle 0.10.2 does not expose quantized DeepSeek2 model weights".to_string()
        }
        CandleModelFamily::Kimi => {
            "Kimi remains fail-closed for all native formats because Candle 0.10.2 does not expose candle_transformers::models::kimi or quantized Kimi GGUF model weights".to_string()
        }
        CandleModelFamily::MiniMax => {
            "MiniMax remains fail-closed for all native formats because Candle 0.10.2 does not expose candle_transformers::models::minimax or quantized MiniMax GGUF model weights".to_string()
        }
    }
}

fn candle_format_generation_status(family: CandleModelFamily, format: NativeModelFormat) -> String {
    match (family, format) {
        (CandleModelFamily::DeepSeek, NativeModelFormat::Safetensors) => {
            "candle-native-deepseek safetensors decoding is wired through candle_transformers::models::deepseek2::DeepSeekV2".to_string()
        }
        (CandleModelFamily::DeepSeek, NativeModelFormat::Gguf) => {
            "candle-native-deepseek GGUF/quantized DeepSeek fails closed because Candle 0.10.2 does not expose quantized DeepSeek2 model weights".to_string()
        }
        (CandleModelFamily::Kimi, NativeModelFormat::Safetensors) => {
            "candle-native-kimi safetensors decoding fails closed because Candle 0.10.2 does not expose candle_transformers::models::kimi".to_string()
        }
        (CandleModelFamily::Kimi, NativeModelFormat::Gguf) => {
            "candle-native-kimi GGUF/quantized Kimi fails closed because Candle 0.10.2 does not expose quantized Kimi GGUF model weights".to_string()
        }
        (CandleModelFamily::MiniMax, NativeModelFormat::Safetensors) => {
            "candle-native-minimax safetensors decoding fails closed because Candle 0.10.2 does not expose candle_transformers::models::minimax".to_string()
        }
        (CandleModelFamily::MiniMax, NativeModelFormat::Gguf) => {
            "candle-native-minimax GGUF/quantized MiniMax fails closed because Candle 0.10.2 does not expose quantized MiniMax GGUF model weights".to_string()
        }
        (_, NativeModelFormat::Unknown) => format!(
            "{} does not support unknown native artifact formats",
            family.engine_name()
        ),
        _ => candle_family_generation_status(family),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleDeviceSelectionContract {
    pub requested: NativeAcceleration,
    pub selected: NativeAcceleration,
    pub compatible_gpu_vendor: Option<GpuVendor>,
    pub selection_reason: String,
    pub fail_closed_if_unavailable: bool,
}

impl CandleDeviceSelectionContract {
    pub fn from_acceleration(acceleration: NativeAcceleration) -> Self {
        let compatible_gpu_vendor = acceleration.compatible_gpu_vendor();
        let selection_reason = match acceleration {
            NativeAcceleration::Cpu => "resources.cpu_only requested CPU execution".to_string(),
            NativeAcceleration::NvidiaCuda => {
                "resources.gpu_vendor requested NVIDIA CUDA execution".to_string()
            }
            NativeAcceleration::AmdRocm => {
                "resources.gpu_vendor requested AMD ROCm execution".to_string()
            }
            NativeAcceleration::AppleMetal => {
                "resources.gpu_vendor requested Apple Metal execution".to_string()
            }
            NativeAcceleration::Auto => {
                "resources.gpu_vendor left device selection to the Candle loader".to_string()
            }
        };

        Self {
            requested: acceleration,
            selected: acceleration,
            compatible_gpu_vendor,
            selection_reason,
            fail_closed_if_unavailable: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleEngineLoadContract {
    pub model_family: CandleModelFamily,
    pub model_format: NativeModelFormat,
    pub artifact_layout: CandleArtifactLayout,
    pub accelerator: NativeAcceleration,
    pub tokenizer: CandleTokenizerRequirement,
    pub supported_operations: Vec<CandleSupportedOperation>,
    pub candle_crates_required: Vec<String>,
    pub device_selection: CandleDeviceSelectionContract,
    pub fail_closed: bool,
    pub fail_closed_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleEngineConfig {
    pub engine: String,
    pub support: CandleFamilySupportMetadata,
    pub load_contract: CandleEngineLoadContract,
}

impl CandleEngineConfig {
    pub fn qwen3(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::Qwen3, format, accelerator)
    }

    pub fn gemma4(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::Gemma4, format, accelerator)
    }

    pub fn kimi(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::Kimi, format, accelerator)
    }

    pub fn mistral(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::Mistral, format, accelerator)
    }

    pub fn deepseek(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::DeepSeek, format, accelerator)
    }

    pub fn minimax(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::MiniMax, format, accelerator)
    }

    pub fn for_family(
        family: CandleModelFamily,
        format: NativeModelFormat,
        accelerator: NativeAcceleration,
    ) -> Self {
        let support = CandleFamilySupportMetadata::for_family(family);
        let supported_operations = if support.supported_formats.contains(&format) {
            support.supported_operations.clone()
        } else {
            Vec::new()
        };
        let fail_closed = supported_operations.is_empty();
        let tokenizer = support.tokenizer_requirement(format);
        let fail_closed_reason = candle_format_generation_status(family, format);

        Self {
            engine: support.engine.clone(),
            support: support.clone(),
            load_contract: CandleEngineLoadContract {
                model_family: family,
                model_format: format,
                artifact_layout: CandleArtifactLayout::for_format(format),
                accelerator,
                tokenizer,
                supported_operations,
                candle_crates_required: support.candle_crates_required.clone(),
                device_selection: CandleDeviceSelectionContract::from_acceleration(accelerator),
                fail_closed,
                fail_closed_reason,
            },
        }
    }

    pub fn is_supported(&self) -> bool {
        matches!(
            self.load_contract.model_format,
            NativeModelFormat::Gguf | NativeModelFormat::Safetensors
        ) && !self.load_contract.supported_operations.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct NativeCandleEngineFactory {
    registry: BTreeMap<CandleModelFamily, CandleFamilySupportMetadata>,
}

impl Default for NativeCandleEngineFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeCandleEngineFactory {
    pub fn new() -> Self {
        let registry = CandleModelFamily::all()
            .iter()
            .copied()
            .map(|family| (family, CandleFamilySupportMetadata::for_family(family)))
            .collect();
        Self { registry }
    }

    pub fn support_metadata(
        &self,
        family: CandleModelFamily,
    ) -> Option<&CandleFamilySupportMetadata> {
        self.registry.get(&family)
    }

    pub fn registered_families(&self) -> Vec<CandleModelFamily> {
        self.registry.keys().copied().collect()
    }

    pub fn plan(
        &self,
        family: CandleModelFamily,
        model: &ModelConfig,
        resources: &ResourceConfig,
    ) -> Result<NativeEngineLoadPlan> {
        let support = self.support_metadata(family).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "Candle model family '{}' is not registered",
                family.as_str()
            )
        })?;
        let format = NativeModelFormat::from_path(&model.path);
        let acceleration = NativeAcceleration::from_resources(resources);
        let candle = CandleEngineConfig::for_family(family, format, acceleration);
        let device_selection = candle.load_contract.device_selection.clone();

        let plan = NativeEngineLoadPlan {
            runtime: RuntimeBackend::CandleNative,
            engine: candle.engine.clone(),
            alias: model.alias.clone(),
            role: normalize_role(&model.role).to_string(),
            family: candle.load_contract.model_family.as_str().to_string(),
            format,
            acceleration,
            candle,
            support,
            device_selection,
            scheduler: NativeSchedulerContract::fifo_runtime(),
            model_path: model.path.clone(),
            budget_fraction: resources.budget,
            implemented: true,
            token_accounting: "native-tokenizer-or-deterministic-estimator".to_string(),
            observability: vec![
                "emit load, request, token, and error telemetry with safe attributes".to_string(),
                "never include prompt content, bearer tokens, API keys, or local paths".to_string(),
            ],
            security: vec![
                "load only configured model aliases".to_string(),
                "validate model artifacts before constructing a native engine".to_string(),
            ],
        };
        validate_native_engine_load_plan(&plan)?;
        Ok(plan)
    }

    pub fn load(&self, plan: &NativeEngineLoadPlan) -> Result<Box<dyn NativeEngine>> {
        validate_native_engine_load_plan(plan)?;
        if !matches!(
            plan.acceleration,
            NativeAcceleration::Cpu | NativeAcceleration::Auto
        ) {
            bail!(
                "native Candle decoding currently supports CPU execution only; requested {} acceleration for model {}",
                plan.acceleration.as_str(),
                plan.alias
            );
        }
        let model = ModelConfig {
            alias: plan.alias.clone(),
            path: plan.model_path.clone(),
            role: plan.role.clone(),
            family: Some(plan.family.clone()),
            weight: 1,
        };
        let artifacts =
            validate_candle_model_artifacts(plan.candle.load_contract.model_family, &model)?;
        verify_candle_artifacts_can_load(&plan.model_path, &artifacts)?;
        let decoder = NativeCandleDecoder::load(
            plan.candle.load_contract.model_family,
            &plan.model_path,
            &artifacts,
        )?;

        Ok(Box::new(ArtifactBackedCandleEngine {
            alias: plan.alias.clone(),
            decoder,
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeEngineLoadPlan {
    pub runtime: RuntimeBackend,
    pub engine: String,
    pub alias: String,
    pub role: String,
    pub family: String,
    pub format: NativeModelFormat,
    pub acceleration: NativeAcceleration,
    pub candle: CandleEngineConfig,
    pub support: CandleFamilySupportMetadata,
    pub device_selection: CandleDeviceSelectionContract,
    pub scheduler: NativeSchedulerContract,
    #[serde(skip)]
    pub model_path: PathBuf,
    pub budget_fraction: f64,
    pub implemented: bool,
    pub token_accounting: String,
    pub observability: Vec<String>,
    pub security: Vec<String>,
}

#[derive(Debug)]
pub struct ArtifactBackedCandleEngine {
    alias: String,
    decoder: NativeCandleDecoder,
}

impl ArtifactBackedCandleEngine {
    fn generate_text(&self, request: &NativeChatRequest) -> Result<String> {
        self.decoder.generate(request)
    }

    fn usage(&self, request: &NativeChatRequest, content: &str) -> Result<NativeTokenUsage> {
        self.decoder.usage(request, content)
    }
}

impl NativeEngine for ArtifactBackedCandleEngine {
    fn model_alias(&self) -> &str {
        &self.alias
    }

    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        Box::pin(async move {
            let content = self.generate_text(&request)?;
            let usage = self.usage(&request, &content)?;
            Ok(NativeChatResponse {
                model: request.model,
                content,
                tool_calls: None,
                finish_reason: "stop".to_string(),
                usage,
            })
        })
    }
}

#[derive(Debug)]
enum NativeCandleDecoder {
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    Real(RealCandleDecoder),
    #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
    Unavailable,
}

impl NativeCandleDecoder {
    fn load(
        family: CandleModelFamily,
        model_path: &Path,
        artifacts: &CandleArtifactValidation,
    ) -> Result<Self> {
        ensure_candle_family_format_supported(family, artifacts.model_format)?;
        load_real_candle_decoder(family, model_path, artifacts)
    }

    fn generate(&self, request: &NativeChatRequest) -> Result<String> {
        #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
        let _ = request;
        match self {
            #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
            Self::Real(decoder) => decoder.generate(request),
            #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
            Self::Unavailable => bail!(
                "native autoregressive decoding requires the native-candle and native-tokenizers features"
            ),
        }
    }

    fn usage(&self, request: &NativeChatRequest, content: &str) -> Result<NativeTokenUsage> {
        match self {
            #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
            Self::Real(decoder) => decoder.usage(request, content),
            #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
            Self::Unavailable => {
                usage_from_native_tokens(&EstimatedNativeTokenCounter, request, content)
            }
        }
    }
}

fn ensure_candle_family_format_supported(
    family: CandleModelFamily,
    format: NativeModelFormat,
) -> Result<()> {
    let config = CandleEngineConfig::for_family(family, format, NativeAcceleration::Cpu);
    if !config.is_supported() {
        bail!("{}", config.load_contract.fail_closed_reason);
    }
    Ok(())
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
fn load_real_candle_decoder(
    _family: CandleModelFamily,
    _model_path: &Path,
    _artifacts: &CandleArtifactValidation,
) -> Result<NativeCandleDecoder> {
    Ok(NativeCandleDecoder::Unavailable)
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
#[derive(Debug)]
struct RealCandleDecoder {
    tokenizer: tokenizers::tokenizer::Tokenizer,
    model: Mutex<RealCandleModel>,
    family: CandleModelFamily,
    /// BOS token id to prepend to the generation prompt's `input_ids`, if the
    /// GGUF tokenizer metadata configures `add_bos_token = true`. See
    /// [`gguf_bos_token_to_prepend`] and [`prepend_bos_if_configured`].
    bos_token_id: Option<u32>,
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
enum RealCandleModel {
    Qwen3(candle_transformers::models::qwen3::ModelForCausalLM),
    Qwen3Gguf(candle_transformers::models::quantized_qwen3::ModelWeights),
    DeepSeek2(candle_transformers::models::deepseek2::DeepSeekV2),
    Gemma3(candle_transformers::models::gemma3::Model),
    Gemma4Gguf(quantized_gemma4::ModelWeights),
    Mistral(candle_transformers::models::mistral::Model),
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl std::fmt::Debug for RealCandleModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant = match self {
            Self::Qwen3(_) => "Qwen3",
            Self::Qwen3Gguf(_) => "Qwen3Gguf",
            Self::DeepSeek2(_) => "DeepSeek2",
            Self::Gemma3(_) => "Gemma3",
            Self::Gemma4Gguf(_) => "Gemma4Gguf",
            Self::Mistral(_) => "Mistral",
        };
        f.debug_tuple("RealCandleModel").field(&variant).finish()
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn load_real_candle_decoder(
    family: CandleModelFamily,
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<NativeCandleDecoder> {
    if !family.has_native_decoder() {
        bail!(
            "{}",
            CandleFamilySupportMetadata::for_family(family).generation_status
        );
    }

    let device = candle_core::Device::Cpu;
    let (tokenizer, bos_token_id) = load_generation_tokenizer(model_path, artifacts)
        .with_context(|| "failed to load native generation tokenizer")?;
    let model = match artifacts.model_format {
        NativeModelFormat::Safetensors => {
            let artifact_dir = safetensors_artifact_dir(model_path);
            let config_path = artifact_dir.join("config.json");
            let weight_paths = artifacts
                .weight_files
                .iter()
                .map(|name| artifact_dir.join(name))
                .collect::<Vec<_>>();
            // Candle exposes safetensors mmap loading as unsafe because it relies on OS mmap invariants.
            // The files were validated above and are used read-only for model weights.
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &weight_paths,
                    candle_core::DType::F32,
                    &device,
                )
            }
            .with_context(|| "failed to mmap safetensors weights with Candle")?;
            match family {
                CandleModelFamily::Qwen3 => {
                    let cfg: candle_transformers::models::qwen3::Config =
                        read_json_config(&config_path)?;
                    RealCandleModel::Qwen3(
                        candle_transformers::models::qwen3::ModelForCausalLM::new(&cfg, vb)
                            .with_context(|| "failed to construct Qwen3 Candle model")?,
                    )
                }
                CandleModelFamily::Gemma4 => {
                    let cfg: candle_transformers::models::gemma3::Config =
                        read_json_config(&config_path)?;
                    RealCandleModel::Gemma3(
                        candle_transformers::models::gemma3::Model::new(false, &cfg, vb)
                            .with_context(|| "failed to construct Gemma Candle model")?,
                    )
                }
                CandleModelFamily::DeepSeek => {
                    let cfg: candle_transformers::models::deepseek2::DeepSeekV2Config =
                        read_json_config(&config_path)?;
                    RealCandleModel::DeepSeek2(
                        candle_transformers::models::deepseek2::DeepSeekV2::new(&cfg, vb)
                            .with_context(|| "failed to construct DeepSeek2 Candle model")?,
                    )
                }
                CandleModelFamily::Mistral => {
                    let cfg: candle_transformers::models::mistral::Config =
                        read_json_config(&config_path)?;
                    RealCandleModel::Mistral(
                        candle_transformers::models::mistral::Model::new(&cfg, vb)
                            .with_context(|| "failed to construct Mistral Candle model")?,
                    )
                }
                CandleModelFamily::Kimi | CandleModelFamily::MiniMax => {
                    unreachable!("blocked families are rejected before loading")
                }
            }
        }
        NativeModelFormat::Gguf => {
            let mut file = fs::File::open(model_path)
                .with_context(|| "failed to open GGUF weights for Candle model loading")?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                .with_context(|| "failed to parse GGUF weights")?;
            match family {
                CandleModelFamily::Qwen3 => RealCandleModel::Qwen3Gguf(
                    candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
                        content, &mut file, &device,
                    )
                    .with_context(|| "failed to construct quantized Qwen3 Candle model")?,
                ),
                CandleModelFamily::Gemma4 => RealCandleModel::Gemma4Gguf(
                    quantized_gemma4::ModelWeights::from_gguf(content, &mut file, &device)
                        .with_context(|| "failed to construct quantized Gemma Candle model")?,
                ),
                CandleModelFamily::Mistral => bail!(
                    "candle-native-mistral GGUF decoding is not wired in Candle 0.10.2; use safetensors with tokenizer.json and config.json"
                ),
                CandleModelFamily::DeepSeek => bail!(
                    "candle-native-deepseek GGUF decoding is not wired because Candle 0.10.2 does not expose quantized DeepSeek2 model weights; use safetensors with tokenizer.json and config.json"
                ),
                CandleModelFamily::Kimi | CandleModelFamily::MiniMax => {
                    unreachable!("blocked families are rejected before loading")
                }
            }
        }
        NativeModelFormat::Unknown => bail!("native artifact format is unsupported"),
    };

    Ok(NativeCandleDecoder::Real(RealCandleDecoder {
        tokenizer,
        model: Mutex::new(model),
        family,
        bos_token_id,
    }))
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl RealCandleDecoder {
    fn generate(&self, request: &NativeChatRequest) -> Result<String> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("native Candle model lock is poisoned"))?;
        model.clear_kv_cache();

        let prompt = match self.family {
            CandleModelFamily::Gemma4 => gemma_chat_input(&request.messages),
            CandleModelFamily::Qwen3
            | CandleModelFamily::DeepSeek
            | CandleModelFamily::Kimi
            | CandleModelFamily::Mistral
            | CandleModelFamily::MiniMax => canonical_native_chat_input(&request.messages),
        };
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize native prompt: {err}"))?;
        let mut input_ids = encoding.get_ids().to_vec();
        prepend_bos_if_configured(&mut input_ids, self.bos_token_id);
        if input_ids.is_empty() {
            bail!("native prompt tokenization produced no tokens");
        }

        let max_tokens = request
            .max_tokens
            .and_then(|tokens| usize::try_from(tokens).ok())
            .unwrap_or(128)
            .clamp(1, 4096);
        let mut generated = Vec::new();
        let mut offset = 0usize;
        for step in 0..max_tokens {
            let step_input = if step == 0 {
                input_ids.clone()
            } else {
                vec![*input_ids.last().expect("input ids are non-empty")]
            };
            let next = model.forward_next(&step_input, offset)?;
            offset = offset.saturating_add(step_input.len());
            input_ids.push(next);
            generated.push(next);
            if is_eos_token(&self.tokenizer, next) {
                break;
            }
        }

        self.tokenizer
            .decode(&generated, true)
            .map_err(|err| anyhow::anyhow!("failed to decode native output tokens: {err}"))
    }

    fn usage(&self, request: &NativeChatRequest, content: &str) -> Result<NativeTokenUsage> {
        let counter = TokenizersNativeTokenCounter::from_tokenizer(self.tokenizer.clone());
        usage_from_native_tokens(&counter, request, content)
    }
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
pub fn generate_gemma4_sources(
    _model_path: &Path,
    _prompts: &[String],
    _max_tokens: u32,
) -> Result<Vec<String>> {
    bail!("generate_gemma4_sources requires the native-candle and native-tokenizers features")
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub fn generate_gemma4_sources(
    model_path: &Path,
    prompts: &[String],
    max_tokens: u32,
) -> Result<Vec<String>> {
    let artifacts = CandleArtifactValidation {
        model_family: CandleModelFamily::Gemma4,
        model_format: NativeModelFormat::Gguf,
        layout: CandleArtifactLayout::for_format(NativeModelFormat::Gguf),
        weight_files: vec![artifact_file_name(model_path)],
        tokenizer_file: None,
        config_file: None,
    };
    let decoder = load_real_candle_decoder(CandleModelFamily::Gemma4, model_path, &artifacts)?;
    prompts
        .iter()
        .map(|prompt| {
            decoder.generate(&NativeChatRequest {
                model: "gemma4-readiness".to_string(),
                messages: vec![NativeChatMessage {
                    role: "user".to_string(),
                    content: Some(Value::String(prompt.clone())),
                    tool_calls: None,
                    tool_call_id: None,
                }],
                temperature: Some(0.0),
                max_tokens: Some(max_tokens),
                tools: None,
                tool_choice: None,
                metadata: BTreeMap::new(),
            })
        })
        .collect()
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl RealCandleModel {
    fn clear_kv_cache(&mut self) {
        match self {
            Self::Qwen3(model) => model.clear_kv_cache(),
            Self::Qwen3Gguf(model) => model.clear_kv_cache(),
            Self::DeepSeek2(model) => model.clear_kv_cache(),
            Self::Gemma3(model) => model.clear_kv_cache(),
            Self::Gemma4Gguf(model) => model.clear_kv_cache(),
            Self::Mistral(model) => model.clear_kv_cache(),
        }
    }

    fn forward_next(&mut self, input_ids: &[u32], offset: usize) -> Result<u32> {
        let device = candle_core::Device::Cpu;
        let input = candle_core::Tensor::new(input_ids, &device)
            .and_then(|tensor| tensor.reshape((1, input_ids.len())))
            .with_context(|| "failed to create native input tensor")?;
        let logits = match self {
            Self::Qwen3(model) => model.forward(&input, offset),
            Self::Qwen3Gguf(model) => model.forward(&input, offset),
            Self::DeepSeek2(model) => model.forward(&input, offset),
            Self::Gemma3(model) => model.forward(&input, offset),
            Self::Gemma4Gguf(model) => model.forward(&input, offset),
            Self::Mistral(model) => model.forward(&input, offset),
        }
        .with_context(|| "native Candle model forward pass failed")?;
        let logits_shape = logits.dims().to_vec();
        let next_logits = match logits.dims() {
            [_, seq_len, _] => logits
                .narrow(1, seq_len.saturating_sub(1), 1)
                .and_then(|tensor| tensor.squeeze(1)),
            [seq_len, _] => logits
                .narrow(0, seq_len.saturating_sub(1), 1)
                .and_then(|tensor| tensor.squeeze(0)),
            [_] => Ok(logits),
            dims => bail!("native Candle model returned unsupported logits shape: {dims:?}"),
        }
        .with_context(|| "failed to select native next-token logits")?;
        let next_logits_shape = next_logits.dims().to_vec();
        let next_token = next_logits
            .argmax(candle_core::D::Minus1)
            .and_then(|tensor| tensor.to_scalar::<u32>())
            .map_err(|err| {
                anyhow::anyhow!(
                    "failed to select next native token from logits shape {logits_shape:?} and next-token logits shape {next_logits_shape:?}: {err}"
                )
            })?;
        Ok(next_token)
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn read_json_config<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let raw = fs::read_to_string(path).with_context(|| "failed to read model config.json")?;
    serde_json::from_str(&raw).with_context(|| "failed to parse model config.json")
}

/// Builds a [`tokenizers::tokenizer::Tokenizer`] from GGUF metadata, dispatching on
/// `tokenizer.ggml.model`.
///
/// Candle's own [`candle_core::quantized::tokenizer::TokenizerFromGguf`] only
/// understands `tokenizer.ggml.model == "gpt2"` (byte-level BPE). Some model
/// families — notably Gemma's `gemma4` GGUF exports — ship a SentencePiece-style
/// BPE vocabulary (metaspace `▁` convention) tagged `tokenizer.ggml.model ==
/// "gemma4"`. This function delegates `"gpt2"` to candle unchanged and adds a
/// native `"gemma4"` code path; any other value is a hard error.
///
/// # Errors
///
/// Returns an error if `tokenizer.ggml.model` is missing or not a recognized
/// value, or if the underlying tokenizer construction fails.
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn tokenizer_from_gguf_content(
    content: &candle_core::quantized::gguf_file::Content,
) -> Result<tokenizers::tokenizer::Tokenizer> {
    let model_kind = gemma4_gguf_tokenizer::metadata_value(content, "tokenizer.ggml.model")
        .and_then(|value| value.to_string().map_err(candle_core::Error::wrap))
        .map_err(|err| anyhow::anyhow!("failed to read GGUF tokenizer model kind: {err}"))?
        .to_lowercase();

    match model_kind.as_str() {
        "gpt2" => {
            <tokenizers::tokenizer::Tokenizer as candle_core::quantized::tokenizer::TokenizerFromGguf>::from_gguf(content)
                .map_err(|err| anyhow::anyhow!("failed to build tokenizer from GGUF metadata: {err}"))
        }
        "gemma4" => gemma4_gguf_tokenizer::build(content)
            .map_err(|err| anyhow::anyhow!("failed to build gemma4 tokenizer from GGUF metadata: {err}")),
        other => bail!("unsupported tokenizer model `{other}` (supported: gpt2, gemma4)"),
    }
}

/// Returns the BOS token id that [`prepend_bos_if_configured`] should insert at
/// the start of `input_ids` for the generation prompt, based on GGUF tokenizer
/// metadata.
///
/// `encode(prompt, false)` bypasses the tokenizer's post-processor, so a
/// `tokenizer.ggml.add_bos_token = true` configuration (as set by
/// [`gemma4_gguf_tokenizer::build`]) never actually adds `<bos>` to the
/// generation prompt. This function reads that intent directly from the GGUF
/// metadata so callers can apply it manually.
///
/// Returns `None` for tokenizer kinds that don't carry this metadata (e.g.
/// `gpt2`), which keeps this a no-op for the existing Qwen3/other native model
/// paths.
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn gguf_bos_token_to_prepend(content: &candle_core::quantized::gguf_file::Content) -> Option<u32> {
    let model_kind = gemma4_gguf_tokenizer::metadata_value(content, "tokenizer.ggml.model")
        .and_then(|value| value.to_string().map_err(candle_core::Error::wrap))
        .ok()?
        .to_lowercase();

    match model_kind.as_str() {
        "gemma4" => gemma4_gguf_tokenizer::bos_token_to_prepend(content),
        _ => None,
    }
}

/// Prepends `bos_token_id` to `input_ids` if it is configured and not already
/// the first element.
///
/// No-op when `bos_token_id` is `None`, or when `input_ids` already starts with
/// that id (e.g. because the tokenizer's post-processor already added it).
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn prepend_bos_if_configured(input_ids: &mut Vec<u32>, bos_token_id: Option<u32>) {
    let Some(bos_token_id) = bos_token_id else {
        return;
    };
    if input_ids.first() != Some(&bos_token_id) {
        input_ids.insert(0, bos_token_id);
    }
}

/// Native (non-candle) construction of a SentencePiece-metaspace BPE tokenizer
/// from `gemma4`-flavoured GGUF metadata.
///
/// Mirrors the structure of candle's `TokenizerFromGguf::from_gguf` for the
/// `gpt2` case, but swaps the byte-level pre-tokenizer/decoder/post-processor
/// for `SentencePiece` [`tokenizers::pre_tokenizers::metaspace::Metaspace`]
/// equivalents, matching the `▁`-based vocabulary that `gemma4` GGUF exports use.
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
mod gemma4_gguf_tokenizer {
    use candle_core::quantized::gguf_file;
    use candle_core::{Context as CandleContext, Error as CandleError, Result as CandleResult};
    use std::collections::HashSet;
    use tokenizers::models::bpe::{Vocab, BPE};
    use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{AddedToken, PostProcessorWrapper, Tokenizer};

    pub(super) fn metadata_value<'a>(
        ct: &'a gguf_file::Content,
        key: &str,
    ) -> CandleResult<&'a gguf_file::Value> {
        ct.metadata
            .get(key)
            .with_context(|| format!("missing GGUF metadata key `{key}`"))
    }

    fn gguf_value_to_u32(v: &gguf_file::Value) -> CandleResult<u32> {
        let as_i64 = match v {
            gguf_file::Value::U8(v) => i64::from(*v),
            gguf_file::Value::I8(v) => i64::from(*v),
            gguf_file::Value::U16(v) => i64::from(*v),
            gguf_file::Value::I16(v) => i64::from(*v),
            gguf_file::Value::U32(v) => i64::from(*v),
            gguf_file::Value::I32(v) => i64::from(*v),
            gguf_file::Value::U64(v) => i64::try_from(*v).map_err(CandleError::wrap)?,
            gguf_file::Value::I64(v) => *v,
            other => candle_core::bail!("expected numeric value for token type/id, got {other:?}"),
        };
        u32::try_from(as_i64)
            .map_err(|_| CandleError::msg(format!("token type/id {as_i64} out of range for u32")))
    }

    fn value_to_string_array(v: &gguf_file::Value, name: &str) -> CandleResult<Vec<String>> {
        let arr = v
            .to_vec()
            .with_context(|| format!("`{name}` is not an array"))?;
        arr.iter()
            .map(|v| {
                v.to_string()
                    .map(std::string::ToString::to_string)
                    .with_context(|| format!("`{name}` element is not a string: {v:?}"))
            })
            .collect()
    }

    fn merges_from_value(v: &gguf_file::Value) -> CandleResult<Vec<(String, String)>> {
        value_to_string_array(v, "tokenizer.ggml.merges")?
            .into_iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .ok_or_else(|| CandleError::msg(format!("invalid merge entry `{m}`")))
            })
            .collect()
    }

    /// Looks up the unknown-token id, checking the `gemma4`-style
    /// `tokenizer.ggml.unknown_token_id` key first and falling back to the
    /// `gpt2`-style `tokenizer.ggml.unk_token_id` key.
    fn unk_token_id(ct: &gguf_file::Content) -> Option<u32> {
        metadata_value(ct, "tokenizer.ggml.unknown_token_id")
            .or_else(|_| metadata_value(ct, "tokenizer.ggml.unk_token_id"))
            .and_then(gguf_value_to_u32)
            .ok()
    }

    /// Returns the BOS token id that callers should prepend to encoded prompts,
    /// if `tokenizer.ggml.add_bos_token` is `true` and `tokenizer.ggml.bos_token_id`
    /// is present.
    ///
    /// `encode(prompt, false)` (used for the generation prompt) skips the
    /// tokenizer's post-processor entirely, so the post-processor configured by
    /// [`build`] never has a chance to add `<bos>`. Callers must prepend it
    /// manually via [`super::prepend_bos_if_configured`]. This is independent of
    /// `tokenizer.ggml.add_eos_token`, which would otherwise also append `<eos>`
    /// if `encode(prompt, true)` were used instead.
    pub(super) fn bos_token_to_prepend(ct: &gguf_file::Content) -> Option<u32> {
        let add_bos = metadata_value(ct, "tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().map_err(CandleError::wrap))
            .unwrap_or(false);
        if !add_bos {
            return None;
        }
        metadata_value(ct, "tokenizer.ggml.bos_token_id")
            .and_then(gguf_value_to_u32)
            .ok()
    }

    /// Builds a BOS/EOS template post-processor, mirroring candle's private
    /// `template_processor` helper for the `gpt2` GGUF tokenizer path.
    fn template_processor(
        tokens: &[String],
        bos_id: Option<u32>,
        eos_id: Option<u32>,
        add_bos: bool,
        add_eos: bool,
    ) -> Option<PostProcessorWrapper> {
        if (!add_bos && !add_eos) || tokens.is_empty() {
            return None;
        }

        let bos = bos_id.and_then(|id| tokens.get(id as usize)).cloned();
        let eos = eos_id.and_then(|id| tokens.get(id as usize)).cloned();

        let mut specials = Vec::new();
        if add_bos {
            let bos_id = bos_id?;
            let bos_tok = bos.clone()?;
            specials.push((bos_tok, bos_id));
        }
        if add_eos {
            let eos_id = eos_id?;
            let eos_tok = eos.clone()?;
            specials.push((eos_tok, eos_id));
        }

        let mut single = Vec::new();
        if add_bos {
            single.push(bos.clone()?);
        }
        single.push("$0".to_string());
        if add_eos {
            single.push(eos.clone()?);
        }

        let mut pair = Vec::new();
        if add_bos {
            pair.push(format!("{}:0", bos.clone()?));
        }
        pair.push("$A:0".to_string());
        if add_eos {
            pair.push(format!("{}:0", eos.clone()?));
        }
        if add_bos {
            pair.push(format!("{}:1", bos.clone()?));
        }
        pair.push("$B:1".to_string());
        if add_eos {
            pair.push(format!("{}:1", eos.clone()?));
        }

        let proc = TemplateProcessing::builder()
            .try_single(single)
            .ok()?
            .try_pair(pair)
            .ok()?
            .special_tokens(specials)
            .build()
            .ok()?;

        Some(PostProcessorWrapper::Template(proc))
    }

    /// Builds a SentencePiece-metaspace BPE [`Tokenizer`] from `gemma4`-flavoured
    /// GGUF metadata (`tokenizer.ggml.model == "gemma4"`).
    pub(super) fn build(ct: &gguf_file::Content) -> CandleResult<Tokenizer> {
        let tokens = value_to_string_array(
            metadata_value(ct, "tokenizer.ggml.tokens")?,
            "tokenizer.ggml.tokens",
        )?;
        let vocab: Vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| -> CandleResult<(String, u32)> {
                let id = u32::try_from(i).map_err(|_| {
                    CandleError::msg(format!("vocab index {i} out of range for u32"))
                })?;
                Ok((t.clone(), id))
            })
            .collect::<CandleResult<Vocab>>()?;
        let merges = merges_from_value(metadata_value(ct, "tokenizer.ggml.merges")?)?;

        let mut builder = BPE::builder().vocab_and_merges(vocab, merges);

        if let Some(token_id) = unk_token_id(ct) {
            if let Some(token) = tokens.get(token_id as usize) {
                builder = builder.unk_token(token.clone());
            }
        }

        if let Ok(val) = metadata_value(ct, "tokenizer.ggml.byte_fallback") {
            builder = builder.byte_fallback(val.to_bool().map_err(CandleError::wrap)?);
        }

        if let Ok(val) = metadata_value(ct, "tokenizer.ggml.ignore_merges") {
            builder = builder.ignore_merges(val.to_bool().map_err(CandleError::wrap)?);
        }

        let bpe = builder.build().map_err(CandleError::wrap)?;
        let mut tokenizer = Tokenizer::new(bpe);

        // SentencePiece convention: prepend a leading metaspace marker unless
        // `tokenizer.ggml.add_space_prefix` is explicitly `false`.
        let add_space_prefix = metadata_value(ct, "tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.to_bool().map_err(CandleError::wrap))
            .unwrap_or(true);
        let prepend_scheme = if add_space_prefix {
            PrependScheme::Always
        } else {
            PrependScheme::Never
        };
        let metaspace = Metaspace::new('▁', prepend_scheme, true);
        tokenizer.with_pre_tokenizer(Some(metaspace.clone()));
        tokenizer.with_decoder(Some(metaspace));

        let add_bos = metadata_value(ct, "tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().map_err(CandleError::wrap))
            .unwrap_or(false);
        let add_eos = metadata_value(ct, "tokenizer.ggml.add_eos_token")
            .and_then(|v| v.to_bool().map_err(CandleError::wrap))
            .unwrap_or(false);
        let bos_id = metadata_value(ct, "tokenizer.ggml.bos_token_id")
            .and_then(gguf_value_to_u32)
            .ok();
        let eos_id = metadata_value(ct, "tokenizer.ggml.eos_token_id")
            .and_then(gguf_value_to_u32)
            .ok();

        if let Some(pp) = template_processor(&tokens, bos_id, eos_id, add_bos, add_eos) {
            tokenizer.with_post_processor(Some(pp));
        }

        // Mark special tokens so decode(skip_special_tokens = true) behaves as expected.
        if let Ok(gguf_file::Value::Array(arr)) = metadata_value(ct, "tokenizer.ggml.token_type") {
            let mut specials = Vec::new();
            for (idx, v) in arr.iter().enumerate() {
                let ty = gguf_value_to_u32(v)?;
                // Aligns with llama_token_type: treat non-normal/non-byte tokens as special.
                let is_special = matches!(ty, 2..=5);
                if is_special {
                    if let Some(tok) = tokens.get(idx) {
                        specials.push(AddedToken::from(tok.clone(), true));
                    }
                }
            }
            if !specials.is_empty() {
                tokenizer.add_special_tokens(&specials);
            }
        }

        let mut explicit_specials = HashSet::new();
        for key in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.pad_token_id",
            "tokenizer.ggml.sep_token_id",
        ] {
            if let Ok(val) = metadata_value(ct, key) {
                explicit_specials.insert(gguf_value_to_u32(val)?);
            }
        }
        if let Some(id) = unk_token_id(ct) {
            explicit_specials.insert(id);
        }
        if !explicit_specials.is_empty() {
            let specials: Vec<_> = explicit_specials
                .into_iter()
                .filter_map(|id| tokens.get(id as usize))
                .map(|tok| AddedToken::from(tok.clone(), true))
                .collect();
            if !specials.is_empty() {
                tokenizer.add_special_tokens(&specials);
            }
        }

        Ok(tokenizer)
    }
}

/// Quantized model weights for `gemma4`-architecture GGUF exports.
///
/// `candle_transformers::models::quantized_gemma3::ModelWeights` reads
/// `attention.head_count`, `attention.key_length`, `rope.freq_base`, and the
/// sliding-window flag as **uniform scalars** applied to every transformer
/// layer. The `gemma4` architecture instead has **per-layer heterogeneous
/// attention configs**: layers alternate between "sliding window" (local) and
/// "global" (full-attention) variants with different head dimensions, KV head
/// counts, and RoPE frequencies. This module re-implements the equivalent of
/// `quantized_gemma3::ModelWeights` with per-layer configuration read from
/// `gemma4.*` metadata, including the `gemma4.attention.head_count_kv` and
/// `gemma4.attention.sliding_window_pattern` per-layer arrays.
///
/// The `forward`/`mask`/`forward_attn`/[`RotaryEmbedding`]/[`Mlp`]/[`QMatMul`]
/// logic is ported from `quantized_gemma3` essentially unchanged, since those
/// types are already per-layer-parametrized; only `from_gguf`'s metadata
/// reading differs.
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
mod quantized_gemma4 {
    use candle_core::quantized::gguf_file;
    use candle_core::quantized::QTensor;
    use candle_core::{Context as CandleContext, DType, Device, IndexOp, Result, Tensor, D};
    use candle_nn::{Embedding, Module};
    use candle_transformers::quantized_nn::RmsNorm;
    use candle_transformers::utils::repeat_kv;

    /// Gemma 3/4 supports a 128K context window.
    const MAX_SEQ_LEN: usize = 131_072;

    #[derive(Debug, Clone)]
    struct QMatMul {
        inner: candle_core::quantized::QMatMul,
        span: tracing::Span,
    }

    impl QMatMul {
        fn from_qtensor(qtensor: QTensor) -> Result<Self> {
            let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
            let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
            Ok(Self { inner, span })
        }

        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let _enter = self.span.enter();
            self.inner.forward(xs)
        }
    }

    #[derive(Debug, Clone)]
    struct Mlp {
        feed_forward_gate: QMatMul,
        feed_forward_up: QMatMul,
        feed_forward_down: QMatMul,
    }

    impl Module for Mlp {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let gate = self.feed_forward_gate.forward(xs)?;
            let up = self.feed_forward_up.forward(xs)?;
            // llama.cpp's Gemma 4 path uses plain GeGLU, whose GELU is the
            // tanh approximation; Candle's Tensor::gelu() is the same variant.
            let gated = (gate.gelu()? * up)?;
            self.feed_forward_down.forward(&gated)
        }
    }

    #[derive(Debug, Clone)]
    struct RotaryEmbedding {
        sin: Tensor,
        cos: Tensor,
    }

    impl RotaryEmbedding {
        /// `freq_factors`, when present, divides each dimension-pair's
        /// rotation rate (`gemma4`'s top-level `rope_freqs.weight`, applied
        /// only to global/non-SWA layers in upstream llama.cpp). A factor of
        /// `1e30` effectively freezes that dimension pair's rotation
        /// (`cos` -> 1, `sin` -> 0 for all positions), which is how gemma4
        /// extends context length without retraining the higher rotary
        /// dimensions.
        fn new(
            head_dim: usize,
            rope_frequency: f32,
            freq_factors: Option<&[f32]>,
            device: &Device,
        ) -> Result<Self> {
            let theta: Vec<_> = (0..head_dim)
                .step_by(2)
                .enumerate()
                .map(|(j, i)| {
                    let base = 1f32 / rope_frequency.powf(i as f32 / head_dim as f32);
                    match freq_factors.and_then(|factors| factors.get(j)) {
                        Some(factor) => base / factor,
                        None => base,
                    }
                })
                .collect();
            let theta = Tensor::new(theta.as_slice(), device)?;
            let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
                .to_dtype(DType::F32)?
                .reshape((MAX_SEQ_LEN, 1))?
                .matmul(&theta.reshape((1, theta.elem_count()))?)?;
            let cos = idx_theta.cos()?;
            let sin = idx_theta.sin()?;
            Ok(Self { sin, cos })
        }

        fn apply_rotary_emb_qkv(
            &self,
            q: &Tensor,
            k: &Tensor,
            index_pos: usize,
        ) -> Result<(Tensor, Tensor)> {
            let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
            let cos = self.cos.narrow(0, index_pos, seq_len)?;
            let sin = self.sin.narrow(0, index_pos, seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        }
    }

    #[derive(Debug, Clone)]
    struct LayerWeights {
        attention_wq: QMatMul,
        attention_wk: QMatMul,
        /// Value projection. Absent on global (non-sliding) layers, which
        /// have no `attn_v.weight` tensor in the GGUF; on those layers the
        /// raw key projection output is reused as the value projection
        /// (`Vcur = Kcur` in upstream llama.cpp's gemma4 graph, before any
        /// reshape/norm/RoPE is applied to `Kcur`).
        attention_wv: Option<QMatMul>,
        attention_wo: QMatMul,

        attention_q_norm: RmsNorm,
        attention_k_norm: RmsNorm,

        attention_norm: RmsNorm,
        post_attention_norm: RmsNorm,
        ffn_norm: RmsNorm,
        post_ffn_norm: RmsNorm,

        mlp: Mlp,

        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        q_dim: usize,

        sliding_window_size: Option<usize>,

        rotary_embedding: RotaryEmbedding,
        neg_inf: Tensor,

        /// `eps` used for the unweighted RMS normalization applied to the
        /// value projection (see [`rms_norm_unweighted`]).
        rms_norm_eps: f64,

        /// Scalar multiplier applied to this block's full output (after both
        /// the attention and feed-forward residual additions) before it is
        /// passed to the next layer, read from `blk.N.layer_output_scale.weight`.
        layer_output_scale: f32,

        kv_cache: Option<(Tensor, Tensor)>,

        span_attn: tracing::Span,
        span_mlp: tracing::Span,
    }

    impl LayerWeights {
        fn mask(
            &self,
            b_sz: usize,
            seq_len: usize,
            index_pos: usize,
            dtype: DType,
            device: &Device,
        ) -> Result<Tensor> {
            let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
                (0..seq_len)
                    .flat_map(|i| {
                        (0..seq_len).map(move |j| {
                            if i < j || j + sliding_window_size < i {
                                0u32
                            } else {
                                1u32
                            }
                        })
                    })
                    .collect()
            } else {
                (0..seq_len)
                    .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                    .collect()
            };
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
            let mask = if index_pos > 0 {
                let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
                Tensor::cat(&[&mask0, &mask], D::Minus1)?
            } else {
                mask
            };
            mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
                .to_dtype(dtype)
        }

        fn forward_attn(
            &mut self,
            x: &Tensor,
            mask: Option<&Tensor>,
            index_pos: usize,
        ) -> Result<Tensor> {
            let _enter = self.span_attn.enter();
            let (b_sz, seq_len, _) = x.dims3()?;

            let q = self.attention_wq.forward(x)?;
            let k = self.attention_wk.forward(x)?;
            // Global (non-sliding) layers have no `attn_v.weight`; upstream
            // llama.cpp reuses the raw key projection output as `Vcur` in
            // that case (`Vcur = Kcur` before any reshape/norm/RoPE).
            let v = match &self.attention_wv {
                Some(wv) => wv.forward(x)?,
                None => k.clone(),
            };

            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;

            let q = self.attention_q_norm.forward(&q.contiguous()?)?;
            let k = self.attention_k_norm.forward(&k.contiguous()?)?;
            // gemma4 applies an unweighted RMS norm to V on every layer
            // (`ggml_rms_norm(Vcur, eps)` in upstream llama.cpp), unlike K
            // which uses the learned `attn_k_norm` weight. V never receives
            // RoPE.
            let v = rms_norm_unweighted(&v.contiguous()?, self.rms_norm_eps)?;

            let (q, k) = self
                .rotary_embedding
                .apply_rotary_emb_qkv(&q, &k, index_pos)?;

            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((k_cache, v_cache)) => {
                    if index_pos == 0 {
                        (k, v)
                    } else {
                        let k = Tensor::cat(&[k_cache, &k], 2)?;
                        let v = Tensor::cat(&[v_cache, &v], 2)?;
                        (k, v)
                    }
                }
            };
            self.kv_cache = Some((k.clone(), v.clone()));

            let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

            // Gemma4 uses `self.scaling = 1.0` (no pre-attention scaling) —
            // unlike the standard `1/sqrt(head_dim)` transformer attention
            // scale, per upstream llama.cpp's
            // `hparams.f_attention_scale = 1.0f`.
            let mut attn_weights = q.matmul(&k.transpose(2, 3)?)?;

            if let Some(mask) = mask {
                let mask = mask.broadcast_as(attn_weights.shape())?;
                let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
                attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;

            let attn_output = attn_output
                .transpose(1, 2)?
                .reshape((b_sz, seq_len, self.q_dim))?;

            self.attention_wo.forward(&attn_output)
        }
    }

    /// Per-layer attention configuration derived from the `gemma4.*`
    /// per-layer arrays (`attention.head_count_kv` and
    /// `attention.sliding_window_pattern`).
    struct LayerConfig {
        n_kv_head: usize,
        head_dim: usize,
        sliding_window_size: Option<usize>,
        rope_freq: f32,
    }

    /// Reads a required `u32` scalar from `gemma4.{suffix}` metadata.
    fn md_u32(ct: &gguf_file::Content, suffix: &str) -> Result<u32> {
        let key = format!("gemma4.{suffix}");
        ct.metadata
            .get(&key)
            .with_context(|| format!("cannot find {key} in metadata"))?
            .to_u32()
    }

    /// Reads a required `f32` scalar from `gemma4.{suffix}` metadata.
    fn md_f32(ct: &gguf_file::Content, suffix: &str) -> Result<f32> {
        let key = format!("gemma4.{suffix}");
        ct.metadata
            .get(&key)
            .with_context(|| format!("cannot find {key} in metadata"))?
            .to_f32()
    }

    /// Converts a GGUF metadata array element to `u32`, accepting any
    /// integral type (`U8`..`I64`) or `Bool` (`true` -> 1, `false` -> 0).
    ///
    /// Real `gemma4` GGUF exports store `attention.head_count_kv` as `I32`
    /// and `attention.sliding_window_pattern` as `Bool`, neither of which
    /// `gguf_file::Value::to_u32` accepts directly.
    fn value_to_u32(v: &gguf_file::Value) -> Result<u32> {
        let as_i64 = match v {
            gguf_file::Value::U8(v) => i64::from(*v),
            gguf_file::Value::I8(v) => i64::from(*v),
            gguf_file::Value::U16(v) => i64::from(*v),
            gguf_file::Value::I16(v) => i64::from(*v),
            gguf_file::Value::U32(v) => i64::from(*v),
            gguf_file::Value::I32(v) => i64::from(*v),
            gguf_file::Value::U64(v) => i64::try_from(*v).map_err(candle_core::Error::wrap)?,
            gguf_file::Value::I64(v) => *v,
            gguf_file::Value::Bool(v) => i64::from(*v),
            other => candle_core::bail!("expected integral or bool value, got {other:?}"),
        };
        u32::try_from(as_i64)
            .map_err(|_| candle_core::Error::msg(format!("value {as_i64} out of range for u32")))
    }

    /// Dequantizes a `[1]`-shaped GGUF tensor (e.g. `blk.N.layer_output_scale.weight`)
    /// to a single `f32` scalar.
    fn dequantize_scalar_f32(tensor: QTensor, device: &Device) -> Result<f32> {
        let values = tensor.dequantize(device)?.flatten_all()?.to_vec1::<f32>()?;
        match values.as_slice() {
            [value] => Ok(*value),
            other => candle_core::bail!(
                "expected a single-element tensor, got {} elements",
                other.len()
            ),
        }
    }

    /// Applies an unweighted RMS normalization (`x / sqrt(mean(x^2) + eps)`)
    /// over the last dimension of `x`, with no learned scale.
    ///
    /// gemma4 applies this to the value projection on every layer (see
    /// `ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps)` in upstream
    /// llama.cpp), in contrast to the key projection which uses the learned
    /// `attn_k_norm` weight.
    fn rms_norm_unweighted(x: &Tensor, eps: f64) -> Result<Tensor> {
        let hidden_size = x.dim(D::Minus1)?;
        let x32 = x.to_dtype(DType::F32)?;
        let norm_x = (x32.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x32.broadcast_div(&(norm_x + eps)?.sqrt()?)?;
        x_normed.to_dtype(x.dtype())
    }

    /// Applies a `tanh`-based softcap to the final logits:
    /// `logits = tanh(logits / cap) * cap`.
    ///
    /// Matches upstream llama.cpp's `hparams.f_final_logit_softcapping`
    /// handling (`ggml_scale` by `1/cap`, `ggml_tanh`, `ggml_scale` by
    /// `cap`), which bounds the final logits to `(-cap, cap)`.
    fn apply_final_logit_softcapping(logits: &Tensor, softcap: f32) -> Result<Tensor> {
        let softcap = f64::from(softcap);
        (logits / softcap)?.tanh()? * softcap
    }

    /// Reads a required `gemma4.{suffix}` array with exactly `expected_len`
    /// elements, converting each element to `u32` via [`value_to_u32`].
    fn md_u32_array(
        ct: &gguf_file::Content,
        suffix: &str,
        expected_len: usize,
    ) -> Result<Vec<u32>> {
        let key = format!("gemma4.{suffix}");
        let values = ct
            .metadata
            .get(&key)
            .with_context(|| format!("cannot find {key} in metadata"))?
            .to_vec()
            .with_context(|| format!("{key} is not an array"))?;
        if values.len() != expected_len {
            candle_core::bail!(
                "{key} has {} elements, expected {expected_len} (one per layer)",
                values.len()
            );
        }
        values.iter().map(value_to_u32).collect()
    }

    #[derive(Debug, Clone)]
    pub struct ModelWeights {
        tok_embeddings: Embedding,
        embedding_length: usize,
        layers: Vec<LayerWeights>,
        norm: RmsNorm,
        output: QMatMul,
        /// `tanh`-based softcap applied to the final logits
        /// (`gemma4.final_logit_softcapping`), e.g. `logits =
        /// tanh(logits / cap) * cap`. Absent in synthetic/test GGUFs and some
        /// exports, in which case the final logits are left unscaled.
        final_logit_softcapping: Option<f32>,
        span: tracing::Span,
        span_output: tracing::Span,
    }

    impl ModelWeights {
        /// Constructs gemma4 quantized model weights from GGUF `Content`.
        ///
        /// Unlike `quantized_gemma3::ModelWeights::from_gguf`, hyperparameters
        /// are read directly from the `gemma4.*` namespace and per-layer
        /// attention configuration (KV head count, head dimension, sliding
        /// window, RoPE frequency) is derived from the per-layer
        /// `gemma4.attention.head_count_kv` and
        /// `gemma4.attention.sliding_window_pattern` arrays.
        ///
        /// # Errors
        ///
        /// Returns an error if required `gemma4.*` metadata keys are missing,
        /// the per-layer arrays have the wrong length, or any tensor cannot be
        /// loaded from the GGUF content.
        pub fn from_gguf<R: std::io::Seek + std::io::Read>(
            ct: gguf_file::Content,
            reader: &mut R,
            device: &Device,
        ) -> Result<Self> {
            let head_count = md_u32(&ct, "attention.head_count")? as usize;
            let block_count = md_u32(&ct, "block_count")? as usize;
            let embedding_length = md_u32(&ct, "embedding_length")? as usize;
            let key_length = md_u32(&ct, "attention.key_length")? as usize;
            let key_length_swa = md_u32(&ct, "attention.key_length_swa")? as usize;
            let rms_norm_eps = f64::from(md_f32(&ct, "attention.layer_norm_rms_epsilon")?);
            let sliding_window = md_u32(&ct, "attention.sliding_window")? as usize;
            let rope_freq_base = md_f32(&ct, "rope.freq_base")?;
            let rope_freq_base_swa = md_f32(&ct, "rope.freq_base_swa")?;

            let head_count_kv = md_u32_array(&ct, "attention.head_count_kv", block_count)?;
            let sliding_window_pattern =
                md_u32_array(&ct, "attention.sliding_window_pattern", block_count)?;

            let layer_configs: Vec<LayerConfig> = (0..block_count)
                .map(|i| {
                    let is_sliding = sliding_window_pattern[i] == 1;
                    if is_sliding {
                        LayerConfig {
                            n_kv_head: head_count_kv[i] as usize,
                            head_dim: key_length_swa,
                            sliding_window_size: Some(sliding_window),
                            rope_freq: rope_freq_base_swa,
                        }
                    } else {
                        LayerConfig {
                            n_kv_head: head_count_kv[i] as usize,
                            head_dim: key_length,
                            sliding_window_size: None,
                            rope_freq: rope_freq_base,
                        }
                    }
                })
                .collect();

            let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

            let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
            let tok_embeddings = tok_embeddings.dequantize(device)?;
            let norm = RmsNorm::from_qtensor(
                ct.tensor(reader, "output_norm.weight", device)?,
                rms_norm_eps,
            )?;
            let output = match ct.tensor(reader, "output.weight", device) {
                Ok(tensor) => tensor,
                Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
            };

            // Shared rope frequency-scaling factors for global (non-SWA)
            // layers, applied per dimension-pair (see `RotaryEmbedding::new`).
            // Absent in synthetic/test GGUFs and some exports, in which case
            // global layers fall back to unscaled RoPE.
            let rope_freqs: Option<Vec<f32>> = ct
                .tensor(reader, "rope_freqs.weight", device)
                .ok()
                .map(|tensor| tensor.dequantize(device)?.flatten_all()?.to_vec1::<f32>())
                .transpose()?;

            // `tanh`-based softcap applied to the final logits
            // (`logits = tanh(logits / cap) * cap`), per upstream
            // llama.cpp's `hparams.f_final_logit_softcapping`. Absent in
            // synthetic/test GGUFs and some exports, in which case the
            // final logits are left unscaled.
            let final_logit_softcapping = md_f32(&ct, "final_logit_softcapping").ok();

            let mut layers = Vec::with_capacity(block_count);
            for (layer_idx, layer_config) in layer_configs.into_iter().enumerate() {
                let prefix = format!("blk.{layer_idx}");

                let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
                let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
                // Global (non-sliding) layers have no `attn_v.weight` tensor in the
                // GGUF at all; `forward_attn` reuses the raw key projection output
                // as `Vcur` in that case, matching upstream llama.cpp.
                let attention_wv = ct
                    .tensor(reader, &format!("{prefix}.attn_v.weight"), device)
                    .ok();
                let attention_wo =
                    ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

                let attention_q_norm = RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                    rms_norm_eps,
                )?;
                let attention_k_norm = RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                    rms_norm_eps,
                )?;
                let attention_norm = RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                    rms_norm_eps,
                )?;
                let post_attention_norm = RmsNorm::from_qtensor(
                    ct.tensor(
                        reader,
                        &format!("{prefix}.post_attention_norm.weight"),
                        device,
                    )?,
                    rms_norm_eps,
                )?;
                let ffn_norm = RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                    rms_norm_eps,
                )?;
                let post_ffn_norm = RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                    rms_norm_eps,
                )?;

                let feed_forward_gate =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_up =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                let feed_forward_down =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;

                let mlp = Mlp {
                    feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                    feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                    feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
                };

                let freq_factors = layer_config
                    .sliding_window_size
                    .is_none()
                    .then_some(rope_freqs.as_deref())
                    .flatten();
                let rotary_embedding = RotaryEmbedding::new(
                    layer_config.head_dim,
                    layer_config.rope_freq,
                    freq_factors,
                    device,
                )?;

                let layer_output_scale = dequantize_scalar_f32(
                    ct.tensor(
                        reader,
                        &format!("{prefix}.layer_output_scale.weight"),
                        device,
                    )?,
                    device,
                )?;

                let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
                let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

                layers.push(LayerWeights {
                    attention_wq: QMatMul::from_qtensor(attention_wq)?,
                    attention_wk: QMatMul::from_qtensor(attention_wk)?,
                    attention_wv: attention_wv.map(QMatMul::from_qtensor).transpose()?,
                    attention_wo: QMatMul::from_qtensor(attention_wo)?,
                    attention_q_norm,
                    attention_k_norm,
                    attention_norm,
                    post_attention_norm,
                    ffn_norm,
                    post_ffn_norm,
                    mlp,
                    n_head: head_count,
                    n_kv_head: layer_config.n_kv_head,
                    head_dim: layer_config.head_dim,
                    q_dim: head_count * layer_config.head_dim,
                    sliding_window_size: layer_config.sliding_window_size,
                    rotary_embedding,
                    neg_inf: neg_inf.clone(),
                    rms_norm_eps,
                    layer_output_scale,
                    kv_cache: None,
                    span_attn,
                    span_mlp,
                });
            }

            let span = tracing::span!(tracing::Level::TRACE, "model");
            let span_output = tracing::span!(tracing::Level::TRACE, "output");

            Ok(Self {
                tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
                embedding_length,
                layers,
                norm,
                output: QMatMul::from_qtensor(output)?,
                final_logit_softcapping,
                span,
                span_output,
            })
        }

        /// Resets the key/value cache for every layer.
        ///
        /// Call this before starting generation for a new prompt so that stale
        /// cached keys/values from a previous request are not reused.
        pub fn clear_kv_cache(&mut self) {
            for layer in &mut self.layers {
                layer.kv_cache = None;
            }
        }

        /// Runs a forward pass over `x` (shape `(batch, seq_len)` token ids),
        /// returning logits for the final position with shape
        /// `(batch, vocab_size)`.
        ///
        /// # Errors
        ///
        /// Returns an error if any tensor operation fails.
        pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
            let (b_sz, seq_len) = x.dims2()?;
            let _enter = self.span.enter();

            let mut layer_in = self.tok_embeddings.forward(x)?;
            layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

            for layer in &mut self.layers {
                let attention_mask = if seq_len == 1 {
                    None
                } else {
                    Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?)
                };

                let residual = &layer_in;
                let x = layer.attention_norm.forward(&layer_in)?;
                let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
                let x = layer.post_attention_norm.forward(&x)?;
                let x = (x + residual)?;

                let _enter = layer.span_mlp.enter();
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let x = layer.mlp.forward(&x)?;
                let x = layer.post_ffn_norm.forward(&x)?;
                let x = (x + residual)?;
                drop(_enter);

                // `layer_output_scale` scales this block's entire output
                // (i.e. the hidden state handed to the next layer), matching
                // upstream llama.cpp's `cur = ggml_mul(cur, out_scale); inpL
                // = cur;` applied after the feed-forward residual add.
                layer_in = (x * f64::from(layer.layer_output_scale))?;
            }

            let _enter = self.span_output.enter();

            let x = layer_in.i((.., seq_len - 1, ..))?;
            let x = self.norm.forward(&x)?;
            let output = self.output.forward(&x)?;
            let output = match self.final_logit_softcapping {
                Some(softcap) if softcap != 0.0 => apply_final_logit_softcapping(&output, softcap)?,
                _ => output,
            };

            Ok(output)
        }
    }

    #[cfg(test)]
    mod softcap_tests {
        use super::apply_final_logit_softcapping;
        use candle_core::{Device, Tensor};

        #[test]
        fn softcapping_bounds_logits_to_plus_minus_cap() {
            let device = Device::Cpu;
            let logits = Tensor::new(&[60.0f32, -60.0, 0.0], &device).expect("logits tensor");

            let capped = apply_final_logit_softcapping(&logits, 30.0)
                .expect("softcapping should succeed")
                .to_vec1::<f32>()
                .expect("capped logits");

            assert!(
                capped.iter().all(|v| v.abs() < 30.0),
                "softcapped logits {capped:?} should be strictly within (-30, 30)"
            );
            assert!(
                (capped[2] - 0.0).abs() < 1e-6,
                "tanh(0) * cap should be 0, got {}",
                capped[2]
            );
        }
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
/// Loads the generation tokenizer for `artifacts`, along with the BOS token id
/// (if any) that [`RealCandleDecoder::generate`] must manually prepend to the
/// generation prompt's `input_ids` via [`prepend_bos_if_configured`].
///
/// `encode(prompt, false)`, used for the generation prompt, bypasses the
/// tokenizer's post-processor, so a GGUF-configured `add_bos_token = true`
/// would otherwise have no effect. The safetensors path returns `None` here
/// unchanged — its `tokenizer.json` is not inspected for this metadata, which
/// keeps this a no-op for the existing Qwen3/other safetensors-backed paths.
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn load_generation_tokenizer(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<(tokenizers::tokenizer::Tokenizer, Option<u32>)> {
    match artifacts.model_format {
        NativeModelFormat::Safetensors => {
            let tokenizer_path = safetensors_artifact_dir(model_path).join("tokenizer.json");
            let tokenizer = tokenizers::tokenizer::Tokenizer::from_file(&tokenizer_path)
                .map_err(|err| anyhow::anyhow!("failed to load tokenizer.json: {err}"))?;
            Ok((tokenizer, None))
        }
        NativeModelFormat::Gguf => {
            let mut file = fs::File::open(model_path)
                .with_context(|| "failed to open GGUF tokenizer metadata")?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                .with_context(|| "failed to read GGUF tokenizer metadata")?;
            let tokenizer = tokenizer_from_gguf_content(&content)?;
            let bos_token_id = gguf_bos_token_to_prepend(&content);
            Ok((tokenizer, bos_token_id))
        }
        NativeModelFormat::Unknown => bail!("native artifact format is unsupported"),
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn is_eos_token(tokenizer: &tokenizers::tokenizer::Tokenizer, token_id: u32) -> bool {
    tokenizer
        .id_to_token(token_id)
        .map(|token| matches!(token.as_str(), "</s>" | "<|endoftext|>" | "<end_of_turn>"))
        .unwrap_or(false)
}

pub fn validate_native_engine_load_plan(plan: &NativeEngineLoadPlan) -> Result<()> {
    if plan.runtime != RuntimeBackend::CandleNative {
        bail!("native load plan runtime must be candle-native");
    }
    if plan.alias.trim().is_empty() {
        bail!("native load plan has an empty model alias");
    }
    if !(0.0..=1.0).contains(&plan.budget_fraction) {
        bail!(
            "native load plan budget_fraction must be between 0.0 and 1.0, got {}",
            plan.budget_fraction
        );
    }
    if plan.engine != plan.candle.engine || plan.engine != plan.support.engine {
        bail!("native load plan engine does not match Candle support metadata");
    }
    if plan.candle.load_contract.model_family != plan.support.model_family {
        bail!("native load plan family does not match Candle support metadata");
    }
    if plan.family != plan.candle.load_contract.model_family.as_str() {
        bail!("native load plan family string does not match Candle load contract");
    }
    if plan.format != plan.candle.load_contract.model_format {
        bail!("native load plan format does not match Candle load contract");
    }
    if plan.acceleration != plan.candle.load_contract.accelerator {
        bail!("native load plan acceleration does not match Candle load contract");
    }
    if plan.device_selection != plan.candle.load_contract.device_selection {
        bail!("native load plan device selection does not match Candle load contract");
    }
    if !plan.support.supported_formats.contains(&plan.format) {
        bail!(
            "{} does not support model format {:?}",
            plan.engine,
            plan.format
        );
    }
    if !plan
        .support
        .supported_accelerators
        .contains(&plan.acceleration)
    {
        bail!(
            "{} does not support acceleration {:?}",
            plan.engine,
            plan.acceleration
        );
    }
    if plan.candle.load_contract.tokenizer == CandleTokenizerRequirement::UnsupportedFormat {
        bail!("native load plan has no tokenizer contract for model format");
    }
    if plan.candle.load_contract.supported_operations.is_empty() {
        bail!("native load plan has no supported Candle operations");
    }
    if !plan.implemented && !plan.candle.load_contract.fail_closed {
        bail!("unimplemented native load plan must fail closed");
    }
    if plan.scheduler.contract_only {
        bail!("native scheduler contract must report implemented FIFO queue runtime");
    }
    if plan.scheduler.queue.discipline != NativeQueueDiscipline::Fifo
        || !plan.scheduler.queue.implemented
        || !plan.scheduler.batching.prefill_decode_phase_scheduling
        || plan.scheduler.batching.implemented
        || plan.scheduler.kv_cache.reuse_implemented
        || plan.scheduler.kv_cache.implemented
        || !plan.scheduler.cancellation.admission_check_implemented
        || plan.scheduler.cancellation.decode_loop_check_implemented
        || plan.scheduler.cancellation.implemented
    {
        bail!(
            "native scheduler must implement FIFO queue and phase metadata while continuous batching, KV-cache reuse, and decode cancellation remain explicit unsupported runtime boundaries"
        );
    }

    Ok(())
}

pub fn validate_candle_model_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
) -> Result<CandleArtifactValidation> {
    let support = CandleFamilySupportMetadata::for_family(family);
    let format = infer_native_artifact_format(&model.path);
    if !support.supported_formats.contains(&format) {
        bail!(
            "{} cannot load model alias '{}' because the artifact format is unsupported; expected a .gguf file or safetensors weights with tokenizer.json and config.json",
            family.engine_name(),
            model.alias
        );
    }

    let layout = CandleArtifactLayout::for_format(format);
    match format {
        NativeModelFormat::Gguf => validate_gguf_artifacts(family, model, layout),
        NativeModelFormat::Safetensors => validate_safetensors_artifacts(family, model, layout),
        NativeModelFormat::Unknown => unreachable!("unsupported formats are rejected above"),
    }
}

fn validate_gguf_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    layout: CandleArtifactLayout,
) -> Result<CandleArtifactValidation> {
    let mut missing = Vec::new();
    if !model.path.is_file() || NativeModelFormat::from_path(&model.path) != NativeModelFormat::Gguf
    {
        missing.push("GGUF weights (*.gguf)".to_string());
    }

    fail_missing_artifacts(family, model, NativeModelFormat::Gguf, &missing)?;
    Ok(CandleArtifactValidation {
        model_family: family,
        model_format: NativeModelFormat::Gguf,
        layout,
        weight_files: vec![artifact_file_name(&model.path)],
        tokenizer_file: None,
        config_file: None,
    })
}

fn validate_safetensors_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    layout: CandleArtifactLayout,
) -> Result<CandleArtifactValidation> {
    let artifact_dir = safetensors_artifact_dir(&model.path);
    let weights = safetensors_weight_files(&model.path, artifact_dir);
    let tokenizer = artifact_dir.join("tokenizer.json");
    let config = artifact_dir.join("config.json");

    let mut missing = Vec::new();
    if weights.is_empty() {
        missing.push("safetensors weights (*.safetensors)".to_string());
    }
    if !tokenizer.is_file() {
        missing.push("tokenizer.json".to_string());
    }
    if !config.is_file() {
        missing.push("config.json".to_string());
    }

    fail_missing_artifacts(family, model, NativeModelFormat::Safetensors, &missing)?;
    Ok(CandleArtifactValidation {
        model_family: family,
        model_format: NativeModelFormat::Safetensors,
        layout,
        weight_files: weights,
        tokenizer_file: Some("tokenizer.json".to_string()),
        config_file: Some("config.json".to_string()),
    })
}

fn verify_candle_artifacts_can_load(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<()> {
    match artifacts.model_format {
        NativeModelFormat::Gguf => verify_gguf_can_load(model_path),
        NativeModelFormat::Safetensors => verify_safetensors_can_load(model_path, artifacts),
        NativeModelFormat::Unknown => bail!("native artifact format is unsupported"),
    }
}

#[cfg(feature = "native-candle")]
fn verify_gguf_can_load(model_path: &Path) -> Result<()> {
    let device = candle_core::Device::Cpu;
    candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_path, &device)
        .map(|_| ())
        .with_context(|| "failed to load GGUF weights with Candle")
}

#[cfg(not(feature = "native-candle"))]
fn verify_gguf_can_load(_model_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(feature = "native-candle")]
fn verify_safetensors_can_load(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<()> {
    let artifact_dir = safetensors_artifact_dir(model_path);
    let paths = artifacts
        .weight_files
        .iter()
        .map(|name| artifact_dir.join(name))
        .collect::<Vec<_>>();
    let device = candle_core::Device::Cpu;
    // Candle exposes safetensors mmap loading as unsafe because it relies on OS mmap invariants.
    // The paths come from validation immediately above and are only used for read-only weight access.
    unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::F32, &device)
    }
    .map(|_| ())
    .with_context(|| "failed to load safetensors weights with Candle")
}

#[cfg(not(feature = "native-candle"))]
fn verify_safetensors_can_load(
    _model_path: &Path,
    _artifacts: &CandleArtifactValidation,
) -> Result<()> {
    Ok(())
}

fn fail_missing_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    format: NativeModelFormat,
    missing: &[String],
) -> Result<()> {
    if missing.is_empty() {
        return Ok(());
    }

    bail!(
        "{} cannot load model alias '{}' as {:?}: missing required artifact(s): {}",
        family.engine_name(),
        model.alias,
        format,
        missing.join(", ")
    )
}

fn infer_native_artifact_format(path: &Path) -> NativeModelFormat {
    let format = NativeModelFormat::from_path(path);
    if format != NativeModelFormat::Unknown {
        return format;
    }

    if path.is_dir() && !safetensors_weight_files(path, path).is_empty() {
        return NativeModelFormat::Safetensors;
    }

    NativeModelFormat::Unknown
}

fn safetensors_artifact_dir(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn safetensors_weight_files(path: &Path, artifact_dir: &Path) -> Vec<String> {
    if path.is_file() && NativeModelFormat::from_path(path) == NativeModelFormat::Safetensors {
        return vec![artifact_file_name(path)];
    }

    let mut weights = fs::read_dir(artifact_dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && NativeModelFormat::from_path(path) == NativeModelFormat::Safetensors
        })
        .map(|path| artifact_file_name(&path))
        .collect::<Vec<_>>();
    weights.sort();
    weights
}

fn artifact_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unnamed>")
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePlacementPlan {
    pub routing_mode: String,
    pub local_node: String,
    pub nodes: Vec<NativePlacementNode>,
    pub unassigned_models: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePlacementNode {
    pub id: String,
    pub base_url: String,
    pub roles: Vec<String>,
    pub model_aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRouteSelection {
    pub query: String,
    pub candidates: Vec<NativePlacementNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeHeartbeat {
    pub node_id: String,
    pub runtime: RuntimeBackend,
    pub routing_mode: String,
    pub healthy: bool,
    pub models: usize,
    pub assigned_models: usize,
    pub unassigned_models: Vec<String>,
    pub budget_fraction: f64,
    pub heartbeat_interval_seconds: u64,
    pub telemetry_event: String,
}

impl NativeHeartbeat {
    pub fn safe_telemetry_attributes(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("cluster.node_id".to_string(), json_value(&self.node_id)),
            ("runtime.backend".to_string(), json_value(self.runtime)),
            (
                "runtime.routing_mode".to_string(),
                json_value(&self.routing_mode),
            ),
            ("runtime.healthy".to_string(), Value::Bool(self.healthy)),
            (
                "runtime.models".to_string(),
                Value::from(self.models as u64),
            ),
            (
                "runtime.assigned_models".to_string(),
                Value::from(self.assigned_models as u64),
            ),
            (
                "runtime.resource.budget_fraction".to_string(),
                Value::from(self.budget_fraction),
            ),
            (
                "runtime.heartbeat_interval_seconds".to_string(),
                Value::from(self.heartbeat_interval_seconds),
            ),
        ])
    }
}

pub fn heartbeat_from_config(cfg: &Config) -> NativeHeartbeat {
    let placement = placement_plan_from_config(cfg);
    let assigned_models = placement
        .nodes
        .iter()
        .map(|node| node.model_aliases.len())
        .sum();
    let healthy = validate_placement_plan(&placement).is_ok();
    NativeHeartbeat {
        node_id: cfg.cluster.node_id.clone(),
        runtime: cfg.runtime.backend,
        routing_mode: placement.routing_mode,
        healthy,
        models: cfg.models.len(),
        assigned_models,
        unassigned_models: placement.unassigned_models,
        budget_fraction: cfg.resources.budget,
        heartbeat_interval_seconds: cfg.runtime.heartbeat_interval_seconds,
        telemetry_event: "llmctl.runtime.heartbeat".to_string(),
    }
}

pub fn placement_plan_from_config(cfg: &Config) -> NativePlacementPlan {
    let nodes = if cfg.cluster.nodes.is_empty() {
        vec![NativePlacementNode {
            id: cfg.cluster.node_id.clone(),
            base_url: format!("http://{}:{}/v1", cfg.server.host, cfg.server.port),
            roles: sorted_roles(&cfg.models),
            model_aliases: cfg.models.iter().map(|model| model.alias.clone()).collect(),
        }]
    } else {
        cfg.cluster
            .nodes
            .iter()
            .map(|node| placement_node(node, &cfg.models))
            .collect()
    };

    let assigned = nodes
        .iter()
        .flat_map(|node| node.model_aliases.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let unassigned_models = cfg
        .models
        .iter()
        .filter(|model| !assigned.contains(&model.alias))
        .map(|model| model.alias.clone())
        .collect();

    NativePlacementPlan {
        routing_mode: if cfg.cluster.nodes.is_empty() {
            "single-node".to_string()
        } else {
            "cluster-role-placement".to_string()
        },
        local_node: cfg.cluster.node_id.clone(),
        nodes,
        unassigned_models,
    }
}

pub fn validate_placement_plan(plan: &NativePlacementPlan) -> Result<()> {
    if !plan.unassigned_models.is_empty() {
        bail!(
            "native placement leaves model aliases unassigned: {}",
            plan.unassigned_models.join(", ")
        );
    }

    let mut owners: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in &plan.nodes {
        if node.id.trim().is_empty() {
            bail!("native placement contains a node with an empty id");
        }
        if node.base_url.trim().is_empty() {
            bail!("native placement node '{}' has an empty base_url", node.id);
        }
        for alias in &node.model_aliases {
            owners
                .entry(alias.as_str())
                .or_default()
                .push(node.id.as_str());
        }
    }

    let duplicate = owners
        .iter()
        .find(|(_, node_ids)| node_ids.len() > 1)
        .map(|(alias, node_ids)| ((*alias).to_string(), node_ids.join(", ")));
    if let Some((alias, node_ids)) = duplicate {
        bail!("native placement assigns model alias '{alias}' to multiple nodes: {node_ids}");
    }

    Ok(())
}

pub fn route_selection_for_model(
    plan: &NativePlacementPlan,
    model_alias: &str,
) -> Result<NativeRouteSelection> {
    let candidates = plan
        .nodes
        .iter()
        .filter(|node| {
            node.model_aliases
                .iter()
                .any(|alias| alias.as_str() == model_alias)
        })
        .cloned()
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        bail!("native placement has no node for model alias '{model_alias}'");
    }
    if candidates.len() > 1 {
        bail!("native placement has multiple nodes for model alias '{model_alias}'");
    }

    Ok(NativeRouteSelection {
        query: format!("model:{model_alias}"),
        candidates,
    })
}

pub fn route_selection_for_role(
    plan: &NativePlacementPlan,
    role: &str,
) -> Result<NativeRouteSelection> {
    let normalized = normalize_role(role);
    let candidates = plan
        .nodes
        .iter()
        .filter(|node| node.roles.iter().any(|node_role| node_role == normalized))
        .cloned()
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        bail!("native placement has no node for role '{normalized}'");
    }

    Ok(NativeRouteSelection {
        query: format!("role:{normalized}"),
        candidates,
    })
}

fn json_value<T: Serialize>(value: T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

fn placement_node(node: &ClusterNodeConfig, models: &[ModelConfig]) -> NativePlacementNode {
    let explicit_aliases = node
        .model_aliases
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let role_set = node
        .roles
        .iter()
        .map(|role| normalize_role(role).to_string())
        .collect::<std::collections::BTreeSet<_>>();
    let model_aliases = models
        .iter()
        .filter(|model| {
            explicit_aliases.contains(&model.alias)
                || role_set.contains(normalize_role(&model.role))
        })
        .map(|model| model.alias.clone())
        .collect();

    NativePlacementNode {
        id: node.id.clone(),
        base_url: node.base_url.clone(),
        roles: node
            .roles
            .iter()
            .map(|role| normalize_role(role).to_string())
            .collect(),
        model_aliases,
    }
}

fn sorted_roles(models: &[ModelConfig]) -> Vec<String> {
    models
        .iter()
        .map(|model| normalize_role(&model.role).to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

impl NativeEngineLoadPlan {
    pub fn safe_telemetry_attributes(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            (
                "runtime.backend".to_string(),
                Value::String("candle-native".to_string()),
            ),
            (
                "runtime.engine".to_string(),
                Value::String(self.engine.clone()),
            ),
            ("model.alias".to_string(), Value::String(self.alias.clone())),
            ("model.role".to_string(), Value::String(self.role.clone())),
            (
                "model.family".to_string(),
                Value::String(self.candle.load_contract.model_family.as_str().to_string()),
            ),
            (
                "model.format".to_string(),
                json_value(self.candle.load_contract.model_format),
            ),
            (
                "runtime.accelerator".to_string(),
                json_value(self.candle.load_contract.accelerator),
            ),
            (
                "runtime.tokenizer_requirement".to_string(),
                json_value(&self.candle.load_contract.tokenizer),
            ),
            (
                "runtime.implemented".to_string(),
                Value::Bool(self.implemented),
            ),
            (
                "runtime.scheduler.contract_only".to_string(),
                Value::Bool(self.scheduler.contract_only),
            ),
            (
                "runtime.fail_closed".to_string(),
                Value::Bool(self.candle.load_contract.fail_closed),
            ),
        ])
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen3CandleEngineLoader;

impl Qwen3CandleEngineLoader {
    pub fn plan(model: &ModelConfig, resources: &ResourceConfig) -> Result<NativeEngineLoadPlan> {
        NativeCandleEngineFactory::default().plan(CandleModelFamily::Qwen3, model, resources)
    }

    pub fn load(&self, plan: &NativeEngineLoadPlan) -> Result<Box<dyn NativeEngine>> {
        NativeCandleEngineFactory::default().load(plan)
    }
}

fn normalize_role(role: &str) -> &str {
    let role = role.trim();
    if STARTER_ROLES.contains(&role) {
        role
    } else {
        "query"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use futures_util::FutureExt;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};
    use tokio::time::{sleep, Duration};

    #[derive(Debug)]
    struct CountingTokenizer;

    impl NativeTokenCounter for CountingTokenizer {
        fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64> {
            Ok(messages
                .iter()
                .map(|message| message_content_text(message).split_whitespace().count() as u64)
                .sum())
        }

        fn count_text(&self, text: &str) -> Result<u64> {
            Ok(text.split_whitespace().count() as u64)
        }
    }

    struct BlockingSchedulerTestEngine {
        tx: mpsc::Sender<BTreeMap<String, Value>>,
        releases: AsyncMutex<Vec<oneshot::Receiver<()>>>,
    }

    impl BlockingSchedulerTestEngine {
        fn new(
            tx: mpsc::Sender<BTreeMap<String, Value>>,
            releases: Vec<oneshot::Receiver<()>>,
        ) -> Self {
            Self {
                tx,
                releases: AsyncMutex::new(releases),
            }
        }
    }

    impl NativeEngine for BlockingSchedulerTestEngine {
        fn model_alias(&self) -> &str {
            "qwen"
        }

        fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
            async move {
                self.tx
                    .send(request.metadata.clone())
                    .await
                    .expect("scheduler metadata captured");
                let release = self.releases.lock().await.pop().expect("release receiver");
                let _ = release.await;
                Ok(NativeChatResponse {
                    model: request.model,
                    content: "ok".to_string(),
                    tool_calls: None,
                    finish_reason: "stop".to_string(),
                    usage: NativeTokenUsage::new(1, 1),
                })
            }
            .boxed()
        }
    }

    fn scheduler_test_request(id: &str) -> NativeChatRequest {
        NativeChatRequest {
            model: "qwen".to_string(),
            messages: vec![NativeChatMessage {
                role: "user".to_string(),
                content: Some(Value::String(id.to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
            metadata: BTreeMap::from([("test.id".to_string(), Value::String(id.to_string()))]),
        }
    }

    #[tokio::test]
    async fn native_scheduler_runs_fifo_and_records_wait_metadata() {
        let (tx, mut rx) = mpsc::channel(3);
        let (first_release_tx, first_release_rx) = oneshot::channel();
        let (second_release_tx, second_release_rx) = oneshot::channel();
        let (third_release_tx, third_release_rx) = oneshot::channel();
        let engine = Arc::new(BlockingSchedulerTestEngine::new(
            tx,
            vec![third_release_rx, second_release_rx, first_release_rx],
        ));
        let scheduler = NativeSchedulerEngine::new(
            engine,
            NativeSchedulerConfig {
                max_concurrent_requests: 1,
                max_queued_requests: 2,
                max_batch_size: 1,
                max_batch_wait_ms: 0,
                kv_cache_budget_bytes: 0,
            },
        );

        let first = tokio::spawn({
            let scheduler = scheduler.clone();
            async move { scheduler.chat(scheduler_test_request("first")).await }
        });
        assert_eq!(
            rx.recv().await.expect("first request")["test.id"],
            Value::String("first".to_string())
        );

        let second = tokio::spawn({
            let scheduler = scheduler.clone();
            async move { scheduler.chat(scheduler_test_request("second")).await }
        });
        sleep(Duration::from_millis(5)).await;
        let third = tokio::spawn({
            let scheduler = scheduler.clone();
            async move { scheduler.chat(scheduler_test_request("third")).await }
        });
        sleep(Duration::from_millis(20)).await;
        first_release_tx.send(()).expect("release first");

        let second_metadata = rx.recv().await.expect("second request");
        assert_eq!(
            second_metadata["test.id"],
            Value::String("second".to_string())
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.discipline"],
            Value::String("fifo".to_string())
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.queue.implemented"],
            Value::Bool(true)
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.batching.continuous.implemented"],
            Value::Bool(false)
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.batching.phase_scheduling.implemented"],
            Value::Bool(true)
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.phase"],
            Value::String("prefill-then-decode".to_string())
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.kv_cache.reuse_implemented"],
            Value::Bool(false)
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.kv_cache.policy"],
            Value::String("request-local-reset".to_string())
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.cancellation.admission_check_implemented"],
            Value::Bool(true)
        );
        assert_eq!(
            second_metadata["llmctl.scheduler.cancellation.decode_loop_check_implemented"],
            Value::Bool(false)
        );
        assert!(
            second_metadata["llmctl.scheduler.queue_wait_ms"]
                .as_u64()
                .expect("wait ms")
                > 0
        );
        second_release_tx.send(()).expect("release second");

        let third_metadata = rx.recv().await.expect("third request");
        assert_eq!(
            third_metadata["test.id"],
            Value::String("third".to_string())
        );
        third_release_tx.send(()).expect("release third");

        first.await.expect("first join").expect("first response");
        second.await.expect("second join").expect("second response");
        third.await.expect("third join").expect("third response");
    }

    #[tokio::test]
    async fn native_scheduler_rejects_when_wait_queue_is_full() {
        let (tx, mut rx) = mpsc::channel(1);
        let (_release_tx, release_rx) = oneshot::channel();
        let scheduler = NativeSchedulerEngine::new(
            Arc::new(BlockingSchedulerTestEngine::new(tx, vec![release_rx])),
            NativeSchedulerConfig {
                max_concurrent_requests: 1,
                max_queued_requests: 0,
                max_batch_size: 1,
                max_batch_wait_ms: 0,
                kv_cache_budget_bytes: 0,
            },
        );

        let first = tokio::spawn({
            let scheduler = scheduler.clone();
            async move { scheduler.chat(scheduler_test_request("first")).await }
        });
        assert_eq!(
            rx.recv().await.expect("first request")["test.id"],
            Value::String("first".to_string())
        );

        let err = scheduler
            .chat(scheduler_test_request("second"))
            .await
            .expect_err("full scheduler queue rejects");
        assert!(err.to_string().contains("native scheduler queue is full"));

        first.abort();
    }

    #[tokio::test]
    async fn native_scheduler_rejects_cancelled_requests_before_admission() {
        let (tx, mut rx) = mpsc::channel(1);
        let scheduler = NativeSchedulerEngine::new(
            Arc::new(BlockingSchedulerTestEngine::new(tx, Vec::new())),
            NativeSchedulerConfig::default(),
        );
        let mut request = scheduler_test_request("cancelled");
        request
            .metadata
            .insert("llmctl.scheduler.cancelled".to_string(), Value::Bool(true));

        let err = scheduler
            .chat(request)
            .await
            .expect_err("cancelled scheduler request rejects");
        assert!(err
            .to_string()
            .contains("native scheduler request was cancelled before decode"));
        assert!(
            rx.try_recv().is_err(),
            "cancelled request must not reach engine"
        );
    }

    #[test]
    fn native_scheduler_contract_exposes_runtime_and_unsupported_boundaries() {
        let contract = NativeSchedulerContract::fifo_runtime();

        assert!(contract.queue.implemented);
        assert!(!contract.batching.implemented);
        assert!(!contract.batching.continuous_batching);
        assert!(contract.batching.prefill_decode_phase_scheduling);
        assert!(contract
            .batching
            .unsupported_reason
            .contains("continuous batching is not active"));
        assert!(!contract.kv_cache.implemented);
        assert!(!contract.kv_cache.reuse_implemented);
        assert_eq!(
            contract.kv_cache.cache_key_metadata_key,
            "llmctl.scheduler.kv_cache_key"
        );
        assert!(contract
            .kv_cache
            .unsupported_reason
            .contains("cross-request KV-cache reuse is disabled"));
        assert!(!contract.cancellation.implemented);
        assert!(contract.cancellation.admission_check_implemented);
        assert!(!contract.cancellation.decode_loop_check_implemented);
        assert_eq!(
            contract.cancellation.cancelled_metadata_key,
            "llmctl.scheduler.cancelled"
        );
    }

    #[test]
    fn native_usage_comes_from_token_counter_not_upstream_metadata() {
        let request = NativeChatRequest {
            model: "qwen-query".to_string(),
            messages: vec![
                NativeChatMessage {
                    role: "system".to_string(),
                    content: Some(Value::String("answer briefly".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                NativeChatMessage {
                    role: "user".to_string(),
                    content: Some(Value::String("hello native runtime".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            temperature: Some(0.2),
            max_tokens: Some(128),
            tools: None,
            tool_choice: None,
            metadata: BTreeMap::new(),
        };

        let usage = usage_from_native_tokens(&CountingTokenizer, &request, "native answer")
            .expect("usage is counted");

        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.total_tokens(), 7);
        assert_eq!(usage.accounting_mode, TokenAccountingMode::NativeExact);
    }

    #[test]
    fn native_usage_reports_estimated_mode_and_nonzero_counts() {
        let request = NativeChatRequest {
            model: "qwen-query".to_string(),
            messages: vec![
                NativeChatMessage {
                    role: "system".to_string(),
                    content: Some(Value::String("answer with operational detail".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                NativeChatMessage {
                    role: "user".to_string(),
                    content: Some(Value::String(
                        "summarize native tokenizer accounting status".to_string(),
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            temperature: None,
            max_tokens: Some(64),
            tools: None,
            tool_choice: None,
            metadata: BTreeMap::new(),
        };

        let usage = usage_from_native_tokens(
            &EstimatedNativeTokenCounter,
            &request,
            "native accounting is estimated until a tokenizer is wired",
        )
        .expect("estimated usage is counted");

        assert_eq!(usage.accounting_mode, TokenAccountingMode::Estimated);
        assert!(usage.input_tokens > 0);
        assert!(usage.output_tokens > 0);
        assert_eq!(
            usage.total_tokens(),
            usage.input_tokens + usage.output_tokens
        );
    }

    #[test]
    fn deterministic_native_embeddings_are_stable_and_marked_non_semantic() {
        let request = NativeEmbeddingRequest {
            model: "embed".to_string(),
            input: vec!["hello".to_string(), "world".to_string()],
            metadata: BTreeMap::new(),
        };

        let first = deterministic_native_embeddings(request.clone()).expect("first embeddings");
        let second = deterministic_native_embeddings(request).expect("second embeddings");

        assert_eq!(first.embeddings, second.embeddings);
        assert_eq!(first.embeddings.len(), 2);
        assert_eq!(
            first.embeddings[0].len(),
            DETERMINISTIC_EMBEDDING_DIMENSIONS
        );
        assert_ne!(first.embeddings[0], first.embeddings[1]);
        assert_eq!(first.backend, "deterministic-local-fallback");
        assert_eq!(first.status, "non-semantic-dev-fallback");
        assert!(!first.semantic);
        assert_eq!(first.usage.accounting_mode, TokenAccountingMode::Estimated);
        assert!(first.usage.input_tokens > 0);
        assert_eq!(first.usage.output_tokens, 0);
    }

    #[test]
    fn estimated_counter_is_deterministic_and_does_not_claim_exact_tokenization() {
        let messages = vec![NativeChatMessage {
            role: "user".to_string(),
            content: Some(Value::String("repeatable fallback accounting".to_string())),
            tool_calls: None,
            tool_call_id: None,
        }];

        let first = EstimatedNativeTokenCounter
            .count_chat_input(&messages)
            .expect("estimated count");
        let second = EstimatedNativeTokenCounter
            .count_chat_input(&messages)
            .expect("estimated count");

        assert_eq!(first, second);
        assert!(first > 0);
        assert_eq!(
            EstimatedNativeTokenCounter.accounting_mode(),
            TokenAccountingMode::Estimated
        );
    }

    #[test]
    fn canonical_native_chat_input_is_explicit_tokenizer_input() {
        let messages = vec![
            NativeChatMessage {
                role: "system".to_string(),
                content: Some(Value::String("answer briefly".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
            NativeChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        assert_eq!(
            canonical_native_chat_input(&messages),
            "<|system|>\nanswer briefly\n<|user|>\nhello\n"
        );
    }

    #[test]
    fn gemma_chat_input_renders_system_as_its_own_turn() {
        let messages = vec![
            NativeChatMessage {
                role: "system".to_string(),
                content: Some(Value::String("answer briefly".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
            NativeChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        assert_eq!(
            gemma_chat_input(&messages),
            "<|turn>system\nanswer briefly<turn|>\n\
             <|turn>user\nhello<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[test]
    fn gemma_chat_input_maps_assistant_role_to_model_and_appends_generation_cue() {
        let messages = vec![
            NativeChatMessage {
                role: "system".to_string(),
                content: Some(Value::String("be terse".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
            NativeChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("hi".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
            NativeChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::String("hello there".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
            NativeChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("how are you?".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        assert_eq!(
            gemma_chat_input(&messages),
            "<|turn>system\nbe terse<turn|>\n\
             <|turn>user\nhi<turn|>\n\
             <|turn>model\nhello there<turn|>\n\
             <|turn>user\nhow are you?<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[cfg(feature = "native-tokenizers")]
    #[test]
    fn tokenizers_counter_feature_exposes_native_accounting_api_shape() {
        use std::path::Path;

        fn assert_counter<T: NativeTokenCounter>() {}

        assert_counter::<TokenizersNativeTokenCounter>();
        let _loader = |path: &Path| TokenizersNativeTokenCounter::from_file(path);
        let _constructor: fn(tokenizers::Tokenizer) -> TokenizersNativeTokenCounter =
            TokenizersNativeTokenCounter::from_tokenizer;
    }

    #[test]
    fn qwen3_loader_plan_covers_starter_role_acceleration_and_safe_status() {
        let model = ModelConfig {
            alias: "qwen-coder".to_string(),
            path: PathBuf::from("/home/alice/models/qwen3-coder.safetensors"),
            role: "coding".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };
        let resources = ResourceConfig {
            budget: 0.8,
            cpu_only: false,
            gpu_vendor: "nvidia".to_string(),
            llama_server_bin: None,
        };

        let plan = Qwen3CandleEngineLoader::plan(&model, &resources).expect("plan validates");

        assert_eq!(plan.runtime, RuntimeBackend::CandleNative);
        assert_eq!(plan.engine, "candle-native-qwen3");
        assert_eq!(plan.alias, "qwen-coder");
        assert_eq!(plan.role, "coding");
        assert_eq!(plan.family, "qwen3");
        assert_eq!(plan.format, NativeModelFormat::Safetensors);
        assert_eq!(plan.acceleration, NativeAcceleration::NvidiaCuda);
        assert_eq!(
            plan.acceleration.compatible_gpu_vendor(),
            Some(GpuVendor::Nvidia)
        );
        assert_eq!(plan.candle.engine, "candle-native-qwen3");
        assert_eq!(
            plan.candle.load_contract.model_family,
            CandleModelFamily::Qwen3
        );
        assert_eq!(
            plan.candle.load_contract.model_format,
            NativeModelFormat::Safetensors
        );
        assert_eq!(
            plan.candle.load_contract.accelerator,
            NativeAcceleration::NvidiaCuda
        );
        assert_eq!(
            plan.device_selection,
            CandleDeviceSelectionContract {
                requested: NativeAcceleration::NvidiaCuda,
                selected: NativeAcceleration::NvidiaCuda,
                compatible_gpu_vendor: Some(GpuVendor::Nvidia),
                selection_reason: "resources.gpu_vendor requested NVIDIA CUDA execution"
                    .to_string(),
                fail_closed_if_unavailable: true,
            }
        );
        assert_eq!(
            plan.candle.load_contract.tokenizer,
            CandleTokenizerRequirement::TokenizerJson
        );
        assert_eq!(
            plan.candle.load_contract.supported_operations,
            vec![
                CandleSupportedOperation::ChatCompletion,
                CandleSupportedOperation::ChatTokenCounting,
                CandleSupportedOperation::CompletionTokenCounting,
            ]
        );
        assert_eq!(
            plan.candle.load_contract.candle_crates_required,
            vec![
                "candle-core".to_string(),
                "candle-nn".to_string(),
                "candle-transformers".to_string(),
                "tokenizers".to_string(),
            ]
        );
        assert!(!plan.candle.load_contract.fail_closed);
        assert!(plan.candle.is_supported());
        assert_eq!(plan.support.model_family, CandleModelFamily::Qwen3);
        assert_eq!(plan.support.engine, "candle-native-qwen3");
        validate_native_engine_load_plan(&plan).expect("plan contract validates");
        assert!(!plan.scheduler.contract_only);
        assert_eq!(plan.scheduler.queue.discipline, NativeQueueDiscipline::Fifo);
        assert!(plan.scheduler.queue.admission_backpressure);
        assert!(plan.scheduler.queue.implemented);
        assert!(!plan.scheduler.batching.continuous_batching);
        assert!(plan.scheduler.batching.prefill_decode_phase_scheduling);
        assert!(!plan.scheduler.batching.implemented);
        assert_eq!(plan.scheduler.kv_cache.cache_scope, "model-worker");
        assert_eq!(
            plan.scheduler.kv_cache.cache_key_metadata_key,
            "llmctl.scheduler.kv_cache_key"
        );
        assert!(!plan.scheduler.kv_cache.reuse_implemented);
        assert!(!plan.scheduler.kv_cache.implemented);
        assert!(plan.scheduler.cancellation.drain_on_cancel);
        assert!(plan.scheduler.cancellation.admission_check_implemented);
        assert!(!plan.scheduler.cancellation.decode_loop_check_implemented);
        assert!(!plan.scheduler.cancellation.implemented);
        assert_eq!(plan.budget_fraction, 0.8);
        assert!(plan.implemented);

        let rendered = serde_json::to_string(&plan).expect("plan serializes");
        assert!(rendered.contains("llmctl.scheduler.kv_cache_budget_bytes"));
        assert!(rendered.contains("llmctl.scheduler.kv_cache_key"));
        assert!(rendered.contains("request-local-reset"));
        assert!(rendered.contains("\"contract_only\":false"));
        assert!(!rendered.contains("/home/alice"));
        assert!(!rendered.contains("qwen3-coder.safetensors"));
        assert!(plan
            .safe_telemetry_attributes()
            .values()
            .all(|value| !value.to_string().contains("/home/alice")));
    }

    #[test]
    fn qwen3_contract_distinguishes_gguf_safetensors_and_unknown_formats() {
        let gguf = CandleEngineConfig::qwen3(NativeModelFormat::Gguf, NativeAcceleration::Cpu);
        assert_eq!(
            gguf.load_contract.tokenizer,
            CandleTokenizerRequirement::GgufMetadata
        );
        assert!(gguf
            .load_contract
            .supported_operations
            .contains(&CandleSupportedOperation::ChatCompletion));
        assert!(gguf.is_supported());

        let safetensors =
            CandleEngineConfig::qwen3(NativeModelFormat::Safetensors, NativeAcceleration::Auto);
        assert_eq!(
            safetensors.load_contract.tokenizer,
            CandleTokenizerRequirement::TokenizerJson
        );
        assert!(safetensors
            .load_contract
            .supported_operations
            .contains(&CandleSupportedOperation::ChatTokenCounting));
        assert!(safetensors.is_supported());

        let unknown =
            CandleEngineConfig::qwen3(NativeModelFormat::Unknown, NativeAcceleration::Auto);
        assert_eq!(
            unknown.load_contract.tokenizer,
            CandleTokenizerRequirement::UnsupportedFormat
        );
        assert!(unknown.load_contract.supported_operations.is_empty());
        assert!(!unknown.is_supported());
        assert!(unknown.load_contract.fail_closed);
    }

    #[test]
    fn candle_contract_includes_gemma_mistral_and_tracked_target_families() {
        let gemma4 =
            CandleEngineConfig::gemma4(NativeModelFormat::Safetensors, NativeAcceleration::Auto);
        assert_eq!(gemma4.engine, "candle-native-gemma4");
        assert_eq!(gemma4.load_contract.model_family, CandleModelFamily::Gemma4);
        assert_eq!(
            gemma4.load_contract.tokenizer,
            CandleTokenizerRequirement::TokenizerJson
        );
        assert!(gemma4.is_supported());
        assert!(!gemma4.load_contract.fail_closed);

        let deepseek =
            CandleEngineConfig::deepseek(NativeModelFormat::Safetensors, NativeAcceleration::Auto);
        assert_eq!(deepseek.engine, "candle-native-deepseek");
        assert_eq!(
            deepseek.load_contract.model_family,
            CandleModelFamily::DeepSeek
        );
        assert!(deepseek.is_supported());
        assert!(!deepseek.load_contract.fail_closed);
        assert!(deepseek
            .load_contract
            .fail_closed_reason
            .contains("deepseek2::DeepSeekV2"));

        let deepseek_gguf =
            CandleEngineConfig::deepseek(NativeModelFormat::Gguf, NativeAcceleration::Auto);
        assert!(!deepseek_gguf.is_supported());
        assert!(deepseek_gguf.load_contract.fail_closed);
        assert!(deepseek_gguf.load_contract.supported_operations.is_empty());
        assert_eq!(
            deepseek_gguf.load_contract.tokenizer,
            CandleTokenizerRequirement::UnsupportedFormat
        );
        assert!(deepseek_gguf
            .load_contract
            .fail_closed_reason
            .contains("GGUF/quantized DeepSeek"));

        let kimi =
            CandleEngineConfig::kimi(NativeModelFormat::Safetensors, NativeAcceleration::Auto);
        assert_eq!(kimi.engine, "candle-native-kimi");
        assert_eq!(kimi.load_contract.model_family, CandleModelFamily::Kimi);
        assert_eq!(
            kimi.load_contract.tokenizer,
            CandleTokenizerRequirement::UnsupportedFormat
        );
        assert!(!kimi.is_supported());
        assert!(kimi.load_contract.fail_closed);
        assert!(kimi
            .load_contract
            .fail_closed_reason
            .contains("models::kimi"));

        let minimax =
            CandleEngineConfig::minimax(NativeModelFormat::Safetensors, NativeAcceleration::Auto);
        assert_eq!(minimax.engine, "candle-native-minimax");
        assert_eq!(
            minimax.load_contract.model_family,
            CandleModelFamily::MiniMax
        );
        assert!(!minimax.is_supported());
        assert!(minimax.load_contract.fail_closed);
        assert_eq!(
            minimax.load_contract.tokenizer,
            CandleTokenizerRequirement::UnsupportedFormat
        );
        assert!(minimax
            .load_contract
            .fail_closed_reason
            .contains("models::minimax"));

        let mistral =
            CandleEngineConfig::mistral(NativeModelFormat::Safetensors, NativeAcceleration::Cpu);
        assert_eq!(mistral.engine, "candle-native-mistral");
        assert_eq!(
            mistral.load_contract.model_family,
            CandleModelFamily::Mistral
        );
        assert_eq!(
            mistral.load_contract.tokenizer,
            CandleTokenizerRequirement::TokenizerJson
        );
        assert!(mistral.is_supported());
        assert!(mistral.load_contract.fail_closed_reason.contains("mistral"));
    }

    #[test]
    fn candle_artifact_layout_distinguishes_gguf_from_safetensors_sidecars() {
        let gguf = CandleArtifactLayout::for_format(NativeModelFormat::Gguf);
        assert_eq!(
            gguf.requirements,
            vec![CandleArtifactRequirement {
                kind: CandleArtifactKind::GgufWeights,
                filename: "*.gguf".to_string(),
                required: true,
            }]
        );

        let safetensors = CandleArtifactLayout::for_format(NativeModelFormat::Safetensors);
        assert_eq!(
            safetensors
                .requirements
                .iter()
                .map(|requirement| requirement.kind)
                .collect::<Vec<_>>(),
            vec![
                CandleArtifactKind::SafetensorsWeights,
                CandleArtifactKind::TokenizerJson,
                CandleArtifactKind::ConfigJson,
            ]
        );
    }

    #[test]
    fn candle_artifact_validation_accepts_real_gguf_weight_file_for_gguf_contracts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let weights = dir.path().join("chat.gguf");
        fs::write(&weights, b"GGUF").expect("write gguf placeholder");

        for family in [CandleModelFamily::Qwen3, CandleModelFamily::Gemma4] {
            let model = ModelConfig {
                alias: format!("{}-chat", family.as_str()),
                path: weights.clone(),
                role: "query".to_string(),
                family: Some("qwen3".to_string()),
                weight: 1,
            };

            let validation = validate_candle_model_artifacts(family, &model)
                .expect("gguf weight file validates");

            assert_eq!(validation.model_family, family);
            assert_eq!(validation.model_format, NativeModelFormat::Gguf);
            assert_eq!(validation.weight_files, vec!["chat.gguf".to_string()]);
            assert_eq!(validation.tokenizer_file, None);
            assert_eq!(validation.config_file, None);
        }
    }

    #[test]
    fn deepseek_gguf_fails_closed_without_claiming_quantized_support() {
        let factory = NativeCandleEngineFactory::default();
        let model = ModelConfig {
            alias: "deepseek-chat".to_string(),
            path: PathBuf::from("/private/deepseek-v2.gguf"),
            role: "thinking".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };

        let err = factory
            .plan(
                CandleModelFamily::DeepSeek,
                &model,
                &ResourceConfig::default(),
            )
            .expect_err("DeepSeek GGUF must not be planned as supported");
        assert!(err.to_string().contains("does not support model format"));

        let config = CandleEngineConfig::deepseek(NativeModelFormat::Gguf, NativeAcceleration::Cpu);
        assert_eq!(config.support.model_family, CandleModelFamily::DeepSeek);
        assert_eq!(
            config.support.supported_formats,
            vec![NativeModelFormat::Safetensors]
        );
        assert!(config.load_contract.fail_closed);
        assert!(config.load_contract.supported_operations.is_empty());
        assert_eq!(
            config.load_contract.tokenizer,
            CandleTokenizerRequirement::UnsupportedFormat
        );
        assert!(config
            .load_contract
            .fail_closed_reason
            .contains("GGUF/quantized DeepSeek"));
    }

    #[test]
    fn native_candle_decoder_rejects_unwired_family_formats_before_loading_artifacts() {
        for (family, format, reason) in [
            (
                CandleModelFamily::DeepSeek,
                NativeModelFormat::Gguf,
                "quantized DeepSeek2 model weights",
            ),
            (
                CandleModelFamily::Kimi,
                NativeModelFormat::Safetensors,
                "models::kimi",
            ),
            (
                CandleModelFamily::Kimi,
                NativeModelFormat::Gguf,
                "quantized Kimi GGUF model weights",
            ),
            (
                CandleModelFamily::MiniMax,
                NativeModelFormat::Safetensors,
                "models::minimax",
            ),
            (
                CandleModelFamily::MiniMax,
                NativeModelFormat::Gguf,
                "quantized MiniMax GGUF model weights",
            ),
        ] {
            let artifacts = CandleArtifactValidation {
                model_family: family,
                model_format: format,
                layout: CandleArtifactLayout::for_format(format),
                weight_files: vec![format!("{}.{}", family.as_str(), format.as_str())],
                tokenizer_file: None,
                config_file: None,
            };

            let err = NativeCandleDecoder::load(
                family,
                Path::new("/private/placeholder-model-file"),
                &artifacts,
            )
            .expect_err("unwired family/format must fail before artifact loading");
            assert!(err.to_string().contains(reason), "{err}");
        }
    }

    #[test]
    fn tracked_unwired_family_artifacts_do_not_validate_as_supported() {
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf = dir.path().join("model.gguf");
        fs::write(&gguf, b"GGUF").expect("write gguf placeholder");
        let safetensors = dir.path().join("model.safetensors");
        fs::write(&safetensors, b"weights").expect("write safetensors placeholder");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), b"{}").expect("write config");

        for family in [CandleModelFamily::Kimi, CandleModelFamily::MiniMax] {
            for path in [&gguf, &safetensors] {
                let model = ModelConfig {
                    alias: format!("{}-chat", family.as_str()),
                    path: path.clone(),
                    role: "thinking".to_string(),
                    family: Some("qwen3".to_string()),
                    weight: 1,
                };

                let err = validate_candle_model_artifacts(family, &model)
                    .expect_err("tracked family without decoder must not validate artifacts");
                assert!(err.to_string().contains("artifact format is unsupported"));
            }
        }
    }

    #[test]
    fn candle_artifact_validation_accepts_safetensors_weights_tokenizer_and_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            dir.path().join("model-00001-of-00002.safetensors"),
            b"weights",
        )
        .expect("write first shard");
        fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            b"weights",
        )
        .expect("write second shard");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), b"{}").expect("write config");

        let model = ModelConfig {
            alias: "deepseek-chat".to_string(),
            path: dir.path().to_path_buf(),
            role: "thinking".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };

        let validation = validate_candle_model_artifacts(CandleModelFamily::DeepSeek, &model)
            .expect("safetensors directory validates");

        assert_eq!(validation.model_family, CandleModelFamily::DeepSeek);
        assert_eq!(validation.model_format, NativeModelFormat::Safetensors);
        assert_eq!(
            validation.weight_files,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ]
        );
        assert_eq!(
            validation.tokenizer_file,
            Some("tokenizer.json".to_string())
        );
        assert_eq!(validation.config_file, Some("config.json".to_string()));
    }

    #[test]
    fn candle_artifact_validation_reports_missing_safetensors_sidecars_actionably() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("model.safetensors"), b"weights").expect("write weights");
        let model = ModelConfig {
            alias: "gemma-chat".to_string(),
            path: dir.path().join("model.safetensors"),
            role: "query".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };

        let err = validate_candle_model_artifacts(CandleModelFamily::Gemma4, &model)
            .expect_err("missing sidecars are rejected");
        let message = err.to_string();

        assert!(message.contains("candle-native-gemma4"));
        assert!(message.contains("gemma-chat"));
        assert!(message.contains("missing required artifact(s)"));
        assert!(message.contains("tokenizer.json"));
        assert!(message.contains("config.json"));
        assert!(!message.contains(dir.path().to_str().expect("utf8 temp path")));
    }

    #[test]
    fn candle_artifact_validation_reports_missing_weights_actionably() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), b"{}").expect("write config");
        let model = ModelConfig {
            alias: "mistral-chat".to_string(),
            path: dir.path().join("model.safetensors"),
            role: "query".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };

        let err = validate_candle_model_artifacts(CandleModelFamily::Mistral, &model)
            .expect_err("missing weights are rejected");
        let message = err.to_string();

        assert!(message.contains("candle-native-mistral"));
        assert!(message.contains("mistral-chat"));
        assert!(message.contains("safetensors weights (*.safetensors)"));
        assert!(!message.contains(dir.path().to_str().expect("utf8 temp path")));
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    fn native_candle_factory_rejects_incomplete_safetensors_model_before_serving() {
        let dir = tempfile::tempdir().expect("tempdir");
        let weights = dir.path().join("model.safetensors");
        let tensor =
            candle_core::Tensor::new(&[1f32, 2f32], &candle_core::Device::Cpu).expect("tensor");
        tensor.save_safetensors("dummy", &weights).expect("weights");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), b"{}").expect("write config");
        let model = ModelConfig {
            alias: "mistral-chat".to_string(),
            path: weights,
            role: "query".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };

        let plan = NativeCandleEngineFactory::default()
            .plan(
                CandleModelFamily::Mistral,
                &model,
                &ResourceConfig::default(),
            )
            .expect("plan");
        let err = match NativeCandleEngineFactory::default().load(&plan) {
            Ok(_) => panic!("dummy safetensors are not a complete Mistral model"),
            Err(err) => err,
        };
        let message = err.to_string();

        assert!(
            message.contains("failed to load native generation tokenizer")
                || message.contains("failed to parse model config.json"),
            "{message}"
        );
        assert!(!message.contains(dir.path().to_str().expect("utf8 temp path")));
    }

    #[test]
    fn native_candle_factory_registers_all_families_with_actionable_metadata() {
        let factory = NativeCandleEngineFactory::default();

        assert_eq!(
            factory.registered_families(),
            vec![
                CandleModelFamily::Qwen3,
                CandleModelFamily::Gemma4,
                CandleModelFamily::DeepSeek,
                CandleModelFamily::Kimi,
                CandleModelFamily::Mistral,
                CandleModelFamily::MiniMax,
            ]
        );

        for family in CandleModelFamily::all() {
            let metadata = factory
                .support_metadata(*family)
                .expect("family is registered");
            assert_eq!(metadata.model_family, *family);
            assert_eq!(metadata.engine, family.engine_name());
            if matches!(family, CandleModelFamily::Kimi | CandleModelFamily::MiniMax) {
                assert!(metadata.supported_formats.is_empty());
                assert!(metadata.tokenizer_contracts.is_empty());
            } else {
                assert!(metadata
                    .supported_formats
                    .contains(&NativeModelFormat::Safetensors));
            }
            assert!(metadata
                .supported_accelerators
                .contains(&NativeAcceleration::Cpu));
            if family.has_native_decoder() {
                assert!(metadata
                    .supported_operations
                    .contains(&CandleSupportedOperation::ChatCompletion));
            } else {
                assert!(metadata.supported_operations.is_empty());
            }
            assert_eq!(
                metadata.tokenizer_requirement(NativeModelFormat::Safetensors),
                if matches!(family, CandleModelFamily::Kimi | CandleModelFamily::MiniMax) {
                    CandleTokenizerRequirement::UnsupportedFormat
                } else {
                    CandleTokenizerRequirement::TokenizerJson
                }
            );
            assert!(metadata
                .generation_status
                .to_ascii_lowercase()
                .contains(family.as_str()));
        }
    }

    #[test]
    fn native_candle_factory_builds_valid_load_plans_for_runnable_families() {
        let factory = NativeCandleEngineFactory::default();
        let resources = ResourceConfig {
            budget: 0.7,
            cpu_only: true,
            gpu_vendor: "nvidia".to_string(),
            llama_server_bin: None,
        };

        for family in [
            CandleModelFamily::Qwen3,
            CandleModelFamily::Gemma4,
            CandleModelFamily::DeepSeek,
            CandleModelFamily::Mistral,
        ] {
            let model = ModelConfig {
                alias: format!("{}-chat", family.as_str()),
                path: PathBuf::from(format!("/private/{}-model.safetensors", family.as_str())),
                role: "thinking".to_string(),
                family: Some("qwen3".to_string()),
                weight: 1,
            };

            let plan = factory
                .plan(family, &model, &resources)
                .expect("registered family plans");

            assert_eq!(plan.runtime, RuntimeBackend::CandleNative);
            assert_eq!(plan.engine, family.engine_name());
            assert_eq!(plan.family, family.as_str());
            assert_eq!(plan.support.model_family, family);
            assert_eq!(plan.format, NativeModelFormat::Safetensors);
            assert_eq!(plan.acceleration, NativeAcceleration::Cpu);
            assert_eq!(plan.device_selection.selected, NativeAcceleration::Cpu);
            assert!(plan.device_selection.fail_closed_if_unavailable);
            assert!(plan.implemented);
            assert!(!plan.candle.load_contract.fail_closed);
            validate_native_engine_load_plan(&plan).expect("load plan validates");

            let rendered = serde_json::to_string(&plan).expect("plan serializes");
            assert!(!rendered.contains("/private"));
            assert!(!rendered.contains("-model.gguf"));
        }
    }

    #[test]
    fn native_candle_factory_fails_closed_for_tracked_unwired_families() {
        let factory = NativeCandleEngineFactory::default();

        for family in [CandleModelFamily::Kimi, CandleModelFamily::MiniMax] {
            for (format, path) in [
                (
                    NativeModelFormat::Safetensors,
                    PathBuf::from(format!("/private/{}-model.safetensors", family.as_str())),
                ),
                (
                    NativeModelFormat::Gguf,
                    PathBuf::from(format!("/private/{}-model.gguf", family.as_str())),
                ),
            ] {
                let model = ModelConfig {
                    alias: format!("{}-chat", family.as_str()),
                    path,
                    role: "thinking".to_string(),
                    family: Some("qwen3".to_string()),
                    weight: 1,
                };

                let err = factory
                    .plan(family, &model, &ResourceConfig::default())
                    .expect_err("unwired family fails closed");
                assert!(err.to_string().contains("does not support model format"));

                let config =
                    CandleEngineConfig::for_family(family, format, NativeAcceleration::Auto);
                assert!(config.load_contract.fail_closed);
                assert!(config.load_contract.supported_operations.is_empty());
                assert_eq!(
                    config.load_contract.tokenizer,
                    CandleTokenizerRequirement::UnsupportedFormat
                );
                assert!(config
                    .load_contract
                    .fail_closed_reason
                    .contains("Candle 0.10.2"));
            }
        }
    }

    #[test]
    fn native_candle_factory_rejects_unactionable_load_plans() {
        let factory = NativeCandleEngineFactory::default();
        let model = ModelConfig {
            alias: "qwen-unknown".to_string(),
            path: PathBuf::from("/private/qwen3.bin"),
            role: "thinking".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };

        let err = factory
            .plan(CandleModelFamily::Qwen3, &model, &ResourceConfig::default())
            .expect_err("unknown model format is rejected");
        assert!(err.to_string().contains("does not support model format"));

        let valid_model = ModelConfig {
            alias: "qwen-ok".to_string(),
            path: PathBuf::from("/private/qwen3.gguf"),
            role: "coding".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };
        let mut plan = factory
            .plan(
                CandleModelFamily::Qwen3,
                &valid_model,
                &ResourceConfig::default(),
            )
            .expect("valid plan");
        plan.engine = "candle-native-mistral".to_string();

        let err = validate_native_engine_load_plan(&plan).expect_err("mismatched engine rejected");
        assert!(err.to_string().contains("engine does not match"));
    }

    #[test]
    fn qwen3_loader_rejects_missing_artifacts_without_leaking_model_path() {
        let model = ModelConfig {
            alias: "qwen-thinking".to_string(),
            path: PathBuf::from("/secret/qwen3-thinking.gguf"),
            role: "thinking".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        };
        let plan = Qwen3CandleEngineLoader::plan(&model, &ResourceConfig::default())
            .expect("plan validates");

        let err = match Qwen3CandleEngineLoader.load(&plan) {
            Ok(_) => panic!("loader should reject missing artifacts"),
            Err(err) => err,
        };
        let message = err.to_string();

        assert!(message.contains("candle-native-qwen3"));
        assert!(message.contains("qwen-thinking"));
        assert!(!plan.candle.load_contract.fail_closed);
        assert_eq!(
            plan.candle.load_contract.tokenizer,
            CandleTokenizerRequirement::GgufMetadata
        );
        assert!(!message.contains("/secret"));
        assert!(!message.contains("qwen3-thinking.gguf"));
    }

    #[test]
    fn placement_plan_assigns_roles_across_two_servers_without_paths() {
        let mut cfg = Config::default();
        cfg.cluster.node_id = "server-a".to_string();
        cfg.cluster.nodes = vec![
            ClusterNodeConfig {
                id: "server-a".to_string(),
                base_url: "http://10.0.0.10:8765/v1".to_string(),
                roles: vec!["thinking".to_string(), "recommendation".to_string()],
                model_aliases: Vec::new(),
            },
            ClusterNodeConfig {
                id: "server-b".to_string(),
                base_url: "http://10.0.0.11:8765/v1".to_string(),
                roles: vec!["coding".to_string()],
                model_aliases: Vec::new(),
            },
        ];
        cfg.models = vec![
            ModelConfig {
                alias: "qwen-think".to_string(),
                path: PathBuf::from("/models/qwen-thinking.gguf"),
                role: "thinking".to_string(),
                family: Some("qwen3".to_string()),
                weight: 1,
            },
            ModelConfig {
                alias: "qwen-reco".to_string(),
                path: PathBuf::from("/models/qwen-reco.gguf"),
                role: "recommendation".to_string(),
                family: Some("qwen3".to_string()),
                weight: 1,
            },
            ModelConfig {
                alias: "qwen-code".to_string(),
                path: PathBuf::from("/models/qwen-code.gguf"),
                role: "coding".to_string(),
                family: Some("qwen3".to_string()),
                weight: 1,
            },
        ];

        let plan = placement_plan_from_config(&cfg);

        assert_eq!(plan.routing_mode, "cluster-role-placement");
        assert_eq!(plan.local_node, "server-a");
        assert_eq!(
            plan.nodes[0].model_aliases,
            vec!["qwen-think".to_string(), "qwen-reco".to_string()]
        );
        assert_eq!(plan.nodes[1].model_aliases, vec!["qwen-code".to_string()]);
        assert!(plan.unassigned_models.is_empty());

        let rendered = serde_json::to_string(&plan).expect("placement serializes");
        assert!(!rendered.contains("/models"));
        assert!(!rendered.contains(".gguf"));
    }

    #[test]
    fn placement_validation_rejects_unassigned_and_duplicate_models() {
        let unassigned = NativePlacementPlan {
            routing_mode: "cluster-role-placement".to_string(),
            local_node: "server-a".to_string(),
            nodes: vec![NativePlacementNode {
                id: "server-a".to_string(),
                base_url: "http://10.0.0.10:8765/v1".to_string(),
                roles: vec!["thinking".to_string()],
                model_aliases: vec!["qwen-think".to_string()],
            }],
            unassigned_models: vec!["qwen-code".to_string()],
        };
        let err = validate_placement_plan(&unassigned).expect_err("unassigned model rejected");
        assert!(err.to_string().contains("qwen-code"));

        let duplicate = NativePlacementPlan {
            routing_mode: "cluster-role-placement".to_string(),
            local_node: "server-a".to_string(),
            nodes: vec![
                NativePlacementNode {
                    id: "server-a".to_string(),
                    base_url: "http://10.0.0.10:8765/v1".to_string(),
                    roles: vec!["coding".to_string()],
                    model_aliases: vec!["qwen-code".to_string()],
                },
                NativePlacementNode {
                    id: "server-b".to_string(),
                    base_url: "http://10.0.0.11:8765/v1".to_string(),
                    roles: vec!["coding".to_string()],
                    model_aliases: vec!["qwen-code".to_string()],
                },
            ],
            unassigned_models: Vec::new(),
        };
        let err = validate_placement_plan(&duplicate).expect_err("duplicate model rejected");
        assert!(err.to_string().contains("multiple nodes"));
    }

    #[test]
    fn route_selection_returns_node_for_model_or_role() {
        let plan = NativePlacementPlan {
            routing_mode: "cluster-role-placement".to_string(),
            local_node: "server-a".to_string(),
            nodes: vec![
                NativePlacementNode {
                    id: "server-a".to_string(),
                    base_url: "http://10.0.0.10:8765/v1".to_string(),
                    roles: vec!["thinking".to_string(), "recommendation".to_string()],
                    model_aliases: vec!["qwen-think".to_string(), "qwen-reco".to_string()],
                },
                NativePlacementNode {
                    id: "server-b".to_string(),
                    base_url: "http://10.0.0.11:8765/v1".to_string(),
                    roles: vec!["coding".to_string()],
                    model_aliases: vec!["qwen-code".to_string()],
                },
            ],
            unassigned_models: Vec::new(),
        };

        let by_model = route_selection_for_model(&plan, "qwen-code").expect("model route");
        assert_eq!(by_model.query, "model:qwen-code");
        assert_eq!(by_model.candidates[0].id, "server-b");

        let by_role = route_selection_for_role(&plan, "thinking").expect("role route");
        assert_eq!(by_role.query, "role:thinking");
        assert_eq!(by_role.candidates[0].id, "server-a");
    }

    #[test]
    fn heartbeat_reports_single_or_cluster_health_without_paths() {
        let mut cfg = Config::default();
        cfg.cluster.node_id = "server-a".to_string();
        cfg.resources.budget = 0.8;
        cfg.models = vec![ModelConfig {
            alias: "qwen-code".to_string(),
            path: PathBuf::from("/private/qwen-code.gguf"),
            role: "coding".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        }];

        let single = heartbeat_from_config(&cfg);
        assert_eq!(single.node_id, "server-a");
        assert_eq!(single.routing_mode, "single-node");
        assert!(single.healthy);
        assert_eq!(single.models, 1);
        assert_eq!(single.assigned_models, 1);
        assert_eq!(single.budget_fraction, 0.8);
        assert_eq!(single.heartbeat_interval_seconds, 30);
        assert_eq!(single.telemetry_event, "llmctl.runtime.heartbeat");

        cfg.cluster.nodes = vec![ClusterNodeConfig {
            id: "server-b".to_string(),
            base_url: "http://10.0.0.11:8765/v1".to_string(),
            roles: vec!["thinking".to_string()],
            model_aliases: Vec::new(),
        }];
        let cluster = heartbeat_from_config(&cfg);
        assert_eq!(cluster.routing_mode, "cluster-role-placement");
        assert!(!cluster.healthy);
        assert_eq!(cluster.unassigned_models, vec!["qwen-code".to_string()]);

        let rendered = serde_json::to_string(&cluster).expect("heartbeat serializes");
        assert!(!rendered.contains("/private"));
        assert!(!rendered.contains(".gguf"));
        assert!(cluster
            .safe_telemetry_attributes()
            .values()
            .all(|value| !value.to_string().contains("/private")));
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers", test))]
mod gguf_tokenizer_tests {
    use super::*;
    use candle_core::quantized::gguf_file::{Content, Value as GgufValue, VersionedMagic};
    use std::collections::HashMap;
    use std::path::Path;

    /// Builds a tiny synthetic GGUF `Content` with a `gemma4`-style SentencePiece
    /// metaspace vocabulary, suitable for exercising the tokenizer builder without
    /// reading a real model file.
    fn gemma4_content() -> Content {
        let tokens = vec![
            "<pad>".to_string(),  // 0
            "<eos>".to_string(),  // 1
            "<bos>".to_string(),  // 2
            "<unk>".to_string(),  // 3
            "<mask>".to_string(), // 4
            "▁".to_string(),      // 5
            "h".to_string(),      // 6
            "i".to_string(),      // 7
            "t".to_string(),      // 8
            "e".to_string(),      // 9
            "r".to_string(),      // 10
            "hi".to_string(),     // 11
            "▁hi".to_string(),    // 12
            "th".to_string(),     // 13
            "the".to_string(),    // 14
            "ther".to_string(),   // 15
            "there".to_string(),  // 16
            "▁there".to_string(), // 17
        ];
        let token_type: Vec<GgufValue> = vec![
            GgufValue::U32(3), // <pad> -> control
            GgufValue::U32(3), // <eos> -> control
            GgufValue::U32(3), // <bos> -> control
            GgufValue::U32(2), // <unk> -> unknown
            GgufValue::U32(3), // <mask> -> control
            GgufValue::U32(1), // ▁ -> normal
            GgufValue::U32(1), // h -> normal
            GgufValue::U32(1), // i -> normal
            GgufValue::U32(1), // t -> normal
            GgufValue::U32(1), // e -> normal
            GgufValue::U32(1), // r -> normal
            GgufValue::U32(1), // hi -> normal
            GgufValue::U32(1), // ▁hi -> normal
            GgufValue::U32(1), // th -> normal
            GgufValue::U32(1), // the -> normal
            GgufValue::U32(1), // ther -> normal
            GgufValue::U32(1), // there -> normal
            GgufValue::U32(1), // ▁there -> normal
        ];
        let merges = vec![
            "h i".to_string(),
            "▁ hi".to_string(),
            "t h".to_string(),
            "th e".to_string(),
            "the r".to_string(),
            "ther e".to_string(),
            "▁ there".to_string(),
        ];

        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gemma4".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufValue::Array(merges.into_iter().map(GgufValue::String).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            GgufValue::Array(token_type),
        );
        metadata.insert("tokenizer.ggml.bos_token_id".to_string(), GgufValue::U32(2));
        metadata.insert("tokenizer.ggml.eos_token_id".to_string(), GgufValue::U32(1));
        metadata.insert(
            "tokenizer.ggml.unknown_token_id".to_string(),
            GgufValue::U32(3),
        );
        metadata.insert(
            "tokenizer.ggml.padding_token_id".to_string(),
            GgufValue::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.add_space_prefix".to_string(),
            GgufValue::Bool(false),
        );
        metadata.insert(
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(true),
        );

        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    /// Builds a minimal gpt2-style GGUF `Content`, just enough metadata for
    /// candle's `TokenizerFromGguf::from_gguf` to be reached and attempted.
    fn gpt2_content() -> Content {
        let tokens = vec![
            "<|endoftext|>".to_string(),
            "h".to_string(),
            "i".to_string(),
            "Ġthere".to_string(),
            "hi".to_string(),
        ];
        let merges = vec!["h i".to_string()];

        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gpt2".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufValue::Array(merges.into_iter().map(GgufValue::String).collect()),
        );

        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    fn unsupported_content() -> Content {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("made-up-model".to_string()),
        );

        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn gemma4_metaspace_tokenizer_builds_and_round_trips() {
        let content = gemma4_content();
        let tokenizer = tokenizer_from_gguf_content(&content)
            .expect("gemma4 metaspace tokenizer should build from synthetic GGUF metadata");

        let encoding = tokenizer
            .encode("hi there", false)
            .expect("encoding should succeed");
        let ids = encoding.get_ids();
        assert!(!ids.is_empty(), "encoding should produce at least one id");

        // With add_space_prefix = false (PrependScheme::Never), the first token
        // must not gain a leading metaspace marker that wasn't in the input.
        let first_token = tokenizer
            .id_to_token(ids[0])
            .expect("first id maps to a token");
        assert!(
            !first_token.starts_with('▁'),
            "leading token `{first_token}` should not carry a metaspace prefix \
             when add_space_prefix is false"
        );

        let decoded = tokenizer
            .decode(ids, true)
            .expect("decoding should succeed");
        assert_eq!(decoded, "hi there");
    }

    #[test]
    fn gguf_bos_token_to_prepend_reads_gemma4_add_bos_token_metadata() {
        let content = gemma4_content();
        assert_eq!(gguf_bos_token_to_prepend(&content), Some(2));
    }

    #[test]
    fn gguf_bos_token_to_prepend_is_none_when_add_bos_token_is_absent() {
        let content = gpt2_content();
        assert_eq!(gguf_bos_token_to_prepend(&content), None);
    }

    #[test]
    fn prepend_bos_if_configured_inserts_missing_bos() {
        let mut input_ids = vec![10, 11, 12];
        prepend_bos_if_configured(&mut input_ids, Some(2));
        assert_eq!(input_ids, vec![2, 10, 11, 12]);
    }

    #[test]
    fn prepend_bos_if_configured_is_noop_when_bos_already_present() {
        let mut input_ids = vec![2, 10, 11, 12];
        prepend_bos_if_configured(&mut input_ids, Some(2));
        assert_eq!(input_ids, vec![2, 10, 11, 12]);
    }

    #[test]
    fn prepend_bos_if_configured_is_noop_when_bos_token_id_is_none() {
        let mut input_ids = vec![10, 11, 12];
        prepend_bos_if_configured(&mut input_ids, None);
        assert_eq!(input_ids, vec![10, 11, 12]);
    }

    #[test]
    fn unsupported_tokenizer_model_is_rejected() {
        let content = unsupported_content();
        let err = tokenizer_from_gguf_content(&content)
            .expect_err("unrecognized tokenizer.ggml.model must be rejected");
        let message = err.to_string();
        assert!(
            message.contains("made-up-model"),
            "error message `{message}` should mention the unsupported model kind"
        );
    }

    #[test]
    fn gpt2_tokenizer_model_delegates_to_candle() {
        let content = gpt2_content();
        // The gpt2 branch must delegate to candle's own
        // `TokenizerFromGguf::from_gguf`, which builds successfully for this
        // minimal-but-valid gpt2 metadata.
        let tokenizer = tokenizer_from_gguf_content(&content)
            .expect("gpt2 metadata should be handled by candle's existing implementation");
        assert!(tokenizer.get_vocab_size(false) >= 4);
    }

    #[test]
    #[ignore = "requires the real gemma-4-12b-it-Q4_K_M.gguf model file on disk"]
    fn real_gemma4_gguf_round_trips_hello_world() {
        let path =
            Path::new("/home/morgan/.local/share/milliways/models/gemma-4-12b-it-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }

        let mut file = fs::File::open(path).expect("open real gemma4 gguf");
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .expect("read real gemma4 gguf metadata");

        let tokenizer = tokenizer_from_gguf_content(&content)
            .expect("build tokenizer from real gemma4 gguf metadata");

        let encoding = tokenizer
            .encode("Hello, world!", false)
            .expect("encode real text");
        let ids = encoding.get_ids().to_vec();
        let decoded = tokenizer.decode(&ids, true).expect("decode real text");

        eprintln!("ids: {ids:?}");
        eprintln!("decoded: {decoded:?}");
        assert_eq!(decoded, "Hello, world!");
    }

    #[test]
    #[ignore = "requires the real gemma-4-12b-it-Q4_K_M.gguf model file on disk"]
    fn real_gemma4_gguf_chat_input_ids_begin_with_bos_token() {
        let path =
            Path::new("/home/morgan/.local/share/milliways/models/gemma-4-12b-it-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }

        let mut file = fs::File::open(path).expect("open real gemma4 gguf");
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .expect("read real gemma4 gguf metadata");

        let tokenizer = tokenizer_from_gguf_content(&content)
            .expect("build tokenizer from real gemma4 gguf metadata");
        let bos_token_id = gguf_bos_token_to_prepend(&content);
        assert_eq!(
            bos_token_id,
            Some(2),
            "gemma4 GGUF should configure bos_token_id=2 with add_bos_token=true"
        );

        let messages = vec![NativeChatMessage {
            role: "user".to_string(),
            content: Some(Value::String(
                "Say hello in one short sentence.".to_string(),
            )),
            tool_calls: None,
            tool_call_id: None,
        }];
        let prompt = gemma_chat_input(&messages);
        let encoding = tokenizer.encode(prompt, false).expect("encode chat prompt");
        let mut input_ids = encoding.get_ids().to_vec();
        prepend_bos_if_configured(&mut input_ids, bos_token_id);

        eprintln!("input_ids: {input_ids:?}");
        assert_eq!(
            input_ids.first().copied(),
            Some(2),
            "constructed input_ids should begin with the gemma BOS token id (2)"
        );
    }

    #[test]
    #[ignore = "requires the real gemma-4-12b-it-Q4_K_M.gguf model file on disk and runs a full 12B forward pass"]
    fn real_gemma4_generation_produces_non_garbage_output() {
        let model_path =
            Path::new("/home/morgan/.local/share/milliways/models/gemma-4-12b-it-Q4_K_M.gguf");
        if !model_path.exists() {
            eprintln!("skipping: {} not present", model_path.display());
            return;
        }

        let artifacts = CandleArtifactValidation {
            model_family: CandleModelFamily::Gemma4,
            model_format: NativeModelFormat::Gguf,
            layout: CandleArtifactLayout::for_format(NativeModelFormat::Gguf),
            weight_files: vec![artifact_file_name(model_path)],
            tokenizer_file: None,
            config_file: None,
        };

        let decoder = load_real_candle_decoder(CandleModelFamily::Gemma4, model_path, &artifacts)
            .expect("load real gemma4 candle decoder from GGUF");

        let request = NativeChatRequest {
            model: "gemma4:12b".to_string(),
            messages: vec![NativeChatMessage {
                role: "user".to_string(),
                content: Some(Value::String(
                    "Say hello in one short sentence.".to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: Some(32),
            tools: None,
            tool_choice: None,
            metadata: BTreeMap::new(),
        };

        let output = decoder
            .generate(&request)
            .expect("real gemma4 generation should succeed");

        eprintln!("decoded output: {output:?}");
        assert!(!output.is_empty(), "decoded output should be non-empty");
        assert!(
            output.chars().any(|ch| ch.is_ascii_alphabetic()),
            "decoded output `{output}` should contain at least one ASCII letter, \
             got what looks like garbage/replacement-character output"
        );
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers", test))]
mod quantized_gemma4_tests {
    use super::quantized_gemma4;
    use candle_core::quantized::gguf_file::{
        Content, TensorInfo, Value as GgufValue, VersionedMagic,
    };
    use candle_core::quantized::GgmlDType;
    use candle_core::{Device, Shape};
    use std::collections::HashMap;
    use std::io::Cursor;

    /// Synthetic gemma4 GGUF config: 2 layers, layer 0 sliding (local) and
    /// layer 1 global, mirroring the real model's alternating attention
    /// pattern but at a tiny scale.
    const EMBEDDING_LENGTH: usize = 8;
    const HEAD_COUNT: usize = 2;
    const KEY_LENGTH: usize = 4; // global head_dim
    const KEY_LENGTH_SWA: usize = 2; // sliding head_dim
    const FFN_DIM: usize = 6;
    const VOCAB_SIZE: usize = 10;
    const BLOCK_COUNT: usize = 2;

    /// Appends an F32 tensor (raw little-endian bytes) to `data` and records
    /// a matching [`TensorInfo`] entry in `tensor_infos`.
    fn push_tensor(
        data: &mut Vec<u8>,
        tensor_infos: &mut HashMap<String, TensorInfo>,
        name: &str,
        shape: &[usize],
    ) {
        let elem_count: usize = shape.iter().product();
        let offset = data.len() as u64;
        for i in 0..elem_count {
            // Small deterministic values keep RmsNorm/softmax well-behaved.
            let value = 0.01 * (i as f32 + 1.0);
            data.extend_from_slice(&value.to_le_bytes());
        }
        tensor_infos.insert(
            name.to_string(),
            TensorInfo {
                ggml_dtype: GgmlDType::F32,
                shape: Shape::from(shape.to_vec()),
                offset,
            },
        );
    }

    /// Builds a synthetic `gemma4` GGUF [`Content`] plus its backing tensor
    /// data, with `block_count` layers alternating sliding/global per
    /// `sliding_window_pattern`.
    fn gemma4_content_and_data(
        head_count_kv: &[u32],
        sliding_window_pattern: &[u32],
    ) -> (Content, Cursor<Vec<u8>>) {
        let mut metadata = HashMap::new();
        metadata.insert(
            "gemma4.block_count".to_string(),
            GgufValue::U32(BLOCK_COUNT as u32),
        );
        metadata.insert(
            "gemma4.embedding_length".to_string(),
            GgufValue::U32(EMBEDDING_LENGTH as u32),
        );
        metadata.insert(
            "gemma4.attention.head_count".to_string(),
            GgufValue::U32(HEAD_COUNT as u32),
        );
        metadata.insert(
            "gemma4.attention.key_length".to_string(),
            GgufValue::U32(KEY_LENGTH as u32),
        );
        metadata.insert(
            "gemma4.attention.value_length".to_string(),
            GgufValue::U32(KEY_LENGTH as u32),
        );
        metadata.insert(
            "gemma4.attention.key_length_swa".to_string(),
            GgufValue::U32(KEY_LENGTH_SWA as u32),
        );
        metadata.insert(
            "gemma4.attention.value_length_swa".to_string(),
            GgufValue::U32(KEY_LENGTH_SWA as u32),
        );
        metadata.insert(
            "gemma4.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::F32(1e-6),
        );
        metadata.insert(
            "gemma4.attention.sliding_window".to_string(),
            GgufValue::U32(4),
        );
        metadata.insert(
            "gemma4.rope.freq_base".to_string(),
            GgufValue::F32(1_000_000.0),
        );
        metadata.insert(
            "gemma4.rope.freq_base_swa".to_string(),
            GgufValue::F32(10_000.0),
        );
        metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufValue::Array(head_count_kv.iter().copied().map(GgufValue::U32).collect()),
        );
        metadata.insert(
            "gemma4.attention.sliding_window_pattern".to_string(),
            GgufValue::Array(
                sliding_window_pattern
                    .iter()
                    .copied()
                    .map(GgufValue::U32)
                    .collect(),
            ),
        );

        let mut data = Vec::new();
        let mut tensor_infos = HashMap::new();

        push_tensor(
            &mut data,
            &mut tensor_infos,
            "token_embd.weight",
            &[VOCAB_SIZE, EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            "output_norm.weight",
            &[EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            "output.weight",
            &[VOCAB_SIZE, EMBEDDING_LENGTH],
        );

        for (layer_idx, &pattern) in sliding_window_pattern.iter().enumerate() {
            let head_dim = if pattern == 1 {
                KEY_LENGTH_SWA
            } else {
                KEY_LENGTH
            };
            let n_kv_head = head_count_kv[layer_idx] as usize;
            let q_dim = HEAD_COUNT * head_dim;
            let kv_dim = n_kv_head * head_dim;
            let prefix = format!("blk.{layer_idx}");

            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.attn_q.weight"),
                &[q_dim, EMBEDDING_LENGTH],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.attn_k.weight"),
                &[kv_dim, EMBEDDING_LENGTH],
            );
            // Global (non-sliding) layers have no `attn_v.weight` in the real
            // GGUF; mirror that here so the fixture exercises the
            // `Vcur = Kcur` fallback path.
            if pattern == 1 {
                push_tensor(
                    &mut data,
                    &mut tensor_infos,
                    &format!("{prefix}.attn_v.weight"),
                    &[kv_dim, EMBEDDING_LENGTH],
                );
            }
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.attn_output.weight"),
                &[EMBEDDING_LENGTH, q_dim],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.attn_q_norm.weight"),
                &[head_dim],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.attn_k_norm.weight"),
                &[head_dim],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.attn_norm.weight"),
                &[EMBEDDING_LENGTH],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.post_attention_norm.weight"),
                &[EMBEDDING_LENGTH],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.ffn_norm.weight"),
                &[EMBEDDING_LENGTH],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.post_ffw_norm.weight"),
                &[EMBEDDING_LENGTH],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.layer_output_scale.weight"),
                &[1],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.ffn_gate.weight"),
                &[FFN_DIM, EMBEDDING_LENGTH],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.ffn_up.weight"),
                &[FFN_DIM, EMBEDDING_LENGTH],
            );
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.ffn_down.weight"),
                &[EMBEDDING_LENGTH, FFN_DIM],
            );
        }

        let content = Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            tensor_data_offset: 0,
        };
        (content, Cursor::new(data))
    }

    #[test]
    fn from_gguf_builds_model_with_alternating_sliding_and_global_layers() {
        let (content, mut reader) = gemma4_content_and_data(&[1, 1], &[1, 0]);
        let device = Device::Cpu;

        let mut model = quantized_gemma4::ModelWeights::from_gguf(content, &mut reader, &device)
            .expect("synthetic gemma4 GGUF should build successfully");

        let input = candle_core::Tensor::new(&[1u32, 2u32, 3u32], &device)
            .and_then(|t| t.reshape((1, 3)))
            .expect("input tensor");

        let logits = model
            .forward(&input, 0)
            .expect("forward pass on synthetic gemma4 model should succeed");

        assert_eq!(logits.dims(), &[1, VOCAB_SIZE]);
    }

    #[test]
    fn from_gguf_rejects_missing_head_count_kv_array() {
        let (mut content, mut reader) = gemma4_content_and_data(&[1, 1], &[1, 0]);
        content.metadata.remove("gemma4.attention.head_count_kv");
        let device = Device::Cpu;

        let err = quantized_gemma4::ModelWeights::from_gguf(content, &mut reader, &device)
            .expect_err("missing head_count_kv array must be rejected");
        let message = err.to_string();
        assert!(
            message.contains("attention.head_count_kv"),
            "error message `{message}` should mention the missing key"
        );
    }

    #[test]
    fn from_gguf_rejects_wrong_length_sliding_window_pattern() {
        let (mut content, mut reader) = gemma4_content_and_data(&[1, 1], &[1, 0]);
        // Replace with an array of the wrong length (1 element instead of 2).
        content.metadata.insert(
            "gemma4.attention.sliding_window_pattern".to_string(),
            GgufValue::Array(vec![GgufValue::U32(1)]),
        );
        let device = Device::Cpu;

        let err = quantized_gemma4::ModelWeights::from_gguf(content, &mut reader, &device)
            .expect_err("wrong-length sliding_window_pattern array must be rejected");
        let message = err.to_string();
        assert!(
            message.contains("sliding_window_pattern"),
            "error message `{message}` should mention the offending key"
        );
        assert!(
            message.contains("expected 2"),
            "error message `{message}` should mention the expected length"
        );
    }

    #[test]
    fn from_gguf_rejects_non_array_head_count_kv() {
        let (mut content, mut reader) = gemma4_content_and_data(&[1, 1], &[1, 0]);
        content.metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        let device = Device::Cpu;

        let err = quantized_gemma4::ModelWeights::from_gguf(content, &mut reader, &device)
            .expect_err("non-array head_count_kv must be rejected");
        let message = err.to_string();
        assert!(
            message.contains("head_count_kv"),
            "error message `{message}` should mention the offending key"
        );
        assert!(
            message.contains("not an array"),
            "error message `{message}` should explain it is not an array"
        );
    }

    #[test]
    #[ignore = "requires the real gemma-4-12b-it-Q4_K_M.gguf model file on disk"]
    fn real_gemma4_gguf_constructs_model_and_runs_forward() {
        let path = std::path::Path::new(
            "/home/morgan/.local/share/milliways/models/gemma-4-12b-it-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }

        let mut file = std::fs::File::open(path).expect("open real gemma4 gguf");
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .expect("read real gemma4 gguf metadata");
        let device = Device::Cpu;

        let mut model = quantized_gemma4::ModelWeights::from_gguf(content, &mut file, &device)
            .expect("construct quantized gemma4 model from real GGUF weights");

        let input = candle_core::Tensor::new(&[2u32, 3u32], &device)
            .and_then(|t| t.reshape((1, 2)))
            .expect("input tensor");

        let logits = model
            .forward(&input, 0)
            .expect("forward pass on real gemma4 model should succeed");

        eprintln!("logits dims: {:?}", logits.dims());
        assert_eq!(logits.dims().len(), 2);
        assert_eq!(logits.dims()[0], 1);
        assert!(logits.dims()[1] > 0);
    }
}
