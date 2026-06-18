use crate::config::{ClusterNodeConfig, Config, ModelConfig, ResourceConfig};
use crate::resources::GpuVendor;
use crate::runtime::RuntimeBackend;
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
use anyhow::Context;
use anyhow::{anyhow, bail, Result};
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

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub fn format_native_chat_prompt(
    family: CandleModelFamily,
    messages: &[NativeChatMessage],
) -> String {
    match family {
        // Qwen3 (dense + MoE) — native ChatML format <|im_start|>role\n...<|im_end|>
        CandleModelFamily::Qwen3 | CandleModelFamily::Qwen3Moe => {
            let mut out = String::new();
            for msg in messages {
                let role = match msg.role.as_str() {
                    "assistant" => "assistant",
                    "system" => "system",
                    _ => "user",
                };
                let content = message_content_text(msg);
                out.push_str("<|im_start|>");
                out.push_str(role);
                out.push('\n');
                out.push_str(&content);
                out.push_str("<|im_end|>\n");
            }
            out.push_str("<|im_start|>assistant\n");
            out
        }
        CandleModelFamily::Gemma4 => {
            // Gemma4 E4B GGUF uses <|turn> (ID 105) to open a turn and <turn|> (ID 106)
            // to close it. The EOS token is also <turn|> (eos_token_id = 106).
            // BOS (add_bos_token = true) is prepended as the string "<bos>" which the
            // tokenizer recognizes as ID 2.
            let mut out = String::from("<bos>");
            for msg in messages {
                let role = msg.role.as_str();
                let content = message_content_text(msg);
                let turn_role = match role {
                    "assistant" => "model",
                    "system" => "system",
                    _ => "user",
                };
                out.push_str("<|turn>");
                out.push_str(turn_role);
                out.push('\n');
                out.push_str(&content);
                out.push_str("<turn|>\n");
            }
            // Open the model's turn without closing it — the model fills the rest
            out.push_str("<|turn>model\n");
            out
        }
        _ => canonical_native_chat_input(messages),
    }
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
    Qwen3Moe,
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
            Self::Qwen3Moe => "qwen3-moe",
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
            Self::Qwen3Moe => "candle-native-qwen3-moe",
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
            Self::Qwen3Moe => "Qwen3 MoE",
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
            Self::Qwen3Moe,
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
            Self::Qwen3 | Self::Qwen3Moe | Self::Gemma4 | Self::DeepSeek | Self::Mistral
        )
    }

    /// Stable identifier for the tool-call protocol the model uses.
    /// Orchestrators (e.g. milliways sommelier) use this to select a
    /// tool-call parser per family. New identifiers require a proposal.
    #[must_use]
    pub const fn tool_protocol(&self) -> &'static str {
        match self {
            Self::Qwen3 | Self::Qwen3Moe => "qwen3-native",
            Self::Gemma4 => "gemma4-native",
            Self::DeepSeek | Self::Kimi | Self::Mistral | Self::MiniMax => "none",
        }
    }

    /// Parse the kebab-case form back into the enum, e.g. `"qwen3-moe"` →
    /// `CandleModelFamily::Qwen3Moe`. Used to interpret the operator's
    /// `family` field in `ModelConfig`.
    #[must_use]
    pub fn from_kebab_str(s: &str) -> Option<Self> {
        Self::all().iter().copied().find(|f| f.as_str() == s)
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
        // Qwen3 MoE is currently only supported via GGUF — candle 0.10.2 does not
        // ship a safetensors path for the MoE variant.
        CandleModelFamily::Qwen3Moe => vec![NativeModelFormat::Gguf],
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
        CandleModelFamily::Qwen3 | CandleModelFamily::Qwen3Moe | CandleModelFamily::Gemma4 | CandleModelFamily::Mistral => {
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
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
enum RealCandleModel {
    Qwen3(candle_transformers::models::qwen3::ModelForCausalLM),
    Qwen3Gguf(candle_transformers::models::quantized_qwen3::ModelWeights),
    Qwen3MoeGguf(candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE),
    DeepSeek2(candle_transformers::models::deepseek2::DeepSeekV2),
    Gemma3(candle_transformers::models::gemma3::Model),
    Gemma4Gguf(crate::gemma4_gguf::ModelWeights),
    Mistral(candle_transformers::models::mistral::Model),
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl std::fmt::Debug for RealCandleModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant = match self {
            Self::Qwen3(_) => "Qwen3",
            Self::Qwen3Gguf(_) => "Qwen3Gguf",
            Self::Qwen3MoeGguf(_) => "Qwen3MoeGguf",
            Self::DeepSeek2(_) => "DeepSeek2",
            Self::Gemma3(_) => "Gemma3",
            Self::Gemma4Gguf(_) => "Gemma4Gguf",
            Self::Mistral(_) => "Mistral",
        };
        f.debug_tuple("RealCandleModel").field(&variant).finish()
    }
}

// Candle's quantized_gemma3 probes for ["gemma3","gemma2","gemma","gemma-embedding"]
// prefixes but not "gemma4". Copy all `from_prefix.*` metadata entries under
// `to_prefix.*` so the probe succeeds and all subsequent key lookups resolve.
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn remap_gguf_arch_prefix(
    mut content: candle_core::quantized::gguf_file::Content,
    from_prefix: &str,
    to_prefix: &str,
) -> candle_core::quantized::gguf_file::Content {
    let prefix = format!("{from_prefix}.");
    let remapped: Vec<(String, candle_core::quantized::gguf_file::Value)> = content
        .metadata
        .iter()
        .filter(|(k, _)| k.starts_with(&prefix))
        .map(|(k, v)| {
            let new_key = format!("{to_prefix}.{}", &k[prefix.len()..]);
            (new_key, v.clone())
        })
        .collect();
    for (k, v) in remapped {
        content.metadata.entry(k).or_insert(v);
    }
    content
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

    let device = crate::gemma4_gguf::best_device();
    let gpu_backend = match &device {
        candle_core::Device::Metal(_) => "metal",
        candle_core::Device::Cuda(_) => "cuda",
        candle_core::Device::Cpu => "cpu",
    };
    // Span + duration metric — recorded on every load attempt (success or fail).
    // The model.quant attribute is filename-derived: pulls the Q-suffix from the
    // file name (e.g. "Qwen3-14B-Q4_K_M.gguf" → "Q4_K_M"); falls back to "unknown".
    let model_quant = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit('-').next())
        .filter(|s| s.starts_with('Q') || s.starts_with('q'))
        .map(str::to_owned)
        .unwrap_or_else(|| "unknown".to_string());
    let load_span = tracing::info_span!(
        "native.model.load",
        model.family = family.as_str(),
        model.quant = %model_quant,
        gpu.backend = gpu_backend,
        gguf.path = %model_path.display(),
    );
    let _load_guard = load_span.enter();
    let load_start = std::time::Instant::now();

    let tokenizer = load_generation_tokenizer(model_path, artifacts)
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
                CandleModelFamily::Qwen3Moe => bail!(
                    "Qwen3 MoE safetensors loading is not wired in Candle 0.10.2; use the GGUF artifact format"
                ),
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
                CandleModelFamily::Qwen3Moe => RealCandleModel::Qwen3MoeGguf(
                    candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE::from_gguf(
                        content,
                        &mut file,
                        &device,
                        // F32 activations on Metal — F16 has been observed to cause
                        // argmax drift in dequant-heavy paths (see Gemma 4 F16 PLE
                        // limitation documented in docs/native-gguf-internals.md).
                        candle_core::DType::F32,
                    )
                    .with_context(|| "failed to construct quantized Qwen3 MoE Candle model")?,
                ),
                CandleModelFamily::Gemma4 => {
                    let profile = crate::gemma4_gguf::detect_profile(&content)
                        .ok_or_else(|| anyhow!(
                            "unsupported Gemma 4 GGUF profile: general.architecture = {:?}; known profiles: {:?}",
                            content.metadata.get("general.architecture"),
                            crate::gemma4_gguf::KNOWN_PROFILES.iter().map(|p| p.source_prefix).collect::<Vec<_>>()
                        ))?;
                    tracing::info!(profile = profile.label, "loading Gemma 4 GGUF");
                    RealCandleModel::Gemma4Gguf(
                        crate::gemma4_gguf::ModelWeights::from_gguf(
                            remap_gguf_arch_prefix(content, profile.source_prefix, "gemma3"),
                            &mut file,
                            &device,
                        )
                        .with_context(|| "failed to construct quantized Gemma4 model")?,
                    )
                }
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

    // Record native.model.load.duration_ms and peak resident memory now that
    // the model has finished constructing. Attributes match the span.
    let load_elapsed_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let load_attrs = [
        opentelemetry::KeyValue::new("model.family", family.as_str()),
        opentelemetry::KeyValue::new("model.quant", model_quant.clone()),
        opentelemetry::KeyValue::new("gpu.backend", gpu_backend),
    ];
    crate::observability::native_model_load_duration_ms().record(load_elapsed_ms, &load_attrs);
    if let Some(mb) = crate::observability::process_peak_resident_mb() {
        let mem_attrs = [opentelemetry::KeyValue::new(
            "model.family",
            family.as_str(),
        )];
        crate::observability::native_model_peak_resident_mb().record(mb, &mem_attrs);
    }
    tracing::info!(
        load_ms = load_elapsed_ms,
        model.family = family.as_str(),
        gpu.backend = gpu_backend,
        "native model load completed"
    );

    Ok(NativeCandleDecoder::Real(RealCandleDecoder {
        tokenizer,
        model: Mutex::new(model),
        family,
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

        let prompt = format_native_chat_prompt(self.family, &request.messages);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize native prompt: {err}"))?;
        let mut input_ids = encoding.get_ids().to_vec();
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
        let family_str = self.family.as_str();
        let input_token_count = input_ids.len();

        // Prefill span: step 0 only. Covers the cost of processing the
        // full input prompt with empty KV cache.
        let prefill_span = tracing::info_span!(
            "native.model.prefill",
            model.family = family_str,
            input_tokens = input_token_count,
            context_pos = 0u64,
        );
        let prefill_start = std::time::Instant::now();
        {
            let _enter = prefill_span.enter();
            let step_input = input_ids.clone();
            let next = model.forward_next(&step_input, offset)?;
            offset = offset.saturating_add(step_input.len());
            input_ids.push(next);
            generated.push(next);
            if is_eos_token(&self.tokenizer, next) {
                // EOS on first token — skip the generation loop entirely.
                let prefill_secs = prefill_start.elapsed().as_secs_f64();
                if prefill_secs > 0.0 {
                    crate::observability::native_model_tokens_per_second().record(
                        input_token_count as f64 / prefill_secs,
                        &[
                            opentelemetry::KeyValue::new("model.family", family_str),
                            opentelemetry::KeyValue::new("phase", "prefill"),
                        ],
                    );
                }
                return self.tokenizer.decode(&generated, true).map_err(|err| {
                    anyhow::anyhow!("failed to decode native output tokens: {err}")
                });
            }
        }
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        if prefill_secs > 0.0 {
            crate::observability::native_model_tokens_per_second().record(
                input_token_count as f64 / prefill_secs,
                &[
                    opentelemetry::KeyValue::new("model.family", family_str),
                    opentelemetry::KeyValue::new("phase", "prefill"),
                ],
            );
        }

        // Generation span: every step after the first. Records tokens-per-second
        // on exit using actual output_tokens generated (not max_tokens).
        let gen_span = tracing::info_span!(
            "native.model.generation",
            model.family = family_str,
            output_tokens = tracing::field::Empty,
        );
        let gen_start = std::time::Instant::now();
        {
            let _enter = gen_span.enter();
            for _ in 1..max_tokens {
                let step_input = vec![*input_ids.last().expect("input ids are non-empty")];
                let next = model.forward_next(&step_input, offset)?;
                offset = offset.saturating_add(step_input.len());
                input_ids.push(next);
                generated.push(next);
                if is_eos_token(&self.tokenizer, next) {
                    break;
                }
            }
            // generated.len() includes the prefill's first token; gen count is the rest.
            let gen_tokens = generated.len().saturating_sub(1);
            tracing::Span::current().record("output_tokens", gen_tokens);
            let gen_secs = gen_start.elapsed().as_secs_f64();
            if gen_secs > 0.0 && gen_tokens > 0 {
                crate::observability::native_model_tokens_per_second().record(
                    gen_tokens as f64 / gen_secs,
                    &[
                        opentelemetry::KeyValue::new("model.family", family_str),
                        opentelemetry::KeyValue::new("phase", "generation"),
                    ],
                );
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

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl RealCandleModel {
    fn clear_kv_cache(&mut self) {
        match self {
            Self::Qwen3(model) => model.clear_kv_cache(),
            Self::Qwen3Gguf(model) => model.clear_kv_cache(),
            // candle 0.10.2's quantized_qwen3_moe does not expose a public
            // clear_kv_cache() method; the inner ConcatKvCache also lacks reset().
            // Independent sessions must reload the model. Filed as follow-up.
            Self::Qwen3MoeGguf(_) => {
                tracing::warn!(
                    "Qwen3 MoE clear_kv_cache is a no-op in candle 0.10.2 — \
                     independent sessions must reload the model"
                );
            }
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
            Self::Qwen3MoeGguf(model) => model.forward(&input, offset),
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

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn load_generation_tokenizer(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<tokenizers::tokenizer::Tokenizer> {
    match artifacts.model_format {
        NativeModelFormat::Safetensors => {
            let tokenizer_path = safetensors_artifact_dir(model_path).join("tokenizer.json");
            tokenizers::tokenizer::Tokenizer::from_file(&tokenizer_path)
                .map_err(|err| anyhow::anyhow!("failed to load tokenizer.json: {err}"))
        }
        NativeModelFormat::Gguf => {
            let mut file = fs::File::open(model_path)
                .with_context(|| "failed to open GGUF tokenizer metadata")?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                .with_context(|| "failed to read GGUF tokenizer metadata")?;
            let result = <tokenizers::tokenizer::Tokenizer as candle_core::quantized::tokenizer::TokenizerFromGguf>::from_gguf(&content);
            match result {
                Ok(tok) => Ok(tok),
                Err(err) => {
                    let msg = err.to_string();
                    if msg.contains("unsupported tokenizer model") {
                        tokenizer_from_gguf_spm(&content).with_context(|| {
                            format!("failed to build SentencePiece tokenizer from GGUF ({msg})")
                        })
                    } else {
                        Err(anyhow::anyhow!(
                            "failed to build tokenizer from GGUF metadata: {err}"
                        ))
                    }
                }
            }
        }
        NativeModelFormat::Unknown => bail!("native artifact format is unsupported"),
    }
}

// Fallback tokenizer for GGUF models that use SPM-style BPE (e.g. Gemma4).
// These models store `tokenizer.ggml.model = "gemma4"` and carry BPE merges,
// but use ▁ (U+2581) whitespace escaping rather than GPT-2 byte-level encoding.
// Candle's TokenizerFromGguf only handles `"gpt2"`, so we build the tokenizer
// manually from the same GGUF metadata.
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn tokenizer_from_gguf_spm(
    content: &candle_core::quantized::gguf_file::Content,
) -> Result<tokenizers::tokenizer::Tokenizer> {
    use tokenizers::{
        decoders::DecoderWrapper,
        models::bpe::{Vocab, BPE},
        pre_tokenizers::{metaspace::Metaspace, metaspace::PrependScheme, PreTokenizerWrapper},
        tokenizer::Tokenizer,
        AddedToken,
    };

    let tokens_val = content
        .metadata
        .get("tokenizer.ggml.tokens")
        .with_context(|| "missing tokenizer.ggml.tokens")?;
    let tokens: Vec<String> = tokens_val
        .to_vec()
        .with_context(|| "tokenizer.ggml.tokens is not an array")?
        .iter()
        .map(|v| {
            v.to_string()
                .cloned()
                .map_err(|e| anyhow::anyhow!("token is not a string: {e}"))
        })
        .collect::<Result<_>>()?;

    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    let merges: Vec<(String, String)> = content
        .metadata
        .get("tokenizer.ggml.merges")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let s = v.to_string().ok()?;
                    let (a, b) = s.split_once(' ')?;
                    Some((a.to_string(), b.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();

    let mut builder = BPE::builder().vocab_and_merges(vocab, merges);

    if let Some(v) = content.metadata.get("tokenizer.ggml.unk_token_id") {
        let id = match v {
            candle_core::quantized::gguf_file::Value::U32(n) => Some(*n as usize),
            candle_core::quantized::gguf_file::Value::U64(n) => Some(*n as usize),
            candle_core::quantized::gguf_file::Value::I32(n) => Some(*n as usize),
            _ => None,
        };
        if let Some(idx) = id {
            if let Some(tok) = tokens.get(idx) {
                builder = builder.unk_token(tok.clone());
            }
        }
    }

    if let Some(v) = content.metadata.get("tokenizer.ggml.ignore_merges") {
        if let Ok(flag) = v.to_bool() {
            builder = builder.ignore_merges(flag);
        }
    }

    let bpe = builder
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build BPE model: {e}"))?;

    let metaspace = Metaspace::new('▁', PrependScheme::Always, true);
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Metaspace(metaspace.clone())));
    tokenizer.with_decoder(Some(DecoderWrapper::Metaspace(metaspace)));

    if let Some(candle_core::quantized::gguf_file::Value::Array(type_arr)) =
        content.metadata.get("tokenizer.ggml.token_type")
    {
        let specials: Vec<AddedToken> = type_arr
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                let ty = match v {
                    candle_core::quantized::gguf_file::Value::U32(n) => *n,
                    candle_core::quantized::gguf_file::Value::I32(n) => *n as u32,
                    _ => return None,
                };
                if matches!(ty, 2..=5) {
                    tokens
                        .get(idx)
                        .map(|tok| AddedToken::from(tok.clone(), true))
                } else {
                    None
                }
            })
            .collect();
        if !specials.is_empty() {
            tokenizer.add_special_tokens(&specials);
        }
    }

    Ok(tokenizer)
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn is_eos_token(tokenizer: &tokenizers::tokenizer::Tokenizer, token_id: u32) -> bool {
    tokenizer
        .id_to_token(token_id)
        .map(|token| {
            matches!(
                token.as_str(),
                "</s>" | "<|endoftext|>" | "<end_of_turn>" | "<turn|>" | "<eos>"
            )
        })
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

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    fn format_native_chat_prompt_gemma4_uses_turn_markers() {
        let messages = vec![NativeChatMessage {
            role: "user".to_string(),
            content: Some(Value::String("Say hello world".to_string())),
            tool_calls: None,
            tool_call_id: None,
        }];
        let prompt = format_native_chat_prompt(CandleModelFamily::Gemma4, &messages);
        assert!(
            prompt.starts_with("<bos>"),
            "Gemma4 prompt must start with <bos>, got: {prompt:?}"
        );
        assert!(
            prompt.contains("<|turn>user"),
            "must contain user turn marker"
        );
        assert!(prompt.contains("<turn|>"), "must close turn with <turn|>");
        assert!(prompt.contains("<|turn>model"), "must open model turn");
        assert!(
            !prompt.contains("<|user|>"),
            "must not use generic role markers"
        );
        assert!(
            !prompt.contains("<start_of_turn>"),
            "must not use Gemma1/2/3 format"
        );
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    #[ignore = "requires GGUF file on disk — slow (~30s)"]
    #[allow(clippy::explicit_counter_loop)] // cur_pos counter is intentional for the LLM-inference loop with early `break` on EOS
    fn gemma4_gguf_forward_pass_produces_coherent_tokens() {
        let gguf_path = std::path::Path::new(
            "/Users/w199447/.local/share/milliways/models/gemma-4-E4B-it-Q4_K_M.gguf",
        );
        if !gguf_path.exists() {
            return;
        }
        let mut file = std::fs::File::open(gguf_path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();

        let tokenizer = tokenizer_from_gguf_spm(&content).unwrap();
        // Use the correct Gemma4 E4B format
        let prompt = "<bos><|turn>user\nSay hello world<turn|>\n<|turn>model\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        eprintln!("Prompt tokens: {input_ids:?}");
        eprintln!("Prompt token strings: {:?}", encoding.get_tokens());

        let device = crate::gemma4_gguf::best_device();
        eprintln!("Device: {device:?}");
        let load_start = std::time::Instant::now();
        let mut model = crate::gemma4_gguf::ModelWeights::from_gguf(
            remap_gguf_arch_prefix(content, "gemma4", "gemma3"),
            &mut file,
            &device,
        )
        .unwrap();
        eprintln!("Model loaded in {:?}", load_start.elapsed());

        // Run prefill
        let input = candle_core::Tensor::new(input_ids.as_slice(), &device)
            .and_then(|t| t.reshape((1, input_ids.len())))
            .unwrap();
        let prefill_start = std::time::Instant::now();
        let logits = model.forward(&input, 0).unwrap();
        eprintln!(
            "Prefill ({} tokens) in {:?}",
            input_ids.len(),
            prefill_start.elapsed()
        );
        eprintln!("Logits shape: {:?}", logits.dims());

        // Check logit stats before argmax
        {
            let flat = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let min = flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = flat.iter().sum::<f32>() / flat.len() as f32;
            let nan_count = flat.iter().filter(|v| v.is_nan()).count();
            let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
            eprintln!("Logit stats: min={min:.3} max={max:.3} mean={mean:.3} nan={nan_count} inf={inf_count}");
            // Check tokens near "Hello" (approx token IDs)
            let check_ids = [1234u32, 2030, 4039, 12468, 37889, 18348, 236862, 245598];
            for id in check_ids {
                if (id as usize) < flat.len() {
                    let s = tokenizer.id_to_token(id).unwrap_or_default();
                    eprintln!("  logit[{id}] ({s:?}) = {:.3}", flat[id as usize]);
                }
            }
            // Check common response words
            for word in [
                "Hello", "hello", "Sure", "Of", "I", "Hi", "▁Hello", "▁Sure", "▁I",
            ] {
                if let Some(id) = tokenizer.token_to_id(word) {
                    eprintln!(
                        "  logit for {:?} (id {id}) = {:.3}",
                        word, flat[id as usize]
                    );
                } else {
                    eprintln!("  {:?} not in vocab", word);
                }
            }
        }

        let next_token = logits
            .squeeze(0)
            .and_then(|t| t.argmax(candle_core::D::Minus1))
            .and_then(|t| t.to_scalar::<u32>())
            .unwrap();
        let next_str = tokenizer.id_to_token(next_token).unwrap_or_default();
        eprintln!("First generated token: {next_token} = {next_str:?}");

        assert!(next_token < 262144, "token ID should be within vocab range");
        assert!(
            !next_str.contains('짤'),
            "model should not produce Korean garbage, got {next_token} = {next_str:?}"
        );

        // Generate 15 more tokens to verify ongoing coherence
        let mut all_tokens = vec![next_token];
        let mut cur_pos = input_ids.len();
        let mut prev_token = next_token;
        for _ in 0..15 {
            let next_input = candle_core::Tensor::new(&[prev_token], &device)
                .and_then(|t| t.reshape((1, 1)))
                .unwrap();
            let logits = model.forward(&next_input, cur_pos).unwrap();
            let tok = logits
                .squeeze(0)
                .and_then(|t| t.argmax(candle_core::D::Minus1))
                .and_then(|t| t.to_scalar::<u32>())
                .unwrap();
            let tok_str = tokenizer.id_to_token(tok).unwrap_or_default();
            assert!(
                !tok_str.contains('짤'),
                "garbage token at step {}: {tok} = {tok_str:?}",
                all_tokens.len()
            );
            all_tokens.push(tok);
            prev_token = tok;
            cur_pos += 1;
            if tok_str == "<turn|>" {
                break;
            }
        }
        let decoded = tokenizer.decode(&all_tokens, true).unwrap_or_default();
        eprintln!("Generated: {decoded:?}");
        // At least some ASCII printable content expected (greeting response)
        assert!(
            decoded.chars().any(|c| c.is_ascii_alphabetic()),
            "expected alphabetic output, got: {decoded:?}"
        );
    }

    #[cfg(all(
        feature = "native-candle",
        feature = "native-tokenizers",
        feature = "gpu-metal"
    ))]
    #[test]
    fn gemma4_gguf_metal_device_init_works() {
        let device = match candle_core::Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => panic!("Metal init failed: {e}"),
        };
        let a = candle_core::Tensor::randn(0f32, 1f32, (256, 256), &device).unwrap();
        let b = a.matmul(&a).unwrap();
        let sum = b.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(sum.is_finite(), "expected finite matmul sum, got {sum}");
        eprintln!("Metal sanity: 256x256 matmul → sum={sum:.3}");
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    #[ignore = "requires GGUF file on disk — slow (~3 min)"]
    #[allow(clippy::explicit_counter_loop)] // cur_pos counter is intentional for the LLM-inference loop with early `break` on EOS
    fn gemma4_gguf_generates_python_counting_program() {
        // Prefer E2B (smaller, ~5 GB working set) — falls back to E4B if absent.
        // Override via GEMMA4_GGUF_PATH env var.
        let candidates: Vec<String> = if let Ok(p) = std::env::var("GEMMA4_GGUF_PATH") {
            vec![p]
        } else {
            vec![
                "/Users/w199447/.local/share/milliways/models/gemma-4-E2B-it-Q4_K_M.gguf".into(),
                "/Users/w199447/.local/share/milliways/models/gemma-4-E4B-it-Q4_K_M.gguf".into(),
            ]
        };
        let gguf_path = candidates
            .iter()
            .map(std::path::PathBuf::from)
            .find(|p| p.exists());
        let Some(gguf_path) = gguf_path else { return };
        eprintln!("Using model: {}", gguf_path.display());
        let mut file = std::fs::File::open(&gguf_path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();

        let tokenizer = tokenizer_from_gguf_spm(&content).unwrap();
        let prompt = "<bos><|turn>user\nWrite a Python program that counts from 1 to 10.<turn|>\n<|turn>model\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        eprintln!("Prompt: {prompt}");

        // best_device() picks Metal/CUDA when those features are compiled in,
        // otherwise falls back to CPU. See Cargo.toml `gpu-metal` / `gpu-cuda`.
        let device = crate::gemma4_gguf::best_device();
        eprintln!("Device: {device:?}");
        let profile = crate::gemma4_gguf::detect_profile(&content).unwrap_or_else(|| {
            panic!(
                "unrecognised Gemma 4 profile: {:?}",
                content.metadata.get("general.architecture")
            )
        });
        eprintln!("Profile: {}", profile.label);
        let mut model = crate::gemma4_gguf::ModelWeights::from_gguf(
            remap_gguf_arch_prefix(content, profile.source_prefix, "gemma3"),
            &mut file,
            &device,
        )
        .unwrap();

        let input = candle_core::Tensor::new(input_ids.as_slice(), &device)
            .and_then(|t| t.reshape((1, input_ids.len())))
            .unwrap();
        let logits = model.forward(&input, 0).unwrap();

        let next_token = logits
            .squeeze(0)
            .and_then(|t| t.argmax(candle_core::D::Minus1))
            .and_then(|t| t.to_scalar::<u32>())
            .unwrap();

        // Greedy generation up to 80 tokens, stopping at <turn|> (token 106).
        let close_turn_id = tokenizer.token_to_id("<turn|>").unwrap_or(106);
        let mut all_tokens = vec![next_token];
        let mut cur_pos = input_ids.len();
        let mut prev_token = next_token;
        for _ in 0..80 {
            if prev_token == close_turn_id {
                break;
            }
            let next_input = candle_core::Tensor::new(&[prev_token], &device)
                .and_then(|t| t.reshape((1, 1)))
                .unwrap();
            let logits = model.forward(&next_input, cur_pos).unwrap();
            let tok = logits
                .squeeze(0)
                .and_then(|t| t.argmax(candle_core::D::Minus1))
                .and_then(|t| t.to_scalar::<u32>())
                .unwrap();
            all_tokens.push(tok);
            prev_token = tok;
            cur_pos += 1;
        }
        let decoded = tokenizer.decode(&all_tokens, true).unwrap_or_default();
        eprintln!("=== MODEL OUTPUT ===\n{decoded}\n=== END ===");
        // Sanity: output should mention either "for", "range", "print", or contain a digit
        let has_code_markers = decoded.contains("for")
            || decoded.contains("range")
            || decoded.contains("print")
            || decoded.chars().any(|c| c.is_ascii_digit());
        assert!(
            has_code_markers,
            "expected Python code markers in output, got: {decoded:?}"
        );
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    #[ignore = "requires GGUF file on disk — slow (~3 min on Metal)"]
    #[allow(clippy::explicit_counter_loop)] // cur_pos counter is intentional for the LLM-inference loop with early `break` on EOS
    fn qwen3_runtime_python_counting_program() {
        // M1 acceptance test for the add-tool-capable-tiered-runtime change.
        // Verifies the model handles both forward AND reverse counting,
        // proving understanding rather than memorisation of one canned example.
        let path_str = std::env::var("QWEN3_GGUF_PATH").unwrap_or_else(|_| {
            "/Users/w199447/.local/share/milliways/models/Qwen3-14B-Instruct-Q4_K_M.gguf".into()
        });
        let gguf_path = std::path::PathBuf::from(&path_str);
        if !gguf_path.exists() {
            eprintln!("skipping qwen3 test, no GGUF at {path_str}");
            return;
        }
        eprintln!("Using model: {}", gguf_path.display());

        let mut file = std::fs::File::open(&gguf_path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();

        // Qwen3 uses standard GPT-2 BPE — TokenizerFromGguf::from_gguf handles it.
        let tokenizer = <tokenizers::tokenizer::Tokenizer as candle_core::quantized::tokenizer::TokenizerFromGguf>::from_gguf(&content).unwrap();
        let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
        let im_start_id = tokenizer.token_to_id("<|im_start|>").unwrap_or(151644);
        eprintln!("Special tokens: <|im_start|> = {im_start_id}, <|im_end|> = {im_end_id}");

        let device = crate::gemma4_gguf::best_device();
        eprintln!("Device: {device:?}");

        let load_start = std::time::Instant::now();
        let mut model = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
            content, &mut file, &device,
        )
        .unwrap();
        let load_elapsed = load_start.elapsed();
        eprintln!("Model loaded in {load_elapsed:?}");

        // Inner helper: greedy decode up to `max_tokens`, stopping at <|im_end|>.
        let run = |model: &mut candle_transformers::models::quantized_qwen3::ModelWeights,
                   user_prompt: &str,
                   max_tokens: usize|
         -> (String, std::time::Duration, std::time::Duration) {
            model.clear_kv_cache();
            let prompt =
                format!("<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n");
            let encoding = tokenizer.encode(prompt.as_str(), false).unwrap();
            let input_ids: Vec<u32> = encoding.get_ids().to_vec();

            let prefill_start = std::time::Instant::now();
            let input = candle_core::Tensor::new(input_ids.as_slice(), &device)
                .and_then(|t| t.reshape((1, input_ids.len())))
                .unwrap();
            let logits = model.forward(&input, 0).unwrap();
            let prefill_elapsed = prefill_start.elapsed();

            let next_token = logits
                .squeeze(0)
                .and_then(|t| t.argmax(candle_core::D::Minus1))
                .and_then(|t| t.to_scalar::<u32>())
                .unwrap();

            let gen_start = std::time::Instant::now();
            let mut all_tokens = vec![next_token];
            let mut cur_pos = input_ids.len();
            let mut prev_token = next_token;
            for _ in 0..max_tokens {
                if prev_token == im_end_id {
                    break;
                }
                let next_input = candle_core::Tensor::new(&[prev_token], &device)
                    .and_then(|t| t.reshape((1, 1)))
                    .unwrap();
                let logits = model.forward(&next_input, cur_pos).unwrap();
                let tok = logits
                    .squeeze(0)
                    .and_then(|t| t.argmax(candle_core::D::Minus1))
                    .and_then(|t| t.to_scalar::<u32>())
                    .unwrap();
                all_tokens.push(tok);
                prev_token = tok;
                cur_pos += 1;
            }
            let gen_elapsed = gen_start.elapsed();
            let decoded = tokenizer.decode(&all_tokens, false).unwrap_or_default();
            (decoded, prefill_elapsed, gen_elapsed)
        };

        // Test 1 — forward counting.  /no_think disables Qwen3's reasoning
        // preamble so the test sees code directly rather than ~300 tokens
        // of "Okay, I need to write..." musing first.
        eprintln!("\n--- PROMPT 1: count from 1 to 10 ---");
        let (forward_out, p1_prefill, p1_gen) = run(
            &mut model,
            "/no_think Write a Python program that counts from 1 to 10.",
            400,
        );
        eprintln!("Prefill: {p1_prefill:?}   Generation: {p1_gen:?}");
        eprintln!("=== FORWARD OUTPUT ===\n{forward_out}\n=== END ===");

        assert!(
            forward_out.contains("print"),
            "forward output missing print() call: {forward_out:?}"
        );
        let forward_has_iteration = forward_out.contains("range(1, 11)")
            || forward_out.contains("range(1,11)")
            || forward_out.contains("range(10)")
            || forward_out.contains("range(11)")
            || (forward_out.contains("range(") && forward_out.contains("11"))
            || forward_out.contains("for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
        assert!(
            forward_has_iteration,
            "forward output missing forward-iteration construct: {forward_out:?}"
        );
        assert!(
            forward_out.contains("1") && forward_out.contains("10"),
            "forward output missing expected digits: {forward_out:?}"
        );

        // Test 2 — reverse counting.  clear_kv_cache happens inside run().
        eprintln!("\n--- PROMPT 2: count from 10 down to 1 ---");
        let (reverse_out, p2_prefill, p2_gen) = run(
            &mut model,
            "/no_think Write a Python program that counts from 10 down to 1.",
            400,
        );
        eprintln!("Prefill: {p2_prefill:?}   Generation: {p2_gen:?}");
        eprintln!("=== REVERSE OUTPUT ===\n{reverse_out}\n=== END ===");

        assert!(
            reverse_out.contains("print"),
            "reverse output missing print() call: {reverse_out:?}"
        );
        let reverse_has_iteration = reverse_out.contains("range(10, 0, -1)")
            || reverse_out.contains("range(10,0,-1)")
            || reverse_out.contains("range(10, -1, -1)")
            || reverse_out.contains("reversed(")
            || reverse_out.contains("for i in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]")
            || (reverse_out.contains("range(") && reverse_out.contains("-1"));
        assert!(
            reverse_has_iteration,
            "reverse output missing reverse-iteration construct: {reverse_out:?}"
        );
        assert!(
            reverse_out.contains("1") && reverse_out.contains("10"),
            "reverse output missing expected digits: {reverse_out:?}"
        );

        // End-to-end real-life check: extract the Python code block from each
        // model output, write it to a file, execute it via python3, and assert
        // the program's stdout matches the requested sequence. This validates
        // the model produces *runnable* code, not just parseable text.
        let extract_python_block = |out: &str| -> Option<String> {
            // Look for the standard ```python ... ``` fence the model emits.
            let start = out.find("```python")?;
            let after = &out[start + "```python".len()..];
            let after = after.strip_prefix('\n').unwrap_or(after);
            let end = after.find("```")?;
            Some(after[..end].to_string())
        };

        let exec_python_script =
            |code: &str, label: &str, fname: &str| -> std::io::Result<String> {
                let path = std::env::temp_dir().join(fname);
                std::fs::write(&path, code)?;
                eprintln!(
                    "\n[{label}] wrote {} bytes of generated code to {}",
                    code.len(),
                    path.display()
                );
                let out = std::process::Command::new("python3").arg(&path).output()?;
                eprintln!(
                    "[{label}] python3 exit={}, stderr={:?}",
                    out.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&out.stderr).trim()
                );
                Ok(String::from_utf8_lossy(&out.stdout).to_string())
            };

        let forward_code = extract_python_block(&forward_out)
            .expect("forward output should contain a ```python``` fenced block");
        let forward_stdout =
            exec_python_script(&forward_code, "forward", "rs_llmctl_count_forward.py")
                .expect("python3 should run forward test.py successfully");
        eprintln!("[forward] stdout:\n{}", forward_stdout);
        let forward_lines: Vec<&str> = forward_stdout.lines().collect();
        assert_eq!(
            forward_lines,
            vec!["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "forward program must print 1..10 line by line; got {forward_stdout:?}"
        );

        let reverse_code = extract_python_block(&reverse_out)
            .expect("reverse output should contain a ```python``` fenced block");
        let reverse_stdout =
            exec_python_script(&reverse_code, "reverse", "rs_llmctl_count_reverse.py")
                .expect("python3 should run reverse test.py successfully");
        eprintln!("[reverse] stdout:\n{}", reverse_stdout);
        let reverse_lines: Vec<&str> = reverse_stdout.lines().collect();
        assert_eq!(
            reverse_lines,
            vec!["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"],
            "reverse program must print 10..1 line by line; got {reverse_stdout:?}"
        );

        // Summary line for task 1.4 — regression-tracking baseline.
        eprintln!("\n=== M1 TIMING SUMMARY ===");
        eprintln!("  Model load:           {load_elapsed:?}");
        eprintln!("  Forward prefill:      {p1_prefill:?}");
        eprintln!("  Forward generation:   {p1_gen:?}");
        eprintln!("  Reverse prefill:      {p2_prefill:?}");
        eprintln!("  Reverse generation:   {p2_gen:?}");
        eprintln!("  Device:               {device:?}");
        eprintln!("  Forward program ran:  /tmp/rs_llmctl_count_forward.py");
        eprintln!("  Reverse program ran:  /tmp/rs_llmctl_count_reverse.py");
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    #[ignore = "Final E2E: model reads chaostooling-otel patterns + adds tracing to its own counter program (slow, requires Qwen3 14B GGUF on disk)"]
    #[allow(clippy::explicit_counter_loop)] // cur_pos counter is intentional for the LLM-inference loop with early `break` on EOS
    fn qwen3_runtime_adds_chaosotel_tracing_to_counter_program() {
        // This is the "no help from outside apart from the prompt" test:
        // we feed the Qwen3 14B model
        //   (a) three real chaostooling-otel instrumentation patterns
        //   (b) the counter program the model wrote in the previous test
        //   (c) the question
        // and capture the FULL pipeline — prompt, the model's <think> reasoning,
        // the final answer, and the extracted Python code — to stderr.
        //
        // No `/no_think` directive: we want to see the model reason about
        // which pattern to use.
        let path_str = std::env::var("QWEN3_GGUF_PATH").unwrap_or_else(|_| {
            "/Users/w199447/.local/share/milliways/models/Qwen3-14B-Instruct-Q4_K_M.gguf".into()
        });
        let gguf_path = std::path::PathBuf::from(&path_str);
        if !gguf_path.exists() {
            eprintln!("skipping chaosotel tracing test, no GGUF at {path_str}");
            return;
        }

        let mut file = std::fs::File::open(&gguf_path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();
        let tokenizer = <tokenizers::tokenizer::Tokenizer as candle_core::quantized::tokenizer::TokenizerFromGguf>::from_gguf(&content).unwrap();
        let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
        let device = crate::gemma4_gguf::best_device();
        eprintln!("Device: {device:?}");
        let load_start = std::time::Instant::now();
        let mut model = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
            content, &mut file, &device,
        )
        .unwrap();
        eprintln!("Model loaded in {:?}", load_start.elapsed());

        // The three real chaostooling-otel patterns, taken verbatim from
        // /Users/w199447/dev/src/pprojects/chaostooling-oss/chaostooling-otel/README.md
        // lines 431-472.
        let chaosotel_context = "\
## chaostooling-otel: Instrumentation Patterns

### Pattern 1: Automatic via Decorators
```python
from chaosotel.decorators import instrument_action

@instrument_action(name=\"my_chaos\", target_type=\"database\", severity=\"medium\")
def my_chaos_action(host: str):
    # Automatic tracing/metrics/logs
    pass
```

### Pattern 2: Manual Span Creation
```python
from chaosotel import get_tracer, ensure_initialized, flush

ensure_initialized()
tracer = get_tracer()

with tracer.start_as_current_span(\"my_operation\") as span:
    span.set_attribute(\"custom_attr\", \"value\")
    result = do_something()

flush()
```

### Pattern 3: Instrumentation Helpers
```python
from chaosotel.core.trace_core import instrument_db_span

with instrument_db_span(
    name=\"query_users\",
    db_system=\"postgresql\",
    db_name=\"production\",
    db_host=\"postgres-primary\",
    db_port=5432,
) as span:
    cursor.execute(\"SELECT * FROM users\")
```";

        // The program the model produced in the previous test, taken from
        // /tmp/rs_llmctl_count_forward.py written by qwen3_runtime_python_counting_program.
        let counter_program = "for i in range(1, 11):\n    print(i)\n";

        let user_prompt = format!(
            "{chaosotel_context}\n\n\
            Here is a small Python program I wrote earlier:\n\n\
            ```python\n{counter_program}```\n\n\
            Read the chaostooling-otel patterns above and rewrite this program so each \
            loop iteration is wrapped in an OpenTelemetry span. Pick the most appropriate \
            pattern (1, 2, or 3) and briefly explain why you picked it. Then output the \
            full updated program as a single ```python``` block."
        );

        // Run with thinking ON — no `/no_think` directive.
        let chat = format!("<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n");
        eprintln!("\n========== PROMPT SENT TO MODEL ==========\n{user_prompt}\n========== END PROMPT ==========\n");

        let encoding = tokenizer.encode(chat.as_str(), false).unwrap();
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        eprintln!("Prompt token count: {}", input_ids.len());

        let prefill_start = std::time::Instant::now();
        let input = candle_core::Tensor::new(input_ids.as_slice(), &device)
            .and_then(|t| t.reshape((1, input_ids.len())))
            .unwrap();
        let logits = model.forward(&input, 0).unwrap();
        eprintln!("Prefill in {:?}", prefill_start.elapsed());

        let next_token = logits
            .squeeze(0)
            .and_then(|t| t.argmax(candle_core::D::Minus1))
            .and_then(|t| t.to_scalar::<u32>())
            .unwrap();

        let gen_start = std::time::Instant::now();
        let mut all_tokens = vec![next_token];
        let mut cur_pos = input_ids.len();
        let mut prev_token = next_token;
        // Generous budget — thinking reasoning + code + explanation can be long.
        for _ in 0..1200 {
            if prev_token == im_end_id {
                break;
            }
            let next_input = candle_core::Tensor::new(&[prev_token], &device)
                .and_then(|t| t.reshape((1, 1)))
                .unwrap();
            let logits = model.forward(&next_input, cur_pos).unwrap();
            let tok = logits
                .squeeze(0)
                .and_then(|t| t.argmax(candle_core::D::Minus1))
                .and_then(|t| t.to_scalar::<u32>())
                .unwrap();
            all_tokens.push(tok);
            prev_token = tok;
            cur_pos += 1;
        }
        let gen_elapsed = gen_start.elapsed();
        eprintln!(
            "Generation in {gen_elapsed:?} ({} tokens, ~{:.1} tok/s)",
            all_tokens.len(),
            all_tokens.len() as f64 / gen_elapsed.as_secs_f64()
        );

        let full_output = tokenizer.decode(&all_tokens, false).unwrap_or_default();

        // Split thinking from output. Qwen3 emits <think>...</think> at the start.
        let (thinking, answer) = if let Some(start) = full_output.find("<think>") {
            let after_open = &full_output[start + "<think>".len()..];
            if let Some(close) = after_open.find("</think>") {
                let thinking = after_open[..close].trim().to_string();
                let answer = after_open[close + "</think>".len()..].trim().to_string();
                (thinking, answer)
            } else {
                (String::new(), full_output.clone())
            }
        } else {
            (String::new(), full_output.clone())
        };

        eprintln!("\n========== MODEL THINKING ==========\n{thinking}\n========== END THINKING ==========\n");
        eprintln!(
            "\n========== MODEL ANSWER ==========\n{answer}\n========== END ANSWER ==========\n"
        );

        // Extract the Python block from the answer (or full output if no <think>).
        let extract_python_block = |out: &str| -> Option<String> {
            let start = out.find("```python")?;
            let after = &out[start + "```python".len()..];
            let after = after.strip_prefix('\n').unwrap_or(after);
            let end = after.find("```")?;
            Some(after[..end].to_string())
        };
        let traced_code = extract_python_block(&answer)
            .or_else(|| extract_python_block(&full_output))
            .expect("model output should contain a ```python``` block");

        let out_path = std::env::temp_dir().join("rs_llmctl_count_traced.py");
        std::fs::write(&out_path, &traced_code).unwrap();
        eprintln!("\n========== EXTRACTED PROGRAM ==========");
        eprintln!(
            "Wrote {} bytes to {}",
            traced_code.len(),
            out_path.display()
        );
        eprintln!("--- BEGIN traced program ---\n{traced_code}\n--- END traced program ---\n");

        // Soft assertions: the model should have recognised a tracing pattern
        // and applied it. We check for any of the three pattern markers; we do
        // NOT execute the program because the chaosotel package is unlikely to
        // be installed in this test env and a strict-import failure would mask
        // the real signal (was the model's *understanding* correct?).
        let mentions_pattern = traced_code.contains("instrument_action")
            || traced_code.contains("start_as_current_span")
            || traced_code.contains("instrument_db_span")
            || traced_code.contains("get_tracer")
            || traced_code.contains("@instrument_");
        assert!(mentions_pattern,
            "model output should reference one of the chaosotel tracing patterns, got: {traced_code:?}");

        // The counter loop should still be there.
        assert!(
            traced_code.contains("range(") && traced_code.contains("print"),
            "traced program should still contain the original counter loop, got: {traced_code:?}"
        );

        // The model should have given some justification for its pattern choice.
        // We check `answer` rather than `full_output` so the <think> reasoning
        // doesn't trivially satisfy this — we want the *user-visible* answer to
        // include a rationale.
        let lower_answer = answer.to_lowercase();
        let answer_mentions_pattern_choice = lower_answer.contains("pattern 1")
            || lower_answer.contains("pattern 2")
            || lower_answer.contains("pattern 3")
            || lower_answer.contains("manual span")
            || lower_answer.contains("decorator");
        assert!(
            answer_mentions_pattern_choice,
            "user-visible answer should explain which pattern was chosen, got: {answer:?}"
        );

        eprintln!("=== chaosotel tracing test PASSED ===");
        eprintln!("  Model reasoning length:  {} chars", thinking.len());
        eprintln!("  Answer length:           {} chars", answer.len());
        eprintln!("  Extracted code length:   {} bytes", traced_code.len());
        eprintln!(
            "  Generation tokens/s:     {:.1}",
            all_tokens.len() as f64 / gen_elapsed.as_secs_f64()
        );
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    #[ignore = "requires Qwen3-Coder MoE GGUF on disk — slow (load ~30-60 s + 2× gen)"]
    #[allow(clippy::explicit_counter_loop)] // cur_pos counter is intentional for the LLM-inference loop with early `break` on EOS
    fn qwen3_moe_coder_python_counting_program() {
        // M2 acceptance test: Qwen3-Coder-30B-A3B (MoE) loaded via candle's
        // quantized_qwen3_moe path, exercises both forward and reverse counting.
        //
        // KV cache reset limitation: candle 0.10.2's GGUFQWenMoE does not expose
        // a public clear_kv_cache(). To run two independent prompts we reload
        // the model file between them (costs another ~load_time on each prompt).
        let path_str = std::env::var("QWEN3_MOE_GGUF_PATH").unwrap_or_else(|_|
            "/Users/w199447/.local/share/milliways/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf".into());
        let gguf_path = std::path::PathBuf::from(&path_str);
        if !gguf_path.exists() {
            eprintln!("skipping qwen3 MoE test, no GGUF at {path_str}");
            return;
        }
        eprintln!("Using model: {}", gguf_path.display());

        let device = crate::gemma4_gguf::best_device();
        eprintln!("Device: {device:?}");

        // Read tokenizer once (no cache state).
        let mut tokfile = std::fs::File::open(&gguf_path).unwrap();
        let tokcontent = candle_core::quantized::gguf_file::Content::read(&mut tokfile).unwrap();
        let tokenizer = <tokenizers::tokenizer::Tokenizer as candle_core::quantized::tokenizer::TokenizerFromGguf>::from_gguf(&tokcontent).unwrap();
        let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
        drop(tokcontent);
        drop(tokfile);

        // Helper that loads a FRESH MoE model, runs one prompt, returns output + timings.
        let run = |user_prompt: &str,
                   max_tokens: usize|
         -> (
            String,
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
        ) {
            let load_start = std::time::Instant::now();
            let mut file = std::fs::File::open(&gguf_path).unwrap();
            let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();
            let mut model =
                candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE::from_gguf(
                    content,
                    &mut file,
                    &device,
                    candle_core::DType::F32,
                )
                .unwrap();
            let load_elapsed = load_start.elapsed();

            let prompt =
                format!("<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n");
            let encoding = tokenizer.encode(prompt.as_str(), false).unwrap();
            let input_ids: Vec<u32> = encoding.get_ids().to_vec();

            let prefill_start = std::time::Instant::now();
            let input = candle_core::Tensor::new(input_ids.as_slice(), &device)
                .and_then(|t| t.reshape((1, input_ids.len())))
                .unwrap();
            let logits = model.forward(&input, 0).unwrap();
            let prefill_elapsed = prefill_start.elapsed();

            let next_token = logits
                .squeeze(0)
                .and_then(|t| t.argmax(candle_core::D::Minus1))
                .and_then(|t| t.to_scalar::<u32>())
                .unwrap();

            let gen_start = std::time::Instant::now();
            let mut all_tokens = vec![next_token];
            let mut cur_pos = input_ids.len();
            let mut prev_token = next_token;
            for _ in 0..max_tokens {
                if prev_token == im_end_id {
                    break;
                }
                let next_input = candle_core::Tensor::new(&[prev_token], &device)
                    .and_then(|t| t.reshape((1, 1)))
                    .unwrap();
                let logits = model.forward(&next_input, cur_pos).unwrap();
                let tok = logits
                    .squeeze(0)
                    .and_then(|t| t.argmax(candle_core::D::Minus1))
                    .and_then(|t| t.to_scalar::<u32>())
                    .unwrap();
                all_tokens.push(tok);
                prev_token = tok;
                cur_pos += 1;
            }
            let gen_elapsed = gen_start.elapsed();
            let decoded = tokenizer.decode(&all_tokens, false).unwrap_or_default();
            (decoded, load_elapsed, prefill_elapsed, gen_elapsed)
        };

        eprintln!("\n--- PROMPT 1: count from 1 to 10 ---");
        let (forward_out, load1, p1_prefill, p1_gen) = run(
            "/no_think Write a Python program that counts from 1 to 10.",
            400,
        );
        eprintln!("Load: {load1:?}   Prefill: {p1_prefill:?}   Generation: {p1_gen:?}");
        eprintln!("=== FORWARD OUTPUT ===\n{forward_out}\n=== END ===");

        assert!(
            forward_out.contains("print"),
            "forward output missing print() call: {forward_out:?}"
        );
        let forward_has_iteration = forward_out.contains("range(1, 11)")
            || forward_out.contains("range(1,11)")
            || forward_out.contains("range(10)")
            || forward_out.contains("range(11)")
            || (forward_out.contains("range(") && forward_out.contains("11"))
            || forward_out.contains("for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
        assert!(
            forward_has_iteration,
            "forward output missing forward-iteration construct: {forward_out:?}"
        );
        assert!(
            forward_out.contains("1") && forward_out.contains("10"),
            "forward output missing expected digits: {forward_out:?}"
        );

        eprintln!("\n--- PROMPT 2: count from 10 down to 1 ---");
        let (reverse_out, load2, p2_prefill, p2_gen) = run(
            "/no_think Write a Python program that counts from 10 down to 1.",
            400,
        );
        eprintln!("Load: {load2:?}   Prefill: {p2_prefill:?}   Generation: {p2_gen:?}");
        eprintln!("=== REVERSE OUTPUT ===\n{reverse_out}\n=== END ===");

        assert!(
            reverse_out.contains("print"),
            "reverse output missing print() call: {reverse_out:?}"
        );
        let reverse_has_iteration = reverse_out.contains("range(10, 0, -1)")
            || reverse_out.contains("range(10,0,-1)")
            || reverse_out.contains("range(10, -1, -1)")
            || reverse_out.contains("reversed(")
            || reverse_out.contains("for i in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]")
            || (reverse_out.contains("range(") && reverse_out.contains("-1"));
        assert!(
            reverse_has_iteration,
            "reverse output missing reverse-iteration construct: {reverse_out:?}"
        );
        assert!(
            reverse_out.contains("1") && reverse_out.contains("10"),
            "reverse output missing expected digits: {reverse_out:?}"
        );

        eprintln!("\n=== M2 TIMING SUMMARY (Qwen3-Coder-30B-A3B MoE) ===");
        eprintln!("  Load 1:               {load1:?}");
        eprintln!("  Forward prefill:      {p1_prefill:?}");
        eprintln!("  Forward generation:   {p1_gen:?}");
        eprintln!("  Load 2 (after drop):  {load2:?}");
        eprintln!("  Reverse prefill:      {p2_prefill:?}");
        eprintln!("  Reverse generation:   {p2_gen:?}");
        eprintln!("  Device:               {device:?}");
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    #[ignore = "requires GGUF file on disk"]
    fn gemma4_gguf_tensor_shapes_match_expected_head_dims() {
        let path_str = std::env::var("GEMMA4_GGUF_PATH").unwrap_or_else(|_| {
            "/Users/w199447/.local/share/milliways/models/gemma-4-E4B-it-Q4_K_M.gguf".into()
        });
        let gguf_path = std::path::PathBuf::from(path_str);
        if !gguf_path.exists() {
            return;
        }
        eprintln!("Inspecting: {}", gguf_path.display());
        let mut file = std::fs::File::open(&gguf_path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();

        // Dump all metadata keys
        let mut meta_keys: Vec<_> = content.metadata.keys().collect();
        meta_keys.sort();
        eprintln!("  All metadata keys ({} total):", meta_keys.len());
        for k in &meta_keys {
            let v = &content.metadata[*k];
            let display = v
                .to_u32()
                .map(|n| format!("{n}"))
                .or_else(|_| v.to_f32().map(|f| format!("{f}")))
                .unwrap_or_else(|_| format!("{v:?}"));
            eprintln!("    {k} = {display}");
        }

        // Dump ALL tensor names to find architecture-specific tensors (altup, sparsity, etc.)
        let mut tensor_names: Vec<&String> = content.tensor_infos.keys().collect();
        tensor_names.sort();
        eprintln!("  All blk.0 tensors:");
        for n in tensor_names.iter().filter(|n| n.starts_with("blk.0.")) {
            eprintln!("    {n} = {:?}", content.tensor_infos[*n].shape);
        }
        eprintln!("  Model-level tensors:");
        for n in tensor_names.iter().filter(|n| !n.starts_with("blk.")) {
            eprintln!("    {n} = {:?}", content.tensor_infos[*n].shape);
        }

        // Dump attn key/value shapes for blk.0 through blk.5 (first global at blk.5)
        eprintln!("  Per-layer attn shapes:");
        for n in 0..6 {
            let q_shape = content
                .tensor_infos
                .get(&format!("blk.{n}.attn_q.weight"))
                .map(|i| i.shape.dims().to_vec());
            let k_shape = content
                .tensor_infos
                .get(&format!("blk.{n}.attn_k.weight"))
                .map(|i| i.shape.dims().to_vec());
            let v_shape = content
                .tensor_infos
                .get(&format!("blk.{n}.attn_v.weight"))
                .map(|i| i.shape.dims().to_vec());
            let out_shape = content
                .tensor_infos
                .get(&format!("blk.{n}.attn_output.weight"))
                .map(|i| i.shape.dims().to_vec());
            let qn_shape = content
                .tensor_infos
                .get(&format!("blk.{n}.attn_q_norm.weight"))
                .map(|i| i.shape.dims().to_vec());
            let kn_shape = content
                .tensor_infos
                .get(&format!("blk.{n}.attn_k_norm.weight"))
                .map(|i| i.shape.dims().to_vec());
            eprintln!("    blk.{n}: q={q_shape:?} k={k_shape:?} v={v_shape:?} out={out_shape:?} q_norm={qn_shape:?} k_norm={kn_shape:?}");
        }

        // Count how many layers have attn_k.weight (vs. shared KV)
        let kv_layers: Vec<usize> = (0..42)
            .filter(|n| {
                content
                    .tensor_infos
                    .contains_key(&format!("blk.{n}.attn_k.weight"))
            })
            .collect();
        eprintln!(
            "  Layers with own attn_k.weight ({}/42): {kv_layers:?}",
            kv_layers.len()
        );

        // Read layer_output_scale values for first 5 layers
        let device = candle_core::Device::Cpu;
        let mut file2 = std::fs::File::open(gguf_path).unwrap();
        let content2 = candle_core::quantized::gguf_file::Content::read(&mut file2).unwrap();
        for layer_idx in 0..5 {
            let key = format!("blk.{layer_idx}.layer_output_scale.weight");
            let val = content2
                .tensor(&mut file2, &key, &device)
                .and_then(|qt| qt.dequantize(&device))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>());
            eprintln!("  blk.{layer_idx}.layer_output_scale = {val:?}");
        }
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    #[ignore = "requires GGUF file on disk"]
    fn gemma4_gguf_tokenizer_assigns_correct_special_token_ids() {
        let gguf_path = std::path::Path::new(
            "/Users/w199447/.local/share/milliways/models/gemma-4-E4B-it-Q4_K_M.gguf",
        );
        if !gguf_path.exists() {
            return;
        }
        let mut file = std::fs::File::open(gguf_path).unwrap();
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();

        let tokenizer = tokenizer_from_gguf_spm(&content).unwrap();
        // Gemma4 E4B uses <|turn> (105) to open turns and <turn|> (106, also EOS) to close them
        let prompt = "<bos><|turn>user\nHello<turn|>\n<|turn>model\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let ids = encoding.get_ids();

        assert_eq!(ids[0], 2, "first token must be BOS (ID 2)");
        assert_eq!(ids[1], 105, "<|turn> must be ID 105");
        let end_pos = ids
            .iter()
            .position(|&id| id == 106)
            .expect("<turn|> (ID 106) must appear");
        assert!(end_pos > 1, "<turn|> must follow the content");
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
                CandleModelFamily::Qwen3Moe,
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
            // Qwen3 MoE is GGUF-only in candle 0.10.2 (no safetensors path),
            // and Kimi / MiniMax are not yet wired at all.
            if matches!(family, CandleModelFamily::Kimi | CandleModelFamily::MiniMax) {
                assert!(metadata.supported_formats.is_empty());
                assert!(metadata.tokenizer_contracts.is_empty());
            } else if matches!(family, CandleModelFamily::Qwen3Moe) {
                assert!(metadata
                    .supported_formats
                    .contains(&NativeModelFormat::Gguf));
                assert!(!metadata
                    .supported_formats
                    .contains(&NativeModelFormat::Safetensors));
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
                if matches!(
                    family,
                    CandleModelFamily::Kimi
                        | CandleModelFamily::MiniMax
                        | CandleModelFamily::Qwen3Moe
                ) {
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
