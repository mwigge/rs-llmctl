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

mod dto;
pub use dto::*;
mod scheduler;
pub use scheduler::*;
mod accounting;
pub use accounting::*;
mod embeddings;
pub use embeddings::*;
mod placement;
pub use placement::*;
mod families;
pub use families::*;
mod decoder;
pub use decoder::*;

const STARTER_ROLES: &[&str] = &["query", "recommendation", "thinking", "coding"];

pub trait NativeEngine: Send + Sync {
    fn model_alias(&self) -> &str;
    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>>;

    fn chat_stream(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        self.chat(request)
    }

    /// Runs a streaming chat, forwarding each decoded content delta to
    /// `token_tx` as it is produced so the SSE layer can emit one
    /// `chat.completion.chunk` per token (Bug 10). The returned response still
    /// carries the full content and final usage.
    ///
    /// The default implementation targets engines without incremental decode:
    /// it runs the buffered `chat` path and emits the whole response as a single
    /// token, preserving prior behavior for those backends.
    fn chat_stream_tokens(
        &self,
        request: NativeChatRequest,
        token_tx: NativeTokenSender,
    ) -> BoxFuture<'_, Result<NativeChatResponse>> {
        Box::pin(async move {
            let response = self.chat(request).await?;
            if !response.content.is_empty() {
                let _ = token_tx.send(response.content.clone());
            }
            Ok(response)
        })
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

/// In-process llama.cpp engine backed by the `llama-cpp-2` Rust bindings.
///
/// This engine runs model inference directly in-process using the llama.cpp C
/// library via FFI.  GPU offload is configured through `gpu_layers`.  When the
/// `dynamic-backends` feature of `llama-cpp-2` is enabled (as it is here) the
/// appropriate GGML backend (`ROCm`, `Vulkan`, CPU) is selected at runtime by
/// loading `.so` plugins from the backends directory rather than being baked
/// in at compile time.
#[cfg(feature = "llama-cpp-native")]
#[derive(Debug, Clone)]
pub struct LlamaCppNativeEngine {
    /// Display name returned by `NativeEngine::model_alias`.
    pub alias: String,
    /// Absolute path to the GGUF model file.
    pub model_path: PathBuf,
    /// Number of model layers to offload to the GPU.  `0` = CPU-only.
    pub gpu_layers: u32,
}

#[cfg(feature = "llama-cpp-native")]
impl LlamaCppNativeEngine {
    /// Constructs a `LlamaCppNativeEngine` after validating that `model_path` exists.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `model_path` does not point to an existing file.
    pub fn load(alias: String, model_path: &Path, gpu_layers: u32) -> Result<Self> {
        if !model_path.exists() {
            anyhow::bail!("model path not found: {}", model_path.display());
        }
        Ok(Self {
            alias,
            model_path: model_path.to_owned(),
            gpu_layers,
        })
    }
}

/// Acquires the llama.cpp process-global backend singleton.
///
/// `LlamaBackend::init()` returns `Err(BackendAlreadyInitialized)` on every
/// call after the first. This helper treats that error as success, so callers
/// can call it once per request without tracking initialisation state.
#[cfg(feature = "llama-cpp-native")]
fn init_llama_backend() -> Result<llama_cpp_2::llama_backend::LlamaBackend> {
    use llama_cpp_2::{llama_backend::LlamaBackend, LlamaCppError};
    LlamaBackend::init()
        .or_else(|e| {
            if e == LlamaCppError::BackendAlreadyInitialized {
                Ok(LlamaBackend {})
            } else {
                Err(e)
            }
        })
        .map_err(|e| anyhow::anyhow!("llama backend init: {e}"))
}

#[cfg(feature = "llama-cpp-native")]
impl NativeEngine for LlamaCppNativeEngine {
    fn model_alias(&self) -> &str {
        &self.alias
    }

    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        use llama_cpp_2::{
            context::params::LlamaContextParams,
            llama_batch::LlamaBatch,
            model::{params::LlamaModelParams, AddBos, LlamaModel},
            sampling::LlamaSampler,
        };
        use std::{fmt::Write as _, num::NonZeroU32};

        let model_path = self.model_path.clone();
        let gpu_layers = self.gpu_layers;

        Box::pin(async move {
            // The entire decode loop runs in a blocking thread so the Tokio
            // thread pool is not starved during FFI calls.
            let response = tokio::task::spawn_blocking(move || -> Result<NativeChatResponse> {
                let backend = init_llama_backend()?;

                let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
                let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
                    .map_err(|e| anyhow::anyhow!("llama model load: {e}"))?;

                // Plain-text prompt: chat template not applied here (requires
                // llama-cpp-2 feature = "common" which adds a heavy C++ dep).
                let mut prompt = request.messages.iter().fold(String::new(), |mut acc, msg| {
                    write!(acc, "<|{}|>\n{}\n", msg.role, message_content_text(msg))
                        .expect("String write is infallible");
                    acc
                });
                prompt.push_str("<|assistant|>\n");

                let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048));
                let mut ctx = model
                    .new_context(&backend, ctx_params)
                    .map_err(|e| anyhow::anyhow!("llama context init: {e}"))?;

                let tokens = model
                    .str_to_token(&prompt, AddBos::Always)
                    .map_err(|e| anyhow::anyhow!("tokenise prompt: {e}"))?;

                let input_token_count = u64::try_from(tokens.len()).unwrap_or(u64::MAX);

                // Prefill: add all prompt tokens to a single batch.
                let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
                for (i, &token) in tokens.iter().enumerate() {
                    let last = i == tokens.len() - 1;
                    batch
                        .add(token, i32::try_from(i).unwrap_or(i32::MAX), &[0], last)
                        .map_err(|e| anyhow::anyhow!("batch add: {e}"))?;
                }
                ctx.decode(&mut batch)
                    .map_err(|e| anyhow::anyhow!("prefill decode: {e}"))?;

                // Greedy sampler — chain_simple wraps it in a performance-
                // tracking chain automatically.
                let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

                let max_new_tokens =
                    usize::try_from(request.max_tokens.unwrap_or(512)).unwrap_or(512);

                let mut output = String::new();
                let mut output_token_count = 0u64;
                let mut pos = tokens.len();
                // Stateful UTF-8 decoder required by `token_to_piece` — tokens
                // may not always map to complete UTF-8 sequences individually.
                let mut decoder = encoding_rs::UTF_8.new_decoder();

                // Decode loop: one token at a time so OTel hooks can fire per token.
                loop {
                    if output_token_count >= u64::try_from(max_new_tokens).unwrap_or(u64::MAX) {
                        break;
                    }

                    let token_id = sampler.sample(&ctx, -1);

                    if model.is_eog_token(token_id) {
                        break;
                    }

                    sampler.accept(token_id);

                    let piece = model
                        .token_to_piece(token_id, &mut decoder, false, None)
                        .map_err(|e| anyhow::anyhow!("token to piece: {e}"))?;
                    output.push_str(&piece);
                    output_token_count += 1;

                    let mut next_batch = LlamaBatch::new(1, 1);
                    next_batch
                        .add(token_id, i32::try_from(pos).unwrap_or(i32::MAX), &[0], true)
                        .map_err(|e| anyhow::anyhow!("next batch add: {e}"))?;
                    ctx.decode(&mut next_batch)
                        .map_err(|e| anyhow::anyhow!("decode step: {e}"))?;
                    pos += 1;
                }

                Ok(NativeChatResponse {
                    model: request.model.clone(),
                    content: output,
                    tool_calls: None,
                    finish_reason: "stop".to_string(),
                    usage: NativeTokenUsage::with_mode(
                        input_token_count,
                        output_token_count,
                        TokenAccountingMode::NativeExact,
                    ),
                })
            })
            .await
            .map_err(|e| anyhow::anyhow!("spawn_blocking join: {e}"))??;

            Ok(response)
        })
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
mod gemma4_gguf_tokenizer;

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
// Kept compiled for synthetic and real-model validation while runtime Gemma4 routing still uses the main GGUF path.
#[allow(dead_code)]
mod quantized_gemma4;

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
            top_p: None,
            top_k: None,
            seed: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
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
            top_p: None,
            top_k: None,
            seed: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
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
            top_p: None,
            top_k: None,
            seed: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
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
        // BOS is handled at token-ID level (prepend_bos_if_configured), not as a
        // string prefix in the rendered prompt. The prompt must use <|turn> markers.
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

        let tokenizer = tokenizer_from_gguf_content(&content).unwrap();
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

        let tokenizer = tokenizer_from_gguf_content(&content).unwrap();
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

    /// Stage-1 spike for the mistral.rs (mistralrs) integration track. Goal:
    /// confirm that mistralrs 0.8.1 can load the Devstral 24B GGUF (which
    /// candle 0.10.2's quantized_llama can NOT, due to its hardcoded
    /// `head_dim = embedding_length / head_count`) and emit at least one
    /// coherent forward-pass result on Metal. This is the "go/no-go" signal
    /// for the full backend integration in Stage 2.
    #[cfg(all(
        feature = "native-candle",
        feature = "native-tokenizers",
        feature = "mistral-rs"
    ))]
    #[test]
    #[ignore = "Stage 1 spike — requires Devstral GGUF on disk + the mistral-rs feature compiled in"]
    fn mistralrs_devstral_spike() {
        // We use the high-level `GgufModelBuilder`. The `model_id` argument
        // accepts a local directory path (HuggingFace convention — the
        // underlying loader resolves it relative to either an HF repo cache
        // or the absolute path we pass in here).
        let model_dir = std::path::PathBuf::from("/Users/w199447/.local/share/milliways/models");
        let model_file = "Devstral-Small-2505-GGUF-Q4_K_M.gguf";
        let full_path = model_dir.join(model_file);
        if !full_path.exists() {
            eprintln!(
                "skipping mistralrs spike, no GGUF at {}",
                full_path.display()
            );
            return;
        }
        eprintln!(
            "Stage 1 spike: loading {} via mistralrs",
            full_path.display()
        );

        // mistralrs is async-tokio. Build a single-thread runtime for the spike
        // — keeps it independent from any global runtime in production paths.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        let result = rt.block_on(async {
            use mistralrs::{GgufModelBuilder, TextMessageRole, TextMessages};

            // Bind Metal explicitly. Without `.with_device()` mistralrs's
            // auto-mapper picked CPU (24 GB headroom) and the forward pass
            // ran painfully slow. The `mistralrs?/metal` feature toggle in
            // Cargo.toml (gated by our gpu-metal feature) is what makes
            // `Device::new_metal(0)` resolve to a real GPU device here.
            #[cfg(feature = "gpu-metal")]
            let device = candle_core::Device::new_metal(0)
                .map_err(|e| format!("Metal device init failed: {e}"))?;
            #[cfg(not(feature = "gpu-metal"))]
            let device = candle_core::Device::Cpu;
            eprintln!("Stage 1 spike: bound device {device:?}");

            let load_start = std::time::Instant::now();
            let model = GgufModelBuilder::new(
                model_dir.to_string_lossy().to_string(),
                vec![model_file.to_string()],
            )
            .with_device(device)
            .with_logging()
            .build()
            .await
            .map_err(|e| format!("mistralrs build failed: {e}"))?;
            let load_elapsed = load_start.elapsed();
            eprintln!("Stage 1 spike: model loaded in {load_elapsed:?}");

            let messages = TextMessages::new().add_message(
                TextMessageRole::User,
                "Write a Python program that counts from 1 to 10. Output only the code in a single ```python``` block.",
            );

            let send_start = std::time::Instant::now();
            let response = model
                .send_chat_request(messages)
                .await
                .map_err(|e| format!("mistralrs chat request failed: {e}"))?;
            let send_elapsed = send_start.elapsed();

            let content = response
                .choices
                .first()
                .and_then(|c| c.message.content.as_ref())
                .cloned()
                .unwrap_or_default();
            eprintln!("Stage 1 spike: chat completed in {send_elapsed:?}");
            eprintln!(
                "Throughput: prompt {:?} tok/s, completion {:?} tok/s",
                response.usage.avg_prompt_tok_per_sec, response.usage.avg_compl_tok_per_sec
            );
            eprintln!("=== MISTRALRS OUTPUT ===\n{content}\n=== END ===");

            Result::<String, String>::Ok(content)
        });

        let content = result.expect("Stage 1 spike: forward pass should succeed");
        assert!(
            content.contains("print") || content.contains("range"),
            "Stage 1 spike output missing Python code markers: {content:?}"
        );
        eprintln!("=== Stage 1 spike PASSED ===");
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    #[allow(clippy::explicit_counter_loop)] // cur_pos counter is intentional for the LLM-inference loop with early `break` on EOS
    #[ignore = "Mac premium: Devstral / Mistral-family GGUF via candle quantized_llama (slow, requires Devstral GGUF on disk)"]
    fn mistral_runtime_python_counting_program() {
        // Devstral Small 24B (Mistral architecture, agentic-tuned by Mistral AI)
        // as the Mac premium path — same dual-direction Python counting test
        // pattern as Qwen3, with the Mistral [INST]/[/INST] chat format.
        //
        // Devstral GGUF tags `general.architecture = "llama"`, so the runtime
        // routes it through candle's `quantized_llama::ModelWeights::from_gguf`.
        // KV cache reset on quantized_llama is a no-op in candle 0.10.2 (same
        // limitation as Qwen3 MoE) — the test reloads the model between prompts.
        let path_str = std::env::var("MISTRAL_GGUF_PATH").unwrap_or_else(|_| {
            "/Users/w199447/.local/share/milliways/models/Devstral-Small-2505-GGUF-Q4_K_M.gguf"
                .into()
        });
        let gguf_path = std::path::PathBuf::from(&path_str);
        if !gguf_path.exists() {
            eprintln!("skipping mistral test, no GGUF at {path_str}");
            return;
        }
        eprintln!("Using model: {}", gguf_path.display());

        let device = crate::gemma4_gguf::best_device();
        eprintln!("Device: {device:?}");

        // Read tokenizer once (no cache state).
        let mut tokfile = std::fs::File::open(&gguf_path).unwrap();
        let tokcontent = candle_core::quantized::gguf_file::Content::read(&mut tokfile).unwrap();
        let tokenizer = <tokenizers::tokenizer::Tokenizer as candle_core::quantized::tokenizer::TokenizerFromGguf>::from_gguf(&tokcontent).unwrap();
        let eos_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(2);
        drop(tokcontent);
        drop(tokfile);

        // Helper: load FRESH model, run one prompt, return decoded output + timings.
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
            let mut model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                content, &mut file, &device,
            )
            .unwrap();
            let load_elapsed = load_start.elapsed();

            // Mistral instruct format. Devstral accepts this baseline.
            let prompt = format!("<s>[INST] {user_prompt} [/INST]");
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
                if prev_token == eos_id {
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
            "Write a Python program that counts from 1 to 10. Output only the code, in a single fenced ```python``` block.",
            400,
        );
        eprintln!("Load: {load1:?}   Prefill: {p1_prefill:?}   Generation: {p1_gen:?}");
        eprintln!("=== FORWARD OUTPUT ===\n{forward_out}\n=== END ===");

        assert!(
            forward_out.contains("print"),
            "forward output missing print(): {forward_out:?}"
        );
        let forward_has_iter = forward_out.contains("range(1, 11)")
            || forward_out.contains("range(1,11)")
            || forward_out.contains("range(10)")
            || forward_out.contains("range(11)")
            || (forward_out.contains("range(") && forward_out.contains("11"))
            || forward_out.contains("for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
        assert!(
            forward_has_iter,
            "forward output missing forward-iteration construct: {forward_out:?}"
        );

        eprintln!("\n--- PROMPT 2: count from 10 down to 1 ---");
        let (reverse_out, load2, p2_prefill, p2_gen) = run(
            "Write a Python program that counts from 10 down to 1. Output only the code, in a single fenced ```python``` block.",
            400,
        );
        eprintln!("Load: {load2:?}   Prefill: {p2_prefill:?}   Generation: {p2_gen:?}");
        eprintln!("=== REVERSE OUTPUT ===\n{reverse_out}\n=== END ===");

        assert!(
            reverse_out.contains("print"),
            "reverse output missing print(): {reverse_out:?}"
        );
        let reverse_has_iter = reverse_out.contains("range(10, 0, -1)")
            || reverse_out.contains("range(10,0,-1)")
            || reverse_out.contains("range(10, -1, -1)")
            || reverse_out.contains("reversed(")
            || reverse_out.contains("for i in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]")
            || (reverse_out.contains("range(") && reverse_out.contains("-1"));
        assert!(
            reverse_has_iter,
            "reverse output missing reverse-iteration construct: {reverse_out:?}"
        );

        eprintln!("\n=== Mistral / Devstral TIMING SUMMARY ===");
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

        let tokenizer = tokenizer_from_gguf_content(&content).unwrap();
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
            top_p: None,
            top_k: None,
            seed: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
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

    #[cfg(feature = "llama-cpp-native")]
    mod llama_cpp_tests {
        use super::super::{LlamaCppNativeEngine, NativeEngine};

        #[test]
        fn load_rejects_missing_path() {
            let result = LlamaCppNativeEngine::load(
                "test".to_string(),
                std::path::Path::new("/nonexistent/model.gguf"),
                32,
            );
            assert!(result.is_err());
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("not found") || msg.contains("nonexistent"),
                "error message `{msg}` should mention missing path"
            );
        }

        #[test]
        fn load_stores_alias_and_gpu_layers() {
            use tempfile::NamedTempFile;
            let tmp = NamedTempFile::new().expect("create temp file");
            let engine = LlamaCppNativeEngine {
                alias: "my-model".to_string(),
                model_path: tmp.path().to_owned(),
                gpu_layers: 32,
            };
            assert_eq!(engine.model_alias(), "my-model");
            assert_eq!(engine.gpu_layers, 32);
        }

        #[test]
        fn llama_cpp_native_engine_implements_native_engine_trait() {
            fn assert_native_engine<T: crate::native::NativeEngine>() {}
            assert_native_engine::<LlamaCppNativeEngine>();
        }

        #[test]
        fn arc_from_box_dyn_engine_preserves_alias() {
            use std::sync::Arc;
            use tempfile::NamedTempFile;
            let tmp = NamedTempFile::new().expect("temp file");
            let engine =
                LlamaCppNativeEngine::load("registry-alias".to_string(), tmp.path(), 0).unwrap();
            let boxed: Box<dyn NativeEngine> = Box::new(engine);
            let arc: Arc<dyn NativeEngine> = Arc::from(boxed);
            assert_eq!(arc.model_alias(), "registry-alias");
        }
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
