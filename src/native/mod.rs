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
mod families_validation;
pub use families_validation::*;
mod decoder;
pub use decoder::*;
mod candle_engine;
pub use candle_engine::*;
mod candle_loader;
pub use candle_loader::*;

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
mod tests;

#[cfg(test)]
mod candle_decoder_tests;

#[cfg(all(feature = "native-candle", feature = "native-tokenizers", test))]
mod gguf_tokenizer_tests;

#[cfg(all(feature = "native-candle", feature = "native-tokenizers", test))]
mod quantized_gemma4_tests;
