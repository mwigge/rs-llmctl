//! Candle decoder engines: artifact-backed engine factory, RealCandleDecoder, and the generation loop.
use super::*;

/// Sender used to forward decoded content deltas out of the (synchronous)
/// decode loop as they are produced, so the SSE layer can emit each token as
/// its own `chat.completion.chunk` (Bug 10).
pub type NativeTokenSender = tokio::sync::mpsc::UnboundedSender<String>;

/// Result of a native generation, carrying the exact token counts observed by
/// the decode loop. `prompt_tokens` is the number of tokens actually fed to the
/// model (the templated prompt plus any prepended BOS); `completion_tokens` is
/// the number of tokens the loop actually produced. Neither is a
/// re-tokenization of the decoded string, so they can be billed as
/// [`TokenAccountingMode::NativeExact`] (Bug 12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeGeneration {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub finish_reason: String,
}

#[derive(Debug)]
pub(crate) enum NativeCandleDecoder {
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    Real(RealCandleDecoder),
    #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
    Unavailable,
}

impl NativeCandleDecoder {
    pub(crate) fn load(
        family: CandleModelFamily,
        model_path: &Path,
        artifacts: &CandleArtifactValidation,
    ) -> Result<Self> {
        ensure_candle_family_format_supported(family, artifacts.model_format)?;
        load_real_candle_decoder(family, model_path, artifacts)
    }

    /// Runs a generation, forwarding each decoded content delta to `on_token`
    /// as it is produced and returning the decoded text plus the exact token
    /// counts the loop observed.
    pub(crate) fn generate_streaming(
        &self,
        request: &NativeChatRequest,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<NativeGeneration> {
        #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
        let _ = (request, on_token);
        match self {
            #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
            Self::Real(decoder) => decoder.generate_streaming(request, on_token),
            #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
            Self::Unavailable => bail!(
                "native autoregressive decoding requires the native-candle and native-tokenizers features"
            ),
        }
    }

    /// Convenience wrapper returning only the decoded text (no streaming).
    /// Retained for the readiness/self-test paths and unit tests.
    pub(crate) fn generate(&self, request: &NativeChatRequest) -> Result<String> {
        let mut noop = |_: &str| {};
        Ok(self.generate_streaming(request, &mut noop)?.text)
    }
}

/// Outcome of attempting to reset a native model's KV cache before a new
/// generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvCacheReset {
    /// The KV cache was genuinely cleared; the next generation starts from a
    /// clean state.
    Cleared,
    /// The model cannot clear its KV cache (candle 0.10.2's
    /// `quantized_qwen3_moe` / `quantized_llama` expose no reset and their
    /// inner `ConcatKvCache` has no `reset()`), so any prior request's KV
    /// state is retained across `generate()` calls.
    Retained,
}

/// Fail-closed guard against KV-cache cross-request contamination.
///
/// A model whose cache is [`KvCacheReset::Retained`] is only safe to serve on
/// a fresh session — its very first generation, when the cache is still empty.
/// Serving a second generation off the same retained cache would reuse (and
/// leak) the previous request's KV state, producing corrupted output and
/// potentially exposing another request's context. This refuses that case, so
/// such families must be served from a fresh session (model reload) per
/// request instead of a shared, un-clearable cache.
///
/// `served` records whether this decoder has already produced a generation; it
/// is flipped to `true` on first use.
pub(crate) fn admit_fresh_kv_session(
    reset: KvCacheReset,
    served: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    let already_served = served.swap(true, std::sync::atomic::Ordering::SeqCst);
    if reset == KvCacheReset::Retained && already_served {
        bail!(
            "native model KV cache cannot be cleared for this family in candle 0.10.2; \
             refusing to serve a second request off retained KV state to avoid \
             cross-request contamination — a fresh session (model reload) is required per request"
        );
    }
    Ok(())
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

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
#[derive(Debug)]
pub(crate) struct RealCandleDecoder {
    pub(crate) tokenizer: tokenizers::tokenizer::Tokenizer,
    pub(crate) model: Mutex<RealCandleModel>,
    /// Device the model's weights were loaded onto (`best_device()`:
    /// Metal/CUDA/CPU). Per-step input tensors MUST be built on this device;
    /// building them on `Device::Cpu` while the model lives on a GPU causes a
    /// device-mismatch failure on the first forward pass (Bug 17).
    pub(crate) device: candle_core::Device,
    pub(crate) family: CandleModelFamily,
    /// BOS token id to prepend to the generation prompt's `input_ids`, if the
    /// GGUF tokenizer metadata configures `add_bos_token = true`. See
    /// [`gguf_bos_token_to_prepend`] and [`prepend_bos_if_configured`].
    pub(crate) bos_token_id: Option<u32>,
    /// Whether this decoder has already produced a generation. Used by
    /// [`admit_fresh_kv_session`] to fail closed for families whose KV cache
    /// cannot be cleared between requests (see [`RealCandleModel::reset_kv_cache`]).
    pub(crate) served: std::sync::atomic::AtomicBool,
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub(crate) enum RealCandleModel {
    Qwen3(candle_transformers::models::qwen3::ModelForCausalLM),
    Qwen3Gguf(candle_transformers::models::quantized_qwen3::ModelWeights),
    Qwen3MoeGguf(candle_transformers::models::quantized_qwen3_moe::GGUFQWenMoE),
    DeepSeek2(candle_transformers::models::deepseek2::DeepSeekV2),
    Gemma3(candle_transformers::models::gemma3::Model),
    Gemma4Gguf(crate::gemma4_gguf::ModelWeights),
    Mistral(candle_transformers::models::mistral::Model),
    // Mistral-family models distributed as GGUFs tagged
    // `general.architecture = "llama"` (Devstral, Mistral Small, etc).
    // candle's quantized_llama covers the runtime; the Mistral family
    // name here is operator-facing.
    MistralGguf(candle_transformers::models::quantized_llama::ModelWeights),
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
            Self::MistralGguf(_) => "MistralGguf",
        };
        f.debug_tuple("RealCandleModel").field(&variant).finish()
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl RealCandleDecoder {
    fn generate_streaming(
        &self,
        request: &NativeChatRequest,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<NativeGeneration> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("native Candle model lock is poisoned"))?;
        let reset = model.reset_kv_cache();
        // Fail closed if this family cannot clear its KV cache and has already
        // served a request: reusing retained KV state across requests corrupts
        // output and can leak a prior request's context.
        admit_fresh_kv_session(reset, &self.served)?;

        let prompt = format_native_chat_prompt(self.family, &request.messages);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize native prompt: {err}"))?;
        let mut input_ids = encoding.get_ids().to_vec();
        prepend_bos_if_configured(&mut input_ids, self.bos_token_id);
        if input_ids.is_empty() {
            bail!("native prompt tokenization produced no tokens");
        }

        // Bug 12: input tokens are counted on the ACTUAL prompt fed to the model
        // (templated + BOS), not a separate canonical serialization.
        let prompt_tokens = input_ids.len() as u64;

        let max_tokens = request
            .max_tokens
            .and_then(|tokens| usize::try_from(tokens).ok())
            .unwrap_or(128)
            .clamp(1, 4096);

        // Bug 11: real sampling. temperature 0/None stays deterministic greedy
        // (Sampling::ArgMax); nonzero temperature builds a top-k/top-p sampler
        // seeded from the request for reproducibility.
        let seed = request.seed.unwrap_or(DEFAULT_SAMPLING_SEED);
        let mut logits_processor = candle_transformers::generation::LogitsProcessor::from_sampling(
            seed,
            sampling_from_request(request),
        );
        let stop_sequences: &[String] = request.stop.as_deref().unwrap_or(&[]);

        let mut generated = Vec::new();
        let mut offset = 0usize;
        let family_str = self.family.as_str();
        let input_token_count = input_ids.len();

        // Incremental decode state: track how many chars of the decoded output
        // have already been forwarded so each step emits only the new suffix.
        let mut emitted_chars = 0usize;
        let mut finish_reason = "length";

        // Emits any newly-decoded content beyond `emitted_chars`, honoring stop
        // sequences. Returns `true` when a stop sequence was hit (generation
        // should end). `on_token` only sees the pre-stop portion.
        let flush_delta = |generated: &[u32],
                           on_token: &mut dyn FnMut(&str),
                           emitted_chars: &mut usize|
         -> Result<bool> {
            let full = self
                .tokenizer
                .decode(generated, true)
                .map_err(|err| anyhow::anyhow!("failed to decode native output tokens: {err}"))?;
            // Truncate at the earliest stop sequence, if any.
            let (visible, hit_stop) = match stop_sequences
                .iter()
                .filter_map(|needle| full.find(needle.as_str()))
                .min()
            {
                Some(idx) => (&full[..idx], true),
                None => (full.as_str(), false),
            };
            let visible_chars = visible.chars().count();
            if visible_chars > *emitted_chars {
                let delta: String = visible.chars().skip(*emitted_chars).collect();
                if !delta.is_empty() {
                    on_token(&delta);
                }
                *emitted_chars = visible_chars;
            }
            Ok(hit_stop)
        };

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
            let logits = model.next_token_logits(&step_input, offset, &self.device)?;
            let next = logits_processor
                .sample(&logits)
                .map_err(|err| anyhow::anyhow!("native token sampling failed: {err}"))?;
            offset = offset.saturating_add(step_input.len());
            input_ids.push(next);
            if is_eos_token(&self.tokenizer, next) {
                // EOS on the very first token — nothing to emit.
                finish_reason = "stop";
            } else {
                generated.push(next);
                if flush_delta(&generated, on_token, &mut emitted_chars)? {
                    finish_reason = "stop";
                }
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
            if finish_reason == "length" {
                for _ in 1..max_tokens {
                    let step_input = vec![*input_ids.last().expect("input ids are non-empty")];
                    let logits = model.next_token_logits(&step_input, offset, &self.device)?;
                    let next = logits_processor
                        .sample(&logits)
                        .map_err(|err| anyhow::anyhow!("native token sampling failed: {err}"))?;
                    offset = offset.saturating_add(step_input.len());
                    input_ids.push(next);
                    if is_eos_token(&self.tokenizer, next) {
                        finish_reason = "stop";
                        break;
                    }
                    generated.push(next);
                    if flush_delta(&generated, on_token, &mut emitted_chars)? {
                        finish_reason = "stop";
                        break;
                    }
                }
            }
            let gen_tokens = generated.len();
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

        // Final text: decode the produced tokens, truncated at any stop
        // sequence so the returned content matches what was streamed.
        let mut text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|err| anyhow::anyhow!("failed to decode native output tokens: {err}"))?;
        if let Some(idx) = stop_sequences
            .iter()
            .filter_map(|needle| text.find(needle.as_str()))
            .min()
        {
            text.truncate(idx);
        }

        Ok(NativeGeneration {
            text,
            prompt_tokens,
            // Bug 12: real generated-token count from the decode loop, not a
            // re-tokenization of the decoded string.
            completion_tokens: generated.len() as u64,
            finish_reason: finish_reason.to_string(),
        })
    }
}

/// Maps a request's sampling parameters onto candle's [`Sampling`] strategy.
///
/// Backward-compatible default: `temperature` of `None` or ~0 yields
/// [`Sampling::ArgMax`] (deterministic greedy), preserving the pre-sampling
/// behavior. A nonzero temperature selects top-k, top-p, combined, or plain
/// temperature sampling depending on which cutoffs the request provides.
///
/// [`Sampling`]: candle_transformers::generation::Sampling
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub(crate) fn sampling_from_request(
    request: &NativeChatRequest,
) -> candle_transformers::generation::Sampling {
    use candle_transformers::generation::Sampling;
    let temperature = request.temperature.map(f64::from).filter(|t| *t >= 1e-7);
    let Some(temperature) = temperature else {
        return Sampling::ArgMax;
    };
    let top_k = request.top_k.map(|k| k as usize).filter(|k| *k > 0);
    let top_p = request
        .top_p
        .map(f64::from)
        .filter(|p| *p > 0.0 && *p < 1.0);
    match (top_k, top_p) {
        (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        (Some(k), None) => Sampling::TopK { k, temperature },
        (None, Some(p)) => Sampling::TopP { p, temperature },
        (None, None) => Sampling::All { temperature },
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl RealCandleModel {
    /// Resets the KV cache before a new generation, reporting whether the
    /// cache was genuinely cleared. Families that candle 0.10.2 cannot reset
    /// return [`KvCacheReset::Retained`] so the caller can fail closed instead
    /// of serving off stale cross-request state (see [`admit_fresh_kv_session`]).
    fn reset_kv_cache(&mut self) -> KvCacheReset {
        match self {
            Self::Qwen3(model) => {
                model.clear_kv_cache();
                KvCacheReset::Cleared
            }
            Self::Qwen3Gguf(model) => {
                model.clear_kv_cache();
                KvCacheReset::Cleared
            }
            // candle 0.10.2's quantized_qwen3_moe does not expose a public
            // clear_kv_cache() method; the inner ConcatKvCache also lacks reset().
            // The cache cannot be cleared, so retained state must not be reused.
            Self::Qwen3MoeGguf(_) => {
                tracing::warn!(
                    "Qwen3 MoE KV cache cannot be cleared in candle 0.10.2 — \
                     a fresh session (model reload) is required per request"
                );
                KvCacheReset::Retained
            }
            Self::DeepSeek2(model) => {
                model.clear_kv_cache();
                KvCacheReset::Cleared
            }
            Self::Gemma3(model) => {
                model.clear_kv_cache();
                KvCacheReset::Cleared
            }
            Self::Gemma4Gguf(model) => {
                model.clear_kv_cache();
                KvCacheReset::Cleared
            }
            Self::Mistral(model) => {
                model.clear_kv_cache();
                KvCacheReset::Cleared
            }
            // candle 0.10.2's quantized_llama also does not expose a public
            // clear_kv_cache(). Same pattern as Qwen3 MoE — retained state must
            // not be reused across requests.
            Self::MistralGguf(_) => {
                tracing::warn!(
                    "Mistral GGUF KV cache cannot be cleared in candle 0.10.2 — \
                     a fresh session (model reload) is required per request"
                );
                KvCacheReset::Retained
            }
        }
    }

    /// Runs one forward pass and returns the 1-D next-token logits tensor.
    /// Token selection (greedy or sampled) is performed by the caller via a
    /// [`candle_transformers::generation::LogitsProcessor`], so sampling
    /// parameters from the request are honored (Bug 11).
    fn next_token_logits(
        &mut self,
        input_ids: &[u32],
        offset: usize,
        device: &candle_core::Device,
    ) -> Result<candle_core::Tensor> {
        let input = candle_core::Tensor::new(input_ids, device)
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
            Self::MistralGguf(model) => model.forward(&input, offset),
        }
        .with_context(|| "native Candle model forward pass failed")?;
        let next_logits = match logits.dims() {
            [_, seq_len, _] => logits
                .narrow(1, seq_len.saturating_sub(1), 1)
                .and_then(|tensor| tensor.squeeze(1))
                .and_then(|tensor| tensor.squeeze(0)),
            [seq_len, _] => logits
                .narrow(0, seq_len.saturating_sub(1), 1)
                .and_then(|tensor| tensor.squeeze(0)),
            [_] => Ok(logits),
            dims => bail!("native Candle model returned unsupported logits shape: {dims:?}"),
        }
        .with_context(|| "failed to select native next-token logits")?;
        Ok(next_logits)
    }
}

/// Prepends `bos_token_id` to `input_ids` if it is configured and not already
/// the first element.
///
/// No-op when `bos_token_id` is `None`, or when `input_ids` already starts with
/// that id (e.g. because the tokenizer's post-processor already added it).
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub(crate) fn prepend_bos_if_configured(input_ids: &mut Vec<u32>, bos_token_id: Option<u32>) {
    let Some(bos_token_id) = bos_token_id else {
        return;
    };
    if input_ids.first() != Some(&bos_token_id) {
        input_ids.insert(0, bos_token_id);
    }
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
