// Vendored Gemma4 quantized model — re-implements the transformer forward pass
// because candle-transformers 0.10.2's quantized_gemma3 lacks every Gemma4
// architectural feature below. See docs/native-gguf-internals.md for the
// derivation; the short list is:
//   1. Per-layer variable head_dim (256 SWA / 512 global) derived from the
//      actual attn_q weight shape, not from sliding_window_pattern metadata.
//   2. Cross-layer KV sharing: the last `shared_kv_layers` layers reuse K/V
//      from layer (n_own_kv - 2) for SWA or (n_own_kv - 1) for Global.
//   3. Per-Layer Embedding (PLE): per-token, per-layer 256-dim conditioning
//      with `1/sqrt(embedding_length)` projection scale, `sqrt(per_layer_dim)`
//      embedding scale, and `1/sqrt(2)` combined-input scale.
//   4. `layer_output_scale` applied to the complete layer output after PLE.
//   5. `final_logit_softcapping = 30` applied after the LM head.
//   6. V RMS-normalised without learnable weights; attention scaling = 1.0
//      (q_norm absorbs the 1/sqrt(head_dim) factor — applying it twice biases
//      argmax toward punctuation tokens).
//   7. Input embedding scaled by sqrt(embedding_length).

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::D;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

const MAX_SEQ_LEN: usize = 131072;
const DEFAULT_ROPE_FREQ_GLOBAL: f32 = 1_000_000.;
const DEFAULT_ROPE_FREQ_SWA: f32 = 10_000.;

/// Known Gemma 4 / Gemma 3n GGUF architecture profile.
///
/// Different publishers stamp the GGUF metadata with different `general.architecture`
/// values and matching key prefixes for the same underlying architecture. The loader
/// reads metadata under the canonical `gemma3.` prefix (inherited from the original
/// candle-transformers adaptation); this struct captures the *source* prefix to
/// remap from, so the rest of the code stays generic.
#[derive(Debug, Clone, Copy)]
pub struct Gemma4Profile {
    /// Architecture identifier in `general.architecture` and the prefix for keys.
    pub source_prefix: &'static str,
    /// Human-readable label for logs and error messages.
    pub label: &'static str,
}

/// Profile for the Gemma 4 E4B Q4_K_M model the vendored loader was originally
/// validated against (`general.architecture = "gemma4"`, 42 layers, 2560 hidden,
/// 10.7 GB F32 PLE table). This is the canonical known-working configuration —
/// do not change without re-running the coherent-output integration test.
pub const PROFILE_GEMMA4_E4B: Gemma4Profile = Gemma4Profile {
    source_prefix: "gemma4",
    label: "Gemma 4 E4B (general.architecture = gemma4)",
};

/// Profile for the Gemma 4 E2B (a.k.a. "Gemma 3n E2B" in Google's official
/// naming) GGUF. Same PLE + shared_kv_layers architecture as E4B; smaller
/// dimensions (~30 layers, narrower hidden, smaller PLE table). Files from
/// the unsloth/`gemma-3n-E2B-it-GGUF` repo stamp `general.architecture` as
/// `"gemma3n"` and use `gemma3n.*` for all attention/rope/ple keys.
pub const PROFILE_GEMMA4_E2B: Gemma4Profile = Gemma4Profile {
    source_prefix: "gemma3n",
    label: "Gemma 4 E2B / Gemma 3n (general.architecture = gemma3n)",
};

/// All known profiles, scanned in order by [`detect_profile`].
pub const KNOWN_PROFILES: &[Gemma4Profile] = &[PROFILE_GEMMA4_E4B, PROFILE_GEMMA4_E2B];

/// Pick a profile by matching `general.architecture` in the GGUF metadata.
/// Returns `None` if the file declares an architecture we have not validated.
#[must_use]
pub fn detect_profile(
    content: &candle_core::quantized::gguf_file::Content,
) -> Option<&'static Gemma4Profile> {
    let arch = content
        .metadata
        .get("general.architecture")?
        .to_string()
        .ok()?;
    KNOWN_PROFILES.iter().find(|p| p.source_prefix == arch)
}

/// Pick the fastest device available at runtime.
///
/// Tries GPU backends compiled in via cargo features (`gpu-metal`, `gpu-cuda`)
/// and falls back to CPU. `gpu-cuda` covers AMD GPUs on Linux when built with
/// ROCm/HIP's CUDA-compatibility shim (`HIP_PLATFORM=amd`).
#[must_use]
pub fn best_device() -> Device {
    #[cfg(feature = "gpu-metal")]
    if let Ok(d) = Device::new_metal(0) {
        tracing::info!(backend = "metal", "using GPU for Candle inference");
        return d;
    }
    #[cfg(feature = "gpu-cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        tracing::info!(backend = "cuda", "using GPU for Candle inference");
        return d;
    }
    tracing::info!(backend = "cpu", "using CPU for Candle inference");
    Device::Cpu
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qt: QTensor) -> Result<Self> {
        Ok(Self {
            inner: candle_core::quantized::QMatMul::from_qtensor(qt)?,
            span: tracing::span!(tracing::Level::TRACE, "qmatmul"),
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _e = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let up = self.up.forward(xs)?;
        self.down.forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, freq_base: f32, device: &Device) -> Result<Self> {
        let theta: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            cos: idx.cos()?,
            sin: idx.sin()?,
        })
    }

    fn apply_one(&self, t: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, _h, seq, _d) = t.dims4()?;
        let cos = self.cos.narrow(0, pos, seq)?;
        let sin = self.sin.narrow(0, pos, seq)?;
        candle_nn::rotary_emb::rope(&t.contiguous()?, &cos, &sin)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    attn_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: Mlp,
    // Per-layer embedding (PLE) components — Gemma4-specific residual modulation.
    // inp_gate projects hidden→per_layer_dim, proj projects back, post_norm normalises.
    inp_gate: QMatMul,
    proj: QMatMul,
    post_norm: RmsNorm,
    // Trained scalar applied to the full layer output after attention + FFN + PLE.
    // Values differ strongly from 1.0 (e.g. layer 0 ≈ 0.061); skipping this causes
    // magnitude blow-up after 42 layers, producing completely garbled output.
    layer_scalar: f32,
    // Cross-layer KV sharing (gemma4.attention.shared_kv_layers).
    // When Some(src), this layer skips K/V projection entirely and reuses the
    // already-computed and cached K/V from layer `src`.  Source layers always
    // execute before sharing layers, so the cache is guaranteed to be populated.
    kv_source_layer: Option<usize>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,
    rms_norm_eps: f64,
    sliding_window: Option<usize>,
    rope: RotaryEmbedding,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn mask(
        &self,
        b: usize,
        seq: usize,
        pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<u32> = if let Some(w) = self.sliding_window {
            (0..seq)
                .flat_map(|i| (0..seq).map(move |j| if i < j || j + w < i { 0u32 } else { 1u32 }))
                .collect()
        } else {
            (0..seq)
                .flat_map(|i| (0..seq).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq, seq), device)?;
        let mask = if pos > 0 {
            let zeros = Tensor::zeros((seq, pos), DType::F32, device)?;
            Tensor::cat(&[&zeros, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b, 1, seq, seq + pos))?.to_dtype(dtype)
    }

    // `ext_kv`: when Some, this is a sharing layer — skip K/V projection and use
    // the provided (already RoPE-encoded, already cached) K/V from the source layer.
    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        pos: usize,
        ext_kv: Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let _e = self.span_attn.enter();
        let (b, seq, _) = x.dims3()?;

        let q = self.wq.forward(x)?;
        let q = q
            .reshape((b, seq, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let q = self.rope.apply_one(&q, pos)?;

        let (k, v) = if let Some((ek, ev)) = ext_kv {
            // Sharing layer: reuse source layer's fully-accumulated, RoPE-encoded cache.
            (ek, ev)
        } else {
            let k = self.wk.forward(x)?;
            let v = self.wv.forward(x)?;
            let k = k
                .reshape((b, seq, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b, seq, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = self.k_norm.forward(&k.contiguous()?)?;
            // V gets plain RMS normalization (no learnable scale) per Gemma4 reference:
            //   v = v / sqrt(mean(v^2) + eps)
            let v = {
                let v = v.contiguous()?;
                let var = (v.sqr()?.mean_keepdim(D::Minus1)? + self.rms_norm_eps)?;
                v.broadcast_div(&var.sqrt()?)?
            };
            let k = self.rope.apply_one(&k, pos)?;

            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((kc, vc)) => {
                    if pos == 0 {
                        (k, v)
                    } else {
                        (Tensor::cat(&[kc, &k], 2)?, Tensor::cat(&[vc, &v], 2)?)
                    }
                }
            };
            self.kv_cache = Some((k.clone(), v.clone()));
            (k, v)
        };

        let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Gemma4 text attention uses scaling=1.0 (the learnable q_norm absorbs the
        // 1/sqrt(head_dim) factor implicitly). Applying it again here over-attenuates
        // softmax and biases argmax toward outlier tokens.
        let mut attn = q.matmul(&k.transpose(2, 3)?)?;

        if let Some(m) = mask {
            let m = m.broadcast_as(attn.shape())?;
            let neg = self.neg_inf.broadcast_as(attn.dims())?;
            attn = m.eq(0u32)?.where_cond(&neg, &attn)?;
        }

        let out = candle_nn::ops::softmax_last_dim(&attn)?.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, seq, self.q_dim))?;
        self.wo.forward(&out)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    // Model-level PLE tensors.  per_layer_token_embd is a separate vocabulary embedding
    // (shape [vocab, num_layers * per_layer_dim]) that provides a per-token, per-layer signal.
    // per_layer_model_proj projects the main hidden state into the same space.
    // per_layer_proj_norm normalises the projected component before adding.
    per_layer_token_embd: Embedding,
    per_layer_model_proj: QMatMul,
    per_layer_proj_norm: RmsNorm,
    num_layers: usize,
    per_layer_dim: usize,
    // tanh(logits / cap) * cap applied to final vocab logits before returning.
    final_logit_softcap: Option<f64>,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Metadata arrives with "gemma3." prefix because remap_gguf_arch_prefix was called
        // before this function. All metadata reads use the remapped prefix.
        let md = |key: &str| {
            let k = format!("gemma3.{key}");
            ct.metadata
                .get(&k)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF key {k}")))
        };

        let head_count = md("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md("block_count")?.to_u32()? as usize;
        let embedding_length = md("embedding_length")?.to_u32()? as usize;
        let key_length_global = md("attention.key_length")?.to_u32()? as usize;
        let key_length_swa = md("attention.key_length_swa")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(key_length_global / 2);
        let rms_norm_eps = md("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md("attention.sliding_window")?.to_u32()? as usize;
        let rope_freq_global = md("rope.freq_base")
            .and_then(|v| v.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQ_GLOBAL);
        let rope_freq_swa = md("rope.freq_base_swa")
            .and_then(|v| v.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQ_SWA);
        let final_logit_softcap = md("final_logit_softcapping")
            .and_then(|v| v.to_f32())
            .map(|v| v as f64)
            .ok();
        // shared_kv_layers: last N layers reuse K/V from layer (i % n_own_kv).
        // For E4B: shared_kv_layers=18, n_own_kv=24; layers 24-41 mirror layers 0-23.
        let shared_kv_layers = md("attention.shared_kv_layers")
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(0);
        let n_own_kv = block_count.saturating_sub(shared_kv_layers);
        tracing::info!(
            shared_kv_layers,
            n_own_kv,
            block_count,
            "cross-layer KV sharing"
        );

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = ct
            .tensor(reader, "output.weight", device)
            .or_else(|_| ct.tensor(reader, "token_embd.weight", device))?;

        let mut layers = Vec::with_capacity(block_count);
        for n in 0..block_count {
            let p = format!("blk.{n}");

            // Derive head_dim from the Q weight shape stored in tensor_infos.
            // GGUF stores weight matrices transposed: shape[0] is the output dim.
            // Global layers: q_out = head_count * key_length_global = 8*512 = 4096
            // SWA layers:    q_out = head_count * key_length_swa   = 8*256 = 2048
            let head_dim = ct
                .tensor_infos
                .get(&format!("{p}.attn_q.weight"))
                .and_then(|info| info.shape.dims().first().copied())
                .map(|q_out| q_out / head_count)
                .unwrap_or(key_length_global);

            let is_swa = head_dim == key_length_swa;
            let sliding_window = is_swa.then_some(sliding_window_size);
            let rope_freq = if is_swa {
                rope_freq_swa
            } else {
                rope_freq_global
            };
            let q_dim = head_count * head_dim;
            // KV sharing mapping (matches llama.cpp gemma4 reuse callback):
            //   SWA sharing layer    → n_own_kv - 2  (last SWA layer with own K/V)
            //   Global sharing layer → n_own_kv - 1  (last global layer with own K/V)
            let kv_source_layer = if n >= n_own_kv {
                Some(n_own_kv.saturating_sub(if is_swa { 2 } else { 1 }))
            } else {
                None
            };

            let wq =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?)?;
            let wk =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?)?;
            let wv =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_v.weight"), device)?)?;
            let wo = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{p}.attn_output.weight"),
                device,
            )?)?;
            let q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_attn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_attention_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let mlp = Mlp {
                gate: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_gate.weight"),
                    device,
                )?)?,
                up: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_up.weight"),
                    device,
                )?)?,
                down: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_down.weight"),
                    device,
                )?)?,
            };

            // PLE layer components
            let inp_gate = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{p}.inp_gate.weight"),
                device,
            )?)?;
            let proj =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.proj.weight"), device)?)?;
            let post_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{p}.post_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let layer_scalar = ct
                .tensor(reader, &format!("{p}.layer_output_scale.weight"), device)
                .and_then(|qt| qt.dequantize(device))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map(|v| v[0])
                .unwrap_or(1.0f32);

            layers.push(LayerWeights {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                attn_norm,
                post_attn_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                inp_gate,
                proj,
                post_norm,
                layer_scalar,
                kv_source_layer,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                q_dim,
                rms_norm_eps,
                sliding_window,
                rope: RotaryEmbedding::new(head_dim, rope_freq, device)?,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                span_mlp: tracing::span!(tracing::Level::TRACE, "attn-mlp"),
            });
        }

        // per_layer_dim inferred from per_layer_proj_norm weight shape (256 for E4B).
        let per_layer_dim = ct
            .tensor_infos
            .get("per_layer_proj_norm.weight")
            .and_then(|info| info.shape.dims().first().copied())
            .unwrap_or(256);

        // per_layer_token_embd is large ([vocab, num_layers * per_layer_dim]).
        // Dequantize to F32 at load time; this is a one-time cost (~10.7 GB for E4B,
        // ~5 GB for E2B). F16 dequantization was tried but introduces precision
        // collapse on Metal — the model produces "()" instead of "Hello". F16
        // appears to work on CPU but the Metal F32→F16 cast in dequantize_f16
        // loses enough fidelity to bias argmax. Keep F32 until a chunked or
        // on-demand dequant path is added.
        tracing::info!(
            per_layer_dim,
            num_layers = block_count,
            "loading per_layer_token_embd (large — dequantizing to F32)"
        );
        let per_layer_token_embd_t = ct
            .tensor(reader, "per_layer_token_embd.weight", device)?
            .dequantize(device)?;
        let per_layer_flat_dim = per_layer_dim * block_count;
        let per_layer_model_proj =
            QMatMul::from_qtensor(ct.tensor(reader, "per_layer_model_proj.weight", device)?)?;
        let per_layer_proj_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "per_layer_proj_norm.weight", device)?,
            rms_norm_eps,
        )?;

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            per_layer_token_embd: Embedding::new(per_layer_token_embd_t, per_layer_flat_dim),
            per_layer_model_proj,
            per_layer_proj_norm,
            num_layers: block_count,
            per_layer_dim,
            final_logit_softcap,
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    pub fn forward(&mut self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (b, seq) = x.dims2()?;
        let _e = self.span.enter();

        let mut h = self.tok_embeddings.forward(x)?;
        h = (h * (self.embedding_length as f64).sqrt())?;

        // PLE pre-loop: compute per-token, per-layer conditioning signals.
        // Mirrors HuggingFace Gemma4 reference exactly:
        //   per_layer_emb  = embedding(tok_ids) * sqrt(per_layer_dim)
        //   per_layer_proj = norm(reshape(W_proj @ h * 1/sqrt(n_embd)))
        //   per_layer_inputs = (per_layer_proj + per_layer_emb) * 1/sqrt(2)
        let flat_ids = x.reshape((b * seq,))?;
        let ple_emb_scale = (self.per_layer_dim as f64).sqrt();
        let per_layer_emb = (self.per_layer_token_embd.forward(&flat_ids)? * ple_emb_scale)?
            .reshape((b, seq, self.num_layers, self.per_layer_dim))?;
        let proj_scale = 1.0f64 / (self.embedding_length as f64).sqrt();
        let per_layer_proj = (self.per_layer_model_proj.forward(&h)? * proj_scale)?.reshape((
            b,
            seq,
            self.num_layers,
            self.per_layer_dim,
        ))?;
        let per_layer_proj = self
            .per_layer_proj_norm
            .forward(&per_layer_proj.contiguous()?)?;
        let inv_sqrt2 = 1.0f64 / 2.0f64.sqrt();
        let per_layer_inputs = ((per_layer_emb + per_layer_proj)? * inv_sqrt2)?;

        for layer_idx in 0..self.num_layers {
            // Resolve external K/V BEFORE mutably borrowing the layer.
            // Sharing layers (kv_source_layer = Some(src)) reuse the source layer's
            // already-populated kv_cache.  Source layers always have lower indices so
            // their cache is guaranteed to be Some by the time we reach here.
            let ext_kv: Option<(Tensor, Tensor)> = match self.layers[layer_idx].kv_source_layer {
                Some(src_idx) => {
                    let (ek, ev) = self.layers[src_idx].kv_cache.as_ref().ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "layer {layer_idx}: source layer {src_idx} kv_cache not populated"
                        ))
                    })?;
                    Some((ek.clone(), ev.clone()))
                }
                None => None,
            };

            let layer = &mut self.layers[layer_idx];

            let mask = if seq == 1 {
                None
            } else {
                Some(layer.mask(b, seq, pos, x.dtype(), x.device())?)
            };

            // Attention block (pre-norm + sub-block + post-norm + residual)
            let residual = h.clone();
            let h_normed = layer.attn_norm.forward(&h)?;
            let attn_out = layer.forward_attn(&h_normed, mask.as_ref(), pos, ext_kv)?;
            let attn_out = layer.post_attn_norm.forward(&attn_out)?;
            h = (attn_out + residual)?;

            // FFN block
            let _e2 = layer.span_mlp.enter();
            let residual = h.clone();
            let ffn_in = layer.ffn_norm.forward(&h)?;
            let ffn_out = layer.mlp.forward(&ffn_in)?;
            let ffn_out = layer.post_ffn_norm.forward(&ffn_out)?;
            h = (ffn_out + residual)?;

            // PLE block: pe_in = current h; output = pe_in + proj_norm(proj(gelu(gate(h)) * ple_in[il]))
            let pe_in = h.clone();
            let ple_in = per_layer_inputs.narrow(2, layer_idx, 1)?.squeeze(2)?;
            let x_g = layer.inp_gate.forward(&h)?;
            let x_g = x_g.gelu()?;
            let x_g = (x_g * ple_in)?;
            let x_g = layer.proj.forward(&x_g)?;
            let x_g = layer.post_norm.forward(&x_g)?;
            h = (pe_in + x_g)?;

            // layer_output_scale: scale the complete layer output (matches llama.cpp).
            h = (h * layer.layer_scalar as f64)?;
        }

        let _e = self.span_output.enter();
        let x = h.i((.., seq - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let logits = self.output.forward(&x)?;
        if let Some(cap) = self.final_logit_softcap {
            (logits.clone() / cap)?.tanh()? * cap
        } else {
            Ok(logits)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache = None;
        }
    }
}
