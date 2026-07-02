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
fn md_u32_array(ct: &gguf_file::Content, suffix: &str, expected_len: usize) -> Result<Vec<u32>> {
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
            let feed_forward_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
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
