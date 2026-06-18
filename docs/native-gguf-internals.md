# Native GGUF Internals

This document explains how rs-llmctl loads GGUF model files, what transformations
are applied, why certain choices diverge from the upstream defaults, and where the
current boundaries of Candle 0.10.2 support lie. It is written for contributors
and operators who need to debug model loading, extend support to new architectures,
or understand why a specific family behaves differently than expected.

---

## GGUF file structure

A GGUF file has two logical sections: **metadata** (key-value pairs describing
the architecture and tokenizer) and **tensors** (the quantized weight matrices).
Metadata keys are prefixed with the architecture name, e.g. `gemma3.block_count`
or `qwen3.attention.head_count`. Tensor names follow a flat convention
(`blk.{layer}.attn_q.weight`) that is architecture-independent.

Candle reads both sections via `candle_core::quantized::gguf_file::Content::read`.
The resulting `Content` struct has a `metadata: HashMap<String, Value>` and a
separate tensor index; the tensor index contains file offsets only — actual weight
data is read lazily during `from_gguf` construction.

---

## Tokenizer loading

### Standard path (GPT-2 BPE)

Candle's `TokenizerFromGguf` trait reads `tokenizer.ggml.model` to select the
tokenizer type. For `"gpt2"` it:

1. Reads `tokenizer.ggml.tokens` → `Vec<String>` (vocab)
2. Reads `tokenizer.ggml.merges` → `Vec<(String, String)>` (BPE merge rules)
3. Builds a `tokenizers::models::bpe::BPE` model
4. Wraps it with a `ByteLevel` pre-tokenizer and decoder (GPT-2 style byte
   encoding: every byte is represented as a unicode character from a fixed 256-
   character alphabet, so the tokenizer never sees raw spaces — only encoded ones
   like `Ġ`)

This covers Qwen3, Llama3, Smaug, and most other modern BPE-based GGUFs.

### Gemma4 path (SPM-style BPE)

Gemma4 GGUFs set `tokenizer.ggml.model = "gemma4"`. Candle rejects this with
`unsupported tokenizer model 'gemma4'` because it is not in the known list.

Despite the name, Gemma4 does **not** use a SentencePiece Unigram model.
It uses **BPE merges** (stored in `tokenizer.ggml.merges`) — the same structure
as GPT-2. The critical difference is in **whitespace handling**:

| Attribute        | GPT-2 BPE                        | Gemma4 SPM-style BPE         |
|------------------|----------------------------------|------------------------------|
| Whitespace rep.  | Byte-encoded (`Ġ` = 0x20)        | Metaspace (`▁` = U+2581)     |
| Pre-tokenizer    | `ByteLevel` (encode every byte)  | `Metaspace` (replace ` `→`▁`)|
| Decoder          | `ByteLevel` (decode byte chars)  | `Metaspace` (`▁`→` `)        |
| Vocab encoding   | Unicode-escaped bytes            | Raw UTF-8 strings with `▁`   |
| Merges present   | Yes                              | Yes                          |

The `▁` approach comes from SentencePiece (Google's tokenizer library). It
prepends `▁` to the start of each word to mark word boundaries, then runs BPE
merges on the escaped text. Decoding simply reverses the substitution.

**Why this matters in practice:** a GPT-2 tokenizer applied to Gemma4 text
would produce entirely wrong token IDs because the vocab maps `▁hello` → 12345
but the byte-encoded form `Ġhello` does not exist in the vocabulary. All
inference output would be garbage or the tokenizer would fail outright.

### Fallback implementation

`load_generation_tokenizer` in `src/native.rs` calls `TokenizerFromGguf::from_gguf`
first. If it returns an error containing `"unsupported tokenizer model"`, the
fallback `tokenizer_from_gguf_spm` runs instead:

```
tokenizer.ggml.tokens   → Vec<String>          vocab strings (with ▁)
tokenizer.ggml.merges   → Vec<(String,String)> BPE merge pairs
tokenizer.ggml.unk_token_id                    optional unknown token index
tokenizer.ggml.byte_fallback                   optional byte fallback flag
tokenizer.ggml.token_type                      special-token marker (types 2–5)
```

The fallback builds `tokenizers::models::bpe::BPE` from these fields and
wraps it with `Metaspace('▁', PrependScheme::Always, split=true)` as both
pre-tokenizer and decoder. `Metaspace` implements both `PreTokenizer` and
`Decoder` in the `tokenizers` crate — the same instance handles both
directions.

**Why not Unigram?** A SentencePiece Unigram model uses `tokenizer.ggml.scores`
(log-probabilities per token) and a Viterbi lattice to find the optimal
segmentation — no merge rules. Gemma4 GGUFs contain `tokenizer.ggml.merges`,
not scores, confirming BPE rather than Unigram. Mistakenly using Unigram would
silently produce different tokenizations than the reference model.

**Why this is not upstream candle's concern yet:** the `TokenizerFromGguf` trait
in candle-core is intentionally minimal. Adding Gemma4 support there would require
candle to expose its BPE builder with a configurable pre/decoder pair, which is
possible but not yet done in 0.10.2. The fallback lives entirely in rs-llmctl and
falls through transparently for any future candle version that adds support.

---

## Metadata key remapping

Candle's `quantized_gemma3::ModelWeights::from_gguf` detects the architecture
prefix by probing:

```
["gemma3", "gemma2", "gemma", "gemma-embedding"]
```

Gemma4 GGUFs use the prefix `"gemma4"`, which is not in this list. The probe
falls back to `"gemma3"`, then fails immediately because `gemma3.attention.head_count`
does not exist in the Gemma4 metadata.

The fix in `load_real_candle_decoder` calls `remap_gguf_arch_prefix` before
passing the content to `from_gguf`:

```rust
remap_gguf_arch_prefix(content, "gemma4", "gemma3")
```

This copies every `gemma4.*` key into a corresponding `gemma3.*` entry using
`HashMap::entry(...).or_insert(...)` — existing keys are never overwritten, so if
a future GGUF includes both prefixes the original values take precedence.

**Why not patch candle instead?** Patching candle would require vendoring the
crate or maintaining a fork. The remap is a two-line shim that covers the metadata
probe with no behavior change for correctly-prefixed GGUFs.

---

## Gemma4 architecture dimensions (E4B Q4_K_M)

The following values are read from GGUF metadata during model construction.
All dimensions are for the Gemma4 E4B (`gemma-4-E4B-it-Q4_K_M.gguf`) variant.

| Metadata key                            | Value | Meaning                                      |
|-----------------------------------------|-------|----------------------------------------------|
| `gemma4.block_count`                    | 42    | Number of transformer layers                 |
| `gemma4.embedding_length`               | 2560  | Residual stream / hidden dimension           |
| `gemma4.embedding_length_per_layer_input` | 256 | Per-layer input projection width             |
| `gemma4.attention.head_count`           | 8     | Query heads (all attention types)            |
| `gemma4.attention.head_count_kv`        | 2     | Key/Value heads (grouped query attention)    |
| `gemma4.attention.key_length`           | 512   | Head dimension for **global** attention      |
| `gemma4.attention.value_length`         | 512   | Value dimension for global attention         |
| `gemma4.attention.key_length_swa`       | 256   | Head dimension for **sliding window** attn   |
| `gemma4.attention.value_length_swa`     | 256   | Value dimension for sliding window attn      |
| `gemma4.attention.sliding_window`       | 512   | SWA receptive field (tokens)                 |
| `gemma4.attention.sliding_window_pattern` | 1   | Layer pattern (1 = alternating global/SWA)   |
| `gemma4.attention.shared_kv_layers`     | 18    | Layers that share KV across groups           |
| `gemma4.rope.freq_base`                 | ~10k  | RoPE base frequency (global layers)          |
| `gemma4.rope.freq_base_swa`             | ~1k   | RoPE base frequency (SWA layers)             |
| `gemma4.rope.dimension_count`           | 512   | RoPE dimension for global layers             |
| `gemma4.rope.dimension_count_swa`       | 256   | RoPE dimension for SWA layers                |

### Layer type pattern

`sliding_window_pattern = 1` means every other layer alternates between global
attention and sliding window attention. With 42 layers the pattern is roughly:
SWA, global, SWA, global, … (exact assignment depends on the implementation's
interpretation of the integer pattern).

### Q projection tensor shape

Because global and SWA layers have different head dimensions, the Q projection
weight matrix is a **different size** depending on the layer type:

```
Global attention layer:  W_Q shape = [embedding_length, head_count × key_length]
                                   = [2560, 8 × 512] = [2560, 4096]

SWA layer:               W_Q shape = [embedding_length, head_count × key_length_swa]
                                   = [2560, 8 × 256] = [2560, 2048]
```

KV projections follow the same split, scaled by `head_count_kv = 2`:

```
Global KV:  W_K = W_V shape = [2560, 2 × 512] = [2560, 1024]
SWA KV:     W_K = W_V shape = [2560, 2 × 256] = [2560, 512]
```

---

## Gemma4 GGUF forward pass — vendored implementation

`quantized_gemma3` in candle-transformers does not support Gemma4 because it
hardcodes a single `head_dim` per model and lacks five Gemma4-specific
architectural features. `src/gemma4_gguf.rs` is a self-contained re-implementation
that reads the remapped `gemma3.*` metadata and exposes a `ModelWeights` with the
same `forward(input, pos) -> logits` surface as the candle types.

The Gemma4-specific features it implements are listed below; each one is required
for the model to produce coherent output. Removing any single one produces
garbage tokens or unstable logit distributions.

### 1. Per-layer variable head dimension

Each layer's `head_dim` is derived from its actual Q weight tensor shape rather
than from a global metadata field:

```rust
let head_dim = ct.tensor_infos.get(&format!("blk.{n}.attn_q.weight"))
    .and_then(|info| info.shape.dims().first().copied())
    .map(|q_out| q_out / head_count)
    .unwrap_or(key_length_global);
let is_swa = head_dim == key_length_swa;
let rope_freq = if is_swa { rope_freq_swa } else { rope_freq_global };
```

- Global layer: `head_dim = 512`, `rope_freq = 1_000_000`
- SWA layer:    `head_dim = 256`, `rope_freq = 10_000`

This makes the loader robust to changes in how `sliding_window_pattern` is
encoded in the GGUF (the actual pattern is a Bool array of 42 entries).

### 2. Cross-layer KV sharing (`shared_kv_layers = 18`)

The last `shared_kv_layers` layers skip K/V projection entirely and reuse the
K/V cache from an earlier layer. The mapping matches llama.cpp's reuse callback:

```
n_own_kv  = block_count - shared_kv_layers          = 42 - 18 = 24
For layer L >= n_own_kv:
  source = n_own_kv - 2  if L is SWA   (= 22, last own-K/V SWA layer)
  source = n_own_kv - 1  if L is Global (= 23, last own-K/V Global layer)
For layer L < n_own_kv:
  source = None — layer computes and caches its own K/V
```

Sharing layers compute their own Q (with their own `q_norm` + RoPE) but read K
and V from `self.layers[source].kv_cache` directly. Because the source layer's
cache already contains K with RoPE applied for the full sequence, sharing layers
do **not** re-apply RoPE to K. The forward loop is index-based (not `iter_mut`)
so the source's cache can be cloned before mutably borrowing the sharing layer.

### 3. Per-Layer Embedding (PLE)

PLE provides per-token, per-layer conditioning via a dedicated 256-dim signal
that gates each layer's hidden state. The pre-loop computation is:

```
per_layer_emb  = embed_per_layer(token_ids) * sqrt(per_layer_dim)   # [B, T, 42, 256]
per_layer_proj = reshape(W_proj(h) * 1/sqrt(embedding_length))      # [B, T, 42, 256]
per_layer_proj = per_layer_proj_norm(per_layer_proj)                # RMSNorm over 256
per_layer_inputs = (per_layer_proj + per_layer_emb) * 1/sqrt(2)     # [B, T, 42, 256]
```

The `per_layer_token_embd` tensor has shape `[262144, 42 * 256] = [262144, 10752]`
and dequantises to roughly 10.7 GB of F32 — paid once at load time.

Inside each layer, after the FFN residual:

```
ple_in   = per_layer_inputs[:, :, layer_idx, :]      # [B, T, 256]
x_gated  = GELU(inp_gate(h)) * ple_in                # [B, T, 256]
x_proj   = post_norm(proj(x_gated))                  # [B, T, 2560]
h        = h + x_proj                                # residual
```

### 4. `layer_output_scale` — per-layer scalar applied to full layer output

Each layer has a trained scalar (`blk.N.layer_output_scale.weight`, shape `[1]`)
applied to the layer's complete output **after** PLE and all residuals:

```
h = h * layer_scalar          # scalar values e.g. 0.061 (layer 0), 0.840 (layer 2)
```

The final RmsNorm at the output rescales the cumulative magnitude. Applying
this scalar to sub-block deltas instead of the full output produces wrong
distributions because that interpretation breaks the model's quantisation-aware
training calibration.

### 5. Final logit softcap (`final_logit_softcapping = 30`)

After the LM head projection, logits are bounded by:

```
logits = tanh(logits / 30) * 30
```

This is the same final softcap used in Gemma2/3 and is read from
`gemma3.final_logit_softcapping` (post-remap).

### 6. Attention details specific to Gemma4 text attention

- **`scaling = 1.0`** (not `1/sqrt(head_dim)`). The learnable `q_norm` (RMSNorm
  with weight) absorbs the magnitude scaling. Applying `1/sqrt(head_dim)` on
  top would over-attenuate softmax and bias argmax toward outlier tokens.
- **V is RMS-normalised** without a learnable weight before attention:
  `v = v / sqrt(mean(v^2) + eps)` per-head.
- Q and K are RMS-normalised with their learned weights (`attn_q_norm`,
  `attn_k_norm`) **before** RoPE.
- Q-norm shape `[head_dim]`: per-layer scale dimension matches the layer's own
  head_dim (256 or 512), not a global constant.

### 7. Embedding scale

Input embeddings are scaled by `sqrt(embedding_length) = sqrt(2560) ≈ 50.6` at
lookup time, matching Gemma3/4's `ScaledWordEmbedding`. The per-layer embedding
table uses a separate scale: `sqrt(per_layer_dim) = sqrt(256) = 16`.

### Layer execution order (each of 42 layers)

```
residual = h
h = attn_norm(h)                                 # pre-norm (RMSNorm, learned)
h = forward_attn(h, mask, pos, ext_kv)            # ext_kv = None for own K/V,
                                                  # Some(src_cache) for sharing
h = post_attention_norm(h)                       # post-norm (RMSNorm, learned)
h = h + residual

residual = h
h = ffn_norm(h)                                  # pre-norm
h = mlp(h)                                       # SiLU(gate) * up → down
h = post_ffw_norm(h)                             # post-norm
h = h + residual

pe_in = h
h = inp_gate(h)                                  # [B, T, 2560] → [B, T, 256]
h = GELU(h)
h = h * per_layer_inputs[:, :, layer_idx, :]
h = proj(h)                                      # [B, T, 256] → [B, T, 2560]
h = post_norm(h)                                 # RMSNorm
h = pe_in + h

h = h * layer_scalar                             # per-layer scalar (e.g. 0.061)
```

After 42 layers:

```
h = output_norm(h)
logits = lm_head(h)                              # weight-tied to token_embd
logits = tanh(logits / 30) * 30                  # final logit softcap
```

---

## Qwen3 dense family (current daily-driver path)

The native runtime's default tool-capable workload is the Qwen3 dense family.
The 14 B variant at Q4_K_M is the baseline recommendation for Apple Silicon
24 GB unified memory and Linux 16 GB-class discrete GPUs; see the tier matrix
in "GPU acceleration" below.

Unlike Gemma 4, Qwen3 GGUF loading goes through candle's stock
`candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf` — no
vendored module, no metadata-prefix remap, no PLE/AltUp/Laurel surgery. This
section documents what the runtime expects so that load/inference failures
can be diagnosed without re-reading candle internals.

### Qwen3 14B Q4_K_M GGUF metadata reference

Read directly from `unsloth/Qwen3-14B-GGUF::Qwen3-14B-Q4_K_M.gguf`:

| Metadata key                            | Value     | Meaning                                      |
|-----------------------------------------|-----------|----------------------------------------------|
| `general.architecture`                  | `qwen3`   | No remap needed — candle reads `qwen3.*` natively. |
| `qwen3.block_count`                     | 40        | Number of decoder layers                     |
| `qwen3.embedding_length`                | 5 120     | Residual stream / hidden dimension           |
| `qwen3.feed_forward_length`             | 17 408    | FFN intermediate width                       |
| `qwen3.attention.head_count`            | 40        | Query heads                                  |
| `qwen3.attention.head_count_kv`         | 8         | KV heads (GQA factor = 5)                    |
| `qwen3.attention.key_length`            | 128       | Per-head dim — **uniform across all layers** |
| `qwen3.attention.value_length`          | 128       | Same as key_length; no SWA split             |
| `qwen3.attention.layer_norm_rms_epsilon`| 1e-6      | RMS norm epsilon                             |
| `qwen3.context_length`                  | 131 072   | 128 k native context                         |
| `qwen3.rope.freq_base`                  | 5 000 000 | Single RoPE base (no SWA-specific)           |
| `tokenizer.ggml.model`                  | `gpt2`    | Standard GPT-2 BPE — candle's `TokenizerFromGguf::from_gguf` handles it directly. |

What is **absent** versus Gemma 4 (important for understanding why the load
path is so much simpler):

- No `attention.key_length_swa` / `value_length_swa` — head_dim is uniform.
- No `attention.sliding_window_pattern` — all layers are full attention.
- No `attention.shared_kv_layers` — every layer computes its own K/V.
- No `embedding_length_per_layer_input` / `per_layer_*` tensors — no PLE.
- No `altup.*` / `laurel_*` / `activation_sparsity_scale` — no Gemma 3n features.
- No `final_logit_softcapping` — no logit softcap.
- No `layer_output_scale` — no per-layer scalar.

### Qwen3 tensor inventory

Tensor names follow candle's standard transformer naming convention. For each
layer `N ∈ [0, block_count)`:

| Tensor name (GGUF)                  | Shape (Q4_K_M packs the data)   | Notes                                          |
|-------------------------------------|----------------------------------|------------------------------------------------|
| `blk.N.attn_q.weight`               | `[head_count × key_length, embedding_length]` = `[5120, 5120]` | Q projection |
| `blk.N.attn_k.weight`               | `[head_count_kv × key_length, embedding_length]` = `[1024, 5120]` | K projection |
| `blk.N.attn_v.weight`               | `[head_count_kv × value_length, embedding_length]` = `[1024, 5120]` | V projection |
| `blk.N.attn_output.weight`          | `[embedding_length, head_count × key_length]` = `[5120, 5120]` | Output projection |
| `blk.N.attn_q_norm.weight`          | `[key_length]` = `[128]`         | Q RMSNorm (learnable scale, per-head)         |
| `blk.N.attn_k_norm.weight`          | `[key_length]` = `[128]`         | K RMSNorm (learnable scale, per-head)         |
| `blk.N.attn_norm.weight`            | `[embedding_length]` = `[5120]`  | Pre-attention RMSNorm                          |
| `blk.N.ffn_norm.weight`             | `[embedding_length]` = `[5120]`  | Pre-FFN RMSNorm                                |
| `blk.N.ffn_gate.weight`             | `[feed_forward_length, embedding_length]` = `[17408, 5120]` | Gate (SwiGLU) |
| `blk.N.ffn_up.weight`               | `[feed_forward_length, embedding_length]` = `[17408, 5120]` | Up |
| `blk.N.ffn_down.weight`             | `[embedding_length, feed_forward_length]` = `[5120, 17408]` | Down |

Model-level (one each):

| Tensor name                  | Shape                          | Notes                                |
|------------------------------|--------------------------------|--------------------------------------|
| `token_embd.weight`          | `[vocab, embedding_length]`    | Token embedding table                |
| `output_norm.weight`         | `[embedding_length]` = `[5120]`| Final RMSNorm                        |
| `output.weight`              | `[vocab, embedding_length]` *or absent* | LM head — weight-tied to `token_embd.weight` when absent in the file. |

### Qwen3 chat template and tool-call protocol

Qwen3 uses the ChatML prompt format. The runtime's
`format_native_chat_prompt(CandleModelFamily::Qwen3, ...)` emits:

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

EOS / end-of-turn token: `<|im_end|>` (ID 151645). Open-turn token:
`<|im_start|>` (ID 151644).

The model has two response modes:

- **Thinking mode** (default): the model emits `<think>...</think>` then the
  user-visible answer. Useful for hard reasoning; costs ~200-800 extra tokens
  per turn.
- **Fast mode** (`/no_think` directive in the user message): skips the
  thinking block. Recommended for short tool-use turns.

Native tool-calling protocol is `qwen3-native` (advertised in `/v1/models`
under `capabilities.tool_protocol`). Tool invocations are emitted between
`<tool_call>...</tool_call>` markers as a single JSON object per call.
Orchestrators using OpenAI-style `tool_calls` arrays need to translate;
those using the Qwen native shape can consume it directly.

### Setup checklist for the Qwen3 family

1. Download the GGUF into `~/.local/share/milliways/models/`. Recommended
   per tier (matches `tier::recommend_model_for_tier`):
   - Tier 1 (6 GB VRAM): `unsloth/Qwen3-4B-GGUF :: Qwen3-4B-Q4_K_M.gguf` (~2.5 GB on disk)
   - Tier 2 (12 GB VRAM): `unsloth/Qwen3-8B-GGUF :: Qwen3-8B-Q4_K_M.gguf` (~5 GB on disk)
   - Tier 3 (16-18 GB): `unsloth/Qwen3-14B-GGUF :: Qwen3-14B-Q4_K_M.gguf` (~9 GB on disk)
2. Build rs-llmctl with the matching GPU feature:
   `cargo build --release --features native-candle,native-tokenizers,gpu-metal`
   (or `gpu-cuda` for NVIDIA / AMD ROCm).
3. The runtime probes the device with `gemma4_gguf::best_device()` at startup
   and emits an info-level log: `detected hardware tier=tier3-mac
   family=qwen3 params_b=14 recommended_quant=Q4_K_M context_window=131072`.
4. No metadata remap, no profile detection — the file loads directly through
   `candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf`.

### Measured baseline on Apple M-series 24 GB unified, Metal

Captured 2026-06-17 with `qwen3_runtime_python_counting_program`:

| Phase | Value |
|---|---|
| Model load (~9 GB file) | 7.3 s |
| Prefill (30-token chat prompt) | 53-236 ms |
| Generation, greedy (140 tokens, /no_think) | 7.4 s → **~19 tok/s** |
| Working-set peak resident | ~11 GB |
| Swap growth during full forward + reverse test | < 2 GB |

Contrast with Gemma 4 E4B on the same hardware (74 s load, dominated by
F32 PLE dequantisation). Qwen3 is the daily-driver path; Gemma 4 remains
available for users who explicitly select it on 24 GB+ deployments.

### Agentic capability sanity test

The `qwen3_runtime_adds_chaosotel_tracing_to_counter_program` integration
test exercises the full read-context-then-modify pipeline:

1. Feeds the model three real chaostooling-otel tracing patterns + the
   counter program from a prior test.
2. Captures the model's `<think>` block separately from the user-visible
   answer.
3. Extracts the regenerated Python and writes it to
   `/tmp/rs_llmctl_count_traced.py`.
4. Asserts the model (a) picked one of the three patterns and (b)
   justified the choice in the user-visible answer.

Typical execution on Metal: 76 s total wall time, 17 tok/s generation.

---

## Comparison: standard vs Gemma4

| Property             | GPT-2 / Qwen3 GGUF      | Gemma3 GGUF          | Gemma4 GGUF                       |
|----------------------|--------------------------|----------------------|-----------------------------------|
| Tokenizer model key  | `"gpt2"`                 | `"gpt2"`             | `"gemma4"`                        |
| Tokenizer type       | BPE + ByteLevel          | BPE + ByteLevel      | BPE + Metaspace (▁)               |
| Merges in GGUF       | Yes                      | Yes                  | Yes                               |
| Scores in GGUF       | No                       | No                   | No (not Unigram)                  |
| Metadata prefix      | `gpt2.` / `qwen3.`       | `gemma3.`            | `gemma4.` (remapped to `gemma3.`) |
| Head dim             | Uniform                  | Uniform              | Per-layer (512 global / 256 SWA)  |
| KV sharing           | No                       | No                   | Yes (`shared_kv_layers`)          |
| SWA                  | No                       | Yes (uniform)        | Yes (with separate head dim)      |
| Candle 0.10.2 status | Fully supported          | Fully supported      | Vendored module in rs-llmctl     |

---

## Current status and path forward

| Component              | Status                                                          |
|------------------------|-----------------------------------------------------------------|
| Tokenizer loading      | Working — SPM-BPE fallback in `tokenizer_from_gguf_spm`        |
| Metadata key probe     | Working — `remap_gguf_arch_prefix("gemma4", "gemma3")`          |
| Model weight loading   | Working — tensors load via standard `blk.{n}.*` names          |
| Forward pass (GGUF)    | **Working** — vendored `gemma4_gguf::ModelWeights` in `src/`   |
| Forward pass (safetensors) | Working — `gemma3::Model` reads config.json directly        |

The forward pass is implemented in `src/gemma4_gguf.rs` (~480 LoC, gated behind
the `native-candle` + `native-tokenizers` feature flags). The vendored module
re-implements the seven Gemma4-specific features above. It deliberately does
not extend `candle_transformers::quantized_gemma3` because the differences in
attention, PLE, and KV sharing span enough call sites that the diff was larger
than the rewrite.

### Known limitation: Qwen3 MoE is CUDA-only in candle 0.10.2

candle-nn 0.10.2's `moe_gemm` and `moe_gemm_gguf` kernels (in
`candle-nn/src/moe.rs`) return a hard `bail!("moe_gemm[_gguf] is only
implemented for the cuda backend")` on any non-CUDA device. There is no
Metal kernel and no CPU fallback.

Practical impact:

| Tier / backend | Qwen3 dense | Qwen3-Coder MoE |
|---|---|---|
| Tier 3 Mac (Metal) | ✅ Works (default daily driver) | ❌ Bails at first forward pass |
| Tier 3+ Linux NVIDIA (CUDA) | ✅ Works | ✅ Works |
| Tier 3+ Linux AMD (ROCm/HIP CUDA shim) | ✅ Works | ⚠️ Untested — depends on whether the HIP shim implements the specific MoE gemm symbol |
| Tier 1 / 2 CPU fallback | ✅ Works | ❌ Bails at first forward pass |

The runtime's `CandleModelFamily::Qwen3Moe` variant and the
`quantized_qwen3_moe::GGUFQWenMoE` load path are still wired and compile
clean on all backends — the failure surfaces only when the first
`forward()` is called. Operators on Mac / CPU receive a clear error
referencing `moe_gemm_gguf` rather than a silent miscompile.

This limitation is upstream-only. Resolving it requires either:

- a Metal kernel implementation in candle-nn for `moe_gemm[_gguf]`, or
- a generic CPU fallback (slow but correct) in the same module.

Until candle ships either, the Mac tier 3+ premium model recommendation
is effectively the same as Tier 3 (Qwen3 14B Q4_K_M dense). Linux users
with NVIDIA hardware can use the MoE variant; AMD users should verify
their ROCm/HIP install before relying on it.

### Variant decision log — why some "obvious" alternatives are NOT the default

This section is the institutional memory for why specific model variants
and inference engines were investigated and **rejected** (or deferred)
as defaults. It exists so that the next person who asks "why don't we
just use X" gets the answer in one place instead of re-litigating it
from scratch.

#### Why Gemma 4 E4B (vendored) is no longer the daily driver

Gemma 4 E4B Q4_K_M was the first model the native runtime supported.
The vendored `src/gemma4_gguf.rs` module implements seven Gemma 4-specific
features (PLE, shared_kv_layers, per-layer head_dim, layer_output_scale,
F32→F16 attention scaling at 1.0, final_logit_softcap, embedding scale)
and produces correct output on Metal end-to-end.

It stays in the codebase but is **no longer recommended** because:

- **Memory profile is hostile on shared-memory Macs.** The Per-Layer
  Embedding (PLE) tensor is dequantised to F32 at load time (~10.7 GB),
  which on a 24 GB unified-memory Mac leaves only 6-8 GB for the OS
  and other apps. Repeated test runs in the same session trigger swap
  thrashing.
- **Slow cold start.** F32 PLE dequantisation takes ~74 s on M-series
  vs ~7 s for the equivalent-class Qwen3 14B. Operators feel this
  every restart.
- **F16 PLE optimisation breaks on Metal.** Attempting to dequantise
  the PLE table to F16 (cutting resident memory to ~5.4 GB) collapses
  the argmax distribution onto punctuation tokens — the model emits
  `()` instead of coherent text. Root-caused to the candle 0.10.2
  Metal F32→F16 cast losing fidelity in the per-layer embedding
  magnitudes. Until candle ships a direct Q4_K_M→F16 Metal kernel
  the PLE stays F32.
- **No quality advantage over Qwen3 14B on the agentic workload.**
  Both models pass the same dual-direction Python counting test and
  the chaostooling-otel pattern-application test. Qwen3 14B is faster
  per-token, has a cleaner tool-call protocol, and uses 60 % less
  resident memory.

The vendored loader still ships so operators on 24 GB+ hardware with
the OS quiesced can opt in via `family = "gemma4"` in their model
config. Gemma 4 E2B (the smaller variant) was NOT vendored: it has
materially different architecture (AltUp + Laurel + per-layer
activation sparsity) requiring another full vendoring session, with
no compensating benefit over Qwen3 4B which serves the same tier.

#### Why Qwen3-Coder-30B-A3B MoE is not the Mac premium model

Qwen3-Coder-30B-A3B is the obvious "premium" pick for a 24 GB Mac
(MoE with ~3 B active parameters per token, agentic-trained, supports
the `qwen3-native` tool protocol). The wiring landed in
`CandleModelFamily::Qwen3Moe`, including the `quantized_qwen3_moe`
load path with `DType::F32` activations.

**It cannot run on Mac in candle 0.10.2.** The forward pass bails at
`candle-nn-0.10.2/src/moe.rs:351`:

```rust
candle::bail!("moe_gemm_gguf is only implemented for the cuda backend")
```

There is no Metal kernel and no CPU fallback for `moe_gemm_gguf` in
this candle version. Linux NVIDIA / AMD-via-ROCm users with `gpu-cuda`
can use the MoE path; macOS users cannot, regardless of available
VRAM. Resolving this requires an upstream candle Metal MoE kernel or
CPU fallback. Tracked as a known limitation in this document under
"Known limitation: Qwen3 MoE is CUDA-only".

#### Why Devstral Small 24B isn't the Mac premium model

Devstral Small 2505 (Mistral AI's agentic-tuned 24 B Mistral) was the
next candidate. It's already on disk (~14 GB) and shares the Mistral
architecture, which means GGUFs with `general.architecture = "llama"`
should route through `candle_transformers::models::quantized_llama`.

**Two cumulative blockers ruled it out as default:**

1. **Candle's `quantized_llama` hardcodes head_dim wrong.** At
   `quantized_llama.rs:458`, candle derives
   `head_dim = embedding_length / head_count` and ignores the
   `llama.attention.key_length` metadata field. Devstral's GGUF
   declares `embedding_length=5120`, `head_count=32`,
   `key_length=128` — i.e. non-canonical 128 ≠ 5120/32 = 160. The
   forward pass fails with a reshape mismatch
   (`[1, N, 4096]` vs `[1, N, 32, 160]`).

2. **The mistralrs alternative needs Xcode on macOS.** mistralrs
   0.8.1 handles Devstral's head_dim correctly (Stage 1 spike loaded
   it cleanly in 26 s) and would have been the cross-platform Rust
   answer. But its `metal` feature transitively pulls in
   `mistralrs-paged-attn`, whose `build.rs` invokes `xcrun metal` to
   compile Metal shader sources. The metal shader compiler ships
   ONLY with full Xcode.app, not with Xcode Command Line Tools.
   In environments where corporate policy forbids installing full
   Xcode, the mistralrs Metal path cannot be built. The mistralrs
   feature flag is kept in `Cargo.toml` as opt-in for Linux CUDA
   users and macOS users who do have Xcode; it's not enabled by
   default and not part of the daily driver.

The candle `Mistral` GGUF path still ships and works for **canonical
Llama-arch GGUFs** (Mistral 7B v0.3, Llama 3.1 8B Instruct,
CodeLlama 13B Instruct). The Devstral / Mistral Small 24B variants
specifically need either an upstream candle patch reading
`key_length`, a vendored Mistral-aware GGUF loader, or external
serving via `llama-server` (Homebrew already ships this binary with
working Metal MoE and non-canonical head_dim support — operators on
Mac can use `milliways /switch-local-server llama-server` when they
want Devstral).

#### Net recommendation by deployment

| Hardware | Daily driver | Premium / specialty |
|---|---|---|
| macOS 24 GB unified | Qwen3 14B Q4_K_M (candle, validated) | `llama-server` subprocess for Devstral / Qwen3-Coder MoE (Homebrew install, Metal works without Xcode) |
| Linux NVIDIA 6 GB | Qwen3 4B Q4_K_M (candle + gpu-cuda) | n/a |
| Linux NVIDIA 12 GB | Qwen3 8B Q4_K_M (candle + gpu-cuda) | n/a |
| Linux NVIDIA 16+ GB | Qwen3 14B Q4_K_M (candle + gpu-cuda) | Qwen3-Coder-30B-A3B (candle + gpu-cuda, MoE works on CUDA) or mistralrs Devstral 24B (with `mistral-rs` + `gpu-cuda` features) |
| Linux AMD 16 GB (ROCm) | Qwen3 14B Q4_K_M (candle + gpu-cuda via HIP shim) | Same as NVIDIA 16+ GB, pending ROCm validation |

The mistralrs feature flag and Cargo dep stay in the workspace
specifically so the Linux deployment story includes
"premium-quality MoE / non-canonical Mistrals" without forcing users
back onto Python. macOS users with company-policy constraints get
the same value via `llama-server` (already installed, Metal works).

### GPU acceleration

The default build is CPU-only. To enable GPU inference, opt in via one of the
mutually-exclusive cargo features:

| Platform                    | Feature       | Build command (release)                                                          |
|-----------------------------|---------------|----------------------------------------------------------------------------------|
| macOS (Apple Silicon)       | `gpu-metal`   | `cargo build --release --features native-candle,native-tokenizers,gpu-metal`     |
| Linux NVIDIA                | `gpu-cuda`    | `cargo build --release --features native-candle,native-tokenizers,gpu-cuda`      |
| Linux AMD (via ROCm/HIP)    | `gpu-cuda`    | `HIP_PLATFORM=amd cargo build --release --features native-candle,native-tokenizers,gpu-cuda` |

#### Measured Metal speedup (Gemma 4 E4B Q4_K_M, Apple M-series 24 GB)

| Phase                  | CPU         | Metal       | Speedup       |
|------------------------|-------------|-------------|---------------|
| Model load (dequant)   | ~75 s       | ~73 s       | ~1× (CPU-bound) |
| Prefill (13 tokens)    | 118 s       | 115 ms      | ~1000×        |
| Per-token (greedy)     | ~3 s        | sub-100 ms (extrapolated) | ~30×+ |

The model load is bandwidth-bound on the ~10.7 GB F32 PLE dequantization and
runs on the CPU regardless of the target device — moving the load to the GPU
would require streaming dequantization, which candle 0.10.2 does not provide.

#### Recommended model per VRAM tier

| Tier | VRAM budget       | Best Gemma 4 variant            | Best non-Gemma alternative | Notes                                                                 |
|------|-------------------|---------------------------------|----------------------------|-----------------------------------------------------------------------|
| 1    | 6 GB              | **(none)**                      | Qwen3 4B Q4_K_M (~2.5 GB)  | E2B's PLE table dequant peaks at ~10 GB on CPU before settling to F32 |
| 2    | 12 GB             | Gemma 4 E2B Q4_K_M              | Qwen3 8B Q4_K_M (~5 GB)    | E2B working set ~7-8 GB once loaded                                   |
| 3    | 24 GB Mac unified | Gemma 4 E4B Q4_K_M (this build) | Qwen3 14B Q4_K_M (~9 GB)   | E4B needs ~17 GB total — fits on Mac but only with the OS idle        |

#### Known limitations

- **F16 PLE dequantisation breaks correctness on Metal.** Calling
  `dequantize_f16` instead of `dequantize` for `per_layer_token_embd` cuts
  resident memory from ~10.7 GB to ~5.4 GB and works correctly on the CPU
  backend, but on Metal the argmax collapses onto punctuation tokens (`()`
  instead of `Hello world`). Root cause is the candle 0.10.2 Metal dequant
  path that goes `Q4_K_M → F32 → F16` — the F32→F16 cast loses enough
  fidelity in the per-layer embedding magnitudes to bias the model. Until
  candle ships a direct Q4_K_M→F16 Metal kernel, the PLE table stays F32.
  For 12 GB GPUs this means **only E2B fits**; E4B is Mac/24 GB+ only.

- **Repeated test runs on a memory-constrained Mac trigger swap thrashing.**
  Each model load reserves ~10 GB for the F32 PLE table. macOS reclaims
  this lazily after the process exits, so back-to-back test runs may
  hang waiting for swap. Mitigation: idle for ~60-120 s between runs, or
  reboot for a clean memory state.

`gpu-cuda` reaches AMD GPUs because ROCm ships a HIP-based CUDA shim that
satisfies candle's CUDA dependencies — no rs-llmctl code change. The ROCm
install is responsible for providing the matching `hipBLAS`/`hipBLASLt`
libraries at link time.

Device selection is centralised in `gemma4_gguf::best_device()`:

```rust
pub fn best_device() -> Device {
    #[cfg(feature = "gpu-metal")]
    if let Ok(d) = Device::new_metal(0) { return d; }
    #[cfg(feature = "gpu-cuda")]
    if let Ok(d) = Device::new_cuda(0) { return d; }
    Device::Cpu
}
```

Compiled-in backends are probed in Metal → CUDA → CPU order. The first one
that initialises successfully wins, so a `gpu-metal` build on a non-Metal box
silently falls back to CPU.

**Memory note.** The Gemma4 `per_layer_token_embd` table is dequantised to F32
at load time (~10.7 GB). On a unified-memory Mac the table lives in shared
memory; on a discrete GPU it has to fit in VRAM alongside the rest of the
model. A future optimisation is on-demand row dequantisation, keeping the
quantised table resident and only converting the rows touched by the current
batch.

### Verifying coherent output

```bash
cargo test --features native-candle,native-tokenizers \
  gemma4_gguf_forward_pass_produces_coherent_tokens -- --ignored
```

The test loads `~/.local/share/milliways/models/gemma-4-E4B-it-Q4_K_M.gguf`,
runs a `<bos><|turn>user\nSay hello world<turn|>\n<|turn>model\n` prompt
through 13-token prefill + 15-token greedy generation on CPU, and asserts that
the decoded output contains ASCII alphabetic content. End-to-end on an M-series
Mac the prefill takes ~80 s and per-token generation ~3 s.
