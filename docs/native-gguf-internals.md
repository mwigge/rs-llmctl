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

## Why Gemma4 GGUF forward pass fails in Candle 0.10.2

`quantized_gemma3::ModelWeights::from_gguf` reads a single `attention.key_length`
value and uses it as `head_dim` for every layer uniformly:

```rust
let head_count = md_get("attention.head_count")?.to_u32()? as usize;  // 8
let key_length = md_get("attention.key_length")?.to_u32()? as usize;  // 512
```

During the forward pass, the Q output tensor is reshaped as:

```rust
q.reshape((batch, seq_len, head_count, key_length))
// = reshape([1, 7, 2048] → [1, 7, 8, 512])
// 8 × 512 = 4096 ≠ 2048  →  shape mismatch error
```

The first layer encountered is an SWA layer (head_dim=256), so the Q tensor is
`2048` elements wide. Candle tries to give it 4096 elements. The mismatch is
fatal and immediate.

**This is not a metadata key issue.** The remap correctly surfaces all `gemma4.*`
values under `gemma3.*`, so Candle reads the right numeric values. The problem is
that Candle's model struct has a single `head_dim` field shared across all layers
and no mechanism to switch it per-layer.

### What a correct implementation would require

1. Read both `attention.key_length` and `attention.key_length_swa` from metadata.
2. Read or derive `sliding_window_pattern` to determine which layer indices use
   which dimension.
3. Store per-layer head_dim in the `Block` struct (or derive it at forward time).
4. Apply the correct reshape for each layer during the forward pass.
5. Apply separate RoPE bases (`rope.freq_base` vs `rope.freq_base_swa`) per layer.

This is architecturally possible within the existing `quantized_gemma3` framework
but requires changes to candle-transformers that go beyond what rs-llmctl can
patch externally without vendoring the crate.

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
| Candle 0.10.2 status | Fully supported          | Fully supported      | Tokenizer ✓, forward pass ✗      |

---

## Current status and path forward

| Component              | Status                                                       |
|------------------------|--------------------------------------------------------------|
| Tokenizer loading      | Working — SPM-BPE fallback in `tokenizer_from_gguf_spm`     |
| Metadata key probe     | Working — `remap_gguf_arch_prefix("gemma4", "gemma3")`       |
| Model weight loading   | Working — tensors load via standard `blk.{n}.*` names       |
| Forward pass (GGUF)    | Blocked — per-layer variable head_dim not in Candle 0.10.2  |
| Forward pass (safetensors) | Working — `gemma3::Model` reads config.json directly    |

To unblock GGUF forward pass without vendoring candle, the most targeted option
is a thin `quantized_gemma4` model implementation in `src/` that replicates
`quantized_gemma3` with per-layer head_dim selection, reading both
`attention.key_length` and `attention.key_length_swa` from the remapped metadata.
The tensor names (`blk.{n}.attn_q.weight`) are identical between Gemma3 and
Gemma4, so only the reshape and RoPE configuration logic needs to change.
