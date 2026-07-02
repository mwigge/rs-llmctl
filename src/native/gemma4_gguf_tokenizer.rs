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
            let id = u32::try_from(i)
                .map_err(|_| CandleError::msg(format!("vocab index {i} out of range for u32")))?;
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
