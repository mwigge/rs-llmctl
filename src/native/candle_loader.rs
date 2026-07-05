//! Candle model construction: GGUF/safetensors loading, tokenizer + config helpers.
use super::*;

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
pub(crate) fn load_real_candle_decoder(
    _family: CandleModelFamily,
    _model_path: &Path,
    _artifacts: &CandleArtifactValidation,
) -> Result<NativeCandleDecoder> {
    Ok(NativeCandleDecoder::Unavailable)
}

// Candle's quantized_gemma3 probes for ["gemma3","gemma2","gemma","gemma-embedding"]
// prefixes but not "gemma4". Copy all `from_prefix.*` metadata entries under
// `to_prefix.*` so the probe succeeds and all subsequent key lookups resolve.
#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub(crate) fn remap_gguf_arch_prefix(
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
pub(crate) fn load_real_candle_decoder(
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
                CandleModelFamily::Mistral => {
                    // Mistral-family GGUFs (Devstral, Mistral Small, etc.) tag
                    // `general.architecture = "llama"` because they share the
                    // Llama transformer shape. candle's `quantized_llama` is
                    // the universal loader for that arch — no remap needed.
                    let arch = content
                        .metadata
                        .get("general.architecture")
                        .and_then(|v| v.to_string().ok().cloned())
                        .unwrap_or_default();
                    if arch != "llama" {
                        bail!(
                            "Mistral GGUF expected general.architecture = \"llama\", got {arch:?}"
                        );
                    }
                    tracing::info!(
                        arch = %arch,
                        "loading Mistral-family GGUF via candle quantized_llama"
                    );
                    RealCandleModel::MistralGguf(
                        candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                            content, &mut file, &device,
                        )
                        .with_context(|| {
                            "failed to construct quantized Llama-arch Candle model for Mistral family"
                        })?,
                    )
                }
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
        device,
        family,
        bos_token_id,
        served: std::sync::atomic::AtomicBool::new(false),
    }))
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
                top_p: None,
                top_k: None,
                seed: None,
                stop: None,
                presence_penalty: None,
                frequency_penalty: None,
                tools: None,
                tool_choice: None,
                metadata: BTreeMap::new(),
            })
        })
        .collect()
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub(crate) fn read_json_config<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
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
pub(crate) fn tokenizer_from_gguf_content(
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
pub(crate) fn gguf_bos_token_to_prepend(
    content: &candle_core::quantized::gguf_file::Content,
) -> Option<u32> {
    let model_kind = gemma4_gguf_tokenizer::metadata_value(content, "tokenizer.ggml.model")
        .and_then(|value| value.to_string().map_err(candle_core::Error::wrap))
        .ok()?
        .to_lowercase();

    match model_kind.as_str() {
        "gemma4" => gemma4_gguf_tokenizer::bos_token_to_prepend(content),
        _ => None,
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
