//! Candle decoder engines: artifact-backed engine factory, RealCandleDecoder, and the generation loop.
use super::*;

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

    pub(crate) fn generate(&self, request: &NativeChatRequest) -> Result<String> {
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
pub(crate) fn load_real_candle_decoder(
    _family: CandleModelFamily,
    _model_path: &Path,
    _artifacts: &CandleArtifactValidation,
) -> Result<NativeCandleDecoder> {
    Ok(NativeCandleDecoder::Unavailable)
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
#[derive(Debug)]
pub(crate) struct RealCandleDecoder {
    tokenizer: tokenizers::tokenizer::Tokenizer,
    model: Mutex<RealCandleModel>,
    family: CandleModelFamily,
    /// BOS token id to prepend to the generation prompt's `input_ids`, if the
    /// GGUF tokenizer metadata configures `add_bos_token = true`. See
    /// [`gguf_bos_token_to_prepend`] and [`prepend_bos_if_configured`].
    bos_token_id: Option<u32>,
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
        family,
        bos_token_id,
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
        prepend_bos_if_configured(&mut input_ids, self.bos_token_id);
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
                tools: None,
                tool_choice: None,
                metadata: BTreeMap::new(),
            })
        })
        .collect()
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
            // candle 0.10.2's quantized_llama also does not expose a public
            // clear_kv_cache(). Same pattern as Qwen3 MoE — independent
            // sessions must reload the model.
            Self::MistralGguf(_) => {
                tracing::warn!(
                    "Mistral GGUF clear_kv_cache is a no-op in candle 0.10.2 — \
                     independent sessions must reload the model"
                );
            }
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
            Self::MistralGguf(model) => model.forward(&input, offset),
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
