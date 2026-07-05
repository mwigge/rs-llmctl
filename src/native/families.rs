//! Candle family/format/acceleration metadata, artifact layouts, and validation contracts.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeModelFormat {
    Gguf,
    Safetensors,
    Unknown,
}

impl NativeModelFormat {
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some(extension) if extension.eq_ignore_ascii_case("gguf") => Self::Gguf,
            Some(extension) if extension.eq_ignore_ascii_case("safetensors") => Self::Safetensors,
            _ => Self::Unknown,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeAcceleration {
    Cpu,
    NvidiaCuda,
    /// Resource-planning hook only — no candle-native execution backend
    /// implements AMD GPU execution yet (candle 0.10.2 has no ROCm/HIP/
    /// Vulkan device backend). Selecting this still fails closed to CPU
    /// via `NativeCandleEngineLoader::load`. See `docs/adr/0001-amd-gpu-acceleration.md`.
    AmdRocm,
    AppleMetal,
    Auto,
}

impl NativeAcceleration {
    pub fn from_resources(resources: &ResourceConfig) -> Self {
        if resources.cpu_only {
            return Self::Cpu;
        }

        match resources.gpu_vendor.trim().to_ascii_lowercase().as_str() {
            "nvidia" | "cuda" => Self::NvidiaCuda,
            "amd" | "rocm" | "hip" => Self::AmdRocm,
            "apple" | "metal" => Self::AppleMetal,
            "auto" | "" => Self::Auto,
            _ => Self::Cpu,
        }
    }

    pub fn compatible_gpu_vendor(&self) -> Option<GpuVendor> {
        match self {
            Self::NvidiaCuda => Some(GpuVendor::Nvidia),
            Self::AmdRocm => Some(GpuVendor::Amd),
            Self::AppleMetal => Some(GpuVendor::Apple),
            Self::Cpu | Self::Auto => None,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::NvidiaCuda => "nvidia-cuda",
            Self::AmdRocm => "amd-rocm",
            Self::AppleMetal => "apple-metal",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleModelFamily {
    Qwen3,
    Qwen3Moe,
    Gemma4,
    DeepSeek,
    Kimi,
    Mistral,
    MiniMax,
}

impl CandleModelFamily {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3-moe",
            Self::Gemma4 => "gemma4",
            Self::DeepSeek => "deepseek",
            Self::Kimi => "kimi",
            Self::Mistral => "mistral",
            Self::MiniMax => "minimax",
        }
    }

    pub const fn engine_name(&self) -> &'static str {
        match self {
            Self::Qwen3 => "candle-native-qwen3",
            Self::Qwen3Moe => "candle-native-qwen3-moe",
            Self::Gemma4 => "candle-native-gemma4",
            Self::DeepSeek => "candle-native-deepseek",
            Self::Kimi => "candle-native-kimi",
            Self::Mistral => "candle-native-mistral",
            Self::MiniMax => "candle-native-minimax",
        }
    }

    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3",
            Self::Qwen3Moe => "Qwen3 MoE",
            Self::Gemma4 => "Gemma 4",
            Self::DeepSeek => "DeepSeek",
            Self::Kimi => "Kimi",
            Self::Mistral => "Mistral",
            Self::MiniMax => "MiniMax",
        }
    }

    pub const fn all() -> &'static [Self] {
        &[
            Self::Qwen3,
            Self::Qwen3Moe,
            Self::Gemma4,
            Self::DeepSeek,
            Self::Kimi,
            Self::Mistral,
            Self::MiniMax,
        ]
    }

    pub const fn has_native_decoder(&self) -> bool {
        matches!(
            self,
            Self::Qwen3 | Self::Qwen3Moe | Self::Gemma4 | Self::DeepSeek | Self::Mistral
        )
    }

    /// Stable identifier for the tool-call protocol the model uses.
    /// Orchestrators (e.g. milliways sommelier) use this to select a
    /// tool-call parser per family. New identifiers require a proposal.
    #[must_use]
    pub const fn tool_protocol(&self) -> &'static str {
        match self {
            Self::Qwen3 | Self::Qwen3Moe => "qwen3-native",
            Self::Gemma4 => "gemma4-native",
            // Devstral and other Mistral instruct-tuned variants emit
            // tool calls inside `[INST]...[/INST]` turns; they don't share
            // a stable cross-vendor tool-call protocol yet. Operators
            // pointing at a Mistral GGUF still get `mistral-instruct`
            // semantics — distinct from the qwen3 / gemma4 protocols.
            Self::Mistral => "mistral-instruct",
            Self::DeepSeek | Self::Kimi | Self::MiniMax => "none",
        }
    }

    /// Parse the kebab-case form back into the enum, e.g. `"qwen3-moe"` →
    /// `CandleModelFamily::Qwen3Moe`. Used to interpret the operator's
    /// `family` field in `ModelConfig`.
    #[must_use]
    pub fn from_kebab_str(s: &str) -> Option<Self> {
        Self::all().iter().copied().find(|f| f.as_str() == s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleTokenizerRequirement {
    GgufMetadata,
    TokenizerJson,
    UnsupportedFormat,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleSupportedOperation {
    ChatCompletion,
    ChatTokenCounting,
    CompletionTokenCounting,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleFamilySupportMetadata {
    pub model_family: CandleModelFamily,
    pub display_name: String,
    pub engine: String,
    pub supported_formats: Vec<NativeModelFormat>,
    pub supported_accelerators: Vec<NativeAcceleration>,
    pub supported_operations: Vec<CandleSupportedOperation>,
    pub candle_crates_required: Vec<String>,
    pub tokenizer_contracts: Vec<CandleTokenizerContract>,
    pub generation_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleTokenizerContract {
    pub model_format: NativeModelFormat,
    pub requirement: CandleTokenizerRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleArtifactKind {
    GgufWeights,
    SafetensorsWeights,
    TokenizerJson,
    ConfigJson,
}

impl CandleArtifactKind {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::GgufWeights => "GGUF weights",
            Self::SafetensorsWeights => "safetensors weights",
            Self::TokenizerJson => "tokenizer.json",
            Self::ConfigJson => "config.json",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleArtifactRequirement {
    pub kind: CandleArtifactKind,
    pub filename: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleArtifactLayout {
    pub model_format: NativeModelFormat,
    pub requirements: Vec<CandleArtifactRequirement>,
}

impl CandleArtifactLayout {
    pub fn for_format(format: NativeModelFormat) -> Self {
        let requirements = match format {
            NativeModelFormat::Gguf => vec![CandleArtifactRequirement {
                kind: CandleArtifactKind::GgufWeights,
                filename: "*.gguf".to_string(),
                required: true,
            }],
            NativeModelFormat::Safetensors => vec![
                CandleArtifactRequirement {
                    kind: CandleArtifactKind::SafetensorsWeights,
                    filename: "*.safetensors".to_string(),
                    required: true,
                },
                CandleArtifactRequirement {
                    kind: CandleArtifactKind::TokenizerJson,
                    filename: "tokenizer.json".to_string(),
                    required: true,
                },
                CandleArtifactRequirement {
                    kind: CandleArtifactKind::ConfigJson,
                    filename: "config.json".to_string(),
                    required: true,
                },
            ],
            NativeModelFormat::Unknown => Vec::new(),
        };

        Self {
            model_format: format,
            requirements,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleArtifactValidation {
    pub model_family: CandleModelFamily,
    pub model_format: NativeModelFormat,
    pub layout: CandleArtifactLayout,
    pub weight_files: Vec<String>,
    pub tokenizer_file: Option<String>,
    pub config_file: Option<String>,
}

impl CandleFamilySupportMetadata {
    pub fn for_family(family: CandleModelFamily) -> Self {
        let supported_formats = supported_candle_formats_for_family(family);
        let supported_operations = if family.has_native_decoder() {
            vec![
                CandleSupportedOperation::ChatCompletion,
                CandleSupportedOperation::ChatTokenCounting,
                CandleSupportedOperation::CompletionTokenCounting,
            ]
        } else {
            Vec::new()
        };
        let tokenizer_contracts = supported_formats
            .iter()
            .copied()
            .map(|model_format| CandleTokenizerContract {
                model_format,
                requirement: tokenizer_requirement_for_supported_format(model_format),
            })
            .collect();

        Self {
            model_family: family,
            display_name: family.display_name().to_string(),
            engine: family.engine_name().to_string(),
            supported_formats,
            supported_accelerators: vec![
                NativeAcceleration::Cpu,
                NativeAcceleration::NvidiaCuda,
                NativeAcceleration::AmdRocm,
                NativeAcceleration::AppleMetal,
                NativeAcceleration::Auto,
            ],
            supported_operations,
            candle_crates_required: vec![
                "candle-core".to_string(),
                "candle-nn".to_string(),
                "candle-transformers".to_string(),
                "tokenizers".to_string(),
            ],
            tokenizer_contracts,
            generation_status: candle_family_generation_status(family),
        }
    }

    pub fn tokenizer_requirement(&self, format: NativeModelFormat) -> CandleTokenizerRequirement {
        self.tokenizer_contracts
            .iter()
            .find(|contract| contract.model_format == format)
            .map(|contract| contract.requirement.clone())
            .unwrap_or(CandleTokenizerRequirement::UnsupportedFormat)
    }
}

fn supported_candle_formats_for_family(family: CandleModelFamily) -> Vec<NativeModelFormat> {
    match family {
        CandleModelFamily::Qwen3 | CandleModelFamily::Gemma4 => {
            vec![NativeModelFormat::Gguf, NativeModelFormat::Safetensors]
        }
        // Qwen3 MoE is currently only supported via GGUF — candle 0.10.2 does not
        // ship a safetensors path for the MoE variant.
        CandleModelFamily::Qwen3Moe => vec![NativeModelFormat::Gguf],
        // Mistral now supports both: safetensors via candle's mistral::Model
        // and GGUF via the llama-arch quantized_llama path (Devstral, etc.).
        CandleModelFamily::Mistral => {
            vec![NativeModelFormat::Gguf, NativeModelFormat::Safetensors]
        }
        CandleModelFamily::DeepSeek => {
            vec![NativeModelFormat::Safetensors]
        }
        CandleModelFamily::Kimi | CandleModelFamily::MiniMax => Vec::new(),
    }
}

fn tokenizer_requirement_for_supported_format(
    format: NativeModelFormat,
) -> CandleTokenizerRequirement {
    match format {
        NativeModelFormat::Gguf => CandleTokenizerRequirement::GgufMetadata,
        NativeModelFormat::Safetensors => CandleTokenizerRequirement::TokenizerJson,
        NativeModelFormat::Unknown => CandleTokenizerRequirement::UnsupportedFormat,
    }
}

fn candle_family_generation_status(family: CandleModelFamily) -> String {
    match family {
        CandleModelFamily::Qwen3 | CandleModelFamily::Qwen3Moe | CandleModelFamily::Gemma4 | CandleModelFamily::Mistral => {
            format!(
                "Candle {} artifact loading and greedy autoregressive decoding are wired where Candle exposes the required architecture and artifact format",
                family.as_str()
            )
        }
        CandleModelFamily::DeepSeek => {
            "Candle deepseek2 safetensors artifact loading and greedy autoregressive decoding are wired through DeepSeekV2; GGUF/quantized DeepSeek remains fail-closed because Candle 0.10.2 does not expose quantized DeepSeek2 model weights".to_string()
        }
        CandleModelFamily::Kimi => {
            "Kimi remains fail-closed for all native formats because Candle 0.10.2 does not expose candle_transformers::models::kimi or quantized Kimi GGUF model weights".to_string()
        }
        CandleModelFamily::MiniMax => {
            "MiniMax remains fail-closed for all native formats because Candle 0.10.2 does not expose candle_transformers::models::minimax or quantized MiniMax GGUF model weights".to_string()
        }
    }
}

fn candle_format_generation_status(family: CandleModelFamily, format: NativeModelFormat) -> String {
    match (family, format) {
        (CandleModelFamily::DeepSeek, NativeModelFormat::Safetensors) => {
            "candle-native-deepseek safetensors decoding is wired through candle_transformers::models::deepseek2::DeepSeekV2".to_string()
        }
        (CandleModelFamily::DeepSeek, NativeModelFormat::Gguf) => {
            "candle-native-deepseek GGUF/quantized DeepSeek fails closed because Candle 0.10.2 does not expose quantized DeepSeek2 model weights".to_string()
        }
        (CandleModelFamily::Kimi, NativeModelFormat::Safetensors) => {
            "candle-native-kimi safetensors decoding fails closed because Candle 0.10.2 does not expose candle_transformers::models::kimi".to_string()
        }
        (CandleModelFamily::Kimi, NativeModelFormat::Gguf) => {
            "candle-native-kimi GGUF/quantized Kimi fails closed because Candle 0.10.2 does not expose quantized Kimi GGUF model weights".to_string()
        }
        (CandleModelFamily::MiniMax, NativeModelFormat::Safetensors) => {
            "candle-native-minimax safetensors decoding fails closed because Candle 0.10.2 does not expose candle_transformers::models::minimax".to_string()
        }
        (CandleModelFamily::MiniMax, NativeModelFormat::Gguf) => {
            "candle-native-minimax GGUF/quantized MiniMax fails closed because Candle 0.10.2 does not expose quantized MiniMax GGUF model weights".to_string()
        }
        (_, NativeModelFormat::Unknown) => format!(
            "{} does not support unknown native artifact formats",
            family.engine_name()
        ),
        _ => candle_family_generation_status(family),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleDeviceSelectionContract {
    pub requested: NativeAcceleration,
    pub selected: NativeAcceleration,
    pub compatible_gpu_vendor: Option<GpuVendor>,
    pub selection_reason: String,
    pub fail_closed_if_unavailable: bool,
}

impl CandleDeviceSelectionContract {
    pub fn from_acceleration(acceleration: NativeAcceleration) -> Self {
        let compatible_gpu_vendor = acceleration.compatible_gpu_vendor();
        let selection_reason = match acceleration {
            NativeAcceleration::Cpu => "resources.cpu_only requested CPU execution".to_string(),
            NativeAcceleration::NvidiaCuda => {
                "resources.gpu_vendor requested NVIDIA CUDA execution".to_string()
            }
            NativeAcceleration::AmdRocm => {
                "resources.gpu_vendor requested AMD ROCm execution".to_string()
            }
            NativeAcceleration::AppleMetal => {
                "resources.gpu_vendor requested Apple Metal execution".to_string()
            }
            NativeAcceleration::Auto => {
                "resources.gpu_vendor left device selection to the Candle loader".to_string()
            }
        };

        Self {
            requested: acceleration,
            selected: acceleration,
            compatible_gpu_vendor,
            selection_reason,
            fail_closed_if_unavailable: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleEngineLoadContract {
    pub model_family: CandleModelFamily,
    pub model_format: NativeModelFormat,
    pub artifact_layout: CandleArtifactLayout,
    pub accelerator: NativeAcceleration,
    pub tokenizer: CandleTokenizerRequirement,
    pub supported_operations: Vec<CandleSupportedOperation>,
    pub candle_crates_required: Vec<String>,
    pub device_selection: CandleDeviceSelectionContract,
    pub fail_closed: bool,
    pub fail_closed_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandleEngineConfig {
    pub engine: String,
    pub support: CandleFamilySupportMetadata,
    pub load_contract: CandleEngineLoadContract,
}

impl CandleEngineConfig {
    pub fn qwen3(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::Qwen3, format, accelerator)
    }

    pub fn gemma4(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::Gemma4, format, accelerator)
    }

    pub fn kimi(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::Kimi, format, accelerator)
    }

    pub fn mistral(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::Mistral, format, accelerator)
    }

    pub fn deepseek(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::DeepSeek, format, accelerator)
    }

    pub fn minimax(format: NativeModelFormat, accelerator: NativeAcceleration) -> Self {
        Self::for_family(CandleModelFamily::MiniMax, format, accelerator)
    }

    pub fn for_family(
        family: CandleModelFamily,
        format: NativeModelFormat,
        accelerator: NativeAcceleration,
    ) -> Self {
        let support = CandleFamilySupportMetadata::for_family(family);
        let supported_operations = if support.supported_formats.contains(&format) {
            support.supported_operations.clone()
        } else {
            Vec::new()
        };
        let fail_closed = supported_operations.is_empty();
        let tokenizer = support.tokenizer_requirement(format);
        let fail_closed_reason = candle_format_generation_status(family, format);

        Self {
            engine: support.engine.clone(),
            support: support.clone(),
            load_contract: CandleEngineLoadContract {
                model_family: family,
                model_format: format,
                artifact_layout: CandleArtifactLayout::for_format(format),
                accelerator,
                tokenizer,
                supported_operations,
                candle_crates_required: support.candle_crates_required.clone(),
                device_selection: CandleDeviceSelectionContract::from_acceleration(accelerator),
                fail_closed,
                fail_closed_reason,
            },
        }
    }

    pub fn is_supported(&self) -> bool {
        matches!(
            self.load_contract.model_format,
            NativeModelFormat::Gguf | NativeModelFormat::Safetensors
        ) && !self.load_contract.supported_operations.is_empty()
    }
}
pub fn validate_native_engine_load_plan(plan: &NativeEngineLoadPlan) -> Result<()> {
    if plan.runtime != RuntimeBackend::CandleNative {
        bail!("native load plan runtime must be candle-native");
    }
    if plan.alias.trim().is_empty() {
        bail!("native load plan has an empty model alias");
    }
    if !(0.0..=1.0).contains(&plan.budget_fraction) {
        bail!(
            "native load plan budget_fraction must be between 0.0 and 1.0, got {}",
            plan.budget_fraction
        );
    }
    if plan.engine != plan.candle.engine || plan.engine != plan.support.engine {
        bail!("native load plan engine does not match Candle support metadata");
    }
    if plan.candle.load_contract.model_family != plan.support.model_family {
        bail!("native load plan family does not match Candle support metadata");
    }
    if plan.family != plan.candle.load_contract.model_family.as_str() {
        bail!("native load plan family string does not match Candle load contract");
    }
    if plan.format != plan.candle.load_contract.model_format {
        bail!("native load plan format does not match Candle load contract");
    }
    if plan.acceleration != plan.candle.load_contract.accelerator {
        bail!("native load plan acceleration does not match Candle load contract");
    }
    if plan.device_selection != plan.candle.load_contract.device_selection {
        bail!("native load plan device selection does not match Candle load contract");
    }
    if !plan.support.supported_formats.contains(&plan.format) {
        bail!(
            "{} does not support model format {:?}",
            plan.engine,
            plan.format
        );
    }
    if !plan
        .support
        .supported_accelerators
        .contains(&plan.acceleration)
    {
        bail!(
            "{} does not support acceleration {:?}",
            plan.engine,
            plan.acceleration
        );
    }
    if plan.candle.load_contract.tokenizer == CandleTokenizerRequirement::UnsupportedFormat {
        bail!("native load plan has no tokenizer contract for model format");
    }
    if plan.candle.load_contract.supported_operations.is_empty() {
        bail!("native load plan has no supported Candle operations");
    }
    if !plan.implemented && !plan.candle.load_contract.fail_closed {
        bail!("unimplemented native load plan must fail closed");
    }
    if plan.scheduler.contract_only {
        bail!("native scheduler contract must report implemented FIFO queue runtime");
    }
    if plan.scheduler.queue.discipline != NativeQueueDiscipline::Fifo
        || !plan.scheduler.queue.implemented
        || !plan.scheduler.batching.prefill_decode_phase_scheduling
        || plan.scheduler.batching.implemented
        || plan.scheduler.kv_cache.reuse_implemented
        || plan.scheduler.kv_cache.implemented
        || !plan.scheduler.cancellation.admission_check_implemented
        || plan.scheduler.cancellation.decode_loop_check_implemented
        || plan.scheduler.cancellation.implemented
    {
        bail!(
            "native scheduler must implement FIFO queue and phase metadata while continuous batching, KV-cache reuse, and decode cancellation remain explicit unsupported runtime boundaries"
        );
    }

    Ok(())
}

pub fn validate_candle_model_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
) -> Result<CandleArtifactValidation> {
    let support = CandleFamilySupportMetadata::for_family(family);
    let format = infer_native_artifact_format(&model.path);
    if !support.supported_formats.contains(&format) {
        bail!(
            "{} cannot load model alias '{}' because the artifact format is unsupported; expected a .gguf file or safetensors weights with tokenizer.json and config.json",
            family.engine_name(),
            model.alias
        );
    }

    let layout = CandleArtifactLayout::for_format(format);
    match format {
        NativeModelFormat::Gguf => validate_gguf_artifacts(family, model, layout),
        NativeModelFormat::Safetensors => validate_safetensors_artifacts(family, model, layout),
        NativeModelFormat::Unknown => unreachable!("unsupported formats are rejected above"),
    }
}

fn validate_gguf_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    layout: CandleArtifactLayout,
) -> Result<CandleArtifactValidation> {
    let mut missing = Vec::new();
    if !model.path.is_file() || NativeModelFormat::from_path(&model.path) != NativeModelFormat::Gguf
    {
        missing.push("GGUF weights (*.gguf)".to_string());
    }

    fail_missing_artifacts(family, model, NativeModelFormat::Gguf, &missing)?;
    Ok(CandleArtifactValidation {
        model_family: family,
        model_format: NativeModelFormat::Gguf,
        layout,
        weight_files: vec![artifact_file_name(&model.path)],
        tokenizer_file: None,
        config_file: None,
    })
}

pub(crate) fn validate_safetensors_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    layout: CandleArtifactLayout,
) -> Result<CandleArtifactValidation> {
    let artifact_dir = safetensors_artifact_dir(&model.path);
    let weights = safetensors_weight_files(&model.path, artifact_dir);
    let tokenizer = artifact_dir.join("tokenizer.json");
    let config = artifact_dir.join("config.json");

    let mut missing = Vec::new();
    if weights.is_empty() {
        missing.push("safetensors weights (*.safetensors)".to_string());
    }
    if !tokenizer.is_file() {
        missing.push("tokenizer.json".to_string());
    }
    if !config.is_file() {
        missing.push("config.json".to_string());
    }

    fail_missing_artifacts(family, model, NativeModelFormat::Safetensors, &missing)?;
    Ok(CandleArtifactValidation {
        model_family: family,
        model_format: NativeModelFormat::Safetensors,
        layout,
        weight_files: weights,
        tokenizer_file: Some("tokenizer.json".to_string()),
        config_file: Some("config.json".to_string()),
    })
}

pub(crate) fn verify_candle_artifacts_can_load(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<()> {
    match artifacts.model_format {
        NativeModelFormat::Gguf => verify_gguf_can_load(model_path),
        NativeModelFormat::Safetensors => verify_safetensors_can_load(model_path, artifacts),
        NativeModelFormat::Unknown => bail!("native artifact format is unsupported"),
    }
}

#[cfg(feature = "native-candle")]
fn verify_gguf_can_load(model_path: &Path) -> Result<()> {
    let device = candle_core::Device::Cpu;
    candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_path, &device)
        .map(|_| ())
        .with_context(|| "failed to load GGUF weights with Candle")
}

#[cfg(not(feature = "native-candle"))]
fn verify_gguf_can_load(_model_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(feature = "native-candle")]
fn verify_safetensors_can_load(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<()> {
    let artifact_dir = safetensors_artifact_dir(model_path);
    let paths = artifacts
        .weight_files
        .iter()
        .map(|name| artifact_dir.join(name))
        .collect::<Vec<_>>();
    let device = candle_core::Device::Cpu;
    // Candle exposes safetensors mmap loading as unsafe because it relies on OS mmap invariants.
    // The paths come from validation immediately above and are only used for read-only weight access.
    unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::F32, &device)
    }
    .map(|_| ())
    .with_context(|| "failed to load safetensors weights with Candle")
}

#[cfg(not(feature = "native-candle"))]
fn verify_safetensors_can_load(
    _model_path: &Path,
    _artifacts: &CandleArtifactValidation,
) -> Result<()> {
    Ok(())
}

fn fail_missing_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    format: NativeModelFormat,
    missing: &[String],
) -> Result<()> {
    if missing.is_empty() {
        return Ok(());
    }

    bail!(
        "{} cannot load model alias '{}' as {:?}: missing required artifact(s): {}",
        family.engine_name(),
        model.alias,
        format,
        missing.join(", ")
    )
}

pub(crate) fn infer_native_artifact_format(path: &Path) -> NativeModelFormat {
    let format = NativeModelFormat::from_path(path);
    if format != NativeModelFormat::Unknown {
        return format;
    }

    if path.is_dir() && !safetensors_weight_files(path, path).is_empty() {
        return NativeModelFormat::Safetensors;
    }

    NativeModelFormat::Unknown
}

pub(crate) fn safetensors_artifact_dir(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn safetensors_weight_files(path: &Path, artifact_dir: &Path) -> Vec<String> {
    if path.is_file() && NativeModelFormat::from_path(path) == NativeModelFormat::Safetensors {
        return vec![artifact_file_name(path)];
    }

    let mut weights = fs::read_dir(artifact_dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && NativeModelFormat::from_path(path) == NativeModelFormat::Safetensors
        })
        .map(|path| artifact_file_name(&path))
        .collect::<Vec<_>>();
    weights.sort();
    weights
}

pub(crate) fn artifact_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unnamed>")
        .to_string()
}
