use crate::config::{ClusterNodeConfig, Config, ModelConfig, ResourceConfig};
use crate::resources::GpuVendor;
use crate::runtime::RuntimeBackend;
use anyhow::{bail, Context, Result};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const STARTER_ROLES: &[&str] = &["query", "recommendation", "thinking", "coding"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeChatRequest {
    pub model: String,
    pub messages: Vec<NativeChatMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TokenAccountingMode {
    /// Counts came from a model-compatible native tokenizer.
    NativeExact,
    /// Counts came from the deterministic fallback estimator and are not exact model tokens.
    Estimated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeTokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub accounting_mode: TokenAccountingMode,
}

impl NativeTokenUsage {
    pub const fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            accounting_mode: TokenAccountingMode::Estimated,
        }
    }

    pub const fn with_mode(
        input_tokens: u64,
        output_tokens: u64,
        accounting_mode: TokenAccountingMode,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            accounting_mode,
        }
    }

    pub const fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeChatResponse {
    pub model: String,
    pub content: String,
    pub finish_reason: String,
    pub usage: NativeTokenUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeQueueDiscipline {
    Fifo,
    WeightedFair,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeQueueContract {
    pub discipline: NativeQueueDiscipline,
    pub admission_backpressure: bool,
    pub priority_metadata_keys: Vec<String>,
    pub implemented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeBatchingContract {
    pub continuous_batching: bool,
    pub max_batch_size_metadata_key: String,
    pub max_wait_ms_metadata_key: String,
    pub implemented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeKvCacheContract {
    pub cache_scope: String,
    pub cache_budget_metadata_key: String,
    pub eviction_policy: String,
    pub implemented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCancellationContract {
    pub cancellation_token_metadata_key: String,
    pub drain_on_cancel: bool,
    pub implemented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeSchedulerContract {
    pub queue: NativeQueueContract,
    pub batching: NativeBatchingContract,
    pub kv_cache: NativeKvCacheContract,
    pub cancellation: NativeCancellationContract,
    pub contract_only: bool,
}

impl NativeSchedulerContract {
    pub fn planned_metadata_only() -> Self {
        Self {
            queue: NativeQueueContract {
                discipline: NativeQueueDiscipline::Fifo,
                admission_backpressure: true,
                priority_metadata_keys: vec![
                    "llmctl.scheduler.priority".to_string(),
                    "llmctl.scheduler.tenant".to_string(),
                ],
                implemented: false,
            },
            batching: NativeBatchingContract {
                continuous_batching: true,
                max_batch_size_metadata_key: "llmctl.scheduler.max_batch_size".to_string(),
                max_wait_ms_metadata_key: "llmctl.scheduler.max_wait_ms".to_string(),
                implemented: false,
            },
            kv_cache: NativeKvCacheContract {
                cache_scope: "model-worker".to_string(),
                cache_budget_metadata_key: "llmctl.scheduler.kv_cache_budget_bytes".to_string(),
                eviction_policy: "metadata-only-lru-target".to_string(),
                implemented: false,
            },
            cancellation: NativeCancellationContract {
                cancellation_token_metadata_key: "llmctl.scheduler.cancel_token".to_string(),
                drain_on_cancel: true,
                implemented: false,
            },
            contract_only: true,
        }
    }
}

pub trait NativeEngine: Send + Sync {
    fn model_alias(&self) -> &str;
    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>>;

    fn chat_stream(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        self.chat(request)
    }
}

pub trait NativeTokenCounter: Send + Sync {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::NativeExact
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64>;
    fn count_text(&self, text: &str) -> Result<u64>;
}

pub trait NativeTokenAccountingAdapter: NativeTokenCounter {}

impl<T> NativeTokenAccountingAdapter for T where T: NativeTokenCounter + ?Sized {}

pub fn canonical_native_chat_input(messages: &[NativeChatMessage]) -> String {
    let mut input = String::new();
    for message in messages {
        input.push_str("<|");
        input.push_str(&message.role);
        input.push_str("|>\n");
        input.push_str(&message.content);
        input.push('\n');
    }
    input
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EstimatedNativeTokenCounter;

impl EstimatedNativeTokenCounter {
    const CHARS_PER_TOKEN: u64 = 4;
    const MESSAGE_OVERHEAD_TOKENS: u64 = 4;

    fn estimate_text_tokens(text: &str) -> u64 {
        let normalized_chars = text.chars().filter(|ch| !ch.is_control()).count() as u64;
        if normalized_chars == 0 {
            return 0;
        }
        normalized_chars
            .saturating_add(Self::CHARS_PER_TOKEN - 1)
            .saturating_div(Self::CHARS_PER_TOKEN)
            .max(1)
    }
}

impl NativeTokenCounter for EstimatedNativeTokenCounter {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::Estimated
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64> {
        Ok(messages
            .iter()
            .map(|message| {
                Self::MESSAGE_OVERHEAD_TOKENS
                    .saturating_add(Self::estimate_text_tokens(&message.role))
                    .saturating_add(Self::estimate_text_tokens(&message.content))
            })
            .sum())
    }

    fn count_text(&self, text: &str) -> Result<u64> {
        Ok(Self::estimate_text_tokens(text))
    }
}

#[cfg(feature = "native-tokenizers")]
#[derive(Debug, Clone)]
pub struct TokenizersNativeTokenCounter {
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "native-tokenizers")]
impl TokenizersNativeTokenCounter {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|err| anyhow::anyhow!("failed to load tokenizer json: {err}"))?;
        Ok(Self::from_tokenizer(tokenizer))
    }

    pub const fn from_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    fn count_serialized_input(&self, input: &str) -> Result<u64> {
        let encoding = self
            .tokenizer
            .encode(input, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize native input: {err}"))?;
        Ok(encoding.len() as u64)
    }
}

#[cfg(feature = "native-tokenizers")]
impl NativeTokenCounter for TokenizersNativeTokenCounter {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::NativeExact
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64> {
        self.count_serialized_input(&canonical_native_chat_input(messages))
    }

    fn count_text(&self, text: &str) -> Result<u64> {
        self.count_serialized_input(text)
    }
}

pub fn usage_from_native_tokens(
    counter: &dyn NativeTokenAccountingAdapter,
    request: &NativeChatRequest,
    response_text: &str,
) -> Result<NativeTokenUsage> {
    Ok(NativeTokenUsage::with_mode(
        counter.count_chat_input(&request.messages)?,
        counter.count_text(response_text)?,
        counter.accounting_mode(),
    ))
}

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleModelFamily {
    Qwen3,
    Gemma4,
    Kimi,
    Mistral,
}

impl CandleModelFamily {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Gemma4 => "gemma4",
            Self::Kimi => "kimi",
            Self::Mistral => "mistral",
        }
    }

    pub const fn engine_name(&self) -> &'static str {
        match self {
            Self::Qwen3 => "candle-native-qwen3",
            Self::Gemma4 => "candle-native-gemma4",
            Self::Kimi => "candle-native-kimi",
            Self::Mistral => "candle-native-mistral",
        }
    }

    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3",
            Self::Gemma4 => "Gemma 4",
            Self::Kimi => "Kimi",
            Self::Mistral => "Mistral",
        }
    }

    pub const fn all() -> &'static [Self] {
        &[Self::Qwen3, Self::Gemma4, Self::Kimi, Self::Mistral]
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
        Self {
            model_family: family,
            display_name: family.display_name().to_string(),
            engine: family.engine_name().to_string(),
            supported_formats: vec![NativeModelFormat::Gguf, NativeModelFormat::Safetensors],
            supported_accelerators: vec![
                NativeAcceleration::Cpu,
                NativeAcceleration::NvidiaCuda,
                NativeAcceleration::AmdRocm,
                NativeAcceleration::AppleMetal,
                NativeAcceleration::Auto,
            ],
            supported_operations: vec![
                CandleSupportedOperation::ChatCompletion,
                CandleSupportedOperation::ChatTokenCounting,
                CandleSupportedOperation::CompletionTokenCounting,
            ],
            candle_crates_required: vec![
                "candle-core".to_string(),
                "candle-nn".to_string(),
                "candle-transformers".to_string(),
                "tokenizers".to_string(),
            ],
            tokenizer_contracts: vec![
                CandleTokenizerContract {
                    model_format: NativeModelFormat::Gguf,
                    requirement: CandleTokenizerRequirement::GgufMetadata,
                },
                CandleTokenizerContract {
                    model_format: NativeModelFormat::Safetensors,
                    requirement: CandleTokenizerRequirement::TokenizerJson,
                },
            ],
            generation_status: format!(
                "Candle {} artifact loading and streaming response surface are wired; family-specific autoregressive decoding remains behind the native engine boundary",
                family.as_str()
            ),
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
                fail_closed_reason: support.generation_status.clone(),
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
            scheduler: NativeSchedulerContract::planned_metadata_only(),
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
        let model = ModelConfig {
            alias: plan.alias.clone(),
            path: plan.model_path.clone(),
            role: plan.role.clone(),
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
                finish_reason: "stop".to_string(),
                usage,
            })
        })
    }
}

#[derive(Debug)]
enum NativeCandleDecoder {
    Real(RealCandleDecoder),
    #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
    Unavailable,
}

impl NativeCandleDecoder {
    fn load(
        family: CandleModelFamily,
        model_path: &Path,
        artifacts: &CandleArtifactValidation,
    ) -> Result<Self> {
        load_real_candle_decoder(family, model_path, artifacts)
    }

    fn generate(&self, request: &NativeChatRequest) -> Result<String> {
        match self {
            Self::Real(decoder) => decoder.generate(request),
            #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
            Self::Unavailable => bail!(
                "native autoregressive decoding requires the native-candle and native-tokenizers features"
            ),
        }
    }

    fn usage(&self, request: &NativeChatRequest, content: &str) -> Result<NativeTokenUsage> {
        match self {
            Self::Real(decoder) => decoder.usage(request, content),
            #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
            Self::Unavailable => {
                usage_from_native_tokens(&EstimatedNativeTokenCounter, request, content)
            }
        }
    }
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
fn load_real_candle_decoder(
    _family: CandleModelFamily,
    _model_path: &Path,
    _artifacts: &CandleArtifactValidation,
) -> Result<NativeCandleDecoder> {
    Ok(NativeCandleDecoder::Unavailable)
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
#[derive(Debug)]
struct RealCandleDecoder {
    tokenizer: tokenizers::tokenizer::Tokenizer,
    model: Mutex<RealCandleModel>,
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
#[derive(Debug)]
struct RealCandleDecoder;

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
#[derive(Debug)]
enum RealCandleModel {
    Qwen3(candle_transformers::models::qwen3::ModelForCausalLM),
    Qwen3Gguf(candle_transformers::models::quantized_qwen3::ModelWeights),
    Gemma3(candle_transformers::models::gemma3::Model),
    Gemma3Gguf(candle_transformers::models::quantized_gemma3::ModelWeights),
    Mistral(candle_transformers::models::mistral::Model),
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn load_real_candle_decoder(
    family: CandleModelFamily,
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<NativeCandleDecoder> {
    if family == CandleModelFamily::Kimi {
        bail!(
            "candle-native-kimi cannot be loaded: Candle 0.10.2 does not expose a Kimi architecture module"
        );
    }

    let device = candle_core::Device::Cpu;
    let tokenizer = load_generation_tokenizer(model_path, artifacts)
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
                CandleModelFamily::Mistral => {
                    let cfg: candle_transformers::models::mistral::Config =
                        read_json_config(&config_path)?;
                    RealCandleModel::Mistral(
                        candle_transformers::models::mistral::Model::new(&cfg, vb)
                            .with_context(|| "failed to construct Mistral Candle model")?,
                    )
                }
                CandleModelFamily::Kimi => unreachable!("Kimi is rejected before loading"),
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
                CandleModelFamily::Gemma4 => RealCandleModel::Gemma3Gguf(
                    candle_transformers::models::quantized_gemma3::ModelWeights::from_gguf(
                        content, &mut file, &device,
                    )
                    .with_context(|| "failed to construct quantized Gemma Candle model")?,
                ),
                CandleModelFamily::Mistral => bail!(
                    "candle-native-mistral GGUF decoding is not wired in Candle 0.10.2; use safetensors with tokenizer.json and config.json"
                ),
                CandleModelFamily::Kimi => unreachable!("Kimi is rejected before loading"),
            }
        }
        NativeModelFormat::Unknown => bail!("native artifact format is unsupported"),
    };

    Ok(NativeCandleDecoder::Real(RealCandleDecoder {
        tokenizer,
        model: Mutex::new(model),
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

        let prompt = canonical_native_chat_input(&request.messages);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize native prompt: {err}"))?;
        let mut input_ids = encoding.get_ids().to_vec();
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
        for step in 0..max_tokens {
            let step_input = if step == 0 {
                input_ids.clone()
            } else {
                vec![*input_ids.last().expect("input ids are non-empty")]
            };
            let next = model.forward_next(&step_input, offset)?;
            offset = offset.saturating_add(step_input.len());
            input_ids.push(next);
            generated.push(next);
            if is_eos_token(&self.tokenizer, next) {
                break;
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

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl RealCandleModel {
    fn clear_kv_cache(&mut self) {
        match self {
            Self::Qwen3(model) => model.clear_kv_cache(),
            Self::Qwen3Gguf(model) => model.clear_kv_cache(),
            Self::Gemma3(model) => model.clear_kv_cache(),
            Self::Gemma3Gguf(_) => {}
            Self::Mistral(model) => model.clear_kv_cache(),
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
            Self::Gemma3(model) => model.forward(&input, offset),
            Self::Gemma3Gguf(model) => model.forward(&input, offset),
            Self::Mistral(model) => model.forward(&input, offset),
        }
        .with_context(|| "native Candle model forward pass failed")?;
        logits
            .argmax(candle_core::D::Minus1)
            .and_then(|tensor| tensor.flatten_all())
            .and_then(|tensor| tensor.to_scalar::<u32>())
            .with_context(|| "failed to select next native token")
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn read_json_config<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let raw = fs::read_to_string(path).with_context(|| "failed to read model config.json")?;
    serde_json::from_str(&raw).with_context(|| "failed to parse model config.json")
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn load_generation_tokenizer(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<tokenizers::tokenizer::Tokenizer> {
    match artifacts.model_format {
        NativeModelFormat::Safetensors => {
            let tokenizer_path = safetensors_artifact_dir(model_path).join("tokenizer.json");
            tokenizers::tokenizer::Tokenizer::from_file(&tokenizer_path)
                .map_err(|err| anyhow::anyhow!("failed to load tokenizer.json: {err}"))
        }
        NativeModelFormat::Gguf => {
            let mut file = fs::File::open(model_path)
                .with_context(|| "failed to open GGUF tokenizer metadata")?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                .with_context(|| "failed to read GGUF tokenizer metadata")?;
            <tokenizers::tokenizer::Tokenizer as candle_core::quantized::tokenizer::TokenizerFromGguf>::from_gguf(&content)
            .map_err(|err| anyhow::anyhow!("failed to build tokenizer from GGUF metadata: {err}"))
        }
        NativeModelFormat::Unknown => bail!("native artifact format is unsupported"),
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn is_eos_token(tokenizer: &tokenizers::tokenizer::Tokenizer, token_id: u32) -> bool {
    tokenizer
        .id_to_token(token_id)
        .map(|token| matches!(token.as_str(), "</s>" | "<|endoftext|>" | "<end_of_turn>"))
        .unwrap_or(false)
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
    if !plan.scheduler.contract_only {
        bail!("native scheduler contract must remain metadata-only until execution is wired");
    }
    if plan.scheduler.queue.implemented
        || plan.scheduler.batching.implemented
        || plan.scheduler.kv_cache.implemented
        || plan.scheduler.cancellation.implemented
    {
        bail!(
            "native scheduler queue, batching, KV cache, and cancellation are not implemented yet"
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

fn validate_safetensors_artifacts(
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

fn verify_candle_artifacts_can_load(
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

fn infer_native_artifact_format(path: &Path) -> NativeModelFormat {
    let format = NativeModelFormat::from_path(path);
    if format != NativeModelFormat::Unknown {
        return format;
    }

    if path.is_dir() && !safetensors_weight_files(path, path).is_empty() {
        return NativeModelFormat::Safetensors;
    }

    NativeModelFormat::Unknown
}

fn safetensors_artifact_dir(path: &Path) -> &Path {
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

fn artifact_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unnamed>")
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePlacementPlan {
    pub routing_mode: String,
    pub local_node: String,
    pub nodes: Vec<NativePlacementNode>,
    pub unassigned_models: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePlacementNode {
    pub id: String,
    pub base_url: String,
    pub roles: Vec<String>,
    pub model_aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRouteSelection {
    pub query: String,
    pub candidates: Vec<NativePlacementNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeHeartbeat {
    pub node_id: String,
    pub runtime: RuntimeBackend,
    pub routing_mode: String,
    pub healthy: bool,
    pub models: usize,
    pub assigned_models: usize,
    pub unassigned_models: Vec<String>,
    pub budget_fraction: f64,
    pub heartbeat_interval_seconds: u64,
    pub telemetry_event: String,
}

impl NativeHeartbeat {
    pub fn safe_telemetry_attributes(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("cluster.node_id".to_string(), json_value(&self.node_id)),
            ("runtime.backend".to_string(), json_value(self.runtime)),
            (
                "runtime.routing_mode".to_string(),
                json_value(&self.routing_mode),
            ),
            ("runtime.healthy".to_string(), Value::Bool(self.healthy)),
            (
                "runtime.models".to_string(),
                Value::from(self.models as u64),
            ),
            (
                "runtime.assigned_models".to_string(),
                Value::from(self.assigned_models as u64),
            ),
            (
                "runtime.resource.budget_fraction".to_string(),
                Value::from(self.budget_fraction),
            ),
            (
                "runtime.heartbeat_interval_seconds".to_string(),
                Value::from(self.heartbeat_interval_seconds),
            ),
        ])
    }
}

pub fn heartbeat_from_config(cfg: &Config) -> NativeHeartbeat {
    let placement = placement_plan_from_config(cfg);
    let assigned_models = placement
        .nodes
        .iter()
        .map(|node| node.model_aliases.len())
        .sum();
    let healthy = validate_placement_plan(&placement).is_ok();
    NativeHeartbeat {
        node_id: cfg.cluster.node_id.clone(),
        runtime: cfg.runtime.backend,
        routing_mode: placement.routing_mode,
        healthy,
        models: cfg.models.len(),
        assigned_models,
        unassigned_models: placement.unassigned_models,
        budget_fraction: cfg.resources.budget,
        heartbeat_interval_seconds: cfg.runtime.heartbeat_interval_seconds,
        telemetry_event: "llmctl.runtime.heartbeat".to_string(),
    }
}

pub fn placement_plan_from_config(cfg: &Config) -> NativePlacementPlan {
    let nodes = if cfg.cluster.nodes.is_empty() {
        vec![NativePlacementNode {
            id: cfg.cluster.node_id.clone(),
            base_url: format!("http://{}:{}/v1", cfg.server.host, cfg.server.port),
            roles: sorted_roles(&cfg.models),
            model_aliases: cfg.models.iter().map(|model| model.alias.clone()).collect(),
        }]
    } else {
        cfg.cluster
            .nodes
            .iter()
            .map(|node| placement_node(node, &cfg.models))
            .collect()
    };

    let assigned = nodes
        .iter()
        .flat_map(|node| node.model_aliases.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let unassigned_models = cfg
        .models
        .iter()
        .filter(|model| !assigned.contains(&model.alias))
        .map(|model| model.alias.clone())
        .collect();

    NativePlacementPlan {
        routing_mode: if cfg.cluster.nodes.is_empty() {
            "single-node".to_string()
        } else {
            "cluster-role-placement".to_string()
        },
        local_node: cfg.cluster.node_id.clone(),
        nodes,
        unassigned_models,
    }
}

pub fn validate_placement_plan(plan: &NativePlacementPlan) -> Result<()> {
    if !plan.unassigned_models.is_empty() {
        bail!(
            "native placement leaves model aliases unassigned: {}",
            plan.unassigned_models.join(", ")
        );
    }

    let mut owners: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in &plan.nodes {
        if node.id.trim().is_empty() {
            bail!("native placement contains a node with an empty id");
        }
        if node.base_url.trim().is_empty() {
            bail!("native placement node '{}' has an empty base_url", node.id);
        }
        for alias in &node.model_aliases {
            owners
                .entry(alias.as_str())
                .or_default()
                .push(node.id.as_str());
        }
    }

    let duplicate = owners
        .iter()
        .find(|(_, node_ids)| node_ids.len() > 1)
        .map(|(alias, node_ids)| ((*alias).to_string(), node_ids.join(", ")));
    if let Some((alias, node_ids)) = duplicate {
        bail!("native placement assigns model alias '{alias}' to multiple nodes: {node_ids}");
    }

    Ok(())
}

pub fn route_selection_for_model(
    plan: &NativePlacementPlan,
    model_alias: &str,
) -> Result<NativeRouteSelection> {
    let candidates = plan
        .nodes
        .iter()
        .filter(|node| {
            node.model_aliases
                .iter()
                .any(|alias| alias.as_str() == model_alias)
        })
        .cloned()
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        bail!("native placement has no node for model alias '{model_alias}'");
    }
    if candidates.len() > 1 {
        bail!("native placement has multiple nodes for model alias '{model_alias}'");
    }

    Ok(NativeRouteSelection {
        query: format!("model:{model_alias}"),
        candidates,
    })
}

pub fn route_selection_for_role(
    plan: &NativePlacementPlan,
    role: &str,
) -> Result<NativeRouteSelection> {
    let normalized = normalize_role(role);
    let candidates = plan
        .nodes
        .iter()
        .filter(|node| node.roles.iter().any(|node_role| node_role == normalized))
        .cloned()
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        bail!("native placement has no node for role '{normalized}'");
    }

    Ok(NativeRouteSelection {
        query: format!("role:{normalized}"),
        candidates,
    })
}

fn json_value<T: Serialize>(value: T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

fn placement_node(node: &ClusterNodeConfig, models: &[ModelConfig]) -> NativePlacementNode {
    let explicit_aliases = node
        .model_aliases
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let role_set = node
        .roles
        .iter()
        .map(|role| normalize_role(role).to_string())
        .collect::<std::collections::BTreeSet<_>>();
    let model_aliases = models
        .iter()
        .filter(|model| {
            explicit_aliases.contains(&model.alias)
                || role_set.contains(normalize_role(&model.role))
        })
        .map(|model| model.alias.clone())
        .collect();

    NativePlacementNode {
        id: node.id.clone(),
        base_url: node.base_url.clone(),
        roles: node
            .roles
            .iter()
            .map(|role| normalize_role(role).to_string())
            .collect(),
        model_aliases,
    }
}

fn sorted_roles(models: &[ModelConfig]) -> Vec<String> {
    models
        .iter()
        .map(|model| normalize_role(&model.role).to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
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

fn normalize_role(role: &str) -> &str {
    let role = role.trim();
    if STARTER_ROLES.contains(&role) {
        role
    } else {
        "query"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use std::fs;
    use std::path::PathBuf;

    #[derive(Debug)]
    struct CountingTokenizer;

    impl NativeTokenCounter for CountingTokenizer {
        fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64> {
            Ok(messages
                .iter()
                .map(|message| message.content.split_whitespace().count() as u64)
                .sum())
        }

        fn count_text(&self, text: &str) -> Result<u64> {
            Ok(text.split_whitespace().count() as u64)
        }
    }

    #[test]
    fn native_usage_comes_from_token_counter_not_upstream_metadata() {
        let request = NativeChatRequest {
            model: "qwen-query".to_string(),
            messages: vec![
                NativeChatMessage {
                    role: "system".to_string(),
                    content: "answer briefly".to_string(),
                },
                NativeChatMessage {
                    role: "user".to_string(),
                    content: "hello native runtime".to_string(),
                },
            ],
            temperature: Some(0.2),
            max_tokens: Some(128),
            metadata: BTreeMap::new(),
        };

        let usage = usage_from_native_tokens(&CountingTokenizer, &request, "native answer")
            .expect("usage is counted");

        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.total_tokens(), 7);
        assert_eq!(usage.accounting_mode, TokenAccountingMode::NativeExact);
    }

    #[test]
    fn native_usage_reports_estimated_mode_and_nonzero_counts() {
        let request = NativeChatRequest {
            model: "qwen-query".to_string(),
            messages: vec![
                NativeChatMessage {
                    role: "system".to_string(),
                    content: "answer with operational detail".to_string(),
                },
                NativeChatMessage {
                    role: "user".to_string(),
                    content: "summarize native tokenizer accounting status".to_string(),
                },
            ],
            temperature: None,
            max_tokens: Some(64),
            metadata: BTreeMap::new(),
        };

        let usage = usage_from_native_tokens(
            &EstimatedNativeTokenCounter,
            &request,
            "native accounting is estimated until a tokenizer is wired",
        )
        .expect("estimated usage is counted");

        assert_eq!(usage.accounting_mode, TokenAccountingMode::Estimated);
        assert!(usage.input_tokens > 0);
        assert!(usage.output_tokens > 0);
        assert_eq!(
            usage.total_tokens(),
            usage.input_tokens + usage.output_tokens
        );
    }

    #[test]
    fn estimated_counter_is_deterministic_and_does_not_claim_exact_tokenization() {
        let messages = vec![NativeChatMessage {
            role: "user".to_string(),
            content: "repeatable fallback accounting".to_string(),
        }];

        let first = EstimatedNativeTokenCounter
            .count_chat_input(&messages)
            .expect("estimated count");
        let second = EstimatedNativeTokenCounter
            .count_chat_input(&messages)
            .expect("estimated count");

        assert_eq!(first, second);
        assert!(first > 0);
        assert_eq!(
            EstimatedNativeTokenCounter.accounting_mode(),
            TokenAccountingMode::Estimated
        );
    }

    #[test]
    fn canonical_native_chat_input_is_explicit_tokenizer_input() {
        let messages = vec![
            NativeChatMessage {
                role: "system".to_string(),
                content: "answer briefly".to_string(),
            },
            NativeChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            },
        ];

        assert_eq!(
            canonical_native_chat_input(&messages),
            "<|system|>\nanswer briefly\n<|user|>\nhello\n"
        );
    }

    #[cfg(feature = "native-tokenizers")]
    #[test]
    fn tokenizers_counter_feature_exposes_native_accounting_api_shape() {
        use std::path::Path;

        fn assert_counter<T: NativeTokenCounter>() {}

        assert_counter::<TokenizersNativeTokenCounter>();
        let _loader = |path: &Path| TokenizersNativeTokenCounter::from_file(path);
        let _constructor: fn(tokenizers::Tokenizer) -> TokenizersNativeTokenCounter =
            TokenizersNativeTokenCounter::from_tokenizer;
    }

    #[test]
    fn qwen3_loader_plan_covers_starter_role_acceleration_and_safe_status() {
        let model = ModelConfig {
            alias: "qwen-coder".to_string(),
            path: PathBuf::from("/home/alice/models/qwen3-coder.safetensors"),
            role: "coding".to_string(),
            weight: 1,
        };
        let resources = ResourceConfig {
            budget: 0.8,
            cpu_only: false,
            gpu_vendor: "nvidia".to_string(),
        };

        let plan = Qwen3CandleEngineLoader::plan(&model, &resources).expect("plan validates");

        assert_eq!(plan.runtime, RuntimeBackend::CandleNative);
        assert_eq!(plan.engine, "candle-native-qwen3");
        assert_eq!(plan.alias, "qwen-coder");
        assert_eq!(plan.role, "coding");
        assert_eq!(plan.family, "qwen3");
        assert_eq!(plan.format, NativeModelFormat::Safetensors);
        assert_eq!(plan.acceleration, NativeAcceleration::NvidiaCuda);
        assert_eq!(
            plan.acceleration.compatible_gpu_vendor(),
            Some(GpuVendor::Nvidia)
        );
        assert_eq!(plan.candle.engine, "candle-native-qwen3");
        assert_eq!(
            plan.candle.load_contract.model_family,
            CandleModelFamily::Qwen3
        );
        assert_eq!(
            plan.candle.load_contract.model_format,
            NativeModelFormat::Safetensors
        );
        assert_eq!(
            plan.candle.load_contract.accelerator,
            NativeAcceleration::NvidiaCuda
        );
        assert_eq!(
            plan.device_selection,
            CandleDeviceSelectionContract {
                requested: NativeAcceleration::NvidiaCuda,
                selected: NativeAcceleration::NvidiaCuda,
                compatible_gpu_vendor: Some(GpuVendor::Nvidia),
                selection_reason: "resources.gpu_vendor requested NVIDIA CUDA execution"
                    .to_string(),
                fail_closed_if_unavailable: true,
            }
        );
        assert_eq!(
            plan.candle.load_contract.tokenizer,
            CandleTokenizerRequirement::TokenizerJson
        );
        assert_eq!(
            plan.candle.load_contract.supported_operations,
            vec![
                CandleSupportedOperation::ChatCompletion,
                CandleSupportedOperation::ChatTokenCounting,
                CandleSupportedOperation::CompletionTokenCounting,
            ]
        );
        assert_eq!(
            plan.candle.load_contract.candle_crates_required,
            vec![
                "candle-core".to_string(),
                "candle-nn".to_string(),
                "candle-transformers".to_string(),
                "tokenizers".to_string(),
            ]
        );
        assert!(!plan.candle.load_contract.fail_closed);
        assert!(plan.candle.is_supported());
        assert_eq!(plan.support.model_family, CandleModelFamily::Qwen3);
        assert_eq!(plan.support.engine, "candle-native-qwen3");
        validate_native_engine_load_plan(&plan).expect("plan contract validates");
        assert!(plan.scheduler.contract_only);
        assert_eq!(plan.scheduler.queue.discipline, NativeQueueDiscipline::Fifo);
        assert!(plan.scheduler.queue.admission_backpressure);
        assert!(!plan.scheduler.queue.implemented);
        assert!(plan.scheduler.batching.continuous_batching);
        assert!(!plan.scheduler.batching.implemented);
        assert_eq!(plan.scheduler.kv_cache.cache_scope, "model-worker");
        assert!(!plan.scheduler.kv_cache.implemented);
        assert!(plan.scheduler.cancellation.drain_on_cancel);
        assert!(!plan.scheduler.cancellation.implemented);
        assert_eq!(plan.budget_fraction, 0.8);
        assert!(plan.implemented);

        let rendered = serde_json::to_string(&plan).expect("plan serializes");
        assert!(rendered.contains("llmctl.scheduler.kv_cache_budget_bytes"));
        assert!(rendered.contains("\"contract_only\":true"));
        assert!(!rendered.contains("/home/alice"));
        assert!(!rendered.contains("qwen3-coder.safetensors"));
        assert!(plan
            .safe_telemetry_attributes()
            .values()
            .all(|value| !value.to_string().contains("/home/alice")));
    }

    #[test]
    fn qwen3_contract_distinguishes_gguf_safetensors_and_unknown_formats() {
        let gguf = CandleEngineConfig::qwen3(NativeModelFormat::Gguf, NativeAcceleration::Cpu);
        assert_eq!(
            gguf.load_contract.tokenizer,
            CandleTokenizerRequirement::GgufMetadata
        );
        assert!(gguf
            .load_contract
            .supported_operations
            .contains(&CandleSupportedOperation::ChatCompletion));
        assert!(gguf.is_supported());

        let safetensors =
            CandleEngineConfig::qwen3(NativeModelFormat::Safetensors, NativeAcceleration::Auto);
        assert_eq!(
            safetensors.load_contract.tokenizer,
            CandleTokenizerRequirement::TokenizerJson
        );
        assert!(safetensors
            .load_contract
            .supported_operations
            .contains(&CandleSupportedOperation::ChatTokenCounting));
        assert!(safetensors.is_supported());

        let unknown =
            CandleEngineConfig::qwen3(NativeModelFormat::Unknown, NativeAcceleration::Auto);
        assert_eq!(
            unknown.load_contract.tokenizer,
            CandleTokenizerRequirement::UnsupportedFormat
        );
        assert!(unknown.load_contract.supported_operations.is_empty());
        assert!(!unknown.is_supported());
        assert!(unknown.load_contract.fail_closed);
    }

    #[test]
    fn candle_contract_includes_gemma4_kimi_and_eu_friendly_mistral_families() {
        let gemma4 =
            CandleEngineConfig::gemma4(NativeModelFormat::Safetensors, NativeAcceleration::Auto);
        assert_eq!(gemma4.engine, "candle-native-gemma4");
        assert_eq!(gemma4.load_contract.model_family, CandleModelFamily::Gemma4);
        assert_eq!(
            gemma4.load_contract.tokenizer,
            CandleTokenizerRequirement::TokenizerJson
        );
        assert!(gemma4.is_supported());
        assert!(!gemma4.load_contract.fail_closed);

        let kimi =
            CandleEngineConfig::kimi(NativeModelFormat::Safetensors, NativeAcceleration::Auto);
        assert_eq!(kimi.engine, "candle-native-kimi");
        assert_eq!(kimi.load_contract.model_family, CandleModelFamily::Kimi);
        assert_eq!(
            kimi.load_contract.tokenizer,
            CandleTokenizerRequirement::TokenizerJson
        );
        assert!(kimi.is_supported());
        assert!(!kimi.load_contract.fail_closed);

        let mistral = CandleEngineConfig::mistral(NativeModelFormat::Gguf, NativeAcceleration::Cpu);
        assert_eq!(mistral.engine, "candle-native-mistral");
        assert_eq!(
            mistral.load_contract.model_family,
            CandleModelFamily::Mistral
        );
        assert_eq!(
            mistral.load_contract.tokenizer,
            CandleTokenizerRequirement::GgufMetadata
        );
        assert!(mistral.is_supported());
        assert!(mistral.load_contract.fail_closed_reason.contains("mistral"));
    }

    #[test]
    fn candle_artifact_layout_distinguishes_gguf_from_safetensors_sidecars() {
        let gguf = CandleArtifactLayout::for_format(NativeModelFormat::Gguf);
        assert_eq!(
            gguf.requirements,
            vec![CandleArtifactRequirement {
                kind: CandleArtifactKind::GgufWeights,
                filename: "*.gguf".to_string(),
                required: true,
            }]
        );

        let safetensors = CandleArtifactLayout::for_format(NativeModelFormat::Safetensors);
        assert_eq!(
            safetensors
                .requirements
                .iter()
                .map(|requirement| requirement.kind)
                .collect::<Vec<_>>(),
            vec![
                CandleArtifactKind::SafetensorsWeights,
                CandleArtifactKind::TokenizerJson,
                CandleArtifactKind::ConfigJson,
            ]
        );
    }

    #[test]
    fn candle_artifact_validation_accepts_real_gguf_weight_file_for_all_families() {
        let dir = tempfile::tempdir().expect("tempdir");
        let weights = dir.path().join("chat.gguf");
        fs::write(&weights, b"GGUF").expect("write gguf placeholder");

        for family in CandleModelFamily::all() {
            let model = ModelConfig {
                alias: format!("{}-chat", family.as_str()),
                path: weights.clone(),
                role: "query".to_string(),
                weight: 1,
            };

            let validation = validate_candle_model_artifacts(*family, &model)
                .expect("gguf weight file validates");

            assert_eq!(validation.model_family, *family);
            assert_eq!(validation.model_format, NativeModelFormat::Gguf);
            assert_eq!(validation.weight_files, vec!["chat.gguf".to_string()]);
            assert_eq!(validation.tokenizer_file, None);
            assert_eq!(validation.config_file, None);
        }
    }

    #[test]
    fn candle_artifact_validation_accepts_safetensors_weights_tokenizer_and_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            dir.path().join("model-00001-of-00002.safetensors"),
            b"weights",
        )
        .expect("write first shard");
        fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            b"weights",
        )
        .expect("write second shard");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), b"{}").expect("write config");

        let model = ModelConfig {
            alias: "kimi-chat".to_string(),
            path: dir.path().to_path_buf(),
            role: "thinking".to_string(),
            weight: 1,
        };

        let validation = validate_candle_model_artifacts(CandleModelFamily::Kimi, &model)
            .expect("safetensors directory validates");

        assert_eq!(validation.model_family, CandleModelFamily::Kimi);
        assert_eq!(validation.model_format, NativeModelFormat::Safetensors);
        assert_eq!(
            validation.weight_files,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ]
        );
        assert_eq!(
            validation.tokenizer_file,
            Some("tokenizer.json".to_string())
        );
        assert_eq!(validation.config_file, Some("config.json".to_string()));
    }

    #[test]
    fn candle_artifact_validation_reports_missing_safetensors_sidecars_actionably() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("model.safetensors"), b"weights").expect("write weights");
        let model = ModelConfig {
            alias: "gemma-chat".to_string(),
            path: dir.path().join("model.safetensors"),
            role: "query".to_string(),
            weight: 1,
        };

        let err = validate_candle_model_artifacts(CandleModelFamily::Gemma4, &model)
            .expect_err("missing sidecars are rejected");
        let message = err.to_string();

        assert!(message.contains("candle-native-gemma4"));
        assert!(message.contains("gemma-chat"));
        assert!(message.contains("missing required artifact(s)"));
        assert!(message.contains("tokenizer.json"));
        assert!(message.contains("config.json"));
        assert!(!message.contains(dir.path().to_str().expect("utf8 temp path")));
    }

    #[test]
    fn candle_artifact_validation_reports_missing_weights_actionably() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), b"{}").expect("write config");
        let model = ModelConfig {
            alias: "mistral-chat".to_string(),
            path: dir.path().join("model.safetensors"),
            role: "query".to_string(),
            weight: 1,
        };

        let err = validate_candle_model_artifacts(CandleModelFamily::Mistral, &model)
            .expect_err("missing weights are rejected");
        let message = err.to_string();

        assert!(message.contains("candle-native-mistral"));
        assert!(message.contains("mistral-chat"));
        assert!(message.contains("safetensors weights (*.safetensors)"));
        assert!(!message.contains(dir.path().to_str().expect("utf8 temp path")));
    }

    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    fn native_candle_factory_rejects_incomplete_safetensors_model_before_serving() {
        let dir = tempfile::tempdir().expect("tempdir");
        let weights = dir.path().join("model.safetensors");
        let tensor =
            candle_core::Tensor::new(&[1f32, 2f32], &candle_core::Device::Cpu).expect("tensor");
        tensor.save_safetensors("dummy", &weights).expect("weights");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), b"{}").expect("write config");
        let model = ModelConfig {
            alias: "mistral-chat".to_string(),
            path: weights,
            role: "query".to_string(),
            weight: 1,
        };

        let plan = NativeCandleEngineFactory::default()
            .plan(
                CandleModelFamily::Mistral,
                &model,
                &ResourceConfig::default(),
            )
            .expect("plan");
        let err = match NativeCandleEngineFactory::default().load(&plan) {
            Ok(_) => panic!("dummy safetensors are not a complete Mistral model"),
            Err(err) => err,
        };
        let message = err.to_string();

        assert!(
            message.contains("failed to load native generation tokenizer")
                || message.contains("failed to parse model config.json"),
            "{message}"
        );
        assert!(!message.contains(dir.path().to_str().expect("utf8 temp path")));
    }

    #[test]
    fn native_candle_factory_registers_all_families_with_actionable_metadata() {
        let factory = NativeCandleEngineFactory::default();

        assert_eq!(
            factory.registered_families(),
            vec![
                CandleModelFamily::Qwen3,
                CandleModelFamily::Gemma4,
                CandleModelFamily::Kimi,
                CandleModelFamily::Mistral,
            ]
        );

        for family in CandleModelFamily::all() {
            let metadata = factory
                .support_metadata(*family)
                .expect("family is registered");
            assert_eq!(metadata.model_family, *family);
            assert_eq!(metadata.engine, family.engine_name());
            assert!(metadata
                .supported_formats
                .contains(&NativeModelFormat::Safetensors));
            assert!(metadata
                .supported_formats
                .contains(&NativeModelFormat::Gguf));
            assert!(metadata
                .supported_accelerators
                .contains(&NativeAcceleration::Cpu));
            assert!(metadata
                .supported_operations
                .contains(&CandleSupportedOperation::ChatCompletion));
            assert_eq!(
                metadata.tokenizer_requirement(NativeModelFormat::Gguf),
                CandleTokenizerRequirement::GgufMetadata
            );
            assert_eq!(
                metadata.tokenizer_requirement(NativeModelFormat::Safetensors),
                CandleTokenizerRequirement::TokenizerJson
            );
            assert!(metadata.generation_status.contains(family.as_str()));
        }
    }

    #[test]
    fn native_candle_factory_builds_valid_load_plans_for_all_registered_families() {
        let factory = NativeCandleEngineFactory::default();
        let resources = ResourceConfig {
            budget: 0.7,
            cpu_only: true,
            gpu_vendor: "nvidia".to_string(),
        };

        for family in CandleModelFamily::all() {
            let model = ModelConfig {
                alias: format!("{}-chat", family.as_str()),
                path: PathBuf::from(format!("/private/{}-model.gguf", family.as_str())),
                role: "thinking".to_string(),
                weight: 1,
            };

            let plan = factory
                .plan(*family, &model, &resources)
                .expect("registered family plans");

            assert_eq!(plan.runtime, RuntimeBackend::CandleNative);
            assert_eq!(plan.engine, family.engine_name());
            assert_eq!(plan.family, family.as_str());
            assert_eq!(plan.support.model_family, *family);
            assert_eq!(plan.format, NativeModelFormat::Gguf);
            assert_eq!(plan.acceleration, NativeAcceleration::Cpu);
            assert_eq!(plan.device_selection.selected, NativeAcceleration::Cpu);
            assert!(plan.device_selection.fail_closed_if_unavailable);
            assert!(plan.implemented);
            assert!(!plan.candle.load_contract.fail_closed);
            validate_native_engine_load_plan(&plan).expect("load plan validates");

            let rendered = serde_json::to_string(&plan).expect("plan serializes");
            assert!(!rendered.contains("/private"));
            assert!(!rendered.contains("-model.gguf"));
        }
    }

    #[test]
    fn native_candle_factory_rejects_unactionable_load_plans() {
        let factory = NativeCandleEngineFactory::default();
        let model = ModelConfig {
            alias: "qwen-unknown".to_string(),
            path: PathBuf::from("/private/qwen3.bin"),
            role: "thinking".to_string(),
            weight: 1,
        };

        let err = factory
            .plan(CandleModelFamily::Qwen3, &model, &ResourceConfig::default())
            .expect_err("unknown model format is rejected");
        assert!(err.to_string().contains("does not support model format"));

        let valid_model = ModelConfig {
            alias: "qwen-ok".to_string(),
            path: PathBuf::from("/private/qwen3.gguf"),
            role: "coding".to_string(),
            weight: 1,
        };
        let mut plan = factory
            .plan(
                CandleModelFamily::Qwen3,
                &valid_model,
                &ResourceConfig::default(),
            )
            .expect("valid plan");
        plan.engine = "candle-native-mistral".to_string();

        let err = validate_native_engine_load_plan(&plan).expect_err("mismatched engine rejected");
        assert!(err.to_string().contains("engine does not match"));
    }

    #[test]
    fn qwen3_loader_rejects_missing_artifacts_without_leaking_model_path() {
        let model = ModelConfig {
            alias: "qwen-thinking".to_string(),
            path: PathBuf::from("/secret/qwen3-thinking.gguf"),
            role: "thinking".to_string(),
            weight: 1,
        };
        let plan = Qwen3CandleEngineLoader::plan(&model, &ResourceConfig::default())
            .expect("plan validates");

        let err = match Qwen3CandleEngineLoader.load(&plan) {
            Ok(_) => panic!("loader should reject missing artifacts"),
            Err(err) => err,
        };
        let message = err.to_string();

        assert!(message.contains("candle-native-qwen3"));
        assert!(message.contains("qwen-thinking"));
        assert!(!plan.candle.load_contract.fail_closed);
        assert_eq!(
            plan.candle.load_contract.tokenizer,
            CandleTokenizerRequirement::GgufMetadata
        );
        assert!(!message.contains("/secret"));
        assert!(!message.contains("qwen3-thinking.gguf"));
    }

    #[test]
    fn placement_plan_assigns_roles_across_two_servers_without_paths() {
        let mut cfg = Config::default();
        cfg.cluster.node_id = "server-a".to_string();
        cfg.cluster.nodes = vec![
            ClusterNodeConfig {
                id: "server-a".to_string(),
                base_url: "http://10.0.0.10:8765/v1".to_string(),
                roles: vec!["thinking".to_string(), "recommendation".to_string()],
                model_aliases: Vec::new(),
            },
            ClusterNodeConfig {
                id: "server-b".to_string(),
                base_url: "http://10.0.0.11:8765/v1".to_string(),
                roles: vec!["coding".to_string()],
                model_aliases: Vec::new(),
            },
        ];
        cfg.models = vec![
            ModelConfig {
                alias: "qwen-think".to_string(),
                path: PathBuf::from("/models/qwen-thinking.gguf"),
                role: "thinking".to_string(),
                weight: 1,
            },
            ModelConfig {
                alias: "qwen-reco".to_string(),
                path: PathBuf::from("/models/qwen-reco.gguf"),
                role: "recommendation".to_string(),
                weight: 1,
            },
            ModelConfig {
                alias: "qwen-code".to_string(),
                path: PathBuf::from("/models/qwen-code.gguf"),
                role: "coding".to_string(),
                weight: 1,
            },
        ];

        let plan = placement_plan_from_config(&cfg);

        assert_eq!(plan.routing_mode, "cluster-role-placement");
        assert_eq!(plan.local_node, "server-a");
        assert_eq!(
            plan.nodes[0].model_aliases,
            vec!["qwen-think".to_string(), "qwen-reco".to_string()]
        );
        assert_eq!(plan.nodes[1].model_aliases, vec!["qwen-code".to_string()]);
        assert!(plan.unassigned_models.is_empty());

        let rendered = serde_json::to_string(&plan).expect("placement serializes");
        assert!(!rendered.contains("/models"));
        assert!(!rendered.contains(".gguf"));
    }

    #[test]
    fn placement_validation_rejects_unassigned_and_duplicate_models() {
        let unassigned = NativePlacementPlan {
            routing_mode: "cluster-role-placement".to_string(),
            local_node: "server-a".to_string(),
            nodes: vec![NativePlacementNode {
                id: "server-a".to_string(),
                base_url: "http://10.0.0.10:8765/v1".to_string(),
                roles: vec!["thinking".to_string()],
                model_aliases: vec!["qwen-think".to_string()],
            }],
            unassigned_models: vec!["qwen-code".to_string()],
        };
        let err = validate_placement_plan(&unassigned).expect_err("unassigned model rejected");
        assert!(err.to_string().contains("qwen-code"));

        let duplicate = NativePlacementPlan {
            routing_mode: "cluster-role-placement".to_string(),
            local_node: "server-a".to_string(),
            nodes: vec![
                NativePlacementNode {
                    id: "server-a".to_string(),
                    base_url: "http://10.0.0.10:8765/v1".to_string(),
                    roles: vec!["coding".to_string()],
                    model_aliases: vec!["qwen-code".to_string()],
                },
                NativePlacementNode {
                    id: "server-b".to_string(),
                    base_url: "http://10.0.0.11:8765/v1".to_string(),
                    roles: vec!["coding".to_string()],
                    model_aliases: vec!["qwen-code".to_string()],
                },
            ],
            unassigned_models: Vec::new(),
        };
        let err = validate_placement_plan(&duplicate).expect_err("duplicate model rejected");
        assert!(err.to_string().contains("multiple nodes"));
    }

    #[test]
    fn route_selection_returns_node_for_model_or_role() {
        let plan = NativePlacementPlan {
            routing_mode: "cluster-role-placement".to_string(),
            local_node: "server-a".to_string(),
            nodes: vec![
                NativePlacementNode {
                    id: "server-a".to_string(),
                    base_url: "http://10.0.0.10:8765/v1".to_string(),
                    roles: vec!["thinking".to_string(), "recommendation".to_string()],
                    model_aliases: vec!["qwen-think".to_string(), "qwen-reco".to_string()],
                },
                NativePlacementNode {
                    id: "server-b".to_string(),
                    base_url: "http://10.0.0.11:8765/v1".to_string(),
                    roles: vec!["coding".to_string()],
                    model_aliases: vec!["qwen-code".to_string()],
                },
            ],
            unassigned_models: Vec::new(),
        };

        let by_model = route_selection_for_model(&plan, "qwen-code").expect("model route");
        assert_eq!(by_model.query, "model:qwen-code");
        assert_eq!(by_model.candidates[0].id, "server-b");

        let by_role = route_selection_for_role(&plan, "thinking").expect("role route");
        assert_eq!(by_role.query, "role:thinking");
        assert_eq!(by_role.candidates[0].id, "server-a");
    }

    #[test]
    fn heartbeat_reports_single_or_cluster_health_without_paths() {
        let mut cfg = Config::default();
        cfg.cluster.node_id = "server-a".to_string();
        cfg.resources.budget = 0.8;
        cfg.models = vec![ModelConfig {
            alias: "qwen-code".to_string(),
            path: PathBuf::from("/private/qwen-code.gguf"),
            role: "coding".to_string(),
            weight: 1,
        }];

        let single = heartbeat_from_config(&cfg);
        assert_eq!(single.node_id, "server-a");
        assert_eq!(single.routing_mode, "single-node");
        assert!(single.healthy);
        assert_eq!(single.models, 1);
        assert_eq!(single.assigned_models, 1);
        assert_eq!(single.budget_fraction, 0.8);
        assert_eq!(single.heartbeat_interval_seconds, 30);
        assert_eq!(single.telemetry_event, "llmctl.runtime.heartbeat");

        cfg.cluster.nodes = vec![ClusterNodeConfig {
            id: "server-b".to_string(),
            base_url: "http://10.0.0.11:8765/v1".to_string(),
            roles: vec!["thinking".to_string()],
            model_aliases: Vec::new(),
        }];
        let cluster = heartbeat_from_config(&cfg);
        assert_eq!(cluster.routing_mode, "cluster-role-placement");
        assert!(!cluster.healthy);
        assert_eq!(cluster.unassigned_models, vec!["qwen-code".to_string()]);

        let rendered = serde_json::to_string(&cluster).expect("heartbeat serializes");
        assert!(!rendered.contains("/private"));
        assert!(!rendered.contains(".gguf"));
        assert!(cluster
            .safe_telemetry_attributes()
            .values()
            .all(|value| !value.to_string().contains("/private")));
    }
}
