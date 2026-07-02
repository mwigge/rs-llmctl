use crate::config::ModelConfig;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(super) struct ModelList {
    pub(super) object: &'static str,
    pub(super) data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub(super) struct ModelObject {
    pub(super) id: String,
    pub(super) object: &'static str,
    pub(super) owned_by: &'static str,
    /// Capability metadata. Additive over the OpenAI Models response shape;
    /// strict clients that ignore unknown fields are unaffected.
    pub(super) capabilities: ModelCapabilities,
}

/// Per-model capability advertisement consumed by external orchestrators
/// (e.g. milliways sommelier) to route requests without out-of-band knowledge.
/// All fields are always present; unknowns surface as `0` or `"unknown"`
/// rather than being omitted, so the schema stays stable.
#[derive(Debug, Serialize)]
pub(super) struct ModelCapabilities {
    /// Maximum context window in tokens. `0` when not known for the configured family.
    pub(super) context_window: u32,
    /// Stable tool-call protocol identifier; see `CandleModelFamily::tool_protocol()`.
    pub(super) tool_protocol: &'static str,
    /// Tool-call wire format consumed by external runners.
    /// `"xml"` for Mistral-family models (Devstral, Mistral-instruct, etc.);
    /// `"openai"` for all others. Allows milliways local runner to select the
    /// correct prompt strategy without out-of-band configuration.
    pub(super) tool_format: &'static str,
    /// Approximate model size in billions of parameters. `0.0` when unknown.
    /// Heuristic: parsed from a `-{N}B` suffix in the alias if present.
    pub(super) model_size_b: f32,
    /// The compute backend bound at startup: `"metal"`, `"cuda"`, or `"cpu"`.
    pub(super) gpu_backend: &'static str,
    /// Hardware tier classification at startup; matches `HardwareTier::as_str()`.
    pub(super) tier: &'static str,
}

/// Snapshot of host-wide runtime facts that apply identically to every model
/// in the response. Computed once per `list_models` request to avoid repeating
/// the Metal/CUDA probe per model.
#[derive(Debug, Clone, Copy)]
pub(super) struct CapabilitySnapshot {
    pub(super) gpu_backend: &'static str,
    pub(super) tier: &'static str,
}

impl CapabilitySnapshot {
    pub(super) fn current() -> Self {
        Self {
            gpu_backend: current_gpu_backend(),
            tier: current_tier_str(),
        }
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub(super) fn current_gpu_backend() -> &'static str {
    match crate::gemma4_gguf::best_device() {
        candle_core::Device::Metal(_) => "metal",
        candle_core::Device::Cuda(_) => "cuda",
        candle_core::Device::Cpu => "cpu",
    }
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
pub(super) fn current_gpu_backend() -> &'static str {
    "cpu"
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub(super) fn current_tier_str() -> &'static str {
    crate::tier::detect().as_str()
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
pub(super) fn current_tier_str() -> &'static str {
    "unknown"
}

/// Default context window per family. Conservative defaults; overrides
/// will land when ModelConfig grows a per-model `context_window` field.
pub(super) fn default_context_window_for_family(family: &str) -> u32 {
    match family {
        "qwen3" | "qwen3-moe" => 131_072,
        "gemma4" => 131_072,
        "mistral" => 32_768,
        "deepseek" => 32_768,
        _ => 0,
    }
}

/// Best-effort parse of "qwen3-14b-q4_k_m" → 14.0. Falls back to 0.0.
pub(super) fn parse_model_size_b_from_alias(alias: &str) -> f32 {
    let lower = alias.to_lowercase();
    for token in lower.split(|c: char| !c.is_ascii_alphanumeric() && c != '.') {
        if let Some(num_part) = token.strip_suffix('b') {
            if let Ok(v) = num_part.parse::<f32>() {
                if v > 0.0 && v < 1_000.0 {
                    return v;
                }
            }
        }
    }
    0.0
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub(super) fn tool_protocol_for_family(family_str: &str) -> &'static str {
    crate::native::CandleModelFamily::from_kebab_str(family_str)
        .map(|f| f.tool_protocol())
        .unwrap_or("none")
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
pub(super) fn tool_protocol_for_family(_family_str: &str) -> &'static str {
    "none"
}

/// Wire format for tool calls. Mistral-family models require XML tool calling
/// (the `<tool_call>…</tool_call>` block format used by Devstral and Mistral
/// instruct variants). All other families use the standard OpenAI JSON
/// `tool_calls` array. Defaults to `"openai"` for unknown families.
pub(super) fn tool_format_for_family(family_str: &str) -> &'static str {
    if family_str == "mistral" {
        "xml"
    } else {
        "openai"
    }
}

pub(super) fn build_model_object(model: &ModelConfig, snap: CapabilitySnapshot) -> ModelObject {
    let family_str = model.family.as_deref().unwrap_or("");
    ModelObject {
        id: model.alias.clone(),
        object: "model",
        owned_by: "rs-llmctl",
        capabilities: ModelCapabilities {
            context_window: default_context_window_for_family(family_str),
            tool_protocol: tool_protocol_for_family(family_str),
            tool_format: tool_format_for_family(family_str),
            model_size_b: parse_model_size_b_from_alias(&model.alias),
            gpu_backend: snap.gpu_backend,
            tier: snap.tier,
        },
    }
}
