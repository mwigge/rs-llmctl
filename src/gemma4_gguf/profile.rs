//! Gemma 4 GGUF architecture-profile catalog and device selection.
use super::*;

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
