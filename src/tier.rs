//! Hardware-tier detection and model recommendation.
//!
//! Classifies the host into one of `HardwareTier::{Tier1Nv6, Tier2Nv12,
//! Tier3Amd16, Tier3Mac, TierUnknown}` based on the highest-priority
//! compiled-in GPU backend (Metal first, CUDA second, CPU fallback). The
//! resulting recommendation is advisory metadata; operators may explicitly
//! select any supported model regardless of tier.
//!
//! This module is feature-gated by `native-candle` because it depends on
//! candle's `Device` API for backend probing. The recommendation lookup
//! itself is data-only and could be lifted out later if needed.

#![cfg(all(feature = "native-candle", feature = "native-tokenizers"))]

use crate::native::CandleModelFamily;

/// Hardware tier classification used by [`recommend_model_for_tier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareTier {
    /// NVIDIA consumer-class GPU with ≤7 GB VRAM (e.g. RTX 2060/3060 6 GB).
    Tier1Nv6,
    /// NVIDIA mid-range GPU with 8-14 GB VRAM (e.g. RTX 3060 12 GB, 4060 Ti).
    Tier2Nv12,
    /// AMD 16 GB-class GPU on Linux (RX 6800, 7700 XT, etc., reached via ROCm/HIP).
    Tier3Amd16,
    /// Apple Silicon Mac with unified memory. Total RAM ≥ 16 GB assumed.
    Tier3Mac,
    /// Unknown / CPU-only / virtualised GPU with unclassifiable VRAM.
    TierUnknown,
}

impl HardwareTier {
    /// Short identifier used in `/v1/models` capability metadata and logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tier1Nv6 => "tier1-nv6",
            Self::Tier2Nv12 => "tier2-nv12",
            Self::Tier3Amd16 => "tier3-amd16",
            Self::Tier3Mac => "tier3-mac",
            Self::TierUnknown => "unknown",
        }
    }
}

/// Concrete model recommendation for a hardware tier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelDescriptor {
    pub family: CandleModelFamily,
    pub params_b: f32,
    pub recommended_quant: &'static str,
    pub context_window: u32,
}

/// Classify the host into a [`HardwareTier`].
///
/// Probes the compiled-in GPU backends in order of priority: Metal first,
/// CUDA second, CPU fallback. The function never panics and never blocks.
#[must_use]
pub fn detect() -> HardwareTier {
    // Metal — succeeds on any Apple Silicon device with Metal available.
    #[cfg(feature = "gpu-metal")]
    if candle_core::Device::new_metal(0).is_ok() {
        return HardwareTier::Tier3Mac;
    }

    // CUDA path covers both NVIDIA and AMD (via ROCm/HIP CUDA shim).
    // Without a stable VRAM-introspection API in candle 0.10.2, we shell
    // out to `nvidia-smi` for NVIDIA cards and fall back to a conservative
    // mid-tier (Tier2Nv12) when the query fails.
    #[cfg(feature = "gpu-cuda")]
    if candle_core::Device::new_cuda(0).is_ok() {
        return classify_cuda_vram_gib(probe_cuda_total_vram_gib());
    }

    HardwareTier::TierUnknown
}

/// Recommended model for a given hardware tier. Stable lookup table; do
/// not change without updating `specs/tiered-model-recommendation/spec.md`.
#[must_use]
pub const fn recommend_model_for_tier(tier: HardwareTier) -> ModelDescriptor {
    match tier {
        HardwareTier::Tier1Nv6 => ModelDescriptor {
            family: CandleModelFamily::Qwen3,
            params_b: 4.0,
            recommended_quant: "Q4_K_M",
            context_window: 32_768,
        },
        HardwareTier::Tier2Nv12 => ModelDescriptor {
            family: CandleModelFamily::Qwen3,
            params_b: 8.0,
            recommended_quant: "Q4_K_M",
            context_window: 131_072,
        },
        HardwareTier::Tier3Amd16 | HardwareTier::Tier3Mac => ModelDescriptor {
            family: CandleModelFamily::Qwen3,
            params_b: 14.0,
            recommended_quant: "Q4_K_M",
            context_window: 131_072,
        },
        HardwareTier::TierUnknown => ModelDescriptor {
            family: CandleModelFamily::Qwen3,
            params_b: 4.0,
            recommended_quant: "Q4_K_M",
            context_window: 16_384,
        },
    }
}

/// Map a probed VRAM size (in GiB) to a tier. Pure data — easy to unit test.
#[must_use]
fn classify_cuda_vram_gib(vram_gib: Option<f64>) -> HardwareTier {
    let Some(gib) = vram_gib else {
        // Query failed — log a warning and default to mid-tier as a
        // conservative middle. The recommendation is advisory anyway.
        tracing::warn!("GPU VRAM probe failed; defaulting tier classification to Tier2Nv12");
        return HardwareTier::Tier2Nv12;
    };
    match gib {
        v if v <= 7.0 => HardwareTier::Tier1Nv6,
        v if v <= 14.0 => HardwareTier::Tier2Nv12,
        // 15-18 GB → Tier3Amd16 (covers AMD 16 GB cards). Above 18 GB
        // (RTX 3090/4090 24 GB) — treat as Tier 3 for now; a future Tier 4
        // could split out 24 GB+ NVIDIA later.
        _ => HardwareTier::Tier3Amd16,
    }
}

/// Shell out to `nvidia-smi` to probe total VRAM on device 0.
/// Returns None if the command is unavailable or output is unparseable.
#[cfg(feature = "gpu-cuda")]
fn probe_cuda_total_vram_gib() -> Option<f64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = std::str::from_utf8(&output.stdout).ok()?.trim();
    // nvidia-smi reports memory.total in MiB.
    let mib: f64 = s.parse().ok()?;
    Some(mib / 1024.0)
}

/// Log the detected tier and recommended model once at startup.
pub fn log_startup_recommendation() {
    let tier = detect();
    let rec = recommend_model_for_tier(tier);
    tracing::info!(
        tier = tier.as_str(),
        family = rec.family.as_str(),
        params_b = rec.params_b,
        recommended_quant = rec.recommended_quant,
        context_window = rec.context_window,
        "detected hardware tier and recommended model"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_strings_are_stable() {
        // These strings appear in /v1/models capability advertisement; changing
        // them is a breaking change for orchestrators.
        assert_eq!(HardwareTier::Tier1Nv6.as_str(), "tier1-nv6");
        assert_eq!(HardwareTier::Tier2Nv12.as_str(), "tier2-nv12");
        assert_eq!(HardwareTier::Tier3Amd16.as_str(), "tier3-amd16");
        assert_eq!(HardwareTier::Tier3Mac.as_str(), "tier3-mac");
        assert_eq!(HardwareTier::TierUnknown.as_str(), "unknown");
    }

    #[test]
    fn recommend_tier1_is_qwen3_4b_with_32k_context() {
        let r = recommend_model_for_tier(HardwareTier::Tier1Nv6);
        assert_eq!(r.family, CandleModelFamily::Qwen3);
        assert!((r.params_b - 4.0).abs() < f32::EPSILON);
        assert_eq!(r.recommended_quant, "Q4_K_M");
        assert_eq!(r.context_window, 32_768);
    }

    #[test]
    fn recommend_tier2_is_qwen3_8b() {
        let r = recommend_model_for_tier(HardwareTier::Tier2Nv12);
        assert!((r.params_b - 8.0).abs() < f32::EPSILON);
        assert_eq!(r.context_window, 131_072);
    }

    #[test]
    fn recommend_tier3_amd_and_mac_both_get_qwen3_14b() {
        let amd = recommend_model_for_tier(HardwareTier::Tier3Amd16);
        let mac = recommend_model_for_tier(HardwareTier::Tier3Mac);
        assert_eq!(amd, mac);
        assert!((amd.params_b - 14.0).abs() < f32::EPSILON);
        assert_eq!(amd.context_window, 131_072);
    }

    #[test]
    fn recommend_unknown_is_conservative() {
        let r = recommend_model_for_tier(HardwareTier::TierUnknown);
        assert!((r.params_b - 4.0).abs() < f32::EPSILON);
        // Tighter context than Tier1 — we don't know what we're running on.
        assert_eq!(r.context_window, 16_384);
    }

    #[test]
    fn classify_cuda_vram_buckets() {
        assert_eq!(classify_cuda_vram_gib(Some(5.5)), HardwareTier::Tier1Nv6);
        assert_eq!(classify_cuda_vram_gib(Some(6.0)), HardwareTier::Tier1Nv6);
        assert_eq!(classify_cuda_vram_gib(Some(7.0)), HardwareTier::Tier1Nv6);
        assert_eq!(classify_cuda_vram_gib(Some(8.0)), HardwareTier::Tier2Nv12);
        assert_eq!(classify_cuda_vram_gib(Some(12.0)), HardwareTier::Tier2Nv12);
        assert_eq!(classify_cuda_vram_gib(Some(14.0)), HardwareTier::Tier2Nv12);
        assert_eq!(classify_cuda_vram_gib(Some(15.0)), HardwareTier::Tier3Amd16);
        assert_eq!(classify_cuda_vram_gib(Some(16.0)), HardwareTier::Tier3Amd16);
        assert_eq!(classify_cuda_vram_gib(Some(24.0)), HardwareTier::Tier3Amd16);
    }

    #[test]
    fn classify_cuda_vram_failure_defaults_to_tier2() {
        // Conservative middle so unknown configurations don't catastrophically
        // misclassify; operators can override.
        assert_eq!(classify_cuda_vram_gib(None), HardwareTier::Tier2Nv12);
    }

    /// End-to-end probe that exercises the actual `detect()` path on this host.
    /// Asserts only that a tier is returned and the recommendation is sane —
    /// the specific tier varies by build/hardware.
    #[test]
    fn detect_returns_a_supported_tier_on_this_host() {
        let t = detect();
        let r = recommend_model_for_tier(t);
        eprintln!(
            "detected: {} → recommended {} {}B ({}, {}k ctx)",
            t.as_str(),
            r.family.as_str(),
            r.params_b,
            r.recommended_quant,
            r.context_window / 1024
        );
        // Whatever tier we got, the recommendation should be a valid Qwen3 model.
        assert!(matches!(
            r.family,
            CandleModelFamily::Qwen3 | CandleModelFamily::Qwen3Moe
        ));
        assert!(r.params_b > 0.0);
    }
}
