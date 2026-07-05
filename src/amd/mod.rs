//! Generic Arch-based ROCm/AMD GPU host qualification.
//!
//! This module discovers AMD GPU hardware and driver state on the local
//! host (`discover_host`), evaluates that discovery against AMD's ROCm
//! support policy (`evaluate_policy`), and produces operator-facing plans:
//! a ROCm installation plan (`installation_plan`), a llama.cpp build plan
//! (`llama_cpp_build_plan`), and a combined qualification report
//! (`qualification_report` / `qualification_report_with_evidence`).
//!
//! The qualification report also selects a runtime backend
//! (`select_backend`) by falling back from ROCm to Vulkan to CPU depending
//! on which layers are actually validated on the host, so that llmctl can
//! make a safe choice even when ROCm is not installed or not yet proven.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output};
use std::sync::mpsc;
use std::time::Duration;

mod types;
pub use types::*;
mod probe;
pub(crate) use probe::*;
mod policy;
pub use policy::*;

#[cfg(test)]
mod tests;

/// Run full AMD GPU host discovery and qualification with no runtime
/// evidence file. Equivalent to
/// `qualification_report_with_evidence(preview_opt_in, arch_opt_in, None)`.
pub fn qualification_report(preview_opt_in: bool, arch_opt_in: bool) -> AmdQualificationReport {
    qualification_report_with_evidence(preview_opt_in, arch_opt_in, None)
}

/// Run full AMD GPU host discovery and qualification, optionally
/// incorporating a `runtime_evidence` JSON file produced by a prior
/// llama.cpp ROCm smoke run.
///
/// This combines [`discover_host`], [`evaluate_policy`],
/// [`installation_plan`], [`llama_cpp_build_plan`], and
/// [`post_install_checks_with_evidence`] into a single
/// [`AmdQualificationReport`], and selects the runtime backend
/// ([`select_backend`]) by falling back from ROCm to Vulkan to CPU. If
/// `runtime_evidence` points at a file proving a qualified ROCm build, the
/// llama.cpp build plan is marked `"qualified"` and stamped with the
/// evidence's source revision.
pub fn qualification_report_with_evidence(
    preview_opt_in: bool,
    arch_opt_in: bool,
    runtime_evidence: Option<&Path>,
) -> AmdQualificationReport {
    let discovery = discover_host();
    let policy = evaluate_policy(&discovery, preview_opt_in, arch_opt_in);
    let install_plan = installation_plan(&discovery, &policy);
    let mut llama_cpp_build = llama_cpp_build_plan(&discovery, &policy);
    if let Some(revision) = runtime_evidence_revision(runtime_evidence) {
        llama_cpp_build.status = "qualified".to_string();
        llama_cpp_build.source_revision = revision;
    }
    let post_install_checks = post_install_checks_with_evidence(&discovery, runtime_evidence);
    let backend = select_backend(&discovery, &policy, &post_install_checks);
    AmdQualificationReport {
        discovery,
        policy,
        install_plan,
        llama_cpp_build,
        post_install_checks,
        backend,
    }
}

/// Probe the local host for AMD GPU hardware, driver, and tooling state.
///
/// Reads `/etc/os-release`, walks `/sys/class/drm` for an AMD DRM device
/// (vendor ID `0x1002`) to read PCI device ID and VRAM size, checks
/// `/dev/kfd` and `/dev/dri` render node access, and probes `vulkaninfo`,
/// `lspci`, `id -Gn`, and the `amd-smi`/`rocm-smi`/`rocminfo` tools (each
/// bounded by [`HOST_PROBE_TIMEOUT`]). Any probe that fails or times out is
/// reflected as `None`/empty/unavailable in the returned discovery rather
/// than causing this function to fail.
pub fn discover_host() -> AmdHostDiscovery {
    let os_release = parse_os_release("/etc/os-release");
    let vulkan = discover_vulkan();
    let drm_device = find_amd_drm_device("/sys/class/drm");
    let pci_device_id = drm_device
        .as_ref()
        .and_then(|device| read_trimmed(&device.join("device")));
    let vram_bytes = drm_device.as_ref().and_then(|device| {
        read_trimmed(&device.join("mem_info_vram_total")).and_then(|v| v.parse().ok())
    });
    let pci_device = command_text("lspci", &["-nn"]).and_then(|output| {
        output
            .lines()
            .find(|line| {
                line.contains("VGA compatible controller")
                    && (line.contains("AMD") || line.contains("ATI"))
            })
            .map(str::to_string)
    });
    let gfx_architecture = vulkan.device.as_deref().and_then(extract_gfx);
    let service_user_groups = match command_text("id", &["-Gn"]) {
        Some(groups) => groups.split_whitespace().map(str::to_string).collect(),
        None => {
            tracing::debug!("`id -Gn` probe failed; service_user_groups will be empty");
            Vec::new()
        }
    };
    let mut tools = BTreeMap::new();
    for (name, version_args) in [
        ("amd-smi", vec!["version"]),
        ("rocm-smi", vec!["--version"]),
        ("rocminfo", vec![]),
    ] {
        tools.insert(name.to_string(), tool_status(name, &version_args));
    }

    AmdHostDiscovery {
        pci_device,
        pci_device_id,
        gfx_architecture,
        vram_bytes,
        os_id: os_release.get("ID").cloned().unwrap_or_default(),
        os_like: os_release
            .get("ID_LIKE")
            .map(|value| value.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default(),
        os_version: os_release.get("VERSION_ID").cloned().unwrap_or_default(),
        kernel: command_text("uname", &["-r"]).unwrap_or_default(),
        driver: vulkan.driver.clone(),
        kfd: device_access("/dev/kfd"),
        dri_render: first_render_device("/dev/dri"),
        service_user_groups,
        tools,
        vulkan,
    }
}
