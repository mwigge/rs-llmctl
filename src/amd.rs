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

pub const ROCM_POLICY_VERSION: &str = "2026-06-15";
pub const PRODUCTION_ROCM_VERSION: &str = "7.2.4";
pub const PREVIEW_ROCM_VERSION: &str = "7.13.0";
pub const MIN_ROCM_VRAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Maximum time to wait for a host-probe subprocess (e.g. `rocminfo`,
/// `vulkaninfo`, `lspci`, `rocm-smi`) before killing it and treating the
/// probe as unavailable. Host discovery must never hang indefinitely on a
/// stuck tool.
const HOST_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmdHostDiscovery {
    pub pci_device: Option<String>,
    pub pci_device_id: Option<String>,
    pub gfx_architecture: Option<String>,
    pub vram_bytes: Option<u64>,
    pub os_id: String,
    pub os_like: Vec<String>,
    pub os_version: String,
    pub kernel: String,
    pub driver: Option<String>,
    pub kfd: DeviceAccess,
    pub dri_render: DeviceAccess,
    pub service_user_groups: Vec<String>,
    pub tools: BTreeMap<String, ToolStatus>,
    pub vulkan: VulkanStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceAccess {
    pub path: String,
    pub exists: bool,
    pub readable: bool,
    pub writable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolStatus {
    pub available: bool,
    pub version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanStatus {
    pub available: bool,
    pub device: Option<String>,
    pub driver: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RocmPolicy {
    pub policy_version: String,
    pub selected_version: String,
    pub production_version: String,
    pub preview_version: String,
    pub preview_opt_in: bool,
    pub arch_opt_in: bool,
    pub support_tier: String,
    pub gpu_supported: bool,
    pub memory_supported: bool,
    pub os_supported: bool,
    pub eligible_for_install: bool,
    pub findings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RocmInstallPlan {
    pub status: String,
    pub version: String,
    pub package_manager: Option<String>,
    pub repository: Option<String>,
    pub packages: Vec<String>,
    pub commands: Vec<Vec<String>>,
    pub source_urls: Vec<String>,
    pub findings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RocmPostInstallCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmdBackendDecision {
    pub selected_backend: String,
    pub rocm_status: String,
    pub vulkan_status: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmdQualificationReport {
    pub discovery: AmdHostDiscovery,
    pub policy: RocmPolicy,
    pub install_plan: RocmInstallPlan,
    pub llama_cpp_build: LlamaCppBuildPlan,
    pub post_install_checks: Vec<RocmPostInstallCheck>,
    pub backend: AmdBackendDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaCppBuildPlan {
    pub status: String,
    pub source_repository: String,
    pub source_revision: String,
    pub backend: String,
    pub gfx_target: Option<String>,
    pub cmake_args: Vec<String>,
    pub qualification_commands: Vec<Vec<String>>,
}

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

/// Evaluate a host's [`AmdHostDiscovery`] against AMD's ROCm support
/// policy and produce a [`RocmPolicy`].
///
/// Checks GPU architecture (`gfx1200`/`gfx1201`), measured VRAM against
/// [`MIN_ROCM_VRAM_BYTES`], and operating system against AMD's production
/// support matrix. `preview_opt_in` selects [`PREVIEW_ROCM_VERSION`] instead
/// of [`PRODUCTION_ROCM_VERSION`]. `arch_opt_in` allows Arch-based
/// distributions (CachyOS, Arch, or any `ID_LIKE=arch` derivative) to be
/// marked eligible under the `milliways-arch-preview` support tier, which is
/// outside AMD's official support matrix.
pub fn evaluate_policy(
    host: &AmdHostDiscovery,
    preview_opt_in: bool,
    arch_opt_in: bool,
) -> RocmPolicy {
    let selected_version = if preview_opt_in {
        PREVIEW_ROCM_VERSION
    } else {
        PRODUCTION_ROCM_VERSION
    };
    let gpu_supported = matches!(
        host.gfx_architecture.as_deref(),
        Some("gfx1200" | "gfx1201")
    );
    let os_supported = supported_os(&host.os_id, &host.os_version);
    let memory_supported = host
        .vram_bytes
        .is_some_and(|bytes| bytes >= MIN_ROCM_VRAM_BYTES);
    let mut findings = Vec::new();
    if !gpu_supported {
        findings.push(format!(
            "GPU architecture {} is outside the ROCm policy",
            host.gfx_architecture.as_deref().unwrap_or("unknown")
        ));
    }
    if !os_supported {
        findings.push(format!(
            "operating system {} {} is outside AMD's production support matrix",
            host.os_id, host.os_version
        ));
    }
    if !memory_supported {
        findings.push(format!(
            "measured VRAM {} is below the {} byte ROCm qualification floor",
            host.vram_bytes.unwrap_or_default(),
            MIN_ROCM_VRAM_BYTES
        ));
    }
    if preview_opt_in {
        findings.push("technology-preview ROCm selected by explicit operator opt-in".to_string());
    }
    if arch_opt_in && !os_supported {
        findings.push(
            "Arch-based installation selected by explicit operator opt-in; this is not AMD production support"
                .to_string(),
        );
    }
    let arch_based = matches!(host.os_id.as_str(), "cachyos" | "arch")
        || host.os_like.iter().any(|id| id == "arch");
    let arch_eligible = arch_opt_in && arch_based && gpu_supported && memory_supported;
    RocmPolicy {
        policy_version: ROCM_POLICY_VERSION.to_string(),
        selected_version: selected_version.to_string(),
        production_version: PRODUCTION_ROCM_VERSION.to_string(),
        preview_version: PREVIEW_ROCM_VERSION.to_string(),
        preview_opt_in,
        arch_opt_in,
        support_tier: if os_supported {
            "amd-production".to_string()
        } else if arch_eligible {
            "milliways-arch-preview".to_string()
        } else {
            "unsupported".to_string()
        },
        gpu_supported,
        memory_supported,
        os_supported,
        eligible_for_install: (gpu_supported && memory_supported && os_supported) || arch_eligible,
        findings,
    }
}

/// Produce an operator-facing ROCm installation plan for `host` given an
/// already-evaluated `policy`.
///
/// If `policy.eligible_for_install` is `false`, returns a
/// `status: "unsupported-host"` plan with no packages or commands and
/// `findings` copied from the policy. Otherwise returns a per-distribution
/// plan (`apt` for Ubuntu/Debian, `dnf` for RHEL/Rocky/OL, `zypper` for
/// SLES, `pacman` for Arch-based distributions under the
/// `milliways-arch-preview` tier). All returned plans require explicit
/// operator approval before execution.
pub fn installation_plan(host: &AmdHostDiscovery, policy: &RocmPolicy) -> RocmInstallPlan {
    if !policy.eligible_for_install {
        return RocmInstallPlan {
            status: "unsupported-host".to_string(),
            version: policy.selected_version.clone(),
            package_manager: None,
            repository: None,
            packages: Vec::new(),
            commands: Vec::new(),
            source_urls: Vec::new(),
            findings: policy.findings.clone(),
        };
    }

    let (manager, repository, packages, commands, source_urls) = match host.os_id.as_str() {
        "ubuntu" | "debian" => (
            "apt",
            "https://repo.radeon.com/rocm/apt",
            vec!["rocm", "amd-smi-lib", "rocminfo"],
            vec![
                vec!["apt-get", "update"],
                vec!["apt-get", "install", "rocm", "amd-smi-lib", "rocminfo"],
            ],
            vec![
                "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html",
            ],
        ),
        "rhel" | "rocky" | "ol" => (
            "dnf",
            "https://repo.radeon.com/rocm/rhel",
            vec!["rocm", "amd-smi-lib", "rocminfo"],
            vec![vec!["dnf", "install", "rocm", "amd-smi-lib", "rocminfo"]],
            vec![
                "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html",
            ],
        ),
        "sles" => (
            "zypper",
            "https://repo.radeon.com/rocm/zyp",
            vec!["rocm", "amd-smi-lib", "rocminfo"],
            vec![vec!["zypper", "install", "rocm", "amd-smi-lib", "rocminfo"]],
            vec![
                "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html",
            ],
        ),
        id if matches!(id, "cachyos" | "arch")
            || host.os_like.iter().any(|like| like == "arch") =>
        {
            (
                "pacman",
                "Arch-compatible distribution repositories",
                vec![
                    "rocm-hip-sdk",
                    "rocm-opencl-sdk",
                    "rocblas",
                    "hipblas",
                    "rocm-smi-lib",
                    "hsa-rocr",
                ],
                vec![vec![
                    "pacman",
                    "-S",
                    "rocm-hip-sdk",
                    "rocm-opencl-sdk",
                    "rocblas",
                    "hipblas",
                    "rocm-smi-lib",
                    "hsa-rocr",
                ]],
                vec![
                    "https://github.com/meltingscales/cachyos-whitedragon-ai-lab/blob/main/LLAMA.CPP-ROCM-ARCH-INSTALL.md",
                    "https://gist.github.com/CyntexMore/aef82f0db72e071253a6f531138adfe4",
                    "https://brian.th3rogers.com/posts/strixhalo-cachyos/",
                    "https://gist.github.com/augustin-laurent/d29f026cdb53a4dff50a400c129d3ea7",
                    "https://codepitbull.medium.com/building-a-llm-server-based-on-cachyos-and-amd-ryzen-ai-max-395-strix-halo-1a2260337a8e",
                    "https://wiki.archlinux.org/title/General-purpose_computing_on_graphics_processing_units",
                ],
            )
        }
        _ => {
            return RocmInstallPlan {
                status: "unsupported-host".to_string(),
                version: policy.selected_version.clone(),
                package_manager: None,
                repository: None,
                packages: Vec::new(),
                commands: Vec::new(),
                source_urls: Vec::new(),
                findings: policy.findings.clone(),
            };
        }
    };
    RocmInstallPlan {
        status: "planned".to_string(),
        version: policy.selected_version.clone(),
        package_manager: Some(manager.to_string()),
        repository: Some(repository.to_string()),
        packages: packages.into_iter().map(str::to_string).collect(),
        commands: commands
            .into_iter()
            .map(|command| command.into_iter().map(str::to_string).collect())
            .collect(),
        source_urls: source_urls.into_iter().map(str::to_string).collect(),
        findings: if policy.support_tier == "amd-production" {
            vec![
                "register the version-pinned official AMD repository before installing".to_string(),
                "do not execute this plan without operator approval".to_string(),
            ]
        } else {
            vec![
                "community package plan; AMD does not list CachyOS/Arch in the production support matrix"
                    .to_string(),
                "pin package versions and preserve package/build provenance before qualification"
                    .to_string(),
                "do not execute this plan without operator approval".to_string(),
            ]
        },
    }
}

/// Produce a llama.cpp build plan that selects a backend by falling back
/// from ROCm (`hip`) to `vulkan` to `cpu` depending on `policy` and the
/// Vulkan availability recorded in `host`.
///
/// When `backend == "hip"`, `-DGGML_HIP=ON` is always added; if
/// `host.gfx_architecture` is known, `-DAMDGPU_TARGETS=<gfx>` pins the
/// target, otherwise cmake is left to auto-detect it. When
/// `backend == "vulkan"`, `-DGGML_VULKAN=ON` is added. The plan's `status`
/// is `"fallback"` for `cpu` and `"planned"` otherwise; `source_revision`
/// is a placeholder until a runtime evidence file supplies a real revision
/// (see [`qualification_report_with_evidence`]).
pub fn llama_cpp_build_plan(host: &AmdHostDiscovery, policy: &RocmPolicy) -> LlamaCppBuildPlan {
    let gfx_target = host.gfx_architecture.clone();
    let backend = if policy.eligible_for_install {
        "hip"
    } else if host.vulkan.available {
        "vulkan"
    } else {
        "cpu"
    };
    let mut cmake_args = vec![
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        "-DLLAMA_CURL=OFF".to_string(),
    ];
    match backend {
        "hip" => {
            cmake_args.push("-DGGML_HIP=ON".to_string());
            if let Some(gfx) = gfx_target.as_ref() {
                cmake_args.push(format!("-DAMDGPU_TARGETS={gfx}"));
            }
        }
        "vulkan" => cmake_args.push("-DGGML_VULKAN=ON".to_string()),
        _ => {}
    }
    LlamaCppBuildPlan {
        status: if backend == "cpu" {
            "fallback".to_string()
        } else {
            "planned".to_string()
        },
        source_repository: "https://github.com/ggml-org/llama.cpp".to_string(),
        source_revision: "must-be-pinned-before-build".to_string(),
        backend: backend.to_string(),
        gfx_target,
        cmake_args,
        qualification_commands: vec![
            vec!["llama-bench", "-m", "<model.gguf>", "-ngl", "99"],
            vec!["test-backend-ops"],
            vec![
                "llama-cli",
                "-m",
                "<gemma4.gguf>",
                "-ngl",
                "99",
                "-p",
                "Say hello.",
            ],
        ]
        .into_iter()
        .map(|command| command.into_iter().map(str::to_string).collect())
        .collect(),
    }
}

/// Run post-install ROCm checks against `host` with no runtime evidence
/// file. Equivalent to `post_install_checks_with_evidence(host, None)`.
pub fn post_install_checks(host: &AmdHostDiscovery) -> Vec<RocmPostInstallCheck> {
    post_install_checks_with_evidence(host, None)
}

/// Run post-install ROCm checks against `host`, optionally validating a
/// `runtime_evidence` JSON file from a llama.cpp ROCm smoke run.
///
/// Checks (in order): an `amd-smi`/`rocm-smi` management tool is available,
/// `rocminfo` is available, `/dev/kfd` and the DRI render node are
/// readable/writable, the current user is in the `video`/`render` groups
/// (or the render node is otherwise writable), and finally the
/// `runtime_evidence` file (if any) proves a qualified ROCm backend with a
/// pinned source revision, passing backend operations, and offloaded GPU
/// layers.
pub fn post_install_checks_with_evidence(
    host: &AmdHostDiscovery,
    runtime_evidence: Option<&Path>,
) -> Vec<RocmPostInstallCheck> {
    let tool_check = |name: &str| {
        let available = host.tools.get(name).is_some_and(|tool| tool.available);
        RocmPostInstallCheck {
            name: name.to_string(),
            passed: available,
            detail: if available {
                "available".to_string()
            } else {
                "not installed or not executable".to_string()
            },
        }
    };
    let smi_available = ["amd-smi", "rocm-smi"]
        .iter()
        .any(|name| host.tools.get(*name).is_some_and(|tool| tool.available));
    vec![
        RocmPostInstallCheck {
            name: "amd-smi-or-rocm-smi".to_string(),
            passed: smi_available,
            detail: if smi_available {
                "AMD SMI-compatible management tool available".to_string()
            } else {
                "neither amd-smi nor rocm-smi is executable".to_string()
            },
        },
        tool_check("rocminfo"),
        RocmPostInstallCheck {
            name: "device-access".to_string(),
            passed: host.kfd.readable
                && host.kfd.writable
                && host.dri_render.readable
                && host.dri_render.writable,
            detail: format!(
                "/dev/kfd rw={}, /dev/dri render rw={}",
                host.kfd.readable && host.kfd.writable,
                host.dri_render.readable && host.dri_render.writable
            ),
        },
        RocmPostInstallCheck {
            name: "service-groups".to_string(),
            passed: host
                .service_user_groups
                .iter()
                .any(|group| group == "video")
                && (host
                    .service_user_groups
                    .iter()
                    .any(|group| group == "render")
                    || host.dri_render.writable),
            detail: host.service_user_groups.join(","),
        },
        validate_runtime_evidence(runtime_evidence),
    ]
}

/// Select the runtime backend for llama.cpp by falling back from `rocm` to
/// `vulkan` to `cpu`.
///
/// `rocm` is selected only when `policy.eligible_for_install` is true *and*
/// every check in `checks` (from [`post_install_checks_with_evidence`])
/// passed — including the runtime evidence check, so package presence alone
/// is never sufficient. Otherwise falls back to `vulkan` if
/// `host.vulkan.available`, and finally to `cpu`. `reasons` explains the
/// decision, carrying forward any unmet [`RocmPolicy::findings`].
pub fn select_backend(
    host: &AmdHostDiscovery,
    policy: &RocmPolicy,
    checks: &[RocmPostInstallCheck],
) -> AmdBackendDecision {
    let rocm_ready = policy.eligible_for_install && checks.iter().all(|check| check.passed);
    if rocm_ready {
        return AmdBackendDecision {
            selected_backend: "rocm".to_string(),
            rocm_status: "qualified".to_string(),
            vulkan_status: if host.vulkan.available {
                "available".to_string()
            } else {
                "unavailable".to_string()
            },
            reasons: vec![
                "ROCm policy, host checks, backend operations, and real-model evidence passed"
                    .to_string(),
            ],
        };
    }
    if host.vulkan.available {
        return AmdBackendDecision {
            selected_backend: "vulkan".to_string(),
            rocm_status: "unavailable".to_string(),
            vulkan_status: "validated".to_string(),
            reasons: policy
                .findings
                .iter()
                .cloned()
                .chain([
                    "ROCm post-install checks are incomplete; validated Vulkan selected"
                        .to_string(),
                ])
                .collect(),
        };
    }
    AmdBackendDecision {
        selected_backend: "cpu".to_string(),
        rocm_status: "unavailable".to_string(),
        vulkan_status: "unavailable".to_string(),
        reasons: policy
            .findings
            .iter()
            .cloned()
            .chain(["no validated AMD GPU backend is available".to_string()])
            .collect(),
    }
}

fn supported_os(id: &str, version: &str) -> bool {
    match id {
        "ubuntu" => version.starts_with("24.04") || version.starts_with("22.04"),
        "debian" => matches!(version, "12" | "13"),
        "rhel" | "rocky" | "ol" => version.starts_with("9") || version.starts_with("10"),
        "sles" => version.starts_with("15"),
        _ => false,
    }
}

fn parse_os_release(path: impl AsRef<Path>) -> BTreeMap<String, String> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once('=')?;
            Some((key.to_string(), value.trim_matches('"').to_string()))
        })
        .collect()
}

/// Sysfs prefix that a `/sys/class/drm/card*/device` symlink must resolve
/// under before its vendor/device/VRAM attributes are trusted. `device` is
/// itself a symlink into the real device tree; without this check a crafted
/// or unusual sysfs layout could point it somewhere outside `/sys/devices`.
const SYSFS_DEVICES_ROOT: &str = "/sys/devices";

fn find_amd_drm_device(root: impl AsRef<Path>) -> Option<std::path::PathBuf> {
    fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .find_map(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("card") || name.contains('-') {
                return None;
            }
            let device = entry.path().join("device");
            let resolved = fs::canonicalize(&device).ok()?;
            if !resolved.starts_with(SYSFS_DEVICES_ROOT) {
                return None;
            }
            (read_trimmed(&device.join("vendor")).as_deref() == Some("0x1002")).then_some(device)
        })
}

fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
}

fn discover_vulkan() -> VulkanStatus {
    let Some(output) = command_text("vulkaninfo", &["--summary"]) else {
        return VulkanStatus {
            available: false,
            device: None,
            driver: None,
        };
    };
    VulkanStatus {
        available: output.contains("vendorID")
            && output.contains("0x1002")
            && output.contains("PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"),
        device: line_value(&output, "deviceName"),
        driver: line_value(&output, "driverInfo"),
    }
}

fn extract_gfx(device: &str) -> Option<String> {
    let start = device.to_ascii_lowercase().find("gfx")?;
    let gfx = device[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    (!gfx.is_empty()).then_some(gfx.to_ascii_lowercase())
}

fn line_value(output: &str, key: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let (left, right) = line.split_once('=')?;
        (left.trim() == key).then(|| right.trim().to_string())
    })
}

/// Spawn `command` and capture its output, killing the child and returning
/// `None` if it does not complete within [`HOST_PROBE_TIMEOUT`].
///
/// Host-probe tools (`rocminfo`, `vulkaninfo`, `lspci`, `rocm-smi`, etc.)
/// occasionally hang on misbehaving hardware or drivers; without a timeout
/// a single stuck probe would hang host discovery (and therefore llmctl
/// startup and runtime status checks) indefinitely. The child's stdout and
/// stderr are read to completion on a dedicated thread; this thread waits
/// for that result with `recv_timeout` and kills the child if it does not
/// arrive in time.
///
/// Known limitation: if a probed tool double-forks a grandchild that
/// inherits the piped stdout/stderr fd, killing the direct child leaves
/// that fd open in the grandchild. The reader thread will then block on
/// `read_to_end` permanently — a leaked thread and fd — because the pipe
/// is never fully closed. This is a narrow edge case for the tools probed
/// here (`lspci`, `rocminfo`, `vulkaninfo`) which do not double-fork in
/// normal operation. The caller still receives `None` within
/// `HOST_PROBE_TIMEOUT`, so discovery fails safe.
fn spawn_with_output(mut command: Command) -> Option<Output> {
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    let mut child: Child = command.spawn().ok()?;

    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        use std::io::Read;
        let mut stdout_buf = Vec::new();
        let mut stderr_buf = Vec::new();
        if let Some(mut out) = stdout.take() {
            let _ = out.read_to_end(&mut stdout_buf);
        }
        if let Some(mut err) = stderr.take() {
            let _ = err.read_to_end(&mut stderr_buf);
        }
        // Ignore send errors: the receiver may have already timed out and
        // dropped, in which case there is nothing left to notify.
        let _ = tx.send((stdout_buf, stderr_buf));
    });

    match rx.recv_timeout(HOST_PROBE_TIMEOUT) {
        Ok((stdout_buf, stderr_buf)) => {
            // Output has fully arrived, so the child has exited or is
            // about to; `wait` here will not block meaningfully.
            let status = child.wait().ok()?;
            Some(Output {
                status,
                stdout: stdout_buf,
                stderr: stderr_buf,
            })
        }
        Err(_) => {
            // The probe exceeded HOST_PROBE_TIMEOUT; kill it so it does not
            // linger as an orphaned process.
            let _ = child.kill();
            let _ = child.wait();
            None
        }
    }
}

fn command_text(program: &str, args: &[&str]) -> Option<String> {
    let output = spawn_with_output(command_with_args(program, args))?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn tool_status(program: &str, args: &[&str]) -> ToolStatus {
    match spawn_with_output(command_with_args(program, args)) {
        Some(output) if output.status.success() => ToolStatus {
            available: true,
            version: Some(
                format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
                .trim()
                .lines()
                .next()
                .unwrap_or_default()
                .to_string(),
            ),
        },
        _ => ToolStatus {
            available: false,
            version: None,
        },
    }
}

fn command_with_args(program: &str, args: &[&str]) -> Command {
    let mut cmd = command(program);
    cmd.args(args);
    cmd
}

fn command(program: &str) -> Command {
    let path = Path::new(program);
    if path.components().count() > 1 || command_exists(program) {
        return Command::new(program);
    }
    let rocm_program = PathBuf::from("/opt/rocm/bin").join(program);
    if rocm_program.is_file() {
        return Command::new(rocm_program);
    }
    Command::new(program)
}

fn command_exists(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|path| path.join(program).is_file()))
}

fn validate_runtime_evidence(path: Option<&Path>) -> RocmPostInstallCheck {
    let Some(path) = path else {
        return RocmPostInstallCheck {
            name: "llama.cpp-runtime-evidence".to_string(),
            passed: false,
            detail: "no runtime qualification evidence supplied".to_string(),
        };
    };
    let passed = fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .is_some_and(|value| {
            value.get("qualified").and_then(|v| v.as_bool()) == Some(true)
                && value.get("backend").and_then(|v| v.as_str()) == Some("rocm")
                && value
                    .pointer("/worker/source_revision")
                    .and_then(|v| v.as_str())
                    .is_some_and(|revision| !revision.is_empty())
                && value
                    .pointer("/backend_operations/passed")
                    .and_then(|v| v.as_u64())
                    .is_some_and(|count| count > 0)
                && value
                    .pointer("/smoke/offloaded_layers")
                    .and_then(|v| v.as_u64())
                    .is_some_and(|count| count > 0)
        });
    RocmPostInstallCheck {
        name: "llama.cpp-runtime-evidence".to_string(),
        passed,
        detail: if passed {
            format!("qualified evidence: {}", path.display())
        } else {
            format!("missing or invalid evidence: {}", path.display())
        },
    }
}

fn runtime_evidence_revision(path: Option<&Path>) -> Option<String> {
    let raw = fs::read_to_string(path?).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .pointer("/worker/source_revision")
        .and_then(|value| value.as_str())
        // `as_str()` only guarantees the JSON value is a string, not that
        // it is non-empty; an empty `source_revision` is treated as "no
        // revision" so callers keep the "must-be-pinned-before-build"
        // placeholder instead of an empty string.
        .filter(|revision| !revision.is_empty())
        .map(str::to_string)
}

fn device_access(path: &str) -> DeviceAccess {
    let exists = Path::new(path).exists();
    DeviceAccess {
        path: path.to_string(),
        exists,
        readable: exists && fs::File::open(path).is_ok(),
        writable: exists && fs::OpenOptions::new().write(true).open(path).is_ok(),
    }
}

fn first_render_device(root: &str) -> DeviceAccess {
    let path = fs::read_dir(root).ok().and_then(|entries| {
        entries.filter_map(Result::ok).find_map(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("renderD")
                .then(|| entry.path())
        })
    });
    path.as_deref()
        .and_then(Path::to_str)
        .map(device_access)
        .unwrap_or_else(|| device_access("/dev/dri/renderD128"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn host(os_id: &str, os_version: &str) -> AmdHostDiscovery {
        AmdHostDiscovery {
            pci_device: Some("AMD Radeon".to_string()),
            pci_device_id: Some("0x7590".to_string()),
            gfx_architecture: Some("gfx1200".to_string()),
            vram_bytes: Some(16 * 1024 * 1024 * 1024),
            os_id: os_id.to_string(),
            os_like: if os_id == "cachyos" {
                vec!["arch".to_string()]
            } else {
                Vec::new()
            },
            os_version: os_version.to_string(),
            kernel: "6.8".to_string(),
            driver: Some("Mesa".to_string()),
            kfd: DeviceAccess {
                path: "/dev/kfd".to_string(),
                exists: true,
                readable: true,
                writable: true,
            },
            dri_render: DeviceAccess {
                path: "/dev/dri/renderD128".to_string(),
                exists: true,
                readable: true,
                writable: true,
            },
            service_user_groups: vec!["video".to_string(), "render".to_string()],
            tools: BTreeMap::from([
                (
                    "amd-smi".to_string(),
                    ToolStatus {
                        available: true,
                        version: Some("26.2.2".to_string()),
                    },
                ),
                (
                    "rocminfo".to_string(),
                    ToolStatus {
                        available: true,
                        version: Some("1.0.0".to_string()),
                    },
                ),
            ]),
            vulkan: VulkanStatus {
                available: true,
                device: Some("AMD Radeon (RADV GFX1200)".to_string()),
                driver: Some("Mesa".to_string()),
            },
        }
    }

    fn runtime_evidence() -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("create runtime evidence tempfile");
        write!(
            file,
            r#"{{"qualified":true,"backend":"rocm","worker":{{"source_revision":"abc"}},"backend_operations":{{"passed":1}},"smoke":{{"offloaded_layers":1}}}}"#
        )
        .expect("write runtime evidence fixture");
        file
    }

    #[test]
    fn unsupported_arch_host_falls_back_to_vulkan() {
        let host = host("cachyos", "");
        let policy = evaluate_policy(&host, false, false);
        let decision = select_backend(&host, &policy, &post_install_checks(&host));
        assert!(!policy.os_supported);
        assert_eq!(decision.selected_backend, "vulkan");
    }

    #[test]
    fn supported_ubuntu_host_gets_official_apt_plan() {
        let host = host("ubuntu", "24.04");
        let policy = evaluate_policy(&host, false, false);
        let plan = installation_plan(&host, &policy);
        assert_eq!(plan.status, "planned");
        assert_eq!(plan.package_manager.as_deref(), Some("apt"));
        assert_eq!(plan.version, PRODUCTION_ROCM_VERSION);
    }

    #[test]
    fn preview_requires_explicit_opt_in() {
        let host = host("ubuntu", "24.04");
        assert_eq!(
            evaluate_policy(&host, false, false).selected_version,
            PRODUCTION_ROCM_VERSION
        );
        assert_eq!(
            evaluate_policy(&host, true, false).selected_version,
            PREVIEW_ROCM_VERSION
        );
    }

    #[test]
    fn arch_derivative_requires_arch_opt_in() {
        let host = host("cachyos", "");
        assert!(!evaluate_policy(&host, false, false).eligible_for_install);
        let policy = evaluate_policy(&host, false, true);
        assert!(policy.eligible_for_install);
        assert_eq!(policy.support_tier, "milliways-arch-preview");
        assert_eq!(
            installation_plan(&host, &policy).package_manager.as_deref(),
            Some("pacman")
        );
    }

    #[test]
    fn plain_arch_host_requires_arch_opt_in() {
        let host = host("arch", "");
        assert!(!evaluate_policy(&host, false, false).eligible_for_install);
        let policy = evaluate_policy(&host, false, true);
        assert!(policy.eligible_for_install);
        assert_eq!(policy.support_tier, "milliways-arch-preview");
        assert_eq!(
            installation_plan(&host, &policy).package_manager.as_deref(),
            Some("pacman")
        );
    }

    #[test]
    fn arch_based_distro_via_os_like_gets_pacman_plan() {
        let mut host = host("ubuntu", "24.04");
        host.os_id = "steamos".to_string();
        host.os_like = vec!["arch".to_string()];
        let policy = evaluate_policy(&host, false, true);
        assert!(policy.eligible_for_install);
        let plan = installation_plan(&host, &policy);
        assert_eq!(plan.package_manager.as_deref(), Some("pacman"));
    }

    #[test]
    fn package_checks_do_not_qualify_rocm_without_runtime_evidence() {
        let host = host("ubuntu", "24.04");
        let policy = evaluate_policy(&host, false, false);
        let decision = select_backend(&host, &policy, &post_install_checks(&host));
        assert_eq!(decision.selected_backend, "vulkan");
        assert_eq!(decision.rocm_status, "unavailable");
    }

    #[test]
    fn supported_measured_host_selects_rocm_with_runtime_evidence() {
        let host = host("ubuntu", "24.04");
        let policy = evaluate_policy(&host, false, false);
        let evidence = runtime_evidence();
        let checks = post_install_checks_with_evidence(&host, Some(evidence.path()));
        assert!(policy.memory_supported);
        assert_eq!(
            select_backend(&host, &policy, &checks).selected_backend,
            "rocm"
        );
    }

    #[test]
    fn missing_device_access_falls_back_to_vulkan() {
        let mut host = host("ubuntu", "24.04");
        host.kfd.writable = false;
        let policy = evaluate_policy(&host, false, false);
        let evidence = runtime_evidence();
        let checks = post_install_checks_with_evidence(&host, Some(evidence.path()));
        assert_eq!(
            select_backend(&host, &policy, &checks).selected_backend,
            "vulkan"
        );
    }

    #[test]
    fn cpu_only_host_is_not_an_amd_gpu_false_positive() {
        let mut host = host("ubuntu", "24.04");
        host.gfx_architecture = None;
        host.pci_device = Some("AMD Ryzen CPU".to_string());
        let policy = evaluate_policy(&host, false, false);
        assert!(!policy.gpu_supported);
        assert!(!policy.eligible_for_install);
    }

    #[test]
    fn measured_vram_below_floor_is_not_eligible() {
        let mut host = host("ubuntu", "24.04");
        host.vram_bytes = Some(4 * 1024 * 1024 * 1024);
        let policy = evaluate_policy(&host, false, false);
        assert!(!policy.memory_supported);
        assert!(!policy.eligible_for_install);
    }

    #[test]
    #[cfg(unix)]
    fn find_amd_drm_device_rejects_device_symlink_outside_sysfs() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("create drm root tempdir");
        let card = root.path().join("card0");
        fs::create_dir_all(&card).expect("create card0 dir");

        // The real target lives inside the tempdir, i.e. outside
        // SYSFS_DEVICES_ROOT ("/sys/devices") — `device` symlinks to it the
        // way a real /sys/class/drm/card*/device entry would symlink into
        // the sysfs device tree.
        let real_device = root.path().join("not-sysfs-device");
        fs::create_dir_all(&real_device).expect("create fake device dir");
        fs::write(real_device.join("vendor"), "0x1002\n").expect("write vendor");

        symlink(&real_device, card.join("device")).expect("symlink device");

        assert_eq!(find_amd_drm_device(root.path()), None);
    }

    #[test]
    #[cfg(unix)]
    fn find_amd_drm_device_accepts_amd_vendor_under_card_dir() {
        // Without a `device` symlink, `card0/device` is a plain directory
        // under the tempdir. canonicalize() will not resolve under
        // /sys/devices on a real system either, so this confirms the
        // negative path covers a host with no AMD DRM device rather than
        // asserting a positive match (which would require a real sysfs
        // layout under /sys/devices).
        let root = tempfile::tempdir().expect("create drm root tempdir");
        let device = root.path().join("card0").join("device");
        fs::create_dir_all(&device).expect("create card0/device dir");
        fs::write(device.join("vendor"), "0x1002\n").expect("write vendor");

        assert_eq!(find_amd_drm_device(root.path()), None);
    }

    #[test]
    fn spawn_with_output_returns_none_for_missing_program() {
        let command = command_with_args("definitely-not-a-real-binary-amd-test", &[]);
        assert_eq!(spawn_with_output(command), None);
    }

    #[test]
    fn spawn_with_output_kills_process_that_exceeds_timeout() {
        // `sleep` well beyond HOST_PROBE_TIMEOUT (5s); spawn_with_output
        // must not block the test for the full sleep duration and must
        // return None once HOST_PROBE_TIMEOUT elapses.
        let command = command_with_args("sleep", &["30"]);
        let started = std::time::Instant::now();

        let output = spawn_with_output(command);

        assert_eq!(output, None);
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "spawn_with_output should time out near HOST_PROBE_TIMEOUT, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn hip_without_gfx_does_not_include_gpu_targets() {
        let mut host = host("ubuntu", "24.04");
        host.gfx_architecture = None;
        let mut policy = evaluate_policy(&host, false, false);
        // Force the `hip` backend even though `gfx_architecture` is
        // unknown, so the build plan must omit -DAMDGPU_TARGETS and let
        // cmake auto-detect the target instead of emitting an empty value.
        policy.eligible_for_install = true;

        let plan = llama_cpp_build_plan(&host, &policy);

        assert_eq!(plan.backend, "hip");
        assert_eq!(plan.gfx_target, None);
        assert!(plan.cmake_args.contains(&"-DGGML_HIP=ON".to_string()));
        assert!(
            !plan
                .cmake_args
                .iter()
                .any(|arg| arg.starts_with("-DAMDGPU_TARGETS=")),
            "cmake args should not include -DAMDGPU_TARGETS when gfx_architecture is unknown: {:?}",
            plan.cmake_args
        );
    }
}
