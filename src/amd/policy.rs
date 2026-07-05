//! ROCm support-policy evaluation, install/build plans, and backend selection.
use super::*;

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
