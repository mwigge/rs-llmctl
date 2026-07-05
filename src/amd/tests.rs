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
