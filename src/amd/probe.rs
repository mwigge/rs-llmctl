//! Low-level host probes: os-release, sysfs DRM, Vulkan, and bounded subprocess execution.
use super::*;

/// Maximum time to wait for a host-probe subprocess (e.g. `rocminfo`,
/// `vulkaninfo`, `lspci`, `rocm-smi`) before killing it and treating the
/// probe as unavailable. Host discovery must never hang indefinitely on a
/// stuck tool.
const HOST_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) fn supported_os(id: &str, version: &str) -> bool {
    match id {
        "ubuntu" => version.starts_with("24.04") || version.starts_with("22.04"),
        "debian" => matches!(version, "12" | "13"),
        "rhel" | "rocky" | "ol" => version.starts_with("9") || version.starts_with("10"),
        "sles" => version.starts_with("15"),
        _ => false,
    }
}

pub(crate) fn parse_os_release(path: impl AsRef<Path>) -> BTreeMap<String, String> {
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

pub(crate) fn find_amd_drm_device(root: impl AsRef<Path>) -> Option<std::path::PathBuf> {
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

pub(crate) fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
}

pub(crate) fn discover_vulkan() -> VulkanStatus {
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

pub(crate) fn extract_gfx(device: &str) -> Option<String> {
    let start = device.to_ascii_lowercase().find("gfx")?;
    let gfx = device[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    (!gfx.is_empty()).then_some(gfx.to_ascii_lowercase())
}

pub(crate) fn line_value(output: &str, key: &str) -> Option<String> {
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
pub(crate) fn spawn_with_output(mut command: Command) -> Option<Output> {
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

pub(crate) fn command_text(program: &str, args: &[&str]) -> Option<String> {
    let output = spawn_with_output(command_with_args(program, args))?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub(crate) fn tool_status(program: &str, args: &[&str]) -> ToolStatus {
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

pub(crate) fn command_with_args(program: &str, args: &[&str]) -> Command {
    let mut cmd = command(program);
    cmd.args(args);
    cmd
}

pub(crate) fn command(program: &str) -> Command {
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

pub(crate) fn command_exists(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|path| path.join(program).is_file()))
}

pub(crate) fn validate_runtime_evidence(path: Option<&Path>) -> RocmPostInstallCheck {
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

pub(crate) fn runtime_evidence_revision(path: Option<&Path>) -> Option<String> {
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

pub(crate) fn device_access(path: &str) -> DeviceAccess {
    let exists = Path::new(path).exists();
    DeviceAccess {
        path: path.to_string(),
        exists,
        readable: exists && fs::File::open(path).is_ok(),
        writable: exists && fs::OpenOptions::new().write(true).open(path).is_ok(),
    }
}

pub(crate) fn first_render_device(root: &str) -> DeviceAccess {
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
