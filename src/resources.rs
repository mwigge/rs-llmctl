use crate::config::ResourceConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;
use sysinfo::System;

const DEFAULT_BUDGET_FRACTION: f64 = 0.80;
const MIN_SAFE_BUDGET_FRACTION: f64 = 0.10;
const MAX_SAFE_BUDGET_FRACTION: f64 = 0.95;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Apple,
    Unknown,
}

impl GpuVendor {
    pub fn from_config_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "nvidia" | "cuda" => Some(Self::Nvidia),
            "amd" | "rocm" | "hip" => Some(Self::Amd),
            "apple" | "metal" => Some(Self::Apple),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuSnapshot {
    pub vendor: GpuVendor,
    pub name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub cpu_threads: usize,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub cpu_only: bool,
    pub gpus: Vec<GpuSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetPlan {
    pub budget_fraction: f64,
    pub cpu_threads: usize,
    pub memory_budget_bytes: u64,
    pub gpu_budgets: Vec<GpuBudget>,
    pub limits: ResourceLimitPlan,
    pub findings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuBudget {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_budget_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimitPlan {
    pub cpu_quota_percent: u32,
    pub memory_max_bytes: u64,
    pub systemd: SystemdResourceProperties,
    pub gpu_vram_budgets: Vec<GpuVramBudgetMetadata>,
}

impl Default for ResourceLimitPlan {
    fn default() -> Self {
        Self {
            cpu_quota_percent: 100,
            memory_max_bytes: u64::MAX,
            systemd: systemd_resource_properties(100, u64::MAX),
            gpu_vram_budgets: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemdResourceProperties {
    #[serde(rename = "CPUQuota")]
    pub cpu_quota: String,
    #[serde(rename = "MemoryMax")]
    pub memory_max: u64,
    pub unit_properties: Vec<String>,
    pub systemd_run_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuVramBudgetMetadata {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_budget_bytes: u64,
    pub enforcement: String,
    pub enforcement_status: String,
    pub hard_enforced: bool,
    pub systemd_property: Option<String>,
}

pub fn snapshot(config: &ResourceConfig) -> ResourceSnapshot {
    let mut system = System::new_all();
    system.refresh_memory();
    let cpu_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let mut snapshot = ResourceSnapshot {
        cpu_threads,
        total_memory_bytes: system.total_memory(),
        available_memory_bytes: system.available_memory(),
        cpu_only: config.cpu_only,
        gpus: Vec::new(),
    };

    if config.cpu_only {
        return snapshot;
    }

    match config.gpu_vendor.trim().to_ascii_lowercase().as_str() {
        "nvidia" => snapshot.gpus.extend(nvidia_smi_gpus()),
        "amd" => snapshot.gpus.extend(amd_gpus()),
        "auto" | "" => {
            snapshot.gpus.extend(nvidia_smi_gpus());
            if snapshot.gpus.is_empty() {
                snapshot.gpus.extend(amd_gpus());
            }
        }
        "apple" | "metal" => {}
        _ => {}
    }
    snapshot
}

pub fn budget_plan(snapshot: &ResourceSnapshot, requested_fraction: f64) -> BudgetPlan {
    let (budget_fraction, findings) = normalized_budget_fraction(requested_fraction);
    let memory_budget_bytes = bytes_fraction(snapshot.available_memory_bytes, budget_fraction);
    let gpu_budgets: Vec<_> = snapshot
        .gpus
        .iter()
        .map(|gpu| GpuBudget {
            vendor: gpu.vendor.clone(),
            name: gpu.name.clone(),
            vram_budget_bytes: bytes_fraction(
                gpu.free_vram_bytes.unwrap_or(gpu.total_vram_bytes),
                budget_fraction,
            ),
        })
        .collect();
    let gpu_vram_budgets = gpu_budgets
        .iter()
        .map(|budget| GpuVramBudgetMetadata {
            vendor: budget.vendor.clone(),
            name: budget.name.clone(),
            vram_budget_bytes: budget.vram_budget_bytes,
            enforcement: "metadata-only".to_string(),
            enforcement_status: "metadata-only".to_string(),
            hard_enforced: false,
            systemd_property: None,
        })
        .collect();
    let cpu_quota_percent = cpu_quota_percent(snapshot.cpu_threads, budget_fraction);
    BudgetPlan {
        budget_fraction,
        cpu_threads: ((snapshot.cpu_threads as f64) * budget_fraction)
            .floor()
            .max(1.0) as usize,
        memory_budget_bytes,
        gpu_budgets,
        limits: ResourceLimitPlan {
            cpu_quota_percent,
            memory_max_bytes: memory_budget_bytes,
            systemd: systemd_resource_properties(cpu_quota_percent, memory_budget_bytes),
            gpu_vram_budgets,
        },
        findings,
    }
}

pub fn snapshot_and_plan(config: &ResourceConfig) -> (ResourceSnapshot, BudgetPlan) {
    let snapshot = snapshot(config);
    let plan = budget_plan(&snapshot, config.budget);
    (snapshot, plan)
}

pub fn cpu_only_snapshot(
    total_memory_bytes: u64,
    available_memory_bytes: u64,
    cpu_threads: usize,
) -> ResourceSnapshot {
    ResourceSnapshot {
        cpu_threads: cpu_threads.max(1),
        total_memory_bytes,
        available_memory_bytes,
        cpu_only: true,
        gpus: Vec::new(),
    }
}

pub fn parse_nvidia_smi_csv(output: &str) -> Vec<GpuSnapshot> {
    output
        .lines()
        .filter_map(|line| {
            let mut columns = line.split(',').map(str::trim);
            let name = columns.next()?.to_string();
            let total_mib = parse_u64_prefix(columns.next()?)?;
            let free_mib = parse_u64_prefix(columns.next()?);
            Some(GpuSnapshot {
                vendor: GpuVendor::Nvidia,
                name,
                total_vram_bytes: mib_to_bytes(total_mib),
                free_vram_bytes: free_mib.map(mib_to_bytes),
            })
        })
        .collect()
}

pub fn parse_rocm_smi_text(output: &str) -> Vec<GpuSnapshot> {
    let mut gpus = Vec::new();
    for line in output.lines() {
        let lower = line.to_ascii_lowercase();
        if !lower.contains("vram") && !lower.contains("memory") {
            continue;
        }
        let Some(value) = line
            .split(|ch: char| !ch.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .filter_map(|part| part.parse::<u64>().ok())
            .next_back()
        else {
            continue;
        };
        let total_vram_bytes = if lower.contains("(b)") || lower.contains(" bytes") {
            value
        } else {
            mib_to_bytes(value)
        };
        gpus.push(GpuSnapshot {
            vendor: GpuVendor::Amd,
            name: format!("AMD GPU {}", gpus.len()),
            total_vram_bytes,
            free_vram_bytes: None,
        });
    }
    gpus
}

fn nvidia_smi_gpus() -> Vec<GpuSnapshot> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            parse_nvidia_smi_csv(&String::from_utf8_lossy(&output.stdout))
        }
        _ => Vec::new(),
    }
}

fn amd_gpus() -> Vec<GpuSnapshot> {
    let mut gpus = amd_sysfs_gpus("/sys/class/drm");
    if !gpus.is_empty() {
        return gpus;
    }
    let output = Command::new("rocm-smi")
        .arg("--showmeminfo")
        .arg("vram")
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            gpus = parse_rocm_smi_text(&String::from_utf8_lossy(&output.stdout));
        }
    }
    gpus
}

fn amd_sysfs_gpus(root: impl AsRef<Path>) -> Vec<GpuSnapshot> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("card") || name.contains('-') {
                return None;
            }
            let device = entry.path().join("device");
            let vendor = fs::read_to_string(device.join("vendor")).ok()?;
            if vendor.trim() != "0x1002" {
                return None;
            }
            let total = read_first_u64(&[
                device.join("mem_info_vram_total"),
                device.join("mem_info_vis_vram_total"),
            ])?;
            let free = read_first_u64(&[
                device.join("mem_info_vram_used"),
                device.join("mem_info_vis_vram_used"),
            ])
            .map(|used| total.saturating_sub(used));
            Some(GpuSnapshot {
                vendor: GpuVendor::Amd,
                name: name.to_string(),
                total_vram_bytes: total,
                free_vram_bytes: free,
            })
        })
        .collect()
}

fn read_first_u64(paths: &[impl AsRef<Path>]) -> Option<u64> {
    paths.iter().find_map(|path| {
        fs::read_to_string(path)
            .ok()
            .and_then(|body| body.trim().parse().ok())
    })
}

fn normalized_budget_fraction(value: f64) -> (f64, Vec<String>) {
    if !value.is_finite() || value <= 0.0 {
        return (
            DEFAULT_BUDGET_FRACTION,
            vec!["resource budget must be finite and greater than zero".to_string()],
        );
    }
    if value < MIN_SAFE_BUDGET_FRACTION {
        return (
            DEFAULT_BUDGET_FRACTION,
            vec![format!(
                "resource budget {value:.2} is below safe minimum {MIN_SAFE_BUDGET_FRACTION:.2}"
            )],
        );
    }
    if value > MAX_SAFE_BUDGET_FRACTION {
        return (
            DEFAULT_BUDGET_FRACTION,
            vec![format!(
                "resource budget {value:.2} exceeds safe maximum {MAX_SAFE_BUDGET_FRACTION:.2}"
            )],
        );
    }
    (value, Vec::new())
}

fn cpu_quota_percent(cpu_threads: usize, fraction: f64) -> u32 {
    ((cpu_threads.max(1) as f64) * fraction * 100.0)
        .round()
        .max(1.0) as u32
}

fn systemd_resource_properties(
    cpu_quota_percent: u32,
    memory_max_bytes: u64,
) -> SystemdResourceProperties {
    let cpu_quota = format!("{cpu_quota_percent}%");
    let unit_properties = vec![
        "CPUAccounting=true".to_string(),
        "MemoryAccounting=true".to_string(),
        format!("CPUQuota={cpu_quota}"),
        format!("MemoryMax={memory_max_bytes}"),
    ];
    let systemd_run_args = unit_properties
        .iter()
        .map(|property| format!("--property={property}"))
        .collect();

    SystemdResourceProperties {
        cpu_quota,
        memory_max: memory_max_bytes,
        unit_properties,
        systemd_run_args,
    }
}

fn bytes_fraction(bytes: u64, fraction: f64) -> u64 {
    ((bytes as f64) * fraction).floor() as u64
}

fn mib_to_bytes(mib: u64) -> u64 {
    mib.saturating_mul(1024 * 1024)
}

fn parse_u64_prefix(value: &str) -> Option<u64> {
    value
        .split_whitespace()
        .next()
        .and_then(|part| part.parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_budget_uses_eighty_percent_default() {
        let snapshot = cpu_only_snapshot(16_000, 10_000, 8);
        let plan = budget_plan(&snapshot, 0.0);
        assert_eq!(plan.budget_fraction, 0.80);
        assert_eq!(plan.memory_budget_bytes, 8_000);
        assert_eq!(plan.cpu_threads, 6);
        assert_eq!(plan.limits.cpu_quota_percent, 640);
        assert_eq!(plan.limits.memory_max_bytes, 8_000);
        assert_eq!(plan.limits.systemd.cpu_quota, "640%");
        assert_eq!(plan.limits.systemd.memory_max, 8_000);
        assert_eq!(
            plan.findings,
            vec!["resource budget must be finite and greater than zero".to_string()]
        );
        assert!(plan.gpu_budgets.is_empty());
        assert!(plan.limits.gpu_vram_budgets.is_empty());
    }

    #[test]
    fn first_time_user_default_surfaces_eighty_percent_cpu_ram_and_vram_budgets() {
        let snapshot = ResourceSnapshot {
            cpu_threads: 10,
            total_memory_bytes: 64_000,
            available_memory_bytes: 40_000,
            cpu_only: false,
            gpus: vec![GpuSnapshot {
                vendor: GpuVendor::Nvidia,
                name: "RTX 6000".to_string(),
                total_vram_bytes: 48_000,
                free_vram_bytes: Some(30_000),
            }],
        };
        let plan = budget_plan(&snapshot, ResourceConfig::default().budget);

        assert_eq!(plan.budget_fraction, 0.80);
        assert_eq!(plan.cpu_threads, 8);
        assert_eq!(plan.memory_budget_bytes, 32_000);
        assert_eq!(plan.gpu_budgets[0].vram_budget_bytes, 24_000);
        assert_eq!(plan.limits.cpu_quota_percent, 800);
        assert_eq!(plan.limits.memory_max_bytes, 32_000);
        assert_eq!(
            plan.limits.systemd.unit_properties,
            vec![
                "CPUAccounting=true",
                "MemoryAccounting=true",
                "CPUQuota=800%",
                "MemoryMax=32000"
            ]
        );
        assert_eq!(
            plan.limits.systemd.systemd_run_args,
            vec![
                "--property=CPUAccounting=true",
                "--property=MemoryAccounting=true",
                "--property=CPUQuota=800%",
                "--property=MemoryMax=32000"
            ]
        );
        assert_eq!(plan.limits.gpu_vram_budgets[0].enforcement, "metadata-only");
        assert_eq!(
            plan.limits.gpu_vram_budgets[0].enforcement_status,
            "metadata-only"
        );
        assert!(!plan.limits.gpu_vram_budgets[0].hard_enforced);
        assert_eq!(plan.limits.gpu_vram_budgets[0].systemd_property, None);
    }

    #[test]
    fn rejects_unsafe_budget_fraction_above_safe_bound() {
        let snapshot = cpu_only_snapshot(16_000, 10_000, 8);
        let plan = budget_plan(&snapshot, 1.0);
        assert_eq!(plan.budget_fraction, 0.80);
        assert_eq!(plan.limits.cpu_quota_percent, 640);
        assert!(plan
            .findings
            .iter()
            .any(|finding| finding == "resource budget 1.00 exceeds safe maximum 0.95"));
    }

    #[test]
    fn parses_nvidia_smi_csv() {
        let gpus = parse_nvidia_smi_csv("NVIDIA T100, 4096, 3072\nRTX 6000, 24576, 20000\n");
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].vendor, GpuVendor::Nvidia);
        assert_eq!(gpus[0].name, "NVIDIA T100");
        assert_eq!(gpus[0].total_vram_bytes, 4096 * 1024 * 1024);
        assert_eq!(gpus[0].free_vram_bytes, Some(3072 * 1024 * 1024));
    }

    #[test]
    fn plans_gpu_budget_from_free_vram() {
        let snapshot = ResourceSnapshot {
            cpu_threads: 12,
            total_memory_bytes: 64_000,
            available_memory_bytes: 32_000,
            cpu_only: false,
            gpus: vec![GpuSnapshot {
                vendor: GpuVendor::Amd,
                name: "card0".to_string(),
                total_vram_bytes: 20_000,
                free_vram_bytes: Some(10_000),
            }],
        };
        let plan = budget_plan(&snapshot, 0.5);
        assert_eq!(plan.cpu_threads, 6);
        assert_eq!(plan.memory_budget_bytes, 16_000);
        assert_eq!(plan.gpu_budgets[0].vram_budget_bytes, 5_000);
        assert_eq!(plan.limits.cpu_quota_percent, 600);
        assert_eq!(plan.limits.memory_max_bytes, 16_000);
        assert_eq!(plan.limits.systemd.cpu_quota, "600%");
        assert_eq!(plan.limits.systemd.memory_max, 16_000);
        assert_eq!(
            plan.limits.gpu_vram_budgets,
            vec![GpuVramBudgetMetadata {
                vendor: GpuVendor::Amd,
                name: "card0".to_string(),
                vram_budget_bytes: 5_000,
                enforcement: "metadata-only".to_string(),
                enforcement_status: "metadata-only".to_string(),
                hard_enforced: false,
                systemd_property: None,
            }]
        );
    }

    #[test]
    fn serialized_budget_plan_includes_enforceable_systemd_properties() {
        let snapshot = cpu_only_snapshot(16_000, 10_000, 8);
        let plan = budget_plan(&snapshot, 0.8);
        let rendered = serde_json::to_value(&plan).expect("budget plan serializes");

        assert_eq!(rendered["limits"]["systemd"]["CPUQuota"], "640%");
        assert_eq!(rendered["limits"]["systemd"]["MemoryMax"], 8_000);
        assert_eq!(
            rendered["limits"]["systemd"]["unit_properties"],
            serde_json::json!([
                "CPUAccounting=true",
                "MemoryAccounting=true",
                "CPUQuota=640%",
                "MemoryMax=8000"
            ])
        );
        assert_eq!(
            rendered["limits"]["systemd"]["systemd_run_args"],
            serde_json::json!([
                "--property=CPUAccounting=true",
                "--property=MemoryAccounting=true",
                "--property=CPUQuota=640%",
                "--property=MemoryMax=8000"
            ])
        );
    }

    #[test]
    fn parses_rocm_smi_vram_lines() {
        let gpus = parse_rocm_smi_text("GPU[0] : VRAM Total Memory (MiB): 8192\n");
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vendor, GpuVendor::Amd);
        assert_eq!(gpus[0].total_vram_bytes, 8192 * 1024 * 1024);
    }

    #[test]
    fn recognizes_configured_apple_metal_gpu_vendor() {
        assert_eq!(
            GpuVendor::from_config_value("metal"),
            Some(GpuVendor::Apple)
        );
        assert_eq!(
            GpuVendor::from_config_value("apple"),
            Some(GpuVendor::Apple)
        );
    }
}
