use crate::config::ResourceConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;
use sysinfo::System;

const DEFAULT_BUDGET_FRACTION: f64 = 0.80;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Unknown,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuBudget {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_budget_bytes: u64,
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

    match config.gpu_vendor.as_str() {
        "nvidia" => snapshot.gpus.extend(nvidia_smi_gpus()),
        "amd" => snapshot.gpus.extend(amd_gpus()),
        "auto" | "" => {
            snapshot.gpus.extend(nvidia_smi_gpus());
            if snapshot.gpus.is_empty() {
                snapshot.gpus.extend(amd_gpus());
            }
        }
        _ => {}
    }
    snapshot
}

pub fn budget_plan(snapshot: &ResourceSnapshot, requested_fraction: f64) -> BudgetPlan {
    let budget_fraction = normalized_budget_fraction(requested_fraction);
    BudgetPlan {
        budget_fraction,
        cpu_threads: ((snapshot.cpu_threads as f64) * budget_fraction)
            .floor()
            .max(1.0) as usize,
        memory_budget_bytes: bytes_fraction(snapshot.available_memory_bytes, budget_fraction),
        gpu_budgets: snapshot
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
            .collect(),
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

fn normalized_budget_fraction(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        value
    } else {
        DEFAULT_BUDGET_FRACTION
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
        assert!(plan.gpu_budgets.is_empty());
    }

    #[test]
    fn parses_nvidia_smi_csv() {
        let gpus = parse_nvidia_smi_csv("NVIDIA T1000, 4096, 3072\nRTX 6000, 24576, 20000\n");
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].vendor, GpuVendor::Nvidia);
        assert_eq!(gpus[0].name, "NVIDIA T1000");
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
    }

    #[test]
    fn parses_rocm_smi_vram_lines() {
        let gpus = parse_rocm_smi_text("GPU[0] : VRAM Total Memory (MiB): 8192\n");
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vendor, GpuVendor::Amd);
        assert_eq!(gpus[0].total_vram_bytes, 8192 * 1024 * 1024);
    }
}
