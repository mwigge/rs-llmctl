//! AMD/ROCm policy constants and host-qualification data types.
use super::*;

pub const ROCM_POLICY_VERSION: &str = "2026-06-15";
pub const PRODUCTION_ROCM_VERSION: &str = "7.2.4";
pub const PREVIEW_ROCM_VERSION: &str = "7.13.0";
pub const MIN_ROCM_VRAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;

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
