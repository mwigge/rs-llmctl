use crate::config::{Config, ModelConfig, ResourceConfig, ServerConfig};
use crate::resources::{budget_plan, snapshot, ResourceLimitPlan};
use crate::runtime::RuntimeBackend;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use tokio::process::Command;

const DEFAULT_GPU_LAYERS: u32 = 99;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorkerId(String);

impl WorkerId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum WorkerBackend {
    Cpu,
    Nvidia {
        gpu_layers: u32,
    },
    /// Resource-planning hook only — candle-native has no AMD execution
    /// backend (candle 0.10.2 lacks ROCm/HIP/Vulkan). Selecting this still
    /// fails closed to CPU. See `docs/adr/0001-amd-gpu-acceleration.md`.
    AmdVulkan {
        gpu_layers: u32,
    },
    Metal {
        gpu_layers: u32,
    },
}

impl WorkerBackend {
    pub fn from_resources(resources: &ResourceConfig) -> Self {
        if resources.cpu_only {
            return Self::Cpu;
        }

        match resources.gpu_vendor.as_str() {
            "nvidia" => Self::Nvidia {
                gpu_layers: DEFAULT_GPU_LAYERS,
            },
            "amd" | "vulkan" | "amd-vulkan" => Self::AmdVulkan {
                gpu_layers: DEFAULT_GPU_LAYERS,
            },
            "metal" | "apple" => Self::Metal {
                gpu_layers: DEFAULT_GPU_LAYERS,
            },
            _ => Self::Cpu,
        }
    }

    fn gpu_layers(&self) -> u32 {
        match self {
            Self::Cpu => 0,
            Self::Nvidia { gpu_layers }
            | Self::AmdVulkan { gpu_layers }
            | Self::Metal { gpu_layers } => *gpu_layers,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerDeviceSelection {
    pub device_index: Option<u32>,
    pub selection: String,
}

impl WorkerDeviceSelection {
    pub fn cpu() -> Self {
        Self {
            device_index: None,
            selection: "cpu-only".to_string(),
        }
    }

    pub fn first_visible_gpu() -> Self {
        Self {
            device_index: Some(0),
            selection: "first-visible-gpu".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerExecutionPlan {
    pub backend: WorkerBackend,
    pub device: WorkerDeviceSelection,
    pub gpu_layers: u32,
}

impl WorkerExecutionPlan {
    pub fn cpu() -> Self {
        Self::from_backend(&WorkerBackend::Cpu)
    }

    pub fn from_backend(backend: &WorkerBackend) -> Self {
        Self {
            backend: backend.clone(),
            device: if matches!(backend, WorkerBackend::Cpu) {
                WorkerDeviceSelection::cpu()
            } else {
                WorkerDeviceSelection::first_visible_gpu()
            },
            gpu_layers: backend.gpu_layers(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub id: WorkerId,
    pub model: ModelConfig,
    pub bind_host: String,
    pub port: u16,
    pub context_size: u32,
    pub backend: WorkerBackend,
}

impl WorkerSpec {
    pub fn from_config(
        id: impl Into<String>,
        model: ModelConfig,
        server: &ServerConfig,
        resources: &ResourceConfig,
        port_offset: u16,
    ) -> Self {
        Self {
            id: WorkerId::new(id),
            model,
            bind_host: server.host.clone(),
            port: server.worker_base_port.saturating_add(port_offset),
            context_size: server.context_size,
            backend: WorkerBackend::from_resources(resources),
        }
    }

    pub fn upstream(&self) -> String {
        format!("http://{}:{}", self.bind_host, self.port)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum WorkerLaunchPlan {
    InProcess {
        backend: RuntimeBackend,
        engine: String,
        implemented: bool,
    },
    /// llama-server subprocess — used for AMD GPU (HIP/Vulkan backend).
    /// See `docs/adr/0001-amd-gpu-acceleration.md` option (b).
    LlamaServerSubprocess { llama_server_path: PathBuf },
    /// In-process llama.cpp FFI engine via the `llama-cpp-2` crate.
    ///
    /// When the `llama-cpp-native` feature is compiled in, AMD GPU workers
    /// prefer this plan over `LlamaServerSubprocess` because it allows
    /// per-token `OTel` hooks inside the sampling loop.
    #[cfg(feature = "llama-cpp-native")]
    LlamaCppNative { gpu_layers: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedWorker {
    pub worker: WorkerSpec,
    #[serde(default = "WorkerLaunchPlan::candle_native")]
    pub launch: WorkerLaunchPlan,
    #[serde(default = "WorkerExecutionPlan::cpu")]
    pub execution: WorkerExecutionPlan,
    pub command: CommandSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartupPlan {
    #[serde(default)]
    pub resource_limits: ResourceLimitPlan,
    pub workers: Vec<PlannedWorker>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDiffCounts {
    pub added: usize,
    pub removed: usize,
    pub changed: usize,
    pub unchanged: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDiff {
    pub added: Vec<WorkerId>,
    pub removed: Vec<WorkerId>,
    pub changed: Vec<WorkerId>,
    pub unchanged: Vec<WorkerId>,
    pub counts: PlanDiffCounts,
}

impl StartupPlan {
    #[must_use]
    pub fn from_config(cfg: &Config) -> Self {
        let is_amd = matches!(
            WorkerBackend::from_resources(&cfg.resources),
            WorkerBackend::AmdVulkan { .. }
        );

        // When the llama-cpp-native feature is compiled in and AMD GPU is
        // detected, prefer the in-process FFI path over the subprocess path.
        // This enables per-token OTel hooks inside the sampling loop.
        #[cfg(feature = "llama-cpp-native")]
        if is_amd {
            let workers = cfg
                .models
                .iter()
                .filter(|m| m.weight > 0)
                .cloned()
                .enumerate()
                .map(|(index, model)| {
                    let port_offset = u16::try_from(index).unwrap_or(u16::MAX);
                    let worker = WorkerSpec::from_config(
                        model.alias.clone(),
                        model,
                        &cfg.server,
                        &cfg.resources,
                        port_offset,
                    );
                    let gpu_layers = worker.backend.gpu_layers();
                    PlannedWorker::llama_cpp_native(worker, gpu_layers)
                })
                .collect();
            let resource_limits =
                budget_plan(&snapshot(&cfg.resources), cfg.resources.budget).limits;
            return Self {
                resource_limits,
                workers,
            };
        }

        // Fallback: subprocess path for AMD without the FFI feature, or CPU.
        let amd_llama_server = if is_amd {
            find_llama_server(cfg.resources.llama_server_bin.as_ref())
        } else {
            None
        };

        let workers = cfg
            .models
            .iter()
            .filter(|model| model.weight > 0)
            .cloned()
            .enumerate()
            .map(|(index, model)| {
                let port_offset = u16::try_from(index).unwrap_or(u16::MAX);
                let worker = WorkerSpec::from_config(
                    model.alias.clone(),
                    model,
                    &cfg.server,
                    &cfg.resources,
                    port_offset,
                );
                if let Some(ref llama_path) = amd_llama_server {
                    PlannedWorker::llama_server_subprocess(worker, llama_path.clone())
                } else {
                    PlannedWorker::in_process_candle_native(worker)
                }
            })
            .collect();

        let resource_limits = budget_plan(&snapshot(&cfg.resources), cfg.resources.budget).limits;

        Self {
            resource_limits,
            workers,
        }
    }

    pub fn diff(&self, next: &Self) -> PlanDiff {
        let old_workers = workers_by_alias(&self.workers);
        let new_workers = workers_by_alias(&next.workers);

        let added = new_workers
            .keys()
            .filter(|alias| !old_workers.contains_key(*alias))
            .map(|alias| WorkerId::new(alias.clone()))
            .collect::<Vec<_>>();
        let removed = old_workers
            .keys()
            .filter(|alias| !new_workers.contains_key(*alias))
            .map(|alias| WorkerId::new(alias.clone()))
            .collect::<Vec<_>>();

        let mut changed = Vec::new();
        let mut unchanged = Vec::new();
        for (alias, old_worker) in &old_workers {
            if let Some(new_worker) = new_workers.get(alias) {
                if old_worker == new_worker {
                    unchanged.push(WorkerId::new(alias.clone()));
                } else {
                    changed.push(WorkerId::new(alias.clone()));
                }
            }
        }

        PlanDiff {
            counts: PlanDiffCounts {
                added: added.len(),
                removed: removed.len(),
                changed: changed.len(),
                unchanged: unchanged.len(),
            },
            added,
            removed,
            changed,
            unchanged,
        }
    }
}

impl WorkerLaunchPlan {
    pub fn candle_native() -> Self {
        Self::InProcess {
            backend: RuntimeBackend::CandleNative,
            engine: "candle-native".to_string(),
            implemented: true,
        }
    }

    pub fn is_in_process(&self) -> bool {
        match self {
            Self::InProcess { .. } => true,
            Self::LlamaServerSubprocess { .. } => false,
            #[cfg(feature = "llama-cpp-native")]
            Self::LlamaCppNative { .. } => true,
        }
    }
}

impl PlannedWorker {
    pub fn in_process_candle_native(worker: WorkerSpec) -> Self {
        let command = CommandSpec::in_process_placeholder(&worker);
        let execution = WorkerExecutionPlan::from_backend(&worker.backend);
        Self {
            worker,
            launch: WorkerLaunchPlan::candle_native(),
            execution,
            command,
        }
    }

    /// Constructs a `PlannedWorker` that will run inference via the
    /// `llama-cpp-2` in-process FFI engine.
    ///
    /// Selecting this plan requires the `llama-cpp-native` Cargo feature.
    #[cfg(feature = "llama-cpp-native")]
    pub fn llama_cpp_native(worker: WorkerSpec, gpu_layers: u32) -> Self {
        let command = CommandSpec::in_process_placeholder(&worker);
        let execution = WorkerExecutionPlan::from_backend(&worker.backend);
        Self {
            worker,
            launch: WorkerLaunchPlan::LlamaCppNative { gpu_layers },
            execution,
            command,
        }
    }

    pub fn llama_server_subprocess(worker: WorkerSpec, llama_server_path: PathBuf) -> Self {
        let command = CommandSpec::llama_server(&worker, &llama_server_path);
        let execution = WorkerExecutionPlan::from_backend(&worker.backend);
        let launch = WorkerLaunchPlan::LlamaServerSubprocess {
            llama_server_path: llama_server_path.clone(),
        };
        Self {
            worker,
            launch,
            execution,
            command,
        }
    }
}

/// Returns the user home directory using the `HOME` environment variable,
/// matching the behaviour of the milliways install scripts.
fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

/// Resolves the path to a HIP-enabled `llama-server` binary. Checks, in order:
/// `~/.local/bin`, `/usr/local/bin`, `/usr/bin`, and the system `PATH`.
pub fn find_llama_server(cfg_path: Option<&PathBuf>) -> Option<PathBuf> {
    if let Some(p) = cfg_path {
        if p.is_file() {
            return Some(p.clone());
        }
    }
    let candidates: Vec<PathBuf> = [
        dirs_home()
            .map(|h| h.join(".local/bin/llama-server"))
            .as_ref()
            .and_then(|p| p.is_file().then(|| p.clone())),
        Some(PathBuf::from("/usr/local/bin/llama-server")),
        Some(PathBuf::from("/usr/bin/llama-server")),
    ]
    .into_iter()
    .flatten()
    .collect();

    for p in candidates {
        if p.is_file() {
            return Some(p);
        }
    }
    which_llama_server()
}

fn which_llama_server() -> Option<PathBuf> {
    let path_env = std::env::var("PATH").unwrap_or_default();
    for dir in path_env.split(':') {
        let p = PathBuf::from(dir).join("llama-server");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn workers_by_alias(workers: &[PlannedWorker]) -> BTreeMap<String, &PlannedWorker> {
    workers
        .iter()
        .map(|worker| (worker.worker.id.as_str().to_string(), worker))
        .collect()
}

impl CommandSpec {
    fn in_process_placeholder(spec: &WorkerSpec) -> Self {
        Self {
            program: PathBuf::from("<in-process:candle-native>"),
            args: vec![
                "--model".to_string(),
                spec.model.path.display().to_string(),
                "--ctx-size".to_string(),
                spec.context_size.to_string(),
            ],
            env: Vec::new(),
        }
    }

    fn llama_server(spec: &WorkerSpec, llama_server_path: &Path) -> Self {
        let gpu_layers = spec.backend.gpu_layers();
        // LD_LIBRARY_PATH must include the rs-llmctl lib dir (libggml-hip.so)
        // and the ROCm runtime libs so the HIP plugin can load at runtime.
        let lib_dir = dirs_home().map_or_else(
            || "/usr/local/lib/llmctl".to_string(),
            |h| format!("{}/.local/lib/llmctl", h.display()),
        );
        let ld_path = format!("{lib_dir}:/opt/rocm/lib:/opt/rocm/lib64");
        let path_env = format!(
            "/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/bin:/usr/bin:/bin:{}",
            std::env::var("PATH").unwrap_or_default()
        );
        Self {
            program: llama_server_path.to_path_buf(),
            args: vec![
                "-m".to_string(),
                spec.model.path.display().to_string(),
                "--alias".to_string(),
                spec.model.alias.clone(),
                "--host".to_string(),
                spec.bind_host.clone(),
                "--port".to_string(),
                spec.port.to_string(),
                "--ctx-size".to_string(),
                spec.context_size.to_string(),
                "-ngl".to_string(),
                gpu_layers.to_string(),
                "--cache-type-k".to_string(),
                "q8_0".to_string(),
                "--cache-type-v".to_string(),
                "q8_0".to_string(),
                "--jinja".to_string(),
                "--metrics".to_string(),
                "--ubatch-size".to_string(),
                "1024".to_string(),
            ],
            env: vec![
                ("HIP_PLATFORM".to_string(), "amd".to_string()),
                ("LD_LIBRARY_PATH".to_string(), ld_path),
                ("PATH".to_string(), path_env),
            ],
        }
    }

    pub fn into_tokio_command(self) -> Command {
        let mut command = Command::new(self.program);
        command.args(self.args);
        for (key, value) in self.env {
            command.env(key, value);
        }
        command.kill_on_drop(true);
        command
    }
}
