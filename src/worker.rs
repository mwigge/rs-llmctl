use crate::config::{Config, ModelConfig, ResourceConfig, ServerConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
    Nvidia { gpu_layers: u32 },
    AmdVulkan { gpu_layers: u32 },
    Metal { gpu_layers: u32 },
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

    fn env(&self) -> Vec<(String, String)> {
        match self {
            Self::Cpu => Vec::new(),
            Self::Nvidia { .. } => vec![("GGML_CUDA_VISIBLE_DEVICES".into(), "0".into())],
            Self::AmdVulkan { .. } => vec![("GGML_VK_VISIBLE_DEVICES".into(), "0".into())],
            Self::Metal { .. } => vec![("GGML_METAL".into(), "1".into())],
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
pub struct PlannedWorker {
    pub worker: WorkerSpec,
    pub command: CommandSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartupPlan {
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
    pub fn from_config(cfg: &Config) -> Self {
        let workers = cfg
            .models
            .iter()
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
                let command =
                    LlamaServerCommand::new(cfg.server.llama_server.clone(), worker.clone())
                        .build();
                PlannedWorker { worker, command }
            })
            .collect();

        Self { workers }
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
                if old_worker.command == new_worker.command {
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

fn workers_by_alias(workers: &[PlannedWorker]) -> BTreeMap<String, &PlannedWorker> {
    workers
        .iter()
        .map(|worker| (worker.worker.id.as_str().to_string(), worker))
        .collect()
}

impl CommandSpec {
    pub fn into_tokio_command(self) -> Command {
        let mut command = Command::new(self.program);
        command.args(self.args);
        for (key, value) in self.env {
            command.env(key, value);
        }
        command
    }
}

#[derive(Debug, Clone)]
pub struct LlamaServerCommand {
    program: PathBuf,
    spec: WorkerSpec,
}

impl LlamaServerCommand {
    pub fn new(program: impl Into<PathBuf>, spec: WorkerSpec) -> Self {
        Self {
            program: program.into(),
            spec,
        }
    }

    pub fn build(&self) -> CommandSpec {
        CommandSpec {
            program: self.program.clone(),
            args: vec![
                "--host".to_string(),
                self.spec.bind_host.clone(),
                "--port".to_string(),
                self.spec.port.to_string(),
                "--model".to_string(),
                self.spec.model.path.display().to_string(),
                "--ctx-size".to_string(),
                self.spec.context_size.to_string(),
                "--n-gpu-layers".to_string(),
                self.spec.backend.gpu_layers().to_string(),
            ],
            env: self.spec.backend.env(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkerState {
    Stopped,
    Starting,
    Warming,
    Ready,
    Draining,
    Stopping,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleStep {
    pub worker_id: WorkerId,
    pub target: WorkerState,
}

impl LifecycleStep {
    pub fn transition(worker_id: WorkerId, target: WorkerState) -> Self {
        Self { worker_id, target }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwapPlan {
    pub steps: Vec<LifecycleStep>,
}

impl SwapPlan {
    pub fn cold(active: WorkerId, replacement: WorkerId) -> Self {
        Self {
            steps: vec![
                LifecycleStep::transition(active.clone(), WorkerState::Draining),
                LifecycleStep::transition(active.clone(), WorkerState::Stopping),
                LifecycleStep::transition(active, WorkerState::Stopped),
                LifecycleStep::transition(replacement.clone(), WorkerState::Starting),
                LifecycleStep::transition(replacement, WorkerState::Ready),
            ],
        }
    }

    pub fn hot(active: WorkerId, replacement: WorkerId) -> Self {
        Self {
            steps: vec![
                LifecycleStep::transition(replacement.clone(), WorkerState::Starting),
                LifecycleStep::transition(replacement.clone(), WorkerState::Warming),
                LifecycleStep::transition(replacement, WorkerState::Ready),
                LifecycleStep::transition(active.clone(), WorkerState::Draining),
                LifecycleStep::transition(active.clone(), WorkerState::Stopping),
                LifecycleStep::transition(active, WorkerState::Stopped),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use std::path::PathBuf;

    fn model(alias: &str, path: &str) -> ModelConfig {
        ModelConfig {
            alias: alias.to_string(),
            path: PathBuf::from(path),
            role: "chat".to_string(),
            weight: 0,
        }
    }

    #[test]
    fn builds_cpu_only_llama_server_command_without_spawning() {
        let spec = WorkerSpec {
            id: WorkerId::new("tiny"),
            model: model("tiny", "/models/tiny.gguf"),
            bind_host: "127.0.0.1".to_string(),
            port: 18765,
            context_size: 4096,
            backend: WorkerBackend::Cpu,
        };

        let command = LlamaServerCommand::new("llama-server", spec).build();

        assert_eq!(command.program, PathBuf::from("llama-server"));
        assert_eq!(
            command.args,
            vec![
                "--host",
                "127.0.0.1",
                "--port",
                "18765",
                "--model",
                "/models/tiny.gguf",
                "--ctx-size",
                "4096",
                "--n-gpu-layers",
                "0",
            ]
        );
        assert!(command.env.is_empty());
    }

    #[test]
    fn builds_gpu_backend_specific_llama_server_commands() {
        let model = model("chat", "/models/chat.gguf");

        let nvidia = LlamaServerCommand::new(
            "/opt/llama-server",
            WorkerSpec {
                id: WorkerId::new("chat-cuda"),
                model: model.clone(),
                bind_host: "127.0.0.1".to_string(),
                port: 18766,
                context_size: 8192,
                backend: WorkerBackend::Nvidia { gpu_layers: 35 },
            },
        )
        .build();
        assert_eq!(
            nvidia.args,
            vec![
                "--host",
                "127.0.0.1",
                "--port",
                "18766",
                "--model",
                "/models/chat.gguf",
                "--ctx-size",
                "8192",
                "--n-gpu-layers",
                "35",
            ]
        );
        assert_eq!(
            nvidia.env,
            vec![("GGML_CUDA_VISIBLE_DEVICES".into(), "0".into())]
        );

        let vulkan = LlamaServerCommand::new(
            "llama-server",
            WorkerSpec {
                id: WorkerId::new("chat-vulkan"),
                model: model.clone(),
                bind_host: "127.0.0.1".to_string(),
                port: 18767,
                context_size: 8192,
                backend: WorkerBackend::AmdVulkan { gpu_layers: 99 },
            },
        )
        .build();
        assert_eq!(vulkan.args.last().map(String::as_str), Some("99"));
        assert_eq!(
            vulkan.env,
            vec![("GGML_VK_VISIBLE_DEVICES".into(), "0".into())]
        );

        let metal = LlamaServerCommand::new(
            "llama-server",
            WorkerSpec {
                id: WorkerId::new("chat-metal"),
                model,
                bind_host: "127.0.0.1".to_string(),
                port: 18768,
                context_size: 8192,
                backend: WorkerBackend::Metal { gpu_layers: 48 },
            },
        )
        .build();
        assert_eq!(metal.args.last().map(String::as_str), Some("48"));
        assert_eq!(metal.env, vec![("GGML_METAL".into(), "1".into())]);
    }

    #[test]
    fn plans_cold_swap_by_stopping_active_before_starting_replacement() {
        let plan = SwapPlan::cold(WorkerId::new("old"), WorkerId::new("new"));

        assert_eq!(
            plan.steps,
            vec![
                LifecycleStep::transition(WorkerId::new("old"), WorkerState::Draining),
                LifecycleStep::transition(WorkerId::new("old"), WorkerState::Stopping),
                LifecycleStep::transition(WorkerId::new("old"), WorkerState::Stopped),
                LifecycleStep::transition(WorkerId::new("new"), WorkerState::Starting),
                LifecycleStep::transition(WorkerId::new("new"), WorkerState::Ready),
            ]
        );
    }

    #[test]
    fn plans_hot_swap_by_warming_replacement_before_draining_active() {
        let plan = SwapPlan::hot(WorkerId::new("old"), WorkerId::new("new"));

        assert_eq!(
            plan.steps,
            vec![
                LifecycleStep::transition(WorkerId::new("new"), WorkerState::Starting),
                LifecycleStep::transition(WorkerId::new("new"), WorkerState::Warming),
                LifecycleStep::transition(WorkerId::new("new"), WorkerState::Ready),
                LifecycleStep::transition(WorkerId::new("old"), WorkerState::Draining),
                LifecycleStep::transition(WorkerId::new("old"), WorkerState::Stopping),
                LifecycleStep::transition(WorkerId::new("old"), WorkerState::Stopped),
            ]
        );
    }

    #[test]
    fn startup_plan_builds_cpu_only_commands_for_configured_models() {
        let cfg = crate::config::Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                worker_base_port: 19000,
                llama_server: "/usr/local/bin/llama-server".to_string(),
                context_size: 4096,
                ..ServerConfig::default()
            },
            resources: ResourceConfig {
                cpu_only: true,
                gpu_vendor: "nvidia".to_string(),
                ..ResourceConfig::default()
            },
            models: vec![
                model("chat", "/models/chat.gguf"),
                model("coder", "/models/coder.gguf"),
            ],
            ..Default::default()
        };

        let plan = StartupPlan::from_config(&cfg);

        assert_eq!(plan.workers.len(), 2);
        assert_eq!(plan.workers[0].worker.id, WorkerId::new("chat"));
        assert_eq!(plan.workers[0].worker.port, 19000);
        assert_eq!(plan.workers[0].worker.backend, WorkerBackend::Cpu);
        assert_eq!(
            plan.workers[0].command.program,
            PathBuf::from("/usr/local/bin/llama-server")
        );
        assert_eq!(
            plan.workers[0].command.args,
            vec![
                "--host",
                "127.0.0.1",
                "--port",
                "19000",
                "--model",
                "/models/chat.gguf",
                "--ctx-size",
                "4096",
                "--n-gpu-layers",
                "0",
            ]
        );
        assert!(plan.workers[0].command.env.is_empty());
        assert_eq!(plan.workers[1].worker.id, WorkerId::new("coder"));
        assert_eq!(plan.workers[1].worker.port, 19001);
    }

    #[test]
    fn startup_plan_selects_gpu_backend_commands_from_resources() {
        let cfg = crate::config::Config {
            server: ServerConfig {
                worker_base_port: 19100,
                llama_server: "llama-server".to_string(),
                ..ServerConfig::default()
            },
            resources: ResourceConfig {
                cpu_only: false,
                gpu_vendor: "nvidia".to_string(),
                ..ResourceConfig::default()
            },
            models: vec![model("chat", "/models/chat.gguf")],
            ..Default::default()
        };

        let plan = StartupPlan::from_config(&cfg);

        assert_eq!(plan.workers.len(), 1);
        assert_eq!(
            plan.workers[0].worker.backend,
            WorkerBackend::Nvidia { gpu_layers: 99 }
        );
        assert_eq!(
            plan.workers[0].command.env,
            vec![("GGML_CUDA_VISIBLE_DEVICES".into(), "0".into())]
        );
        assert_eq!(
            plan.workers[0].command.args.last().map(String::as_str),
            Some("99")
        );
    }

    #[test]
    fn startup_plan_diff_reports_added_removed_and_changed_worker_aliases() {
        let old = StartupPlan {
            workers: vec![
                PlannedWorker {
                    worker: WorkerSpec {
                        id: WorkerId::new("chat"),
                        model: model("chat", "/models/chat-v1.gguf"),
                        bind_host: "127.0.0.1".to_string(),
                        port: 19000,
                        context_size: 4096,
                        backend: WorkerBackend::Cpu,
                    },
                    command: CommandSpec {
                        program: PathBuf::from("llama-server"),
                        args: vec!["--model".to_string(), "/models/chat-v1.gguf".to_string()],
                        env: Vec::new(),
                    },
                },
                PlannedWorker {
                    worker: WorkerSpec {
                        id: WorkerId::new("embed"),
                        model: model("embed", "/models/embed.gguf"),
                        bind_host: "127.0.0.1".to_string(),
                        port: 19001,
                        context_size: 4096,
                        backend: WorkerBackend::Cpu,
                    },
                    command: CommandSpec {
                        program: PathBuf::from("llama-server"),
                        args: vec!["--model".to_string(), "/models/embed.gguf".to_string()],
                        env: Vec::new(),
                    },
                },
            ],
        };
        let new = StartupPlan {
            workers: vec![
                PlannedWorker {
                    worker: WorkerSpec {
                        id: WorkerId::new("chat"),
                        model: model("chat", "/models/chat-v2.gguf"),
                        bind_host: "127.0.0.1".to_string(),
                        port: 19000,
                        context_size: 4096,
                        backend: WorkerBackend::Cpu,
                    },
                    command: CommandSpec {
                        program: PathBuf::from("llama-server"),
                        args: vec!["--model".to_string(), "/models/chat-v2.gguf".to_string()],
                        env: Vec::new(),
                    },
                },
                PlannedWorker {
                    worker: WorkerSpec {
                        id: WorkerId::new("coder"),
                        model: model("coder", "/models/coder.gguf"),
                        bind_host: "127.0.0.1".to_string(),
                        port: 19001,
                        context_size: 4096,
                        backend: WorkerBackend::Cpu,
                    },
                    command: CommandSpec {
                        program: PathBuf::from("llama-server"),
                        args: vec!["--model".to_string(), "/models/coder.gguf".to_string()],
                        env: Vec::new(),
                    },
                },
            ],
        };

        let diff = old.diff(&new);

        assert_eq!(diff.added, vec![WorkerId::new("coder")]);
        assert_eq!(diff.removed, vec![WorkerId::new("embed")]);
        assert_eq!(diff.changed, vec![WorkerId::new("chat")]);
        assert_eq!(diff.unchanged, vec![]);
    }
}
