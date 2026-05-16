use crate::config::{ModelConfig, ResourceConfig, ServerConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::process::Command;

const DEFAULT_GPU_LAYERS: u32 = 99;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
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
}
