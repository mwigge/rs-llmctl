use crate::config::{Config, ModelConfig, ResourceConfig, ServerConfig};
use crate::observability::{
    emit_runtime_telemetry, RuntimeTelemetryEvent, TelemetryEventName, TelemetrySignal,
};
use crate::resources::{budget_plan, snapshot, ResourceLimitPlan};
use crate::runtime::RuntimeBackend;
use chrono::Utc;
use futures_util::future::{BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::{Child, Command};

const DEFAULT_GPU_LAYERS: u32 = 99;
const READY_PROBE_ATTEMPTS: usize = 120;
const READY_PROBE_INTERVAL: Duration = Duration::from_millis(500);

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
    pub fn from_config(cfg: &Config) -> Self {
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
                PlannedWorker::in_process_candle_native(worker)
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
        matches!(self, Self::InProcess { .. })
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
pub struct WorkerStatus {
    pub worker_id: WorkerId,
    pub pid: Option<u32>,
    pub state: WorkerState,
    pub restart_count: u32,
    pub last_error: Option<String>,
}

impl WorkerStatus {
    fn new(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            pid: None,
            state: WorkerState::Stopped,
            restart_count: 0,
            last_error: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnedWorker {
    pub pid: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRunnerError {
    message: String,
}

impl WorkerRunnerError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for WorkerRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for WorkerRunnerError {}

impl From<std::io::Error> for WorkerRunnerError {
    fn from(value: std::io::Error) -> Self {
        Self::new(value.to_string())
    }
}

pub trait WorkerRunner {
    fn spawn<'a>(
        &'a mut self,
        planned: &'a PlannedWorker,
    ) -> BoxFuture<'a, Result<SpawnedWorker, WorkerRunnerError>>;

    fn stop<'a>(
        &'a mut self,
        worker_id: &'a WorkerId,
    ) -> BoxFuture<'a, Result<(), WorkerRunnerError>>;

    fn wait_ready<'a>(
        &'a mut self,
        _planned: &'a PlannedWorker,
    ) -> BoxFuture<'a, Result<(), WorkerRunnerError>> {
        async { Ok(()) }.boxed()
    }
}

#[derive(Debug, Default)]
pub struct TokioWorkerRunner {
    children: BTreeMap<WorkerId, Child>,
}

impl TokioWorkerRunner {
    pub fn new() -> Self {
        Self::default()
    }
}

impl WorkerRunner for TokioWorkerRunner {
    fn spawn<'a>(
        &'a mut self,
        planned: &'a PlannedWorker,
    ) -> BoxFuture<'a, Result<SpawnedWorker, WorkerRunnerError>> {
        async move {
            if planned.launch.is_in_process() {
                return Err(WorkerRunnerError::new(
                    "candle-native runtime is planned as an in-process worker, but the engine implementation is not available yet",
                ));
            }

            let child = planned.command.clone().into_tokio_command().spawn()?;
            let pid = child.id().unwrap_or_default();
            self.children.insert(planned.worker.id.clone(), child);
            Ok(SpawnedWorker { pid })
        }
        .boxed()
    }

    fn stop<'a>(
        &'a mut self,
        worker_id: &'a WorkerId,
    ) -> BoxFuture<'a, Result<(), WorkerRunnerError>> {
        async move {
            if let Some(mut child) = self.children.remove(worker_id) {
                child.kill().await?;
            }

            Ok(())
        }
        .boxed()
    }

    fn wait_ready<'a>(
        &'a mut self,
        planned: &'a PlannedWorker,
    ) -> BoxFuture<'a, Result<(), WorkerRunnerError>> {
        async move {
            let client = reqwest::Client::new();
            let urls = [
                format!("{}/health", planned.worker.upstream()),
                format!("{}/healthz", planned.worker.upstream()),
                format!("{}/v1/models", planned.worker.upstream()),
            ];

            for _ in 0..READY_PROBE_ATTEMPTS {
                for url in &urls {
                    match client.get(url).send().await {
                        Ok(response) if response.status().is_success() => return Ok(()),
                        Ok(_) | Err(_) => {}
                    }
                }
                tokio::time::sleep(READY_PROBE_INTERVAL).await;
            }

            Err(WorkerRunnerError::new(format!(
                "worker {} did not become ready at {}",
                planned.worker.id.as_str(),
                planned.worker.upstream()
            )))
        }
        .boxed()
    }
}

#[derive(Debug)]
pub struct WorkerSupervisor<R> {
    runner: R,
    statuses: BTreeMap<WorkerId, WorkerStatus>,
}

impl<R> WorkerSupervisor<R> {
    pub fn new(runner: R) -> Self {
        Self {
            runner,
            statuses: BTreeMap::new(),
        }
    }

    pub fn runner(&self) -> &R {
        &self.runner
    }

    pub fn statuses(&self) -> Vec<WorkerStatus> {
        self.statuses.values().cloned().collect()
    }
}

impl<R: WorkerRunner> WorkerSupervisor<R> {
    pub async fn start_all(&mut self, plan: &StartupPlan) -> Vec<WorkerStatus> {
        for planned in &plan.workers {
            self.start(planned).await;
        }

        self.statuses()
    }

    pub async fn start(&mut self, planned: &PlannedWorker) -> WorkerStatus {
        let worker_id = planned.worker.id.clone();
        let restart_count = self
            .statuses
            .get(&worker_id)
            .map(|status| status.restart_count)
            .unwrap_or_default();

        self.statuses.insert(
            worker_id.clone(),
            WorkerStatus {
                worker_id: worker_id.clone(),
                pid: None,
                state: WorkerState::Starting,
                restart_count,
                last_error: None,
            },
        );
        if let Some(status) = self.statuses.get(&worker_id) {
            emit_worker_lifecycle_transition(status, WorkerState::Stopped);
        }

        match self.runner.spawn(planned).await {
            Ok(spawned) => {
                self.update_status(&worker_id, |status| {
                    status.pid = Some(spawned.pid);
                    status.state = WorkerState::Warming;
                    status.last_error = None;
                });

                match self.runner.wait_ready(planned).await {
                    Ok(()) => self.update_status(&worker_id, |status| {
                        status.state = WorkerState::Ready;
                        status.last_error = None;
                    }),
                    Err(error) => {
                        let _ = self.runner.stop(&worker_id).await;
                        self.update_status(&worker_id, |status| {
                            status.pid = None;
                            status.state = WorkerState::Failed;
                            status.last_error = Some(error.to_string());
                        })
                    }
                }
            }
            Err(error) => self.update_status(&worker_id, |status| {
                status.pid = None;
                status.state = WorkerState::Failed;
                status.last_error = Some(error.to_string());
            }),
        }
    }

    pub async fn drain(&mut self, worker_id: &WorkerId) -> WorkerStatus {
        self.update_status(worker_id, |status| {
            status.state = WorkerState::Draining;
            status.last_error = None;
        })
    }

    pub async fn stop(&mut self, worker_id: &WorkerId) -> WorkerStatus {
        self.update_status(worker_id, |status| {
            status.state = WorkerState::Stopping;
            status.last_error = None;
        });

        match self.runner.stop(worker_id).await {
            Ok(()) => self.update_status(worker_id, |status| {
                status.pid = None;
                status.state = WorkerState::Stopped;
                status.last_error = None;
            }),
            Err(error) => self.update_status(worker_id, |status| {
                status.state = WorkerState::Failed;
                status.last_error = Some(error.to_string());
            }),
        }
    }

    pub async fn restart(&mut self, planned: &PlannedWorker) -> WorkerStatus {
        let worker_id = planned.worker.id.clone();
        let stopped = self.stop(&worker_id).await;
        if stopped.state == WorkerState::Failed {
            return stopped;
        }

        self.update_status(&worker_id, |status| {
            status.restart_count = status.restart_count.saturating_add(1);
        });

        self.start(planned).await
    }

    pub async fn stop_all(&mut self) -> Vec<WorkerStatus> {
        let worker_ids = self.statuses.keys().cloned().collect::<Vec<WorkerId>>();
        let mut statuses = Vec::with_capacity(worker_ids.len());
        for worker_id in worker_ids {
            statuses.push(self.stop(&worker_id).await);
        }
        statuses
    }

    fn update_status(
        &mut self,
        worker_id: &WorkerId,
        update: impl FnOnce(&mut WorkerStatus),
    ) -> WorkerStatus {
        let status = self
            .statuses
            .entry(worker_id.clone())
            .or_insert_with(|| WorkerStatus::new(worker_id.clone()));
        let previous = status.state;
        update(status);
        let updated = status.clone();
        if previous != updated.state {
            emit_worker_lifecycle_transition(&updated, previous);
        }
        updated
    }
}

fn emit_worker_lifecycle_transition(status: &WorkerStatus, previous: WorkerState) {
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::WorkerLifecycle,
        Utc::now(),
        BTreeMap::from([
            (
                "llmctl.worker.id".to_string(),
                json!(status.worker_id.as_str()),
            ),
            ("llmctl.worker.previous_state".to_string(), json!(previous)),
            ("llmctl.worker.state".to_string(), json!(status.state)),
            (
                "llmctl.worker.restart_count".to_string(),
                json!(status.restart_count),
            ),
            ("llmctl.worker.pid".to_string(), json!(status.pid)),
            (
                "llmctl.worker.failed".to_string(),
                json!(status.state == WorkerState::Failed),
            ),
        ]),
    ));
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
    pub active: WorkerId,
    pub replacement: WorkerId,
    pub steps: Vec<LifecycleStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SwapMode {
    Cold,
    Hot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwapExecution {
    pub mode: SwapMode,
    pub plan: SwapPlan,
    pub statuses: Vec<WorkerStatus>,
    pub success: bool,
}

impl SwapPlan {
    pub fn cold(active: WorkerId, replacement: WorkerId) -> Self {
        Self {
            active: active.clone(),
            replacement: replacement.clone(),
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
            active: active.clone(),
            replacement: replacement.clone(),
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

impl<R: WorkerRunner> WorkerSupervisor<R> {
    pub async fn execute_swap(
        &mut self,
        mode: SwapMode,
        active: &WorkerId,
        replacement: &PlannedWorker,
    ) -> SwapExecution {
        match mode {
            SwapMode::Cold => self.execute_cold_swap(active, replacement).await,
            SwapMode::Hot => self.execute_hot_swap(active, replacement).await,
        }
    }

    async fn execute_cold_swap(
        &mut self,
        active: &WorkerId,
        replacement: &PlannedWorker,
    ) -> SwapExecution {
        let plan = SwapPlan::cold(active.clone(), replacement.worker.id.clone());
        let mut statuses = Vec::new();
        statuses.push(self.drain(active).await);
        let stopped = self.stop(active).await;
        let stop_ok = stopped.state == WorkerState::Stopped;
        statuses.push(stopped);
        if !stop_ok {
            return SwapExecution {
                mode: SwapMode::Cold,
                plan,
                statuses,
                success: false,
            };
        }

        let started = self.start(replacement).await;
        let success = started.state == WorkerState::Ready;
        statuses.push(started);
        SwapExecution {
            mode: SwapMode::Cold,
            plan,
            statuses,
            success,
        }
    }

    async fn execute_hot_swap(
        &mut self,
        active: &WorkerId,
        replacement: &PlannedWorker,
    ) -> SwapExecution {
        let plan = SwapPlan::hot(active.clone(), replacement.worker.id.clone());
        let mut statuses = Vec::new();
        let started = self.start(replacement).await;
        let start_ok = started.state == WorkerState::Ready;
        statuses.push(started);
        if !start_ok {
            return SwapExecution {
                mode: SwapMode::Hot,
                plan,
                statuses,
                success: false,
            };
        }

        statuses.push(self.drain(active).await);
        let stopped = self.stop(active).await;
        let success = stopped.state == WorkerState::Stopped;
        statuses.push(stopped);
        SwapExecution {
            mode: SwapMode::Hot,
            plan,
            statuses,
            success,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use futures_util::future::{ready, BoxFuture, FutureExt};
    use std::path::PathBuf;

    fn model(alias: &str, path: &str) -> ModelConfig {
        ModelConfig {
            alias: alias.to_string(),
            path: PathBuf::from(path),
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        }
    }

    fn planned_worker(alias: &str) -> PlannedWorker {
        let worker = WorkerSpec {
            id: WorkerId::new(alias),
            model: model(alias, &format!("/models/{alias}.gguf")),
            bind_host: "127.0.0.1".to_string(),
            port: 19000,
            context_size: 4096,
            backend: WorkerBackend::Cpu,
        };
        PlannedWorker::in_process_candle_native(worker)
    }

    fn startup_plan(workers: Vec<PlannedWorker>) -> StartupPlan {
        StartupPlan {
            resource_limits: ResourceLimitPlan::default(),
            workers,
        }
    }

    #[derive(Debug, Default)]
    struct FakeRunner {
        next_pid: u32,
        fail_spawn_for: Vec<WorkerId>,
        fail_stop_for: Vec<WorkerId>,
        fail_ready_for: Vec<WorkerId>,
        spawned: Vec<WorkerId>,
        stopped: Vec<WorkerId>,
    }

    impl WorkerRunner for FakeRunner {
        fn spawn<'a>(
            &'a mut self,
            planned: &'a PlannedWorker,
        ) -> BoxFuture<'a, Result<SpawnedWorker, WorkerRunnerError>> {
            self.spawned.push(planned.worker.id.clone());
            if self.fail_spawn_for.contains(&planned.worker.id) {
                return ready(Err(WorkerRunnerError::new("spawn failed"))).boxed();
            }

            let pid = self.next_pid;
            self.next_pid += 1;
            ready(Ok(SpawnedWorker { pid })).boxed()
        }

        fn stop<'a>(
            &'a mut self,
            worker_id: &'a WorkerId,
        ) -> BoxFuture<'a, Result<(), WorkerRunnerError>> {
            self.stopped.push(worker_id.clone());
            if self.fail_stop_for.contains(worker_id) {
                return ready(Err(WorkerRunnerError::new("stop failed"))).boxed();
            }

            ready(Ok(())).boxed()
        }

        fn wait_ready<'a>(
            &'a mut self,
            planned: &'a PlannedWorker,
        ) -> BoxFuture<'a, Result<(), WorkerRunnerError>> {
            if self.fail_ready_for.contains(&planned.worker.id) {
                return ready(Err(WorkerRunnerError::new("ready probe failed"))).boxed();
            }

            ready(Ok(())).boxed()
        }
    }

    #[tokio::test]
    async fn supervisor_starts_all_workers_and_reports_runtime_status() {
        let plan = startup_plan(vec![planned_worker("chat"), planned_worker("coder")]);
        let runner = FakeRunner {
            next_pid: 5000,
            ..FakeRunner::default()
        };
        let mut supervisor = WorkerSupervisor::new(runner);

        let statuses = supervisor.start_all(&plan).await;

        assert_eq!(statuses.len(), 2);
        assert_eq!(statuses[0].worker_id, WorkerId::new("chat"));
        assert_eq!(statuses[0].pid, Some(5000));
        assert_eq!(statuses[0].state, WorkerState::Ready);
        assert_eq!(statuses[0].restart_count, 0);
        assert_eq!(statuses[0].last_error, None);
        assert_eq!(statuses[1].worker_id, WorkerId::new("coder"));
        assert_eq!(statuses[1].pid, Some(5001));
        assert_eq!(statuses[1].state, WorkerState::Ready);
    }

    #[tokio::test]
    async fn supervisor_records_spawn_failures_without_leaking_command_env() {
        let plan = startup_plan(vec![planned_worker("chat")]);
        let runner = FakeRunner {
            fail_spawn_for: vec![WorkerId::new("chat")],
            ..FakeRunner::default()
        };
        let mut supervisor = WorkerSupervisor::new(runner);

        let statuses = supervisor.start_all(&plan).await;
        let status_json = serde_json::to_string(&statuses[0]).expect("serialize status");

        assert_eq!(statuses[0].worker_id, WorkerId::new("chat"));
        assert_eq!(statuses[0].pid, None);
        assert_eq!(statuses[0].state, WorkerState::Failed);
        assert_eq!(statuses[0].last_error.as_deref(), Some("spawn failed"));
        assert!(!status_json.contains("API_TOKEN"));
        assert!(!status_json.contains("super-secret"));
    }

    #[tokio::test]
    async fn supervisor_stops_worker_when_readiness_probe_fails() {
        let plan = startup_plan(vec![planned_worker("chat")]);
        let runner = FakeRunner {
            next_pid: 6000,
            fail_ready_for: vec![WorkerId::new("chat")],
            ..FakeRunner::default()
        };
        let mut supervisor = WorkerSupervisor::new(runner);

        let statuses = supervisor.start_all(&plan).await;

        assert_eq!(statuses[0].worker_id, WorkerId::new("chat"));
        assert_eq!(statuses[0].pid, None);
        assert_eq!(statuses[0].state, WorkerState::Failed);
        assert_eq!(
            statuses[0].last_error.as_deref(),
            Some("ready probe failed")
        );
        assert_eq!(supervisor.runner().stopped, vec![WorkerId::new("chat")]);
    }

    #[tokio::test]
    async fn supervisor_drains_and_stops_running_workers() {
        let worker = planned_worker("chat");
        let plan = startup_plan(vec![worker.clone()]);
        let mut supervisor = WorkerSupervisor::new(FakeRunner {
            next_pid: 7000,
            ..FakeRunner::default()
        });
        supervisor.start_all(&plan).await;

        let drained = supervisor.drain(&worker.worker.id).await;
        let stopped = supervisor.stop(&worker.worker.id).await;

        assert_eq!(drained.state, WorkerState::Draining);
        assert_eq!(drained.pid, Some(7000));
        assert_eq!(stopped.state, WorkerState::Stopped);
        assert_eq!(stopped.pid, None);
    }

    #[tokio::test]
    async fn supervisor_restart_stops_then_starts_worker_and_increments_count() {
        let worker = planned_worker("chat");
        let plan = startup_plan(vec![worker.clone()]);
        let mut supervisor = WorkerSupervisor::new(FakeRunner {
            next_pid: 8000,
            ..FakeRunner::default()
        });
        supervisor.start_all(&plan).await;

        let restarted = supervisor.restart(&worker).await;

        assert_eq!(restarted.worker_id, WorkerId::new("chat"));
        assert_eq!(restarted.pid, Some(8001));
        assert_eq!(restarted.state, WorkerState::Ready);
        assert_eq!(restarted.restart_count, 1);
        assert_eq!(supervisor.runner().stopped, vec![WorkerId::new("chat")]);
    }

    #[tokio::test]
    async fn supervisor_restart_does_not_spawn_when_stop_fails() {
        let worker = planned_worker("chat");
        let plan = startup_plan(vec![worker.clone()]);
        let mut supervisor = WorkerSupervisor::new(FakeRunner {
            next_pid: 9000,
            fail_stop_for: vec![WorkerId::new("chat")],
            ..FakeRunner::default()
        });
        supervisor.start_all(&plan).await;

        let restarted = supervisor.restart(&worker).await;

        assert_eq!(restarted.state, WorkerState::Failed);
        assert_eq!(restarted.pid, Some(9000));
        assert_eq!(restarted.restart_count, 0);
        assert_eq!(restarted.last_error.as_deref(), Some("stop failed"));
        assert_eq!(supervisor.runner().spawned, vec![WorkerId::new("chat")]);
    }

    #[tokio::test]
    async fn supervisor_stop_all_cleans_up_every_known_worker() {
        let plan = startup_plan(vec![planned_worker("chat"), planned_worker("coder")]);
        let mut supervisor = WorkerSupervisor::new(FakeRunner {
            next_pid: 10_000,
            ..FakeRunner::default()
        });
        supervisor.start_all(&plan).await;

        let statuses = supervisor.stop_all().await;

        assert_eq!(statuses.len(), 2);
        assert!(statuses
            .iter()
            .all(|status| status.state == WorkerState::Stopped));
        assert_eq!(
            supervisor.runner().stopped,
            vec![WorkerId::new("chat"), WorkerId::new("coder")]
        );
    }

    #[tokio::test]
    async fn supervisor_executes_cold_swap_by_stopping_active_before_replacement() {
        let active = planned_worker("old");
        let replacement = planned_worker("new");
        let mut supervisor = WorkerSupervisor::new(FakeRunner {
            next_pid: 11_000,
            ..FakeRunner::default()
        });
        supervisor
            .start_all(&startup_plan(vec![active.clone()]))
            .await;

        let execution = supervisor
            .execute_swap(SwapMode::Cold, &active.worker.id, &replacement)
            .await;

        assert!(execution.success);
        assert_eq!(execution.mode, SwapMode::Cold);
        assert_eq!(
            execution
                .statuses
                .iter()
                .map(|status| (&status.worker_id, status.state))
                .collect::<Vec<_>>(),
            vec![
                (&WorkerId::new("old"), WorkerState::Draining),
                (&WorkerId::new("old"), WorkerState::Stopped),
                (&WorkerId::new("new"), WorkerState::Ready),
            ]
        );
        assert_eq!(
            supervisor.runner().spawned,
            vec![WorkerId::new("old"), WorkerId::new("new")]
        );
        assert_eq!(supervisor.runner().stopped, vec![WorkerId::new("old")]);
    }

    #[tokio::test]
    async fn supervisor_executes_hot_swap_by_warming_replacement_before_active_stop() {
        let active = planned_worker("old");
        let replacement = planned_worker("new");
        let mut supervisor = WorkerSupervisor::new(FakeRunner {
            next_pid: 12_000,
            ..FakeRunner::default()
        });
        supervisor
            .start_all(&startup_plan(vec![active.clone()]))
            .await;

        let execution = supervisor
            .execute_swap(SwapMode::Hot, &active.worker.id, &replacement)
            .await;

        assert!(execution.success);
        assert_eq!(execution.mode, SwapMode::Hot);
        assert_eq!(
            execution
                .statuses
                .iter()
                .map(|status| (&status.worker_id, status.state))
                .collect::<Vec<_>>(),
            vec![
                (&WorkerId::new("new"), WorkerState::Ready),
                (&WorkerId::new("old"), WorkerState::Draining),
                (&WorkerId::new("old"), WorkerState::Stopped),
            ]
        );
        assert_eq!(
            supervisor.runner().spawned,
            vec![WorkerId::new("old"), WorkerId::new("new")]
        );
        assert_eq!(supervisor.runner().stopped, vec![WorkerId::new("old")]);
    }

    #[tokio::test]
    async fn supervisor_hot_swap_keeps_active_when_replacement_readiness_fails() {
        let active = planned_worker("old");
        let replacement = planned_worker("new");
        let mut supervisor = WorkerSupervisor::new(FakeRunner {
            next_pid: 13_000,
            fail_ready_for: vec![WorkerId::new("new")],
            ..FakeRunner::default()
        });
        supervisor
            .start_all(&startup_plan(vec![active.clone()]))
            .await;

        let execution = supervisor
            .execute_swap(SwapMode::Hot, &active.worker.id, &replacement)
            .await;

        assert!(!execution.success);
        assert_eq!(
            execution
                .statuses
                .iter()
                .map(|status| (&status.worker_id, status.state))
                .collect::<Vec<_>>(),
            vec![(&WorkerId::new("new"), WorkerState::Failed)]
        );
        assert_eq!(
            supervisor.runner().spawned,
            vec![WorkerId::new("old"), WorkerId::new("new")]
        );
        assert_eq!(supervisor.runner().stopped, vec![WorkerId::new("new")]);
        let active_status = supervisor
            .statuses()
            .into_iter()
            .find(|status| status.worker_id == active.worker.id)
            .expect("active status");
        assert_eq!(active_status.state, WorkerState::Ready);
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
    fn startup_plan_builds_cpu_only_in_process_workers_for_configured_models() {
        let cfg = crate::config::Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                worker_base_port: 19000,
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
        let rendered = serde_json::to_value(&plan).expect("startup plan serializes");

        assert_eq!(plan.workers.len(), 2);
        assert!(rendered["resource_limits"]["systemd"]["CPUQuota"]
            .as_str()
            .expect("CPUQuota")
            .ends_with('%'));
        assert!(
            rendered["resource_limits"]["systemd"]["MemoryMax"]
                .as_u64()
                .expect("MemoryMax")
                > 0
        );
        assert!(rendered["resource_limits"]["systemd"]["unit_properties"]
            .as_array()
            .expect("unit properties")
            .iter()
            .any(|property| property == "CPUAccounting=true"));
        assert!(rendered["resource_limits"]["systemd"]["systemd_run_args"]
            .as_array()
            .expect("systemd-run args")
            .iter()
            .any(|property| property
                .as_str()
                .expect("systemd-run arg")
                .starts_with("--property=MemoryMax=")));
        assert_eq!(plan.workers[0].worker.id, WorkerId::new("chat"));
        assert_eq!(plan.workers[0].launch, WorkerLaunchPlan::candle_native());
        assert_eq!(plan.workers[0].worker.port, 19000);
        assert_eq!(plan.workers[0].worker.backend, WorkerBackend::Cpu);
        assert_eq!(
            plan.workers[0].command.program,
            PathBuf::from("<in-process:candle-native>")
        );
        assert_eq!(
            plan.workers[0].command.args,
            vec!["--model", "/models/chat.gguf", "--ctx-size", "4096"]
        );
        assert!(plan.workers[0].command.env.is_empty());
        assert_eq!(plan.workers[1].worker.id, WorkerId::new("coder"));
        assert_eq!(plan.workers[1].worker.port, 19001);
    }

    #[test]
    fn startup_plan_selects_gpu_execution_metadata_from_resources() {
        let cfg = crate::config::Config {
            server: ServerConfig {
                worker_base_port: 19100,
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
        assert_eq!(plan.workers[0].launch, WorkerLaunchPlan::candle_native());
        assert_eq!(
            plan.workers[0].worker.backend,
            WorkerBackend::Nvidia { gpu_layers: 99 }
        );
        assert_eq!(
            plan.workers[0].execution,
            WorkerExecutionPlan {
                backend: WorkerBackend::Nvidia { gpu_layers: 99 },
                device: WorkerDeviceSelection {
                    device_index: Some(0),
                    selection: "first-visible-gpu".to_string(),
                },
                gpu_layers: 99,
            }
        );
        assert!(plan.workers[0].command.env.is_empty());
        assert_eq!(
            plan.workers[0].command.program,
            PathBuf::from("<in-process:candle-native>")
        );
    }

    #[test]
    fn startup_plan_defaults_to_in_process_candle_native() {
        let cfg = crate::config::Config {
            models: vec![model("chat", "/models/chat.gguf")],
            ..Default::default()
        };

        let plan = StartupPlan::from_config(&cfg);

        assert_eq!(plan.workers.len(), 1);
        assert_eq!(
            plan.workers[0].launch,
            WorkerLaunchPlan::InProcess {
                backend: RuntimeBackend::CandleNative,
                engine: "candle-native".to_string(),
                implemented: true,
            }
        );
        assert_eq!(
            plan.workers[0].command.program,
            PathBuf::from("<in-process:candle-native>")
        );
    }

    #[tokio::test]
    async fn candle_native_placeholder_fails_clearly_if_started() {
        let cfg = crate::config::Config {
            models: vec![model("chat", "/models/chat.gguf")],
            ..Default::default()
        };
        let plan = StartupPlan::from_config(&cfg);
        let mut supervisor = WorkerSupervisor::new(TokioWorkerRunner::new());

        let statuses = supervisor.start_all(&plan).await;

        assert_eq!(statuses[0].worker_id, WorkerId::new("chat"));
        assert_eq!(statuses[0].pid, None);
        assert_eq!(statuses[0].state, WorkerState::Failed);
        assert!(statuses[0]
            .last_error
            .as_deref()
            .expect("last error")
            .contains("candle-native runtime is planned as an in-process worker"));
    }

    #[test]
    fn startup_plan_diff_reports_added_removed_and_changed_worker_aliases() {
        let old = StartupPlan {
            resource_limits: ResourceLimitPlan::default(),
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
                    launch: WorkerLaunchPlan::candle_native(),
                    execution: WorkerExecutionPlan::cpu(),
                    command: CommandSpec {
                        program: PathBuf::from("<in-process:candle-native>"),
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
                    launch: WorkerLaunchPlan::candle_native(),
                    execution: WorkerExecutionPlan::cpu(),
                    command: CommandSpec {
                        program: PathBuf::from("<in-process:candle-native>"),
                        args: vec!["--model".to_string(), "/models/embed.gguf".to_string()],
                        env: Vec::new(),
                    },
                },
            ],
        };
        let new = StartupPlan {
            resource_limits: ResourceLimitPlan::default(),
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
                    launch: WorkerLaunchPlan::candle_native(),
                    execution: WorkerExecutionPlan::cpu(),
                    command: CommandSpec {
                        program: PathBuf::from("<in-process:candle-native>"),
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
                    launch: WorkerLaunchPlan::candle_native(),
                    execution: WorkerExecutionPlan::cpu(),
                    command: CommandSpec {
                        program: PathBuf::from("<in-process:candle-native>"),
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
