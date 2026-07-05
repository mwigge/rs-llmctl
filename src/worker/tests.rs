
use super::*;
use crate::config::{ModelConfig, ResourceConfig, ServerConfig};
use crate::resources::ResourceLimitPlan;
use crate::runtime::RuntimeBackend;
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
    /// Ordered spawn/stop event log, used to prove interleaving (e.g. that a
    /// downgraded swap stops the active worker before spawning the
    /// replacement, so both are never resident at once).
    events: Vec<String>,
    /// Workers whose process the next `poll_exited` should report as exited.
    exited: Vec<WorkerId>,
}

impl WorkerRunner for FakeRunner {
    fn spawn<'a>(
        &'a mut self,
        planned: &'a PlannedWorker,
    ) -> BoxFuture<'a, Result<SpawnedWorker, WorkerRunnerError>> {
        self.spawned.push(planned.worker.id.clone());
        self.events
            .push(format!("spawn:{}", planned.worker.id.as_str()));
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
        self.events.push(format!("stop:{}", worker_id.as_str()));
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

    fn poll_exited(&mut self) -> Vec<WorkerId> {
        std::mem::take(&mut self.exited)
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

// Bug 4 regression: a hot swap whose active+replacement footprints exceed
// the resource budget must NOT double-load (start replacement while active
// is still resident). Before the fix, `execute_hot_swap` always spawned the
// replacement first; here the over-budget hot swap is downgraded to a cold
// swap so the active worker is stopped before the replacement is spawned.
#[tokio::test]
async fn hot_swap_over_budget_downgrades_to_cold_to_avoid_double_allocation() {
    let active = planned_worker("old");
    let replacement = planned_worker("new");
    let mut supervisor = WorkerSupervisor::new(FakeRunner {
        next_pid: 20_000,
        ..FakeRunner::default()
    });
    supervisor
        .start_all(&startup_plan(vec![active.clone()]))
        .await;

    // active 6GB + replacement 6GB = 12GB > 10GB budget → cannot co-reside.
    let budget = SwapBudget {
        active_bytes: 6_000_000_000,
        replacement_bytes: 6_000_000_000,
        budget_bytes: 10_000_000_000,
    };
    assert!(!budget.hot_swap_fits());

    let execution = supervisor
        .execute_swap_with_budget(SwapMode::Hot, &active.worker.id, &replacement, Some(budget))
        .await;

    assert!(execution.success);
    // Downgraded to cold: mode reported as Cold, and the event log proves the
    // active worker was stopped BEFORE the replacement was ever spawned.
    assert_eq!(execution.mode, SwapMode::Cold);
    assert_eq!(
        supervisor.runner().events,
        vec![
            "spawn:old".to_string(),
            "stop:old".to_string(),
            "spawn:new".to_string(),
        ]
    );
}

// Companion: when active+replacement fit the budget, the hot swap proceeds
// as requested (replacement warmed before the active worker is drained).
#[tokio::test]
async fn hot_swap_within_budget_stays_hot() {
    let active = planned_worker("old");
    let replacement = planned_worker("new");
    let mut supervisor = WorkerSupervisor::new(FakeRunner {
        next_pid: 21_000,
        ..FakeRunner::default()
    });
    supervisor
        .start_all(&startup_plan(vec![active.clone()]))
        .await;

    let budget = SwapBudget {
        active_bytes: 4_000_000_000,
        replacement_bytes: 4_000_000_000,
        budget_bytes: 10_000_000_000,
    };
    assert!(budget.hot_swap_fits());

    let execution = supervisor
        .execute_swap_with_budget(SwapMode::Hot, &active.worker.id, &replacement, Some(budget))
        .await;

    assert!(execution.success);
    assert_eq!(execution.mode, SwapMode::Hot);
    // Hot: replacement spawned before the active worker is stopped.
    assert_eq!(
        supervisor.runner().events,
        vec![
            "spawn:old".to_string(),
            "spawn:new".to_string(),
            "stop:old".to_string(),
        ]
    );
}

// Bug 3 regression: drain must close the worker's admission gate so routing
// immediately stops selecting it. Before the fix, drain only flipped the
// status enum while the admission gate (and routing) stayed open.
#[tokio::test]
async fn drain_closes_admission_gate_so_routing_stops_selecting_worker() {
    let worker = planned_worker("chat");
    let mut supervisor = WorkerSupervisor::new(FakeRunner {
        next_pid: 22_000,
        ..FakeRunner::default()
    });
    supervisor
        .start_all(&startup_plan(vec![worker.clone()]))
        .await;

    let admission = supervisor
        .worker_admission(&worker.worker.id)
        .expect("ready worker exposes an admission gate");
    assert!(admission.is_admitting());
    assert!(admission.try_enter().is_some());

    let drained = supervisor.drain(&worker.worker.id).await;

    assert_eq!(drained.state, WorkerState::Draining);
    assert!(!admission.is_admitting());
    assert!(
        admission.try_enter().is_none(),
        "a draining worker must not admit new requests"
    );
}

// Bug 3 regression: drain must wait for in-flight requests to finish before
// completing, rather than tearing the worker down under live traffic.
#[tokio::test]
async fn drain_waits_for_in_flight_requests_before_completing() {
    let worker = planned_worker("chat");
    let mut supervisor = WorkerSupervisor::new(FakeRunner {
        next_pid: 23_000,
        ..FakeRunner::default()
    })
    .with_drain_timeout(Duration::from_millis(150));
    supervisor
        .start_all(&startup_plan(vec![worker.clone()]))
        .await;

    let admission = supervisor
        .worker_admission(&worker.worker.id)
        .expect("admission gate");
    let guard = admission.try_enter().expect("admitted");
    assert_eq!(admission.in_flight(), 1);

    // With an in-flight request held, drain blocks until the drain timeout.
    let started = Instant::now();
    supervisor.drain(&worker.worker.id).await;
    assert!(
        started.elapsed() >= Duration::from_millis(120),
        "drain returned before waiting out in-flight work: {:?}",
        started.elapsed()
    );

    // Dropping the guard releases the in-flight count.
    drop(guard);
    assert_eq!(admission.in_flight(), 0);
}

// Bug 3 regression: a crashed worker must be detected and marked not-ready so
// routing avoids the dead port. Before the fix there was no reaping loop and
// a crashed worker kept its Ready status (and open admission).
#[tokio::test]
async fn reap_crashed_marks_exited_worker_not_ready() {
    let worker = planned_worker("chat");
    // The fake reports the worker's process as exited on the next poll,
    // simulating a crash after it became ready.
    let mut supervisor = WorkerSupervisor::new(FakeRunner {
        next_pid: 24_000,
        exited: vec![worker.worker.id.clone()],
        ..FakeRunner::default()
    });
    supervisor
        .start_all(&startup_plan(vec![worker.clone()]))
        .await;
    assert_eq!(
        supervisor.worker_state(&worker.worker.id),
        Some(WorkerState::Ready)
    );

    let reaped = supervisor.reap_crashed();

    assert_eq!(reaped.len(), 1);
    assert_eq!(reaped[0].state, WorkerState::Failed);
    assert_eq!(
        supervisor.worker_state(&worker.worker.id),
        Some(WorkerState::Failed)
    );
    assert!(
        !supervisor
            .worker_admission(&worker.worker.id)
            .expect("admission gate")
            .is_admitting(),
        "a crashed worker must stop admitting requests"
    );
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

// When llama-cpp-native is compiled in, AMD GPU workers use the in-process
// engine rather than the subprocess, so this test only applies to builds
// without that feature.
#[cfg(not(feature = "llama-cpp-native"))]
#[test]
fn startup_plan_selects_llama_server_subprocess_for_amd_when_binary_configured() {
    let llama_path = PathBuf::from("/usr/local/bin/llama-server");
    let cfg = crate::config::Config {
        server: ServerConfig {
            worker_base_port: 19200,
            ..ServerConfig::default()
        },
        resources: ResourceConfig {
            cpu_only: false,
            gpu_vendor: "amd".to_string(),
            llama_server_bin: Some(llama_path.clone()),
            ..ResourceConfig::default()
        },
        models: vec![model("chat", "/models/qwen3-14b.gguf")],
        ..Default::default()
    };

    let plan = StartupPlan::from_config(&cfg);

    assert_eq!(plan.workers.len(), 1);
    assert!(matches!(
        plan.workers[0].launch,
        WorkerLaunchPlan::LlamaServerSubprocess { .. }
    ));
    assert_eq!(
        plan.workers[0].worker.backend,
        WorkerBackend::AmdVulkan { gpu_layers: 99 }
    );
    assert_eq!(plan.workers[0].command.program, llama_path);
    assert!(plan.workers[0].command.args.contains(&"-ngl".to_string()));
    assert!(plan.workers[0]
        .command
        .env
        .iter()
        .any(|(k, _)| k == "HIP_PLATFORM"));
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

#[cfg(feature = "llama-cpp-native")]
#[test]
fn amd_gpu_prefers_llama_cpp_native_when_feature_enabled() {
    use crate::config::{Config, ResourceConfig};

    let mut cfg = Config::default();
    cfg.resources = ResourceConfig {
        gpu_vendor: "amd".to_string(),
        cpu_only: false,
        ..ResourceConfig::default()
    };
    cfg.models = vec![model("my-model", "/models/my-model.gguf")];

    let plan = StartupPlan::from_config(&cfg);

    assert_eq!(plan.workers.len(), 1);
    assert!(
        matches!(
            plan.workers[0].launch,
            WorkerLaunchPlan::LlamaCppNative { .. }
        ),
        "AMD GPU with llama-cpp-native feature should use LlamaCppNative launch plan, got {:?}",
        plan.workers[0].launch
    );
}

#[test]
fn cpu_resources_use_in_process_candle_native_plan() {
    use crate::config::{Config, ResourceConfig};

    let mut cfg = Config::default();
    cfg.resources = ResourceConfig {
        gpu_vendor: String::new(),
        cpu_only: true,
        ..ResourceConfig::default()
    };
    cfg.models = vec![model("my-model", "/models/my-model.gguf")];

    let plan = StartupPlan::from_config(&cfg);

    assert_eq!(plan.workers.len(), 1);
    assert!(
        matches!(plan.workers[0].launch, WorkerLaunchPlan::InProcess { .. }),
        "CPU-only resources should use InProcess launch plan"
    );
}
