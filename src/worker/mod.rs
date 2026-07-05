use crate::observability::{
    emit_runtime_telemetry, RuntimeTelemetryEvent, TelemetryEventName, TelemetrySignal,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod plan;
pub use plan::*;
mod admission;
pub use admission::*;
mod runner;
pub use runner::*;
mod swap;
pub use swap::*;

/// Upper bound on how long `drain` waits for in-flight requests to finish
/// before proceeding to tear the worker down.
const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Poll interval used while waiting for a worker's in-flight count to reach 0.
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(50);

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

#[derive(Debug)]
pub struct WorkerSupervisor<R> {
    runner: R,
    statuses: BTreeMap<WorkerId, WorkerStatus>,
    admissions: WorkerAdmissionRegistry,
    drain_timeout: Duration,
}

impl<R> WorkerSupervisor<R> {
    pub fn new(runner: R) -> Self {
        Self {
            runner,
            statuses: BTreeMap::new(),
            admissions: Arc::new(std::sync::RwLock::new(BTreeMap::new())),
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
        }
    }

    /// Returns a clone of the shared admission registry for the request path to
    /// consult without locking the supervisor.
    pub fn admissions(&self) -> WorkerAdmissionRegistry {
        self.admissions.clone()
    }

    fn insert_admission(&self, worker_id: WorkerId, admission: Arc<WorkerAdmission>) {
        if let Ok(mut registry) = self.admissions.write() {
            registry.insert(worker_id, admission);
        }
    }

    fn get_admission(&self, worker_id: &WorkerId) -> Option<Arc<WorkerAdmission>> {
        self.admissions.read().ok()?.get(worker_id).cloned()
    }

    /// Overrides how long `drain` waits for in-flight requests to finish.
    #[must_use]
    pub fn with_drain_timeout(mut self, drain_timeout: Duration) -> Self {
        self.drain_timeout = drain_timeout;
        self
    }

    pub fn runner(&self) -> &R {
        &self.runner
    }

    pub fn statuses(&self) -> Vec<WorkerStatus> {
        self.statuses.values().cloned().collect()
    }

    /// Current lifecycle state of a worker, if known.
    pub fn worker_state(&self, worker_id: &WorkerId) -> Option<WorkerState> {
        self.statuses.get(worker_id).map(|status| status.state)
    }

    /// Live admission gate for a worker. Routing clones this to gate requests
    /// and to hold an in-flight guard while proxying, without holding the
    /// supervisor lock for the request's lifetime.
    pub fn worker_admission(&self, worker_id: &WorkerId) -> Option<Arc<WorkerAdmission>> {
        self.get_admission(worker_id)
    }

    /// Number of workers currently reporting [`WorkerState::Ready`].
    pub fn ready_worker_count(&self) -> usize {
        self.statuses
            .values()
            .filter(|status| status.state == WorkerState::Ready)
            .count()
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
                    Ok(()) => {
                        // A freshly-ready worker admits new requests. Install a
                        // fresh admission gate so any stale (draining) gate from
                        // a prior incarnation is replaced.
                        self.insert_admission(worker_id.clone(), WorkerAdmission::ready());
                        self.update_status(&worker_id, |status| {
                            status.state = WorkerState::Ready;
                            status.last_error = None;
                        })
                    }
                    Err(error) => {
                        let _ = self.runner.stop(&worker_id).await;
                        self.stop_admitting(&worker_id);
                        self.update_status(&worker_id, |status| {
                            status.pid = None;
                            status.state = WorkerState::Failed;
                            status.last_error = Some(error.to_string());
                        })
                    }
                }
            }
            Err(error) => {
                self.stop_admitting(&worker_id);
                self.update_status(&worker_id, |status| {
                    status.pid = None;
                    status.state = WorkerState::Failed;
                    status.last_error = Some(error.to_string());
                })
            }
        }
    }

    /// Drains a worker: stops admitting new requests, waits (up to the drain
    /// timeout) for in-flight requests to finish, then marks it draining. This
    /// is the real drain gate — routing stops selecting the worker as soon as
    /// admission is closed, and teardown waits for outstanding work.
    pub async fn drain(&mut self, worker_id: &WorkerId) -> WorkerStatus {
        self.stop_admitting(worker_id);
        if let Some(admission) = self.get_admission(worker_id) {
            let deadline = Instant::now() + self.drain_timeout;
            while admission.in_flight() > 0 && Instant::now() < deadline {
                tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
            }
        }
        self.update_status(worker_id, |status| {
            status.state = WorkerState::Draining;
            status.last_error = None;
        })
    }

    pub async fn stop(&mut self, worker_id: &WorkerId) -> WorkerStatus {
        self.stop_admitting(worker_id);
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

    /// Closes a worker's admission gate so routing immediately stops selecting
    /// it. Safe to call for unknown workers.
    fn stop_admitting(&self, worker_id: &WorkerId) {
        if let Some(admission) = self.get_admission(worker_id) {
            admission.set_admitting(false);
        }
    }

    /// Detects workers whose process has crashed and marks them not-ready so
    /// routing avoids them. Returns the statuses of any workers reaped. Intended
    /// to be called periodically by a supervision loop.
    pub fn reap_crashed(&mut self) -> Vec<WorkerStatus> {
        let exited = self.runner.poll_exited();
        let mut reaped = Vec::new();
        for worker_id in exited {
            // Only workers we believed were live are worth transitioning; a
            // worker already stopped/stopping/failed needs no change.
            let was_live = matches!(
                self.worker_state(&worker_id),
                Some(WorkerState::Ready | WorkerState::Warming | WorkerState::Draining)
            );
            if !was_live {
                continue;
            }
            self.stop_admitting(&worker_id);
            reaped.push(self.update_status(&worker_id, |status| {
                status.pid = None;
                status.state = WorkerState::Failed;
                status.last_error = Some("worker process exited unexpectedly".to_string());
            }));
        }
        reaped
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

#[cfg(test)]
mod tests;
