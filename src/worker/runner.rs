use super::{PlannedWorker, WorkerId};
use futures_util::future::{BoxFuture, FutureExt};
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;
use tokio::process::Child;

const READY_PROBE_ATTEMPTS: usize = 120;
const READY_PROBE_INTERVAL: Duration = Duration::from_millis(500);

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

    /// Returns the ids of workers whose underlying process has exited since the
    /// last poll. Used by the supervisor's crash-reaping loop to detect dead
    /// workers so routing can avoid them. Runners without a real process
    /// (test fakes) report none by default.
    fn poll_exited(&mut self) -> Vec<WorkerId> {
        Vec::new()
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

            // Ensure any prior child for this worker id is fully terminated (and
            // its bound port released) before the replacement binds the same
            // fixed port. Without this, spawning the replacement first and then
            // dropping the old `Child` transiently double-binds the port.
            if let Some(mut existing) = self.children.remove(&planned.worker.id) {
                let _ = existing.kill().await;
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

    fn poll_exited(&mut self) -> Vec<WorkerId> {
        let mut exited = Vec::new();
        let ids: Vec<WorkerId> = self.children.keys().cloned().collect();
        for id in ids {
            if let Some(child) = self.children.get_mut(&id) {
                // `try_wait` reaps without blocking; `Ok(Some(_))` means the
                // process has exited, `Err` means it can no longer be waited on.
                match child.try_wait() {
                    Ok(Some(_)) | Err(_) => {
                        self.children.remove(&id);
                        exited.push(id);
                    }
                    Ok(None) => {}
                }
            }
        }
        exited
    }
}
