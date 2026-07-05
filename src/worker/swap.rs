use super::{PlannedWorker, WorkerId, WorkerRunner, WorkerState, WorkerStatus, WorkerSupervisor};
use serde::{Deserialize, Serialize};

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

/// Resource footprints and budget used to decide whether a hot swap can safely
/// co-resident the active and replacement models. All values are in the same
/// unit (bytes) and against the same resource (VRAM on a GPU box, otherwise
/// system memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapBudget {
    /// Resident footprint of the currently-active worker.
    pub active_bytes: u64,
    /// Resident footprint of the replacement worker.
    pub replacement_bytes: u64,
    /// Total budget both models must fit within to co-reside.
    pub budget_bytes: u64,
}

impl SwapBudget {
    /// Whether the active and replacement models can be resident simultaneously
    /// (as a hot swap requires) without exceeding the budget.
    #[must_use]
    pub fn hot_swap_fits(&self) -> bool {
        self.active_bytes.saturating_add(self.replacement_bytes) <= self.budget_bytes
    }
}

impl<R: WorkerRunner> WorkerSupervisor<R> {
    pub async fn execute_swap(
        &mut self,
        mode: SwapMode,
        active: &WorkerId,
        replacement: &PlannedWorker,
    ) -> SwapExecution {
        self.execute_swap_with_budget(mode, active, replacement, None)
            .await
    }

    /// Executes a swap, optionally enforcing a resource budget for hot swaps.
    ///
    /// A hot swap loads the replacement while the active model is still
    /// resident. On a constrained box that double-allocation OOMs. When `budget`
    /// is supplied and the two models cannot co-reside, the hot swap is
    /// automatically downgraded to a cold swap (which stops the active worker
    /// before loading the replacement) so the operation stays within budget
    /// instead of risking OOM during a "zero-downtime" swap. `None` preserves
    /// the caller-selected mode unchanged.
    pub async fn execute_swap_with_budget(
        &mut self,
        mode: SwapMode,
        active: &WorkerId,
        replacement: &PlannedWorker,
        budget: Option<SwapBudget>,
    ) -> SwapExecution {
        let effective_mode = match (mode, budget) {
            (SwapMode::Hot, Some(budget)) if !budget.hot_swap_fits() => SwapMode::Cold,
            (mode, _) => mode,
        };
        match effective_mode {
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
