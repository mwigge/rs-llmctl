use super::{AdmissionController, AuthFailureLimiter, CircuitBreakers, DEFAULT_MAX_IN_FLIGHT};
use crate::config::Config;
use crate::native;
use crate::storage::Storage;
use crate::worker::{TokioWorkerRunner, WorkerAdmissionRegistry, WorkerSupervisor};
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;

pub type NativeEngineRegistry = BTreeMap<String, Arc<dyn native::NativeEngine>>;

#[derive(Clone)]
pub struct ServerState {
    pub(super) cfg: Arc<Config>,
    pub(super) storage: Storage,
    pub(super) client: reqwest::Client,
    pub(super) upstreams: BTreeMap<String, String>,
    pub(super) admission: AdmissionController,
    pub(super) serving_limits: ServingLimits,
    pub(super) native_engines: NativeEngineRegistry,
    pub(super) worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    /// Lock-free-to-read view of live worker admission gates, cloned from the
    /// supervisor. Present when `worker_control` is; lets the request path gate
    /// on live worker state without contending on the supervisor mutex.
    pub(super) worker_admissions: Option<WorkerAdmissionRegistry>,
    pub(super) draining: Arc<AtomicBool>,
    pub(super) circuit_breakers: CircuitBreakers,
    pub(super) auth_failures: AuthFailureLimiter,
}

#[derive(Debug, Clone, Copy)]
pub struct ServingLimits {
    pub(super) max_in_flight: usize,
    pub(super) upstream_timeout: Duration,
}

impl ServingLimits {
    pub fn new(max_in_flight: usize, upstream_timeout: Duration) -> Self {
        Self {
            max_in_flight: max_in_flight.max(1),
            upstream_timeout: upstream_timeout.max(Duration::from_millis(1)),
        }
    }

    pub(super) fn from_config(cfg: &Config) -> Self {
        let configured_max = cfg
            .quotas
            .iter()
            .filter_map(|quota| usize::try_from(quota.max_concurrency).ok())
            .filter(|limit| *limit > 0)
            .fold(0usize, usize::saturating_add);
        let max_in_flight = if configured_max > 0 {
            configured_max
        } else {
            DEFAULT_MAX_IN_FLIGHT
        };

        Self::new(
            max_in_flight,
            Duration::from_secs(cfg.server.upstream_timeout_seconds),
        )
    }

    pub(super) fn upstream_timeout(&self) -> Duration {
        self.upstream_timeout
    }
}
