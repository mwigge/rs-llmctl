use super::WorkerId;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Live admission gate for a single worker, shared between the supervisor and
/// the request-routing path.
///
/// Routing consults `is_admitting()` to decide whether a worker may receive new
/// requests, and holds an [`InFlightGuard`] for the duration of each proxied
/// request so `drain` can wait for outstanding work to finish before the worker
/// is torn down. This is the piece that connects live worker lifecycle state to
/// the request path — without it, `drain`/`stop` only flip an enum while the
/// router keeps proxying to a dead port.
#[derive(Debug)]
pub struct WorkerAdmission {
    admitting: AtomicBool,
    in_flight: AtomicUsize,
}

impl WorkerAdmission {
    pub(super) fn ready() -> Arc<Self> {
        Arc::new(Self {
            admitting: AtomicBool::new(true),
            in_flight: AtomicUsize::new(0),
        })
    }

    /// Whether the worker is currently accepting new requests.
    pub fn is_admitting(&self) -> bool {
        self.admitting.load(Ordering::SeqCst)
    }

    /// Number of requests currently in flight against this worker.
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    pub(super) fn set_admitting(&self, value: bool) {
        self.admitting.store(value, Ordering::SeqCst);
    }

    /// Attempts to admit a new request. Returns a guard that keeps the worker's
    /// in-flight count raised until dropped, or `None` when the worker is not
    /// currently admitting (draining/stopping/stopped/failed).
    pub fn try_enter(self: &Arc<Self>) -> Option<InFlightGuard> {
        if !self.is_admitting() {
            return None;
        }
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        // Re-check after incrementing to close the race where `drain` flipped
        // `admitting` to false between the check above and the increment.
        if !self.is_admitting() {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(InFlightGuard {
            admission: self.clone(),
        })
    }
}

/// RAII guard that decrements a worker's in-flight count when dropped. Held for
/// the duration of a proxied request.
#[derive(Debug)]
pub struct InFlightGuard {
    admission: Arc<WorkerAdmission>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.admission.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Shared, lock-free-to-read registry mapping each worker to its live admission
/// gate. The supervisor owns the authoritative copy and the request-routing
/// path holds a clone (see [`WorkerSupervisor::admissions`]), so routing can
/// consult live worker admission state without contending on the supervisor's
/// async mutex — which a swap/drain may hold for the duration of a model load.
pub type WorkerAdmissionRegistry = Arc<std::sync::RwLock<BTreeMap<WorkerId, Arc<WorkerAdmission>>>>;
