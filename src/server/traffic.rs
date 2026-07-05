use super::record_circuit_breaker_state;
use opentelemetry::global;
use opentelemetry::KeyValue;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Caps on how many distinct scopes/upstreams/keys each in-memory map tracks
/// before idle entries are pruned, so a stream of distinct subjects cannot leak
/// memory unboundedly.
const SCOPED_ADMISSION_MAX_TRACKED: usize = 4096;
const CIRCUIT_BREAKER_MAX_TRACKED: usize = 4096;
const AUTH_FAILURE_MAX_TRACKED: usize = 8192;

#[derive(Clone)]
pub(super) struct AdmissionController {
    global: Arc<Semaphore>,
    scoped: Arc<Mutex<BTreeMap<String, Arc<Semaphore>>>>,
}

impl AdmissionController {
    pub(super) fn new(max_in_flight: usize) -> Self {
        Self {
            global: Arc::new(Semaphore::new(max_in_flight.max(1))),
            scoped: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Number of global admission permits currently available. The count of
    /// in-flight requests is `max_in_flight - available_permits()`. Read-only;
    /// used by the admin status endpoint.
    pub(super) fn available_permits(&self) -> usize {
        self.global.available_permits()
    }

    #[cfg(test)]
    pub(super) fn try_acquire_for(
        &self,
        scope: Option<(String, usize)>,
    ) -> std::result::Result<AdmissionPermit, AdmissionError> {
        self.try_acquire_for_all(scope.into_iter().collect())
    }

    pub(super) fn try_acquire_for_all(
        &self,
        scopes: Vec<(String, usize)>,
    ) -> std::result::Result<AdmissionPermit, AdmissionError> {
        let global = self
            .global
            .clone()
            .try_acquire_owned()
            .map_err(|_| AdmissionError::Busy)?;

        let mut scoped_permits = Vec::with_capacity(scopes.len());
        for (scope, limit) in scopes {
            let permit = {
                let semaphore = {
                    let mut scoped = self.scoped.lock().map_err(|_| AdmissionError::Busy)?;
                    // Bound growth from many distinct scopes: drop entries that
                    // no in-flight request holds. A held permit keeps an extra
                    // `Arc` clone alive, so `strong_count == 1` means only the
                    // map references the semaphore and it can be safely
                    // recreated on demand.
                    if scoped.len() >= SCOPED_ADMISSION_MAX_TRACKED {
                        scoped.retain(|_, sem| Arc::strong_count(sem) > 1);
                    }
                    scoped
                        .entry(scope)
                        .or_insert_with(|| Arc::new(Semaphore::new(limit.max(1))))
                        .clone()
                };
                semaphore
                    .try_acquire_owned()
                    .map_err(|_| AdmissionError::Busy)?
            };
            scoped_permits.push(permit);
        }

        Ok(AdmissionPermit {
            _global: global,
            _scoped: scoped_permits,
        })
    }
}

pub(super) struct AdmissionPermit {
    _global: OwnedSemaphorePermit,
    _scoped: Vec<OwnedSemaphorePermit>,
}

impl std::fmt::Debug for AdmissionPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmissionPermit").finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AdmissionError {
    Busy,
}

#[derive(Debug, Clone, Default)]
pub(super) struct CircuitBreakers {
    states: Arc<Mutex<BTreeMap<String, CircuitBreakerState>>>,
}

#[derive(Debug, Clone)]
struct CircuitBreakerState {
    consecutive_failures: u32,
    opened_at: Option<Instant>,
    half_open_probe_in_flight: bool,
}

impl CircuitBreakers {
    pub(super) fn allow_request(&self, upstream: &str, reset_after: Duration) -> bool {
        // Recover from a poisoned lock instead of panicking: a single request
        // that panicked while holding this lock must not wedge circuit
        // breaking for every subsequent request (fail-open degradation).
        let mut states = lock_recover(&self.states);
        let Some(state) = states.get_mut(upstream) else {
            return true;
        };
        let Some(opened_at) = state.opened_at else {
            return true;
        };
        if opened_at.elapsed() >= reset_after {
            if state.half_open_probe_in_flight {
                record_circuit_breaker_state(
                    upstream,
                    "half_open_busy",
                    state.consecutive_failures,
                );
                false
            } else {
                state.half_open_probe_in_flight = true;
                record_circuit_breaker_state(upstream, "half_open", state.consecutive_failures);
                true
            }
        } else {
            record_circuit_breaker_state(upstream, "open", state.consecutive_failures);
            false
        }
    }

    pub(super) fn record_success(&self, upstream: &str) {
        let mut states = lock_recover(&self.states);
        prune_idle_circuit_breakers(&mut states);
        let state = states
            .entry(upstream.to_string())
            .or_insert(CircuitBreakerState {
                consecutive_failures: 0,
                opened_at: None,
                half_open_probe_in_flight: false,
            });
        state.consecutive_failures = 0;
        state.opened_at = None;
        state.half_open_probe_in_flight = false;
        record_circuit_breaker_state(upstream, "closed", 0);
    }

    pub(super) fn record_failure(&self, upstream: &str, threshold: u32) {
        let mut states = lock_recover(&self.states);
        prune_idle_circuit_breakers(&mut states);
        let state = states
            .entry(upstream.to_string())
            .or_insert(CircuitBreakerState {
                consecutive_failures: 0,
                opened_at: None,
                half_open_probe_in_flight: false,
            });
        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        state.half_open_probe_in_flight = false;
        if threshold > 0 && state.consecutive_failures >= threshold {
            state.opened_at = Some(Instant::now());
            record_circuit_breaker_state(upstream, "open", state.consecutive_failures);
        } else {
            record_circuit_breaker_state(upstream, "closed", state.consecutive_failures);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct AuthFailureLimiter {
    failures: Arc<Mutex<BTreeMap<String, AuthFailureWindow>>>,
}

#[derive(Debug, Clone)]
struct AuthFailureWindow {
    started: Instant,
    count: u32,
}

impl AuthFailureLimiter {
    pub(super) fn is_limited(&self, key: &str, limit: u32) -> bool {
        if limit == 0 {
            return false;
        }
        // Recover from a poisoned lock instead of panicking, so one panicked
        // request can't wedge auth throttling for everyone.
        let mut failures = lock_recover(&self.failures);
        // Read-only check path: use `get_mut` (not `entry`) so merely testing a
        // key never inserts an entry — that would let unauthenticated traffic
        // with fresh keys grow the map unboundedly.
        let Some(window) = failures.get_mut(key) else {
            return false;
        };
        if window.started.elapsed() >= Duration::from_secs(60) {
            window.started = Instant::now();
            window.count = 0;
        }
        window.count >= limit
    }

    pub(super) fn record_failure(&self, key: &str, limit: u32) {
        if limit == 0 {
            return;
        }
        let mut failures = lock_recover(&self.failures);
        prune_stale_auth_windows(&mut failures);
        let window = failures
            .entry(key.to_string())
            .or_insert(AuthFailureWindow {
                started: Instant::now(),
                count: 0,
            });
        if window.started.elapsed() >= Duration::from_secs(60) {
            window.started = Instant::now();
            window.count = 0;
        }
        window.count = window.count.saturating_add(1);
        let meter = global::meter(crate::SERVICE_NAME);
        meter
            .u64_counter("llmctl_auth_failures_total")
            .with_description("Failed bearer authentication attempts by throttle state")
            .build()
            .add(
                1,
                &[
                    KeyValue::new("limited", window.count >= limit),
                    KeyValue::new(
                        "status",
                        if window.count >= limit {
                            "limited"
                        } else {
                            "failed"
                        },
                    ),
                ],
            );
    }

    pub(super) fn record_success(&self, key: &str) {
        let mut failures = lock_recover(&self.failures);
        failures.remove(key);
    }
}

/// Locks a mutex, recovering the guard if the lock was poisoned by a panic
/// while held. Hot-path traffic controls must degrade gracefully rather than
/// propagate a poison panic to every subsequent request.
fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Drops circuit-breaker entries that carry no state (closed, no failures, no
/// in-flight probe) when the map grows large. A recreated entry is equivalent,
/// so this is behavior-preserving.
fn prune_idle_circuit_breakers(states: &mut BTreeMap<String, CircuitBreakerState>) {
    if states.len() < CIRCUIT_BREAKER_MAX_TRACKED {
        return;
    }
    states.retain(|_, state| {
        state.opened_at.is_some()
            || state.consecutive_failures > 0
            || state.half_open_probe_in_flight
    });
}

/// Drops auth-failure windows that have expired (older than the 60s window) or
/// carry no failures when the map grows large. Such entries are treated as
/// absent by `is_limited`/`record_failure`, so removing them is
/// behavior-preserving.
fn prune_stale_auth_windows(failures: &mut BTreeMap<String, AuthFailureWindow>) {
    if failures.len() < AUTH_FAILURE_MAX_TRACKED {
        return;
    }
    failures
        .retain(|_, window| window.count > 0 && window.started.elapsed() < Duration::from_secs(60));
}
