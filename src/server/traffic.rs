use super::record_circuit_breaker_state;
use opentelemetry::global;
use opentelemetry::KeyValue;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

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
        let mut states = self.states.lock().expect("circuit breaker mutex poisoned");
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
        let mut states = self.states.lock().expect("circuit breaker mutex poisoned");
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
        let mut states = self.states.lock().expect("circuit breaker mutex poisoned");
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
        let mut failures = self.failures.lock().expect("auth limiter mutex poisoned");
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
        window.count >= limit
    }

    pub(super) fn record_failure(&self, key: &str, limit: u32) {
        if limit == 0 {
            return;
        }
        let mut failures = self.failures.lock().expect("auth limiter mutex poisoned");
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
        let mut failures = self.failures.lock().expect("auth limiter mutex poisoned");
        failures.remove(key);
    }
}
