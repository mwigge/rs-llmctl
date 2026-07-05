use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetrySignal {
    Metric,
    Span,
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryEventName {
    RequestRouting,
    QuotaDecision,
    WorkerLifecycle,
    CircuitBreaker,
    ResourceSnapshot,
    DriftObservation,
    ModelInstallVerification,
    NativeRuntimeStatus,
    RuntimeHeartbeat,
}

impl TelemetryEventName {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RequestRouting => "llmctl.request.routing",
            Self::QuotaDecision => "llmctl.quota.decision",
            Self::WorkerLifecycle => "llmctl.worker.lifecycle",
            Self::CircuitBreaker => "llmctl.upstream.circuit_breaker",
            Self::ResourceSnapshot => "llmctl.resource.snapshot",
            Self::DriftObservation => "llmctl.drift.observation",
            Self::ModelInstallVerification => "llmctl.model.install.verification",
            Self::NativeRuntimeStatus => "llmctl.runtime.native.status",
            Self::RuntimeHeartbeat => "llmctl.runtime.heartbeat",
        }
    }
}

impl Serialize for TelemetryEventName {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for TelemetryEventName {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "llmctl.request.routing" => Ok(Self::RequestRouting),
            "llmctl.quota.decision" => Ok(Self::QuotaDecision),
            "llmctl.worker.lifecycle" => Ok(Self::WorkerLifecycle),
            "llmctl.upstream.circuit_breaker" => Ok(Self::CircuitBreaker),
            "llmctl.resource.snapshot" => Ok(Self::ResourceSnapshot),
            "llmctl.drift.observation" => Ok(Self::DriftObservation),
            "llmctl.model.install.verification" => Ok(Self::ModelInstallVerification),
            "llmctl.runtime.native.status" => Ok(Self::NativeRuntimeStatus),
            "llmctl.runtime.heartbeat" => Ok(Self::RuntimeHeartbeat),
            _ => Err(serde::de::Error::custom(format!(
                "unknown telemetry event name `{value}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeTelemetryEvent {
    pub signal: TelemetrySignal,
    pub name: TelemetryEventName,
    pub timestamp: DateTime<Utc>,
    pub attributes: BTreeMap<String, Value>,
}

impl RuntimeTelemetryEvent {
    pub fn new(
        signal: TelemetrySignal,
        name: TelemetryEventName,
        timestamp: DateTime<Utc>,
        attributes: BTreeMap<String, Value>,
    ) -> Self {
        Self {
            signal,
            name,
            timestamp,
            attributes: sanitize_otel_attributes(attributes),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TelemetryBatch {
    pub events: Vec<RuntimeTelemetryEvent>,
}

impl TelemetryBatch {
    pub fn new(events: Vec<RuntimeTelemetryEvent>) -> Self {
        Self { events }
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }
}

#[derive(Debug, Clone, Default)]
pub struct NoopTelemetryExporter;

impl NoopTelemetryExporter {
    pub fn export(&self, batch: &TelemetryBatch) -> Result<usize> {
        Ok(batch.len())
    }
}

pub fn emit_runtime_telemetry(event: &RuntimeTelemetryEvent) {
    let attributes = telemetry_key_values(event);
    runtime_events_counter().add(1, &attributes);
    if matches!(event.name, TelemetryEventName::RuntimeHeartbeat) {
        heartbeat_timestamp_gauge().record(event.timestamp.timestamp().max(0) as u64, &attributes);
        let healthy = event
            .attributes
            .get("runtime.healthy")
            .or_else(|| event.attributes.get("llmctl.runtime.healthy"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        heartbeat_healthy_gauge().record(u64::from(healthy), &attributes);
        let draining = event
            .attributes
            .get("llmctl.server.draining")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        server_draining_gauge().record(u64::from(draining), &attributes);
    }

    if matches!(event.signal, TelemetrySignal::Span) {
        let tracer = global::tracer(crate::SERVICE_NAME);
        let mut span = tracer.start(event.name.as_str().to_string());
        for attribute in attributes {
            span.set_attribute(attribute);
        }
        if event
            .attributes
            .get("llmctl.status")
            .and_then(Value::as_str)
            .is_some_and(|status| status != "ok")
        {
            span.set_status(Status::error("llmctl status is not ok"));
        }
        span.end();
    }

    tracing::info!(
        telemetry_event = event.name.as_str(),
        telemetry_signal = ?event.signal,
        "runtime telemetry event emitted"
    );
}

fn runtime_events_counter() -> &'static Counter<u64> {
    static COUNTER: OnceLock<Counter<u64>> = OnceLock::new();
    COUNTER.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_counter("llmctl.runtime.events")
            .with_description("Runtime telemetry events emitted by rs-llmctl")
            .build()
    })
}

fn heartbeat_timestamp_gauge() -> &'static Gauge<u64> {
    static GAUGE: OnceLock<Gauge<u64>> = OnceLock::new();
    GAUGE.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_gauge("llmctl_runtime_heartbeat_timestamp_seconds")
            .with_description("Unix timestamp of the most recent rs-llmctl runtime heartbeat")
            .build()
    })
}

fn heartbeat_healthy_gauge() -> &'static Gauge<u64> {
    static GAUGE: OnceLock<Gauge<u64>> = OnceLock::new();
    GAUGE.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_gauge("llmctl_runtime_heartbeat_healthy")
            .with_description("Runtime heartbeat health as 1 for healthy and 0 for unhealthy")
            .build()
    })
}

fn server_draining_gauge() -> &'static Gauge<u64> {
    static GAUGE: OnceLock<Gauge<u64>> = OnceLock::new();
    GAUGE.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_gauge("llmctl_server_draining")
            .with_description("Server drain state as 1 while graceful shutdown drain is active")
            .build()
    })
}

fn telemetry_key_values(event: &RuntimeTelemetryEvent) -> Vec<KeyValue> {
    let mut attributes = vec![
        KeyValue::new("llmctl.telemetry.name", event.name.as_str()),
        KeyValue::new(
            "llmctl.telemetry.signal",
            match event.signal {
                TelemetrySignal::Metric => "metric",
                TelemetrySignal::Span => "span",
                TelemetrySignal::Log => "log",
            },
        ),
    ];
    attributes.extend(
        event
            .attributes
            .iter()
            .filter_map(|(key, value)| json_to_key_value(key, value)),
    );
    attributes
}

fn json_to_key_value(key: &str, value: &Value) -> Option<KeyValue> {
    match value {
        Value::Bool(value) => Some(KeyValue::new(key.to_string(), *value)),
        Value::Number(value) => value
            .as_i64()
            .map(|value| KeyValue::new(key.to_string(), value))
            .or_else(|| {
                value
                    .as_u64()
                    .and_then(|value| i64::try_from(value).ok())
                    .map(|value| KeyValue::new(key.to_string(), value))
            })
            .or_else(|| {
                value
                    .as_f64()
                    .map(|value| KeyValue::new(key.to_string(), value))
            }),
        Value::String(value) => Some(KeyValue::new(key.to_string(), value.clone())),
        _ => None,
    }
}
