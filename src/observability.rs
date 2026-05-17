use crate::config::{Config, OtlpProtocol};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use opentelemetry::global;
use opentelemetry::metrics::{Counter, Gauge};
use opentelemetry::trace::{Span, Status, Tracer};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig, WithTonicConfig};
use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Duration;
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

pub const REDACTED_ATTRIBUTE_VALUE: &str = "[REDACTED]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservabilityPlan {
    pub service_name: String,
    pub service_version: Option<String>,
    pub environment: Option<String>,
    pub traces_enabled: bool,
    pub metrics_enabled: bool,
    pub logs_enabled: bool,
    pub resource_attributes: BTreeMap<String, String>,
    pub exporter: Exporter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exporter {
    None,
    Otlp {
        endpoint: String,
        protocol: OtlpProtocol,
        headers: BTreeMap<String, String>,
        timeout_ms: u64,
    },
}

#[derive(Debug, Default)]
pub struct TelemetryRuntime {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
    logger_provider: Option<SdkLoggerProvider>,
}

impl TelemetryRuntime {
    pub fn install(cfg: &Config, json_logs: bool) -> Result<Self> {
        let plan = ObservabilityPlan::from_config(cfg)?;
        Self::install_plan(&plan, json_logs)
    }

    pub fn install_plan(plan: &ObservabilityPlan, json_logs: bool) -> Result<Self> {
        let mut runtime = Self::from_plan(plan)?;
        runtime.install_tracing(json_logs)?;
        Ok(runtime)
    }

    pub fn from_plan(plan: &ObservabilityPlan) -> Result<Self> {
        let Exporter::Otlp {
            endpoint,
            protocol,
            headers,
            timeout_ms,
        } = &plan.exporter
        else {
            return Ok(Self::default());
        };

        let resource = telemetry_resource(plan);
        let timeout = Duration::from_millis(*timeout_ms);
        let mut runtime = Self::default();

        if plan.traces_enabled {
            let exporter = build_span_exporter(endpoint, *protocol, headers, timeout)
                .context("build OTLP trace exporter")?;
            let provider = SdkTracerProvider::builder()
                .with_resource(resource.clone())
                .with_batch_exporter(exporter)
                .build();
            global::set_tracer_provider(provider.clone());
            runtime.tracer_provider = Some(provider);
        }

        if plan.metrics_enabled {
            let exporter = build_metric_exporter(endpoint, *protocol, headers, timeout)
                .context("build OTLP metric exporter")?;
            let provider = SdkMeterProvider::builder()
                .with_resource(resource.clone())
                .with_periodic_exporter(exporter)
                .build();
            global::set_meter_provider(provider.clone());
            runtime.meter_provider = Some(provider);
        }

        if plan.logs_enabled {
            let exporter = build_log_exporter(endpoint, *protocol, headers, timeout)
                .context("build OTLP log exporter")?;
            let provider = SdkLoggerProvider::builder()
                .with_resource(resource)
                .with_batch_exporter(exporter)
                .build();
            runtime.logger_provider = Some(provider);
        }

        Ok(runtime)
    }

    pub fn shutdown(self) -> Result<()> {
        if let Some(provider) = self.tracer_provider {
            provider
                .shutdown()
                .context("shutdown OTLP trace provider")?;
        }
        if let Some(provider) = self.meter_provider {
            provider
                .shutdown()
                .context("shutdown OTLP meter provider")?;
        }
        if let Some(provider) = self.logger_provider {
            provider.shutdown().context("shutdown OTLP log provider")?;
        }
        Ok(())
    }

    fn install_tracing(&mut self, json_logs: bool) -> Result<()> {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let fmt_layer = if json_logs {
            tracing_subscriber::fmt::layer().json().boxed()
        } else {
            tracing_subscriber::fmt::layer().boxed()
        };
        let mut layers = vec![fmt_layer];

        if let Some(logger_provider) = &self.logger_provider {
            layers.push(
                opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
                    logger_provider,
                )
                .boxed(),
            );
        }
        tracing_subscriber::registry()
            .with(filter)
            .with(layers)
            .try_init()
            .context("install tracing subscriber")?;
        Ok(())
    }
}

fn telemetry_resource(plan: &ObservabilityPlan) -> Resource {
    let mut attributes = vec![KeyValue::new("service.name", plan.service_name.clone())];
    if let Some(version) = &plan.service_version {
        attributes.push(KeyValue::new("service.version", version.clone()));
    }
    if let Some(environment) = &plan.environment {
        attributes.push(KeyValue::new(
            "deployment.environment.name",
            environment.clone(),
        ));
    }
    attributes.extend(
        plan.resource_attributes
            .iter()
            .map(|(key, value)| KeyValue::new(key.clone(), value.clone())),
    );
    Resource::builder().with_attributes(attributes).build()
}

fn otlp_protocol(protocol: OtlpProtocol) -> Protocol {
    match protocol {
        OtlpProtocol::HttpProtobuf => Protocol::HttpBinary,
        OtlpProtocol::Grpc => Protocol::Grpc,
    }
}

fn otlp_headers(headers: &BTreeMap<String, String>) -> HashMap<String, String> {
    headers
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn otlp_metadata(headers: &BTreeMap<String, String>) -> Result<MetadataMap> {
    let mut metadata = MetadataMap::new();
    for (key, value) in headers {
        metadata.insert(
            key.parse::<MetadataKey<_>>()
                .with_context(|| format!("parse OTLP gRPC metadata key `{key}`"))?,
            value
                .parse::<MetadataValue<_>>()
                .with_context(|| format!("parse OTLP gRPC metadata value for `{key}`"))?,
        );
    }
    Ok(metadata)
}

fn build_span_exporter(
    endpoint: &str,
    protocol: OtlpProtocol,
    headers: &BTreeMap<String, String>,
    timeout: Duration,
) -> Result<opentelemetry_otlp::SpanExporter> {
    match protocol {
        OtlpProtocol::HttpProtobuf => Ok(opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .with_protocol(otlp_protocol(protocol))
            .with_timeout(timeout)
            .with_headers(otlp_headers(headers))
            .build()?),
        OtlpProtocol::Grpc => Ok(opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_timeout(timeout)
            .with_metadata(otlp_metadata(headers)?)
            .build()?),
    }
}

fn build_metric_exporter(
    endpoint: &str,
    protocol: OtlpProtocol,
    headers: &BTreeMap<String, String>,
    timeout: Duration,
) -> Result<opentelemetry_otlp::MetricExporter> {
    match protocol {
        OtlpProtocol::HttpProtobuf => Ok(opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .with_protocol(otlp_protocol(protocol))
            .with_timeout(timeout)
            .with_headers(otlp_headers(headers))
            .build()?),
        OtlpProtocol::Grpc => Ok(opentelemetry_otlp::MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_timeout(timeout)
            .with_metadata(otlp_metadata(headers)?)
            .build()?),
    }
}

fn build_log_exporter(
    endpoint: &str,
    protocol: OtlpProtocol,
    headers: &BTreeMap<String, String>,
    timeout: Duration,
) -> Result<opentelemetry_otlp::LogExporter> {
    match protocol {
        OtlpProtocol::HttpProtobuf => Ok(opentelemetry_otlp::LogExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .with_protocol(otlp_protocol(protocol))
            .with_timeout(timeout)
            .with_headers(otlp_headers(headers))
            .build()?),
        OtlpProtocol::Grpc => Ok(opentelemetry_otlp::LogExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_timeout(timeout)
            .with_metadata(otlp_metadata(headers)?)
            .build()?),
    }
}

impl ObservabilityPlan {
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let observability = &cfg.observability;
        let endpoint = observability
            .exporter
            .endpoint
            .clone()
            .or_else(|| observability.otlp_endpoint.clone());

        let exporter = match endpoint {
            Some(endpoint) if !endpoint.trim().is_empty() => Exporter::Otlp {
                endpoint,
                protocol: observability.exporter.protocol,
                headers: observability.exporter.headers.clone(),
                timeout_ms: observability.exporter.timeout_ms,
            },
            _ => Exporter::None,
        };

        anyhow::ensure!(
            observability.exporter.timeout_ms > 0,
            "observability exporter timeout must be greater than zero"
        );

        Ok(Self {
            service_name: observability
                .service_name
                .clone()
                .unwrap_or_else(|| crate::SERVICE_NAME.to_string()),
            service_version: observability.service_version.clone(),
            environment: observability.environment.clone(),
            traces_enabled: observability.traces_enabled,
            metrics_enabled: observability.metrics_enabled,
            logs_enabled: observability.logs_enabled,
            resource_attributes: observability.resource_attributes.clone(),
            exporter,
        })
    }
}

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

pub fn sanitize_otel_attributes(attributes: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    attributes
        .into_iter()
        .map(|(key, value)| {
            if should_redact_attribute(&key, &value) {
                (key, Value::String(REDACTED_ATTRIBUTE_VALUE.to_string()))
            } else {
                (key, value)
            }
        })
        .collect()
}

fn should_redact_attribute(key: &str, value: &Value) -> bool {
    let normalized = key.to_ascii_lowercase().replace(['-', '.'], "_");

    if normalized.contains("authorization")
        || normalized.contains("api_key")
        || normalized.contains("apikey")
        || normalized.contains("secret")
        || normalized.contains("password")
        || normalized.contains("token")
        || normalized.contains("bearer")
        || normalized.contains("prompt")
        || normalized.contains("message")
        || normalized == "content"
        || normalized.ends_with("_content")
        || normalized.contains("otlp_headers")
        || normalized.contains("collector_header")
        || normalized.contains("exporter_header")
        || normalized.contains("header_authorization")
        || normalized.ends_with("_path")
        || normalized.contains("_path_")
        || normalized == "path"
    {
        return true;
    }

    string_value_contains_secret(value)
}

fn string_value_contains_secret(value: &Value) -> bool {
    value.as_str().is_some_and(|value| {
        let normalized = value.to_ascii_lowercase();
        normalized.contains("bearer ")
            || normalized.contains("api_key=")
            || normalized.contains("authorization=")
            || normalized.contains("x-api-key")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn runtime_telemetry_event_names_cover_runtime_surfaces() {
        assert_eq!(
            TelemetryEventName::RequestRouting.as_str(),
            "llmctl.request.routing"
        );
        assert_eq!(
            TelemetryEventName::QuotaDecision.as_str(),
            "llmctl.quota.decision"
        );
        assert_eq!(
            TelemetryEventName::WorkerLifecycle.as_str(),
            "llmctl.worker.lifecycle"
        );
        assert_eq!(
            TelemetryEventName::CircuitBreaker.as_str(),
            "llmctl.upstream.circuit_breaker"
        );
        assert_eq!(
            TelemetryEventName::ResourceSnapshot.as_str(),
            "llmctl.resource.snapshot"
        );
        assert_eq!(
            TelemetryEventName::DriftObservation.as_str(),
            "llmctl.drift.observation"
        );
        assert_eq!(
            TelemetryEventName::ModelInstallVerification.as_str(),
            "llmctl.model.install.verification"
        );
        assert_eq!(
            TelemetryEventName::NativeRuntimeStatus.as_str(),
            "llmctl.runtime.native.status"
        );
        assert_eq!(
            TelemetryEventName::RuntimeHeartbeat.as_str(),
            "llmctl.runtime.heartbeat"
        );
    }

    #[test]
    fn telemetry_batch_serializes_events_without_network_exporter() {
        let mut attrs = BTreeMap::new();
        attrs.insert("route".to_string(), json!("/v1/chat/completions"));
        attrs.insert("quota.allowed".to_string(), json!(true));

        let event = RuntimeTelemetryEvent::new(
            TelemetrySignal::Metric,
            TelemetryEventName::QuotaDecision,
            Utc::now(),
            attrs,
        );
        let batch = TelemetryBatch::new(vec![event]);

        let serialized = serde_json::to_value(&batch).expect("batch serializes");
        assert_eq!(serialized["events"][0]["signal"], "metric");
        assert_eq!(serialized["events"][0]["name"], "llmctl.quota.decision");
        assert_eq!(
            serialized["events"][0]["attributes"]["route"],
            "/v1/chat/completions"
        );
        assert_eq!(serialized["events"][0]["attributes"]["quota.allowed"], true);
    }

    #[test]
    fn runtime_telemetry_emitter_accepts_sanitized_span_attributes() {
        let mut attrs = BTreeMap::new();
        attrs.insert("llmctl.model".to_string(), json!("qwen"));
        attrs.insert("llmctl.allowed".to_string(), json!(true));
        attrs.insert("llmctl.tokens".to_string(), json!(42));
        let event = RuntimeTelemetryEvent::new(
            TelemetrySignal::Span,
            TelemetryEventName::RequestRouting,
            Utc::now(),
            attrs,
        );

        emit_runtime_telemetry(&event);
    }

    #[test]
    fn telemetry_attribute_sanitizer_redacts_secret_and_content_values() {
        let mut attrs = BTreeMap::new();
        attrs.insert("authorization".to_string(), json!("Bearer collector-token"));
        attrs.insert("api_key".to_string(), json!("sk-live-secret"));
        attrs.insert("prompt".to_string(), json!("tell me the admin password"));
        attrs.insert(
            "messages".to_string(),
            json!([{ "role": "user", "content": "private" }]),
        );
        attrs.insert(
            "otel.exporter.otlp.headers".to_string(),
            json!("Authorization=Bearer abc"),
        );
        attrs.insert(
            "model.path".to_string(),
            json!("/home/alice/.cache/llmctl/model.gguf"),
        );
        attrs.insert("route".to_string(), json!("/v1/chat/completions"));
        attrs.insert("quota.allowed".to_string(), json!(false));

        let sanitized = sanitize_otel_attributes(attrs);

        assert_eq!(sanitized.get("authorization"), Some(&json!("[REDACTED]")));
        assert_eq!(sanitized.get("api_key"), Some(&json!("[REDACTED]")));
        assert_eq!(sanitized.get("prompt"), Some(&json!("[REDACTED]")));
        assert_eq!(sanitized.get("messages"), Some(&json!("[REDACTED]")));
        assert_eq!(
            sanitized.get("otel.exporter.otlp.headers"),
            Some(&json!("[REDACTED]"))
        );
        assert_eq!(sanitized.get("model.path"), Some(&json!("[REDACTED]")));
        assert_eq!(sanitized.get("route"), Some(&json!("/v1/chat/completions")));
        assert_eq!(sanitized.get("quota.allowed"), Some(&json!(false)));
    }
}
