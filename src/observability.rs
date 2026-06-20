use crate::config::{Config, LangfuseExporterConfig, ObservabilityExporterConfig, OtlpProtocol};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use opentelemetry::global;
use opentelemetry::metrics::{Counter, Gauge, Histogram};
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

/// Translate Langfuse project credentials into an OTLP/HTTP exporter aimed at
/// Langfuse's `/api/public/otel` ingestion endpoint, authenticated with HTTP
/// Basic auth (`base64(public_key:secret_key)`) — the scheme Langfuse expects
/// from generic OTLP producers. Returns `None` when disabled or incomplete,
/// so callers can fall back to the explicit exporter configuration.
pub fn langfuse_otlp_exporter(cfg: &LangfuseExporterConfig) -> Option<Exporter> {
    if !cfg.enabled {
        return None;
    }
    let host = cfg
        .host
        .as_deref()
        .unwrap_or("")
        .trim()
        .trim_end_matches('/');
    let public_key = cfg.public_key.as_deref().unwrap_or("").trim();
    let secret_key = cfg.secret_key.as_deref().unwrap_or("").trim();
    if host.is_empty() || public_key.is_empty() || secret_key.is_empty() {
        return None;
    }

    let mut headers = BTreeMap::new();
    headers.insert(
        "Authorization".to_string(),
        langfuse_basic_auth_header(public_key, secret_key),
    );

    Some(Exporter::Otlp {
        endpoint: format!("{host}/api/public/otel"),
        protocol: OtlpProtocol::HttpProtobuf,
        headers,
        timeout_ms: ObservabilityExporterConfig::default().timeout_ms,
    })
}

fn langfuse_basic_auth_header(public_key: &str, secret_key: &str) -> String {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    let credentials = format!("{public_key}:{secret_key}");
    format!("Basic {}", STANDARD.encode(credentials))
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
            _ => langfuse_otlp_exporter(&observability.langfuse).unwrap_or(Exporter::None),
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

/// Returns the histogram that tracks thinking-phase token counts per model.
pub fn gen_ai_thinking_tokens_histogram() -> &'static Histogram<u64> {
    static HIST: OnceLock<Histogram<u64>> = OnceLock::new();
    HIST.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_histogram("gen_ai.thinking_tokens")
            .with_description("Number of thinking-phase content deltas per inference request")
            .build()
    })
}

macro_rules! static_f64_gauge {
    ($name:expr, $desc:expr) => {{
        static GAUGE: OnceLock<Gauge<f64>> = OnceLock::new();
        GAUGE.get_or_init(|| {
            global::meter(crate::SERVICE_NAME)
                .f64_gauge($name)
                .with_description($desc)
                .build()
        })
    }};
}

/// Returns the gauge that tracks the fraction of output that was thinking content.
pub fn gen_ai_thinking_ratio_gauge() -> &'static Gauge<f64> {
    static_f64_gauge!(
        "gen_ai.thinking_ratio",
        "Fraction of content deltas that were thinking-phase (0.0–1.0) per request"
    )
}

/// Returns the gauge that tracks the KV-cache occupancy ratio for one model worker.
///
/// Records values in the range `[0.0, 1.0]` where `1.0` means the cache is
/// completely full.  Tagged with `gen_ai.request.model`.
pub fn gen_ai_kv_cache_usage_ratio_gauge() -> &'static Gauge<f64> {
    static_f64_gauge!(
        "gen_ai.kv_cache.usage_ratio",
        "KV-cache occupancy ratio per model worker (0.0 = empty, 1.0 = full)"
    )
}

fn add_thinking_phase_event(name: &'static str, attrs: Vec<KeyValue>) {
    use opentelemetry::trace::get_active_span;
    get_active_span(|span| span.add_event(name, attrs));
}

/// Adds a `gen_ai.thinking.started` event to the current span.
pub fn emit_gen_ai_thinking_phase_started(model: &str, position: u64) {
    add_thinking_phase_event(
        "gen_ai.thinking.started",
        vec![
            KeyValue::new("gen_ai.request.model", model.to_string()),
            KeyValue::new(
                "gen_ai.token.position",
                i64::try_from(position).unwrap_or(i64::MAX),
            ),
        ],
    );
}

/// Adds a `gen_ai.thinking.ended` event to the current span.
pub fn emit_gen_ai_thinking_phase_ended(model: &str, thinking_tokens: u64, duration_seconds: f64) {
    add_thinking_phase_event(
        "gen_ai.thinking.ended",
        vec![
            KeyValue::new("gen_ai.request.model", model.to_string()),
            KeyValue::new(
                "gen_ai.thinking.tokens",
                i64::try_from(thinking_tokens).unwrap_or(i64::MAX),
            ),
            KeyValue::new("gen_ai.thinking.duration_s", duration_seconds),
        ],
    );
}

/// Emits `gen_ai.thinking_tokens` and `gen_ai.thinking_ratio` metrics for one
/// completed inference request, tagged with the serving model name.
pub fn emit_gen_ai_thinking_metrics(model: &str, thinking_deltas: u64, output_deltas: u64) {
    let attrs = [KeyValue::new("gen_ai.request.model", model.to_string())];
    gen_ai_thinking_tokens_histogram().record(thinking_deltas, &attrs);
    let total = thinking_deltas + output_deltas;
    let ratio = if total > 0 {
        thinking_deltas as f64 / total as f64
    } else {
        0.0
    };
    gen_ai_thinking_ratio_gauge().record(ratio, &attrs);
}

/// Maximum bytes buffered per SSE frame before the parser gives up.
const MAX_SSE_CONTENT_BUFFER_BYTES: usize = 512 * 1024;

/// Tail length kept for cross-chunk `<think>` / `</think>` tag detection.
const TAG_TAIL_LEN: usize = 8;

/// Streaming phase tracked by [`SseContentParser`].
#[derive(Debug, Default, PartialEq, Eq)]
enum ContentPhase {
    #[default]
    Output,
    Thinking,
}

/// Parses a Server-Sent Events stream and counts output versus thinking-phase
/// content deltas.
///
/// Thinking content is any delta that arrives between a `<think>` and a
/// `</think>` tag, even when the tags are split across consecutive SSE frames.
/// Delta frames that carry only the tag itself are not counted in either
/// category.
#[derive(Debug, Default)]
pub struct SseContentParser {
    buffer: String,
    phase: ContentPhase,
    thinking_deltas: u64,
    output_deltas: u64,
    /// Last [`TAG_TAIL_LEN`] chars of previously-seen content, used to detect
    /// tags that are split across frame boundaries.
    tag_tail: String,
}

impl SseContentParser {
    /// Feeds raw SSE bytes into the parser, updating internal delta counters.
    pub fn push(&mut self, bytes: &[u8]) {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);
        while let Some(frame_end) = self.buffer.find("\n\n") {
            if frame_end > MAX_SSE_CONTENT_BUFFER_BYTES {
                self.buffer.drain(..frame_end + 2);
                continue;
            }
            let frame = self.buffer[..frame_end].to_string();
            self.buffer.drain(..frame_end + 2);
            self.process_frame(&frame);
        }
    }

    /// Returns the number of thinking-phase content deltas seen so far.
    pub fn thinking_deltas(&self) -> u64 {
        self.thinking_deltas
    }

    /// Returns the number of output-phase content deltas seen so far.
    pub fn output_deltas(&self) -> u64 {
        self.output_deltas
    }

    fn process_frame(&mut self, frame: &str) {
        for line in frame.lines() {
            let Some(data) = line.trim_start().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            self.process_delta_value(&value);
        }
    }

    fn process_delta_value(&mut self, value: &Value) {
        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            return;
        };
        for choice in choices {
            let Some(content) = choice
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            self.classify_delta(content);
        }
    }

    fn classify_delta(&mut self, delta: &str) {
        let combined = format!("{}{}", self.tag_tail, delta);
        // Only look for a transition into thinking when we're currently in output phase.
        let entered_thinking = self.phase == ContentPhase::Output && combined.contains("<think>");
        // Only look for a transition out of thinking when we're currently in thinking phase.
        let left_thinking = self.phase == ContentPhase::Thinking && combined.contains("</think>");

        // Update tail with just the last TAG_TAIL_LEN chars of this delta
        // (not combined) so it remains the suffix of actual content seen.
        self.tag_tail = tail_str(&combined, TAG_TAIL_LEN);

        // A delta that carries a phase-transition tag is not counted in either bucket.
        if entered_thinking {
            self.phase = ContentPhase::Thinking;
            return;
        }
        if left_thinking {
            self.phase = ContentPhase::Output;
            return;
        }

        match self.phase {
            ContentPhase::Thinking => self.thinking_deltas += 1,
            ContentPhase::Output => self.output_deltas += 1,
        }
    }
}

/// Returns the last `n` chars of `s` as a new `String`.
fn tail_str(s: &str, n: usize) -> String {
    s.chars()
        .rev()
        .take(n)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
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
    fn langfuse_exporter_derives_otlp_endpoint_and_basic_auth_header() {
        let cfg = LangfuseExporterConfig {
            enabled: true,
            host: Some("https://cloud.langfuse.com/".to_string()),
            public_key: Some("pk-lf-abc".to_string()),
            secret_key: Some("sk-lf-xyz".to_string()),
        };

        let exporter = langfuse_otlp_exporter(&cfg).expect("langfuse exporter should be derived");
        match exporter {
            Exporter::Otlp {
                endpoint,
                protocol,
                headers,
                ..
            } => {
                assert_eq!(endpoint, "https://cloud.langfuse.com/api/public/otel");
                assert_eq!(protocol, OtlpProtocol::HttpProtobuf);
                assert_eq!(
                    headers.get("Authorization").map(String::as_str),
                    Some("Basic cGstbGYtYWJjOnNrLWxmLXh5eg==")
                );
            }
            Exporter::None => panic!("expected an OTLP exporter"),
        }
    }

    #[test]
    fn langfuse_exporter_is_none_when_disabled_or_missing_keys() {
        assert!(langfuse_otlp_exporter(&LangfuseExporterConfig::default()).is_none());

        let missing_secret = LangfuseExporterConfig {
            enabled: true,
            host: Some("https://cloud.langfuse.com".to_string()),
            public_key: Some("pk-lf-abc".to_string()),
            secret_key: None,
        };
        assert!(langfuse_otlp_exporter(&missing_secret).is_none());

        let missing_host = LangfuseExporterConfig {
            enabled: true,
            host: None,
            public_key: Some("pk-lf-abc".to_string()),
            secret_key: Some("sk-lf-xyz".to_string()),
        };
        assert!(langfuse_otlp_exporter(&missing_host).is_none());
    }

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

    #[test]
    fn sse_content_parser_counts_output_deltas_without_thinking_phase() {
        let mut parser = SseContentParser::default();
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n");
        assert_eq!(parser.output_deltas(), 2);
        assert_eq!(parser.thinking_deltas(), 0);
    }

    #[test]
    fn sse_content_parser_separates_thinking_from_output_deltas() {
        let mut parser = SseContentParser::default();
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"<think>\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"reasoning\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"</think>\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n");
        assert_eq!(parser.output_deltas(), 2); // "Hello" + "answer"
        assert_eq!(parser.thinking_deltas(), 1); // "reasoning"
    }

    #[test]
    fn sse_content_parser_handles_think_tag_split_across_chunks() {
        let mut parser = SseContentParser::default();
        // "<think>" split across two SSE frames: the first partial chunk is
        // counted as output because the tag cannot be detected until the second
        // chunk arrives and completes it.
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"<th\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"ink>\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"deep thought\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"</think>\"}}]}\n\n");
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n");
        assert_eq!(parser.thinking_deltas(), 1); // "deep thought"
                                                 // "<th" is counted as output (partial tag not yet detected), "done" too.
        assert_eq!(parser.output_deltas(), 2);
    }

    #[test]
    fn sse_content_parser_ignores_done_sentinel() {
        let mut parser = SseContentParser::default();
        parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
        parser.push(b"data: [DONE]\n\n");
        assert_eq!(parser.output_deltas(), 1);
        assert_eq!(parser.thinking_deltas(), 0);
    }

    #[test]
    fn emit_gen_ai_thinking_metrics_does_not_panic_on_zero_deltas() {
        emit_gen_ai_thinking_metrics("test-model", 0, 0);
    }

    #[test]
    fn gen_ai_kv_cache_usage_ratio_gauge_returns_stable_reference() {
        let g1 = gen_ai_kv_cache_usage_ratio_gauge();
        let g2 = gen_ai_kv_cache_usage_ratio_gauge();
        // Both references must point to the same static gauge (pointer equality).
        assert!(std::ptr::eq(g1, g2));
        // Recording must not panic.
        g1.record(
            0.42,
            &[opentelemetry::KeyValue::new("gen_ai.request.model", "test")],
        );
    }

    #[test]
    fn emit_gen_ai_thinking_phase_started_does_not_panic() {
        emit_gen_ai_thinking_phase_started("test-model", 0);
    }

    #[test]
    fn emit_gen_ai_thinking_phase_ended_does_not_panic() {
        emit_gen_ai_thinking_phase_ended("test-model", 128, 1.5);
    }
}
