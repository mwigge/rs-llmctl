use crate::config::{Config, LangfuseExporterConfig, ObservabilityExporterConfig, OtlpProtocol};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use opentelemetry::global;
use opentelemetry::metrics::{Counter, Gauge, Histogram};
use opentelemetry::propagation::Injector;
use opentelemetry::trace::{Span, Status, Tracer, TracerProvider as _};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig, WithTonicConfig};
use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::propagation::TraceContextPropagator;
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

static GLOBAL_TRACER_PROVIDER_REGISTERED: OnceLock<()> = OnceLock::new();
static GLOBAL_METER_PROVIDER_REGISTERED: OnceLock<()> = OnceLock::new();
static TRACING_SUBSCRIBER_INITIALIZED: OnceLock<()> = OnceLock::new();

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
            GLOBAL_TRACER_PROVIDER_REGISTERED.get_or_init(|| {
                global::set_tracer_provider(provider.clone());
                global::set_text_map_propagator(TraceContextPropagator::new());
            });
            runtime.tracer_provider = Some(provider);
        }

        if plan.metrics_enabled {
            let exporter = build_metric_exporter(endpoint, *protocol, headers, timeout)
                .context("build OTLP metric exporter")?;
            let provider = SdkMeterProvider::builder()
                .with_resource(resource.clone())
                .with_periodic_exporter(exporter)
                .build();
            GLOBAL_METER_PROVIDER_REGISTERED.get_or_init(|| {
                global::set_meter_provider(provider.clone());
            });
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

        if let Some(provider) = &self.tracer_provider {
            let tracer = provider.tracer(crate::SERVICE_NAME);
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
        }

        if let Some(logger_provider) = &self.logger_provider {
            layers.push(
                opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
                    logger_provider,
                )
                .boxed(),
            );
        }
        if TRACING_SUBSCRIBER_INITIALIZED.get().is_none() {
            tracing_subscriber::registry()
                .with(filter)
                .with(layers)
                .try_init()
                .context("install tracing subscriber")?;
            let _ = TRACING_SUBSCRIBER_INITIALIZED.set(());
        }
        Ok(())
    }
}

struct HeaderInjector<'a>(&'a mut reqwest::header::HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(val)) = (
            reqwest::header::HeaderName::from_bytes(key.as_bytes()),
            reqwest::header::HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, val);
        }
    }
}

/// Inject the active span's W3C `traceparent` (and `tracestate`) header into a
/// reqwest request builder. When no active span or propagator is present the
/// builder is returned unchanged; callers do not need to gate on OTel config.
pub fn inject_trace_context(builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    let cx = opentelemetry::Context::current();
    let mut headers = reqwest::header::HeaderMap::new();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&cx, &mut HeaderInjector(&mut headers));
    });
    headers
        .iter()
        .fold(builder, |b, (name, value)| b.header(name, value))
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
/// from generic OTLP producers. Returns `Ok(None)` when disabled or incomplete,
/// so callers can fall back to the explicit exporter configuration.
///
/// # Errors
/// Returns an error if the configured host is not a valid http/https URL, which
/// prevents path-injection attacks via a crafted host value.
pub fn langfuse_otlp_exporter(cfg: &LangfuseExporterConfig) -> Result<Option<Exporter>> {
    if !cfg.enabled {
        return Ok(None);
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
        return Ok(None);
    }

    // Parse the configured host to extract only scheme+authority — reject embedded paths
    // that could redirect OTLP traffic to an arbitrary endpoint.
    let parsed_host =
        url::Url::parse(host).with_context(|| format!("invalid langfuse host: {host:?}"))?;
    anyhow::ensure!(
        matches!(parsed_host.scheme(), "http" | "https"),
        "langfuse host must use http or https scheme"
    );
    let hostname = parsed_host
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("langfuse host has no hostname"))?;
    let origin = format!("{}://{}", parsed_host.scheme(), hostname);
    let origin = if let Some(port) = parsed_host.port() {
        format!("{origin}:{port}")
    } else {
        origin
    };
    let endpoint = format!("{origin}/api/public/otel");

    let mut headers = BTreeMap::new();
    headers.insert(
        "Authorization".to_string(),
        langfuse_basic_auth_header(public_key, secret_key),
    );

    Ok(Some(Exporter::Otlp {
        endpoint,
        protocol: OtlpProtocol::HttpProtobuf,
        headers,
        timeout_ms: ObservabilityExporterConfig::default().timeout_ms,
    }))
}

fn langfuse_basic_auth_header(public_key: &str, secret_key: &str) -> String {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    let credentials = format!("{public_key}:{secret_key}");
    format!("Basic {}", STANDARD.encode(credentials))
}

// DNS-based IMDS hostnames. IP-based IMDS addresses (169.254.169.254, fd00:ec2::254, etc.)
// are covered by `is_blocked_endpoint_ip` via range checks and do not need to appear here.
const SSRF_BLOCKED_METADATA_HOSTS: &[&str] =
    &["metadata.google.internal", "metadata.goog", "instance-data"];

/// Validate that an HTTP endpoint URL is safe to use.
///
/// Rejects non-http/https schemes, embedded credentials (userinfo), known cloud
/// instance metadata DNS hostnames, and all non-routable IP ranges (loopback,
/// link-local, private, shared address space, and IPv6 ULA/link-local) to
/// prevent SSRF exfiltration of trace data or webhook payloads.
///
/// # Errors
/// Returns an error if the URL is malformed, uses a non-http/https scheme,
/// contains credentials, or targets a non-routable or blocked address.
pub fn validate_http_endpoint(endpoint: &str) -> Result<()> {
    let parsed = url::Url::parse(endpoint.trim())
        .with_context(|| format!("invalid endpoint URL: {endpoint:?}"))?;

    anyhow::ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "endpoint scheme must be http or https, got {:?}",
        parsed.scheme()
    );

    // Reject credentials embedded in the URL — no legitimate OTLP/webhook URL needs them,
    // and userinfo can be used to smuggle a blocked hostname as the "password".
    anyhow::ensure!(
        parsed.username().is_empty() && parsed.password().is_none(),
        "endpoint URL must not contain credentials (user:pass@ form)"
    );

    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("endpoint URL must include a host"))?;
    let host_lower = host.to_ascii_lowercase();

    // Block known cloud metadata DNS hostnames.
    for &blocked in SSRF_BLOCKED_METADATA_HOSTS {
        anyhow::ensure!(
            host_lower != blocked,
            "endpoint host {host:?} is not permitted"
        );
    }

    // Block loopback, link-local, private, and unspecified IP ranges.
    if let Some(ip) = parsed.host().and_then(|h| match h {
        url::Host::Ipv4(ip) => Some(std::net::IpAddr::V4(ip)),
        url::Host::Ipv6(ip) => Some(std::net::IpAddr::V6(ip)),
        url::Host::Domain(_) => None,
    }) {
        anyhow::ensure!(
            !is_blocked_endpoint_ip(ip),
            "endpoint IP {ip} is not routable (loopback, link-local, or private address)"
        );
    }

    Ok(())
}

fn is_blocked_endpoint_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()           // 127.0.0.0/8
                || v4.is_link_local()  // 169.254.0.0/16
                || v4.is_private()     // 10/8, 172.16/12, 192.168/16
                || v4.is_unspecified() // 0.0.0.0
                || is_shared_address_space(v4) // 100.64.0.0/10 (RFC 6598, incl. Alibaba 100.100.100.200)
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()                // ::1
                || v6.is_unspecified()      // ::
                || is_ipv6_link_local(v6)   // fe80::/10
                || is_ipv6_unique_local(v6) // fc00::/7 (includes fd00:ec2::254)
        }
    }
}

/// Returns true if the IPv4 address falls within the RFC 6598 shared address
/// space 100.64.0.0/10, which covers the Alibaba Cloud IMDS at 100.100.100.200.
fn is_shared_address_space(ip: std::net::Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 100 && (octets[1] & 0xC0) == 64
}

/// Returns true if the IPv6 address falls within the link-local range `fe80::/10`.
///
/// `Ipv6Addr::is_unicast_link_local()` is not yet stable, so the check is
/// implemented manually.
fn is_ipv6_link_local(ip: std::net::Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xffc0) == 0xfe80
}

/// Returns true if the IPv6 address falls within the unique-local range
/// `fc00::/7`, which covers `fd00:ec2::254` and all other ULA addresses.
///
/// `Ipv6Addr::is_unique_local()` is not yet stable, so the check is
/// implemented manually.
fn is_ipv6_unique_local(ip: std::net::Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xfe00) == 0xfc00
}

impl ObservabilityPlan {
    /// Build an `ObservabilityPlan` from a configuration value.
    ///
    /// # Errors
    /// Returns an error if the OTLP endpoint URL fails SSRF validation or if
    /// the exporter timeout is zero.
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let observability = &cfg.observability;
        let endpoint = observability
            .exporter
            .endpoint
            .clone()
            .or_else(|| observability.otlp_endpoint.clone());

        let exporter = match endpoint {
            Some(endpoint) if !endpoint.trim().is_empty() => {
                validate_http_endpoint(&endpoint)?;
                Exporter::Otlp {
                    endpoint,
                    protocol: observability.exporter.protocol,
                    headers: observability.exporter.headers.clone(),
                    timeout_ms: observability.exporter.timeout_ms,
                }
            }
            _ => {
                let exporter =
                    langfuse_otlp_exporter(&observability.langfuse)?.unwrap_or(Exporter::None);
                if let Exporter::Otlp { ref endpoint, .. } = exporter {
                    validate_http_endpoint(endpoint)?;
                }
                exporter
            }
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

pub fn genai_input_tokens_counter() -> &'static Counter<u64> {
    static COUNTER: OnceLock<Counter<u64>> = OnceLock::new();
    COUNTER.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_counter("gen_ai.usage.input_tokens")
            .with_description("Input tokens per GenAI semantic conventions")
            .build()
    })
}

pub fn genai_output_tokens_counter() -> &'static Counter<u64> {
    static COUNTER: OnceLock<Counter<u64>> = OnceLock::new();
    COUNTER.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_counter("gen_ai.usage.output_tokens")
            .with_description("Output tokens per GenAI semantic conventions")
            .build()
    })
}

/// Histogram of wall-clock time spent constructing a native model from a GGUF
/// or safetensors artifact. Attributes: `model.family`, `model.quant`,
/// `gpu.backend`. Recorded once per model load.
pub fn native_model_load_duration_ms() -> &'static Histogram<f64> {
    static H: OnceLock<Histogram<f64>> = OnceLock::new();
    H.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .f64_histogram("native.model.load.duration_ms")
            .with_description("Wall-clock time to load a native model artifact, in milliseconds")
            .with_unit("ms")
            .build()
    })
}

/// Histogram of throughput in tokens-per-second for native inference, split
/// by phase. Attributes: `model.family`, `phase` (`"prefill"` or `"generation"`).
pub fn native_model_tokens_per_second() -> &'static Histogram<f64> {
    static H: OnceLock<Histogram<f64>> = OnceLock::new();
    H.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .f64_histogram("native.model.tokens_per_second")
            .with_description("Native model throughput in tokens per second, split by phase")
            .with_unit("token/s")
            .build()
    })
}

/// Gauge of peak resident memory observed after a native model load completes.
/// Attribute: `model.family`. Sampled via `getrusage(RUSAGE_SELF).ru_maxrss`
/// (macOS reports bytes; Linux reports KiB).
pub fn native_model_peak_resident_mb() -> &'static Gauge<f64> {
    static G: OnceLock<Gauge<f64>> = OnceLock::new();
    G.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .f64_gauge("native.model.peak_resident_mb")
            .with_description("Peak resident memory after native model load, in MB")
            .with_unit("MB")
            .build()
    })
}

/// Sample the current process's peak resident memory in MB.
/// Returns `None` if the syscall is unavailable or fails.
///
/// macOS `ru_maxrss` is in bytes; Linux `ru_maxrss` is in kilobytes (KiB).
/// Other Unix platforms report bytes per BSD convention.
#[must_use]
pub fn process_peak_resident_mb() -> Option<f64> {
    #[cfg(unix)]
    {
        // SAFETY: getrusage is async-signal-safe; the rusage struct is POD;
        // we zero-init it before the call.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc != 0 {
            return None;
        }
        let maxrss = usage.ru_maxrss as f64;
        // Linux reports `ru_maxrss` in KiB; macOS and most BSDs report it in
        // bytes. Both cases normalise to megabytes here.
        #[cfg(target_os = "linux")]
        let mb = maxrss / 1024.0;
        #[cfg(not(target_os = "linux"))]
        let mb = maxrss / (1024.0 * 1024.0);
        Some(mb)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[cfg(test)]
mod native_observability_tests {
    use super::*;

    #[test]
    fn peak_resident_mb_returns_a_positive_value_on_unix() {
        // Sanity check the cross-platform sampler. The actual value depends on
        // the test process; we just assert it's positive and plausible (< 32 GB
        // for a unit-test process is a very loose upper bound).
        #[cfg(unix)]
        {
            let mb = process_peak_resident_mb().expect("getrusage should succeed on Unix");
            assert!(mb > 0.0, "expected positive RSS, got {mb}");
            assert!(
                mb < 32_768.0,
                "implausibly large RSS for a test process: {mb} MB"
            );
        }
        #[cfg(not(unix))]
        {
            assert_eq!(process_peak_resident_mb(), None);
        }
    }

    #[test]
    fn native_metric_instruments_share_a_single_global_meter() {
        // The OnceLock initialisation pattern means calling each accessor
        // twice should hand back the same instance — guards against accidental
        // duplicate metric registration that would silently fan-out exports.
        let load_a = native_model_load_duration_ms() as *const _;
        let load_b = native_model_load_duration_ms() as *const _;
        assert_eq!(
            load_a, load_b,
            "load duration histogram must be a OnceLock-cached singleton"
        );

        let tps_a = native_model_tokens_per_second() as *const _;
        let tps_b = native_model_tokens_per_second() as *const _;
        assert_eq!(
            tps_a, tps_b,
            "tokens/s histogram must be a OnceLock-cached singleton"
        );

        let rss_a = native_model_peak_resident_mb() as *const _;
        let rss_b = native_model_peak_resident_mb() as *const _;
        assert_eq!(
            rss_a, rss_b,
            "peak resident gauge must be a OnceLock-cached singleton"
        );
    }
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

        let exporter = langfuse_otlp_exporter(&cfg)
            .expect("langfuse exporter result should be Ok")
            .expect("langfuse exporter should be derived");
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
        assert!(langfuse_otlp_exporter(&LangfuseExporterConfig::default())
            .expect("disabled exporter should be Ok")
            .is_none());

        let missing_secret = LangfuseExporterConfig {
            enabled: true,
            host: Some("https://cloud.langfuse.com".to_string()),
            public_key: Some("pk-lf-abc".to_string()),
            secret_key: None,
        };
        assert!(langfuse_otlp_exporter(&missing_secret)
            .expect("missing secret should be Ok(None)")
            .is_none());

        let missing_host = LangfuseExporterConfig {
            enabled: true,
            host: None,
            public_key: Some("pk-lf-abc".to_string()),
            secret_key: Some("sk-lf-xyz".to_string()),
        };
        assert!(langfuse_otlp_exporter(&missing_host)
            .expect("missing host should be Ok(None)")
            .is_none());
    }

    #[test]
    fn langfuse_exporter_rejects_host_with_injected_path() {
        // A host value with an embedded path must not silently redirect OTLP traffic.
        let cfg = LangfuseExporterConfig {
            enabled: true,
            host: Some("https://legit.langfuse.com/injected-path".to_string()),
            public_key: Some("pk-lf-abc".to_string()),
            secret_key: Some("sk-lf-xyz".to_string()),
        };
        let exporter = langfuse_otlp_exporter(&cfg)
            .expect("parse should succeed")
            .expect("should produce an exporter");
        // The endpoint must use only the origin — the injected path must be stripped.
        match exporter {
            Exporter::Otlp { endpoint, .. } => {
                assert_eq!(endpoint, "https://legit.langfuse.com/api/public/otel");
            }
            Exporter::None => panic!("expected an OTLP exporter"),
        }
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
    fn inject_trace_context_is_noop_with_no_active_span() {
        let client = reqwest::Client::new();
        let builder = client.get("http://localhost/");
        let result = inject_trace_context(builder);
        // Should return without panic; we verify by successfully building the request.
        let request = result
            .build()
            .expect("request builds after inject_trace_context");
        assert!(!request.headers().contains_key("traceparent"));
    }

    #[test]
    fn install_plan_with_no_exporter_succeeds_without_otel_layer() {
        let plan = ObservabilityPlan {
            service_name: "test".to_string(),
            service_version: None,
            environment: None,
            traces_enabled: false,
            metrics_enabled: false,
            logs_enabled: false,
            resource_attributes: BTreeMap::new(),
            exporter: Exporter::None,
        };
        let runtime = TelemetryRuntime::from_plan(&plan).expect("from_plan with None exporter");
        assert!(runtime.tracer_provider.is_none());
        assert!(runtime.meter_provider.is_none());
        assert!(runtime.logger_provider.is_none());
    }

    #[test]
    fn validate_http_endpoint_accepts_http_and_https() {
        assert!(validate_http_endpoint("http://otel-collector:4318").is_ok());
        assert!(validate_http_endpoint("https://ingest.signoz.io:443").is_ok());
        assert!(validate_http_endpoint("https://collector.internal:4318/v1/traces").is_ok());
    }

    #[test]
    fn validate_http_endpoint_rejects_non_http_scheme() {
        let err = validate_http_endpoint("file:///etc/secrets/dump").unwrap_err();
        assert!(err.to_string().contains("file"), "error: {err}");

        assert!(validate_http_endpoint("grpc://collector:4317").is_err());
        assert!(validate_http_endpoint("ftp://collector:21").is_err());
    }

    #[test]
    fn validate_http_endpoint_rejects_missing_scheme() {
        assert!(validate_http_endpoint("collector.internal:4318").is_err());
        assert!(validate_http_endpoint("/etc/otel.sock").is_err());
    }

    #[test]
    fn validate_http_endpoint_blocks_aws_azure_gcp_imds() {
        // 169.254.169.254 is link-local and is now blocked by IP range check.
        let err = validate_http_endpoint("http://169.254.169.254/latest/meta-data").unwrap_err();
        assert!(err.to_string().contains("169.254.169.254"), "error: {err}");
    }

    #[test]
    fn validate_http_endpoint_blocks_gcp_metadata_hostname() {
        let err = validate_http_endpoint("http://metadata.google.internal/computeMetadata/v1/")
            .unwrap_err();
        assert!(
            err.to_string().contains("metadata.google.internal"),
            "error: {err}"
        );

        assert!(validate_http_endpoint("http://metadata.goog/").is_err());
        assert!(validate_http_endpoint("http://instance-data/").is_err());
    }

    #[test]
    fn validate_http_endpoint_block_is_case_insensitive() {
        assert!(validate_http_endpoint("http://METADATA.GOOG/").is_err());
        assert!(validate_http_endpoint("http://Metadata.Google.Internal/").is_err());
        assert!(validate_http_endpoint("HTTP://169.254.169.254/").is_err());
    }

    #[test]
    fn validate_http_endpoint_standalone_without_config() {
        assert!(validate_http_endpoint("http://169.254.169.254/").is_err());
        assert!(validate_http_endpoint("https://collector.internal:4318").is_ok());
    }

    #[test]
    fn validate_http_endpoint_handles_ipv6_bracketed_address() {
        // Loopback IPv6 must be rejected.
        assert!(validate_http_endpoint("http://[::1]:4318").is_err());
        // Non-special global unicast IPv6 must be accepted.
        assert!(validate_http_endpoint("http://[2001:db8::1]:4318").is_ok());
        // ULA (fc00::/7) must be rejected.
        assert!(validate_http_endpoint("http://[fd00:ec2::254]/latest").is_err());
        assert!(validate_http_endpoint("http://[fc00::1]/path").is_err());
    }

    #[test]
    fn validate_http_endpoint_rejects_userinfo_in_url() {
        // Userinfo can smuggle a blocked host as the "password" field.
        assert!(validate_http_endpoint("https://x:@169.254.169.254/meta").is_err());
        assert!(validate_http_endpoint("https://user:pass@internal.service/").is_err());
        assert!(validate_http_endpoint("http://admin@localhost/").is_err());
    }

    #[test]
    fn validate_http_endpoint_rejects_loopback_and_private_ips() {
        assert!(validate_http_endpoint("http://127.0.0.1:4318").is_err());
        assert!(validate_http_endpoint("http://[::1]:4318").is_err());
        assert!(validate_http_endpoint("http://0.0.0.0:4318").is_err());
        assert!(validate_http_endpoint("http://10.0.0.1:4318").is_err());
        assert!(validate_http_endpoint("http://192.168.1.1:4318").is_err());
        assert!(validate_http_endpoint("http://172.16.0.1:4318").is_err());
        assert!(validate_http_endpoint("http://100.100.100.200:4318").is_err()); // Alibaba IMDS
        assert!(validate_http_endpoint("http://169.254.1.1:4318").is_err()); // link-local (not just .169.254)
        assert!(validate_http_endpoint("http://[fd00:ec2::254]:4318").is_err());
        assert!(validate_http_endpoint("http://[fc00::1]:4318").is_err());
        assert!(validate_http_endpoint("http://[fe80::1]:4318").is_err()); // IPv6 link-local
    }

    #[test]
    fn from_config_rejects_blocked_langfuse_host() {
        use crate::config::{
            LangfuseExporterConfig, ObservabilityConfig, ObservabilityExporterConfig,
        };
        use std::collections::BTreeMap;

        // Build a minimal Config with a Langfuse host that resolves to a blocked address.
        // We use a host that starts with the metadata IP so the derived OTLP endpoint
        // will be rejected by validate_http_endpoint.
        let cfg = Config {
            observability: ObservabilityConfig {
                otlp_endpoint: None,
                service_name: None,
                service_version: None,
                environment: None,
                traces_enabled: true,
                metrics_enabled: true,
                logs_enabled: true,
                resource_attributes: BTreeMap::new(),
                exporter: ObservabilityExporterConfig {
                    endpoint: None,
                    ..ObservabilityExporterConfig::default()
                },
                langfuse: LangfuseExporterConfig {
                    enabled: true,
                    host: Some("http://169.254.169.254".to_string()),
                    public_key: Some("pk-lf-test".to_string()),
                    secret_key: Some("sk-lf-test".to_string()),
                },
                webhook: crate::config::WebhookExporterConfig::default(),
                gen_ai: crate::config::GenAiObservabilityConfig::default(),
            },
            ..Default::default()
        };

        let result = ObservabilityPlan::from_config(&cfg);
        assert!(
            result.is_err(),
            "expected SSRF block for langfuse host pointing to IMDS"
        );
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
