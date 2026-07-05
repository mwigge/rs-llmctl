use super::*;

static GLOBAL_TRACER_PROVIDER_REGISTERED: OnceLock<()> = OnceLock::new();
static GLOBAL_METER_PROVIDER_REGISTERED: OnceLock<()> = OnceLock::new();
static TRACING_SUBSCRIBER_INITIALIZED: OnceLock<()> = OnceLock::new();
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
    pub(super) tracer_provider: Option<SdkTracerProvider>,
    pub(super) meter_provider: Option<SdkMeterProvider>,
    pub(super) logger_provider: Option<SdkLoggerProvider>,
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
