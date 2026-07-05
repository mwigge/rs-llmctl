//! Observability, exporters, logging, SSE, and event configuration.
use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct ObservabilityConfig {
    /// Deprecated shorthand retained for older configs; prefer exporter.endpoint.
    pub otlp_endpoint: Option<String>,
    pub service_name: Option<String>,
    pub service_version: Option<String>,
    pub environment: Option<String>,
    pub traces_enabled: bool,
    pub metrics_enabled: bool,
    pub logs_enabled: bool,
    pub resource_attributes: BTreeMap<String, String>,
    pub exporter: ObservabilityExporterConfig,
    /// Derives an OTLP exporter targeting Langfuse's ingestion endpoint from
    /// project keys, when no explicit `exporter.endpoint`/`otlp_endpoint` is set.
    pub langfuse: LangfuseExporterConfig,
    /// Fire-and-forget HTTP callback fired with usage/lineage metadata after
    /// every completion — for ecosystems without an OTLP receiver.
    pub webhook: WebhookExporterConfig,
    /// Controls gen_ai semantic-convention data captured in OTel spans.
    pub gen_ai: GenAiObservabilityConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            otlp_endpoint: None,
            service_name: None,
            service_version: None,
            environment: None,
            traces_enabled: true,
            metrics_enabled: true,
            logs_enabled: true,
            resource_attributes: BTreeMap::new(),
            exporter: ObservabilityExporterConfig::default(),
            langfuse: LangfuseExporterConfig::default(),
            webhook: WebhookExporterConfig::default(),
            gen_ai: GenAiObservabilityConfig::default(),
        }
    }
}

/// Langfuse project credentials. When `enabled` and both keys are present,
/// these are translated into an OTLP/HTTP exporter targeting Langfuse's
/// `/api/public/otel` ingestion path with HTTP Basic auth — see
/// [`crate::observability::langfuse_otlp_exporter`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct LangfuseExporterConfig {
    pub enabled: bool,
    /// Langfuse host, e.g. `https://cloud.langfuse.com` or a self-hosted URL.
    pub host: Option<String>,
    pub public_key: Option<String>,
    pub secret_key: Option<String>,
}

/// Fire-and-forget webhook delivered after every completion, carrying the
/// same usage/lineage metadata recorded in the audit trail — for ecosystems
/// (chat ops, custom dashboards, ticketing) that consume callbacks rather
/// than OTLP.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct WebhookExporterConfig {
    pub enabled: bool,
    pub url: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub timeout_ms: u64,
}

impl Default for WebhookExporterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            url: None,
            headers: BTreeMap::new(),
            timeout_ms: 5_000,
        }
    }
}

/// Controls which gen_ai semantic-convention data is captured in OTel spans.
///
/// Sensitive prompt content can be suppressed per-environment so traces never
/// carry user input outside the host.
// All four fields are independent on/off knobs for different observability
// categories; a bitfield would lose TOML readability and serde round-trip
// clarity, so a struct of bools is the right shape here.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct GenAiObservabilityConfig {
    /// When `true` (default), `gen_ai.user.message` and `gen_ai.system.message`
    /// span events include the actual message body.  Set to `false` to emit
    /// `[REDACTED]` instead — required in environments where prompt content
    /// must not leave the host.
    pub capture_message_content: bool,
    /// Emit a `gen_ai.token` span event for every decoded token.  Defaults to
    /// `false` — enable only in development; production traces will be very
    /// high volume if this is on.
    pub token_events: bool,
    /// Record a per-token logit entropy histogram.  Defaults to `false` —
    /// computing entropy over the full vocabulary adds latency on every token.
    pub logit_entropy: bool,
    /// Emit `gen_ai.thinking.started` and `gen_ai.thinking.ended` span events
    /// when the model enters and exits a thinking phase.  Defaults to `true`.
    pub thinking_phase_events: bool,
}

impl Default for GenAiObservabilityConfig {
    fn default() -> Self {
        Self {
            capture_message_content: true,
            token_events: false,
            logit_entropy: false,
            thinking_phase_events: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct ObservabilityExporterConfig {
    pub endpoint: Option<String>,
    pub protocol: OtlpProtocol,
    pub headers: BTreeMap<String, String>,
    pub timeout_ms: u64,
}

impl Default for ObservabilityExporterConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            protocol: OtlpProtocol::HttpProtobuf,
            headers: BTreeMap::new(),
            timeout_ms: 5_000,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OtlpProtocol {
    #[default]
    HttpProtobuf,
    Grpc,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct SseConfig {
    pub enabled: bool,
    pub heartbeat_seconds: u64,
    pub max_stream_seconds: u64,
}

impl Default for SseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            heartbeat_seconds: 15,
            max_stream_seconds: 3_600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct LogConfig {
    pub format: LogFormat,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            format: LogFormat::Pretty,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct EventConfig {
    pub format: EventFormat,
    pub schema_version: u32,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            format: EventFormat::Json,
            schema_version: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum EventFormat {
    #[default]
    Json,
    Jsonl,
    CloudEvents,
}
