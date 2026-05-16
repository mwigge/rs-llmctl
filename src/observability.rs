use crate::config::{Config, OtlpProtocol};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

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
    ResourceSnapshot,
    DriftObservation,
    ModelInstallVerification,
}

impl TelemetryEventName {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RequestRouting => "llmctl.request.routing",
            Self::QuotaDecision => "llmctl.quota.decision",
            Self::WorkerLifecycle => "llmctl.worker.lifecycle",
            Self::ResourceSnapshot => "llmctl.resource.snapshot",
            Self::DriftObservation => "llmctl.drift.observation",
            Self::ModelInstallVerification => "llmctl.model.install.verification",
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
            "llmctl.resource.snapshot" => Ok(Self::ResourceSnapshot),
            "llmctl.drift.observation" => Ok(Self::DriftObservation),
            "llmctl.model.install.verification" => Ok(Self::ModelInstallVerification),
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
