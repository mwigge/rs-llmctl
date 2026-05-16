use crate::config::{Config, OtlpProtocol};
use anyhow::Result;
use std::collections::BTreeMap;

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
