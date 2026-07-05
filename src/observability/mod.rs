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

mod runtime;
pub use runtime::*;
mod events;
pub use events::*;
mod metrics;
pub use metrics::*;
mod redaction;
pub use redaction::*;
mod sse;
pub use sse::*;

#[cfg(test)]
mod native_observability_tests;

#[cfg(test)]
mod tests;
