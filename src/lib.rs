//! # rs-llmctl
//!
//! `rs-llmctl` is a Rust-native local LLM control plane and serving runtime.
//! It provides OpenAI-compatible HTTP endpoints, Candle-native model execution
//! contracts, model lifecycle management, quota enforcement, audit/reporting,
//! data-fabric exports, and OpenTelemetry instrumentation.
//!
//! The library modules are intentionally split by operator concern:
//! configuration and production validation live in [`config`] and [`security`],
//! runtime/model execution contracts live in [`runtime`] and [`native`], serving
//! lives in [`server`], and persistence/reporting live in [`storage`],
//! [`reporting`], [`contracts`], and [`data_fabric`]. The `llmctl` binary builds
//! on these modules for the CLI and daemon entrypoint.
//!
//! Production security validation is exposed through
//! [`config::validate_production_security`]. It checks hashed API-key material,
//! active auth posture, TLS evidence, audit retention, monthly reporting, and
//! observability requirements before an externally bound production service can
//! start.

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
mod gemma4_gguf;

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub mod tier;

pub mod amd;
pub mod audit;
pub mod config;
pub mod contracts;
pub mod data_fabric;
pub mod guardrails;
pub mod integrations;
pub mod model;
pub mod native;
pub mod observability;
pub mod policy_sign;
pub mod profiles;
pub mod quota;
pub mod rag;
pub mod readiness;
pub mod reporting;
pub mod resources;
pub mod runtime;
pub mod security;
pub mod server;
pub mod storage;
pub mod worker;

pub const SERVICE_NAME: &str = "rs-llmctl";
