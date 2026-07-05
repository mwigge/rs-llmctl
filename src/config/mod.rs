use crate::guardrails::GuardrailsConfig;
use crate::runtime::RuntimeBackend;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokio::fs;

pub fn is_external_host(host: &str) -> bool {
    !matches!(host.trim(), "127.0.0.1" | "localhost" | "::1")
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    #[default]
    Single,
    ColdSwap,
    HotSwap,
    Weighted,
    Fallback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub mode: Mode,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub security: SecurityConfig,
    #[serde(default)]
    pub resources: ResourceConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    #[serde(default)]
    pub sse: SseConfig,
    #[serde(default)]
    pub log: LogConfig,
    #[serde(default)]
    pub events: EventConfig,
    #[serde(default, rename = "data-fabric", alias = "data_fabric")]
    pub data_fabric: DataFabricConfig,
    #[serde(default)]
    pub audit: AuditConfig,
    #[serde(default)]
    pub guardrails: GuardrailsConfig,
    #[serde(default, rename = "external-providers", alias = "external_providers")]
    pub external_providers: ExternalProvidersConfig,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub quotas: Vec<QuotaConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Single,
            server: ServerConfig::default(),
            security: SecurityConfig::default(),
            resources: ResourceConfig::default(),
            runtime: RuntimeConfig::default(),
            cluster: ClusterConfig::default(),
            storage: StorageConfig::default(),
            observability: ObservabilityConfig::default(),
            sse: SseConfig::default(),
            log: LogConfig::default(),
            events: EventConfig::default(),
            data_fabric: DataFabricConfig::default(),
            audit: AuditConfig::default(),
            guardrails: GuardrailsConfig::default(),
            external_providers: ExternalProvidersConfig::default(),
            models: vec![],
            quotas: vec![],
        }
    }
}

mod runtime_cluster;
pub use runtime_cluster::*;
mod server;
pub use server::*;
mod security;
pub use security::*;
mod resources;
pub use resources::*;
mod observability;
pub use observability::*;
mod providers;
pub use providers::*;

#[cfg(test)]
mod tests;

pub fn default_config_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(format!("{home}/.config/rs-llmctl/config.toml"))
}

pub async fn load(path: &Path) -> Result<Config> {
    let body = fs::read_to_string(path)
        .await
        .with_context(|| format!("read config {}", path.display()))?;
    let cfg = toml::from_str(&body).with_context(|| format!("parse config {}", path.display()))?;
    Ok(cfg)
}

pub async fn save(path: &Path, cfg: &Config) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent).await?;
    }
    let body = toml::to_string_pretty(cfg)?;

    // Atomic write: serialize into a sibling temp file in the same directory,
    // fsync it, then rename it over the target. `rename(2)` within a single
    // directory is atomic, so a crash or full disk mid-write cannot leave a
    // truncated live config.toml — which holds the API-key hash table and the
    // security posture. A plain `fs::write` (truncate-then-write) could.
    let dir = parent.unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config.toml".to_string());
    let unique = format!(
        "{}-{}",
        std::process::id(),
        SAVE_TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let tmp_path = dir.join(format!(".{file_name}.tmp-{unique}"));

    let write_result = async {
        let mut file = fs::File::create(&tmp_path).await?;
        file.write_all(body.as_bytes()).await?;
        file.sync_all().await?;
        drop(file);
        fs::rename(&tmp_path, path).await?;
        Ok::<(), std::io::Error>(())
    }
    .await;

    if write_result.is_err() {
        // Best-effort cleanup so a failed save doesn't leak temp files.
        let _ = fs::remove_file(&tmp_path).await;
    }
    write_result.with_context(|| format!("atomically write config {}", path.display()))?;
    Ok(())
}

static SAVE_TEMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Validate that a configuration is safe enough for production or external bind.
///
/// This is the public configuration-layer entrypoint used by the CLI, tests,
/// release checks, and embedders. It delegates to [`crate::security`] and
/// verifies the active security posture, including hashed API keys, supported
/// scopes, key lifecycle metadata, plaintext-secret rejection, native TLS
/// shape, external-provider egress constraints, audit retention, monthly
/// reports, and OpenTelemetry exporter coverage when production/external
/// serving is enabled.
pub fn validate_production_security(cfg: &Config) -> Result<()> {
    crate::security::validate_production_security(cfg)
}
