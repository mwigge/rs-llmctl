use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokio::fs;

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
    pub storage: StorageConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    #[serde(default)]
    pub audit: AuditConfig,
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
            storage: StorageConfig::default(),
            observability: ObservabilityConfig::default(),
            audit: AuditConfig::default(),
            models: vec![],
            quotas: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub worker_base_port: u16,
    pub llama_server: String,
    pub context_size: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8765,
            worker_base_port: 18765,
            llama_server: "llama-server".to_string(),
            context_size: 8192,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct SecurityConfig {
    pub production: bool,
    #[serde(alias = "require_auth")]
    pub require_auth: bool,
    #[serde(alias = "bind_external")]
    pub bind_external: bool,
    #[serde(alias = "api_keys")]
    pub api_keys: Vec<ApiKeyConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKeyConfig {
    pub id: String,
    pub sha256: String,
    pub subject: String,
    pub team: String,
    pub scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub budget: f64,
    pub cpu_only: bool,
    pub gpu_vendor: String,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            budget: 0.80,
            cpu_only: false,
            gpu_vendor: "auto".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub db_path: PathBuf,
    pub model_dir: PathBuf,
}

impl Default for StorageConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Self {
            db_path: PathBuf::from(format!("{home}/.local/share/rs-llmctl/llmctl.db")),
            model_dir: PathBuf::from(format!("{home}/.local/share/rs-llmctl/models")),
        }
    }
}

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
pub struct AuditConfig {
    pub retention_days: u32,
    pub report_directory: Option<PathBuf>,
    pub report_formats: Vec<String>,
    pub monthly_reports: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            retention_days: 365,
            report_directory: None,
            report_formats: vec!["json".to_string()],
            monthly_reports: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConfig {
    pub alias: String,
    pub path: PathBuf,
    #[serde(default = "default_role")]
    pub role: String,
    #[serde(default)]
    pub weight: u32,
}

fn default_role() -> String {
    "chat".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaConfig {
    pub subject: String,
    pub team: String,
    pub requests_per_minute: u32,
    pub tokens_per_day: u64,
    pub max_concurrency: u32,
    pub allowed_models: Vec<String>,
}

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
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let body = toml::to_string_pretty(cfg)?;
    fs::write(path, body).await?;
    Ok(())
}

pub fn validate_production_security(cfg: &Config) -> Result<()> {
    crate::security::validate_production_security(cfg)
}
