use crate::runtime::RuntimeBackend;
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
            models: vec![],
            quotas: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct RuntimeConfig {
    pub backend: RuntimeBackend,
    pub heartbeat_interval_seconds: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            backend: RuntimeBackend::CandleNative,
            heartbeat_interval_seconds: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct ClusterConfig {
    pub node_id: String,
    pub nodes: Vec<ClusterNodeConfig>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            node_id: "local".to_string(),
            nodes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct ClusterNodeConfig {
    pub id: String,
    pub base_url: String,
    pub roles: Vec<String>,
    pub model_aliases: Vec<String>,
}

impl Default for ClusterNodeConfig {
    fn default() -> Self {
        Self {
            id: "local".to_string(),
            base_url: "http://127.0.0.1:8765/v1".to_string(),
            roles: Vec::new(),
            model_aliases: Vec::new(),
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
    #[serde(default = "default_upstream_timeout_seconds")]
    pub upstream_timeout_seconds: u64,
    #[serde(default = "default_graceful_drain_seconds")]
    pub graceful_drain_seconds: u64,
    #[serde(default = "default_circuit_breaker_failures")]
    pub circuit_breaker_failures: u32,
    #[serde(default = "default_circuit_breaker_reset_seconds")]
    pub circuit_breaker_reset_seconds: u64,
    #[serde(default)]
    pub model_upstream_timeout_seconds: BTreeMap<String, u64>,
    #[serde(default)]
    pub cors_allowed_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8765,
            worker_base_port: 18765,
            llama_server: "llama-server".to_string(),
            context_size: 8192,
            upstream_timeout_seconds: default_upstream_timeout_seconds(),
            graceful_drain_seconds: default_graceful_drain_seconds(),
            circuit_breaker_failures: default_circuit_breaker_failures(),
            circuit_breaker_reset_seconds: default_circuit_breaker_reset_seconds(),
            model_upstream_timeout_seconds: BTreeMap::new(),
            cors_allowed_origins: Vec::new(),
        }
    }
}

fn default_upstream_timeout_seconds() -> u64 {
    300
}

fn default_graceful_drain_seconds() -> u64 {
    5
}

fn default_circuit_breaker_failures() -> u32 {
    3
}

fn default_circuit_breaker_reset_seconds() -> u64 {
    30
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct SecurityConfig {
    pub production: bool,
    #[serde(alias = "require_auth")]
    pub require_auth: bool,
    #[serde(alias = "bind_external")]
    pub bind_external: bool,
    #[serde(default = "default_auth_failure_limit_per_minute")]
    pub auth_failure_limit_per_minute: u32,
    #[serde(default, alias = "tls_termination")]
    pub tls_termination: TlsTerminationConfig,
    #[serde(alias = "api_keys")]
    pub api_keys: Vec<ApiKeyConfig>,
}

fn default_auth_failure_limit_per_minute() -> u32 {
    60
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            production: false,
            require_auth: false,
            bind_external: false,
            auth_failure_limit_per_minute: default_auth_failure_limit_per_minute(),
            tls_termination: TlsTerminationConfig::default(),
            api_keys: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct TlsTerminationConfig {
    pub enabled: bool,
    pub provider: Option<String>,
    pub evidence: Option<String>,
    #[serde(alias = "mtls")]
    pub m_tls: bool,
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
#[serde(default)]
pub struct StorageConfig {
    pub db_path: PathBuf,
    pub model_dir: PathBuf,
    #[serde(default = "default_storage_max_connections")]
    pub max_connections: u32,
    #[serde(default, alias = "database-url")]
    pub database_url: Option<String>,
    #[serde(default)]
    pub backend: Option<crate::storage::StorageBackend>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Self {
            db_path: PathBuf::from(format!("{home}/.local/share/rs-llmctl/llmctl.db")),
            model_dir: PathBuf::from(format!("{home}/.local/share/rs-llmctl/models")),
            max_connections: default_storage_max_connections(),
            database_url: None,
            backend: None,
        }
    }
}

fn default_storage_max_connections() -> u32 {
    5
}

impl StorageConfig {
    pub fn connection_plan(&self) -> Result<crate::storage::StorageConnectionPlan> {
        crate::storage::StorageConnectionPlan::from_config(self)
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct DataFabricConfig {
    pub enabled: bool,
    pub format: DataFabricFormat,
    pub schema_version: u32,
    pub output_dir: Option<PathBuf>,
    pub datasets: DataFabricDatasets,
}

impl Default for DataFabricConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            format: DataFabricFormat::Json,
            schema_version: 1,
            output_dir: None,
            datasets: DataFabricDatasets::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DataFabricFormat {
    #[default]
    Json,
    Jsonl,
    ArrowJson,
    ArrowIpc,
    Parquet,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct DataFabricDatasets {
    pub security: bool,
    pub observability: bool,
    pub usage: bool,
    pub user: bool,
    pub finops: bool,
    pub models: bool,
    pub drift: bool,
    pub audit: bool,
}

impl Default for DataFabricDatasets {
    fn default() -> Self {
        Self {
            security: true,
            observability: true,
            usage: true,
            user: true,
            finops: true,
            models: true,
            drift: true,
            audit: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
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
    let mut cfg =
        toml::from_str(&body).with_context(|| format!("parse config {}", path.display()))?;
    apply_legacy_runtime_backend(&body, &mut cfg);
    Ok(cfg)
}

fn apply_legacy_runtime_backend(body: &str, cfg: &mut Config) {
    let Ok(value) = toml::from_str::<toml::Value>(body) else {
        return;
    };
    if value.get("runtime").is_some() {
        return;
    }

    let has_legacy_llama_server = value
        .get("server")
        .and_then(toml::Value::as_table)
        .is_some_and(|server| {
            server.contains_key("llama_server") || server.contains_key("llama-server")
        });
    if has_legacy_llama_server {
        cfg.runtime.backend = RuntimeBackend::LlamaServer;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::RuntimeBackend;

    #[test]
    fn default_runtime_backend_is_candle_native() {
        let cfg = Config::default();

        assert_eq!(cfg.runtime.backend, RuntimeBackend::CandleNative);
        assert_eq!(cfg.runtime.heartbeat_interval_seconds, 30);
        assert_eq!(cfg.storage.max_connections, 5);
        assert_eq!(cfg.server.upstream_timeout_seconds, 300);
        assert_eq!(cfg.server.graceful_drain_seconds, 5);
        assert_eq!(cfg.server.circuit_breaker_failures, 3);
        assert_eq!(cfg.server.circuit_breaker_reset_seconds, 30);
        assert_eq!(cfg.security.auth_failure_limit_per_minute, 60);
    }

    #[test]
    fn parses_compatibility_llama_server_runtime_backend() {
        let cfg: Config = toml::from_str(
            r#"
[runtime]
backend = "llama-server"
heartbeat-interval-seconds = 10
"#,
        )
        .expect("parse config");

        assert_eq!(cfg.runtime.backend, RuntimeBackend::LlamaServer);
        assert_eq!(cfg.runtime.heartbeat_interval_seconds, 10);
    }

    #[tokio::test]
    async fn load_treats_legacy_llama_server_field_as_compatibility_backend() {
        let path = std::env::temp_dir().join(format!(
            "rs-llmctl-runtime-compat-{}.toml",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"
[server]
host = "127.0.0.1"
port = 8765
worker_base_port = 18765
llama_server = "/usr/local/bin/llama-server"
context_size = 4096
"#,
        )
        .await
        .expect("write config");

        let cfg = load(&path).await.expect("load config");
        let _ = fs::remove_file(&path).await;

        assert_eq!(cfg.runtime.backend, RuntimeBackend::LlamaServer);
    }
}
