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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct RuntimeConfig {
    pub backend: RuntimeBackend,
    pub heartbeat_interval_seconds: u64,
    pub embeddings: NativeEmbeddingRuntimeConfig,
    pub scheduler: NativeSchedulerRuntimeConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            backend: RuntimeBackend::CandleNative,
            heartbeat_interval_seconds: 30,
            embeddings: NativeEmbeddingRuntimeConfig::default(),
            scheduler: NativeSchedulerRuntimeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct NativeSchedulerRuntimeConfig {
    pub max_concurrent_requests: usize,
    pub max_queued_requests: usize,
    pub max_batch_size: usize,
    pub max_batch_wait_ms: u64,
    pub kv_cache_budget_bytes: u64,
}

impl Default for NativeSchedulerRuntimeConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 1,
            max_queued_requests: 127,
            max_batch_size: 1,
            max_batch_wait_ms: 0,
            kv_cache_budget_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "kebab-case")]
pub struct NativeEmbeddingRuntimeConfig {
    pub mode: NativeEmbeddingMode,
    #[serde(alias = "model_alias")]
    pub model_alias: Option<String>,
}

impl Default for NativeEmbeddingRuntimeConfig {
    fn default() -> Self {
        Self {
            mode: NativeEmbeddingMode::Semantic,
            model_alias: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum NativeEmbeddingMode {
    #[default]
    Semantic,
    DevFallback,
}

impl NativeEmbeddingMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::DevFallback => "dev-fallback",
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
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub tls: ServerTlsConfig,
    pub worker_base_port: u16,
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
            tls: ServerTlsConfig::default(),
            worker_base_port: 18765,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ServerTlsConfig {
    pub enabled: bool,
    #[serde(alias = "cert_path")]
    pub cert_path: Option<PathBuf>,
    #[serde(alias = "key_path")]
    pub key_path: Option<PathBuf>,
    #[serde(alias = "require_client_cert")]
    pub require_client_cert: bool,
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
    #[serde(default)]
    pub trusted_proxies: Vec<String>,
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
            trusted_proxies: Vec::new(),
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
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ApiKeyConfig {
    pub id: String,
    pub sha256: String,
    pub subject: String,
    pub team: String,
    pub scopes: Vec<String>,
    #[serde(alias = "created_at")]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(alias = "expires_at")]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(alias = "rotated_at")]
    pub rotated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub owner: Option<String>,
    pub purpose: Option<String>,
    #[serde(alias = "last_four")]
    pub last_four: Option<String>,
    pub fingerprint: Option<String>,
    #[serde(default = "default_api_key_status")]
    pub status: String,
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            sha256: String::new(),
            subject: String::new(),
            team: String::new(),
            scopes: Vec::new(),
            created_at: None,
            expires_at: None,
            rotated_at: None,
            owner: None,
            purpose: None,
            last_four: None,
            fingerprint: None,
            status: default_api_key_status(),
        }
    }
}

fn default_api_key_status() -> String {
    "active".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub budget: f64,
    pub cpu_only: bool,
    pub gpu_vendor: String,
    /// Path to a HIP-enabled `llama-server` binary used when `gpu_vendor` is
    /// `"amd"`. When absent, rs-llmctl searches `~/.local/bin`, `/usr/local/bin`,
    /// and `/usr/bin` for `llama-server`. See ADR-0001 option (b).
    pub llama_server_bin: Option<std::path::PathBuf>,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            budget: 0.80,
            cpu_only: false,
            gpu_vendor: "auto".to_string(),
            llama_server_bin: None,
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
    /// Derives an OTLP exporter targeting Langfuse's ingestion endpoint from
    /// project keys, when no explicit `exporter.endpoint`/`otlp_endpoint` is set.
    pub langfuse: LangfuseExporterConfig,
    /// Fire-and-forget HTTP callback fired with usage/lineage metadata after
    /// every completion — for ecosystems without an OTLP receiver.
    pub webhook: WebhookExporterConfig,
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
    pub lineage: bool,
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
            lineage: true,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ExternalProvidersConfig {
    pub enabled: bool,
    pub providers: Vec<ExternalProviderConfig>,
    pub routes: Vec<ExternalProviderRouteConfig>,
}

impl ExternalProvidersConfig {
    pub fn provider(&self, id: &str) -> Option<&ExternalProviderConfig> {
        self.providers.iter().find(|provider| provider.id == id)
    }

    pub fn route_for_model(&self, alias: &str) -> Option<&ExternalProviderRouteConfig> {
        self.routes.iter().find(|route| route.model_alias == alias)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ExternalProviderConfig {
    pub id: String,
    pub kind: ExternalProviderKind,
    pub base_url: String,
    pub api_key_env: String,
}

impl Default for ExternalProviderConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: ExternalProviderKind::OpenAiCompatible,
            base_url: String::new(),
            api_key_env: String::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ExternalProviderKind {
    #[default]
    OpenAiCompatible,
    VertexAi,
    OpenRouter,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ExternalProviderRouteConfig {
    pub model_alias: String,
    pub provider: String,
    pub provider_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConfig {
    pub alias: String,
    pub path: PathBuf,
    #[serde(default = "default_role")]
    pub role: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default = "default_model_weight")]
    pub weight: u32,
}

fn default_role() -> String {
    "chat".to_string()
}

fn default_model_weight() -> u32 {
    1
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            alias: String::new(),
            path: PathBuf::new(),
            role: default_role(),
            family: None,
            weight: default_model_weight(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::RuntimeBackend;

    #[test]
    fn default_runtime_backend_is_candle_native() {
        let cfg = Config::default();

        assert_eq!(cfg.runtime.backend, RuntimeBackend::CandleNative);
        assert_eq!(cfg.runtime.heartbeat_interval_seconds, 30);
        assert_eq!(cfg.runtime.embeddings.mode, NativeEmbeddingMode::Semantic);
        assert_eq!(cfg.runtime.embeddings.model_alias, None);
        assert_eq!(cfg.storage.max_connections, 5);
        assert_eq!(cfg.server.upstream_timeout_seconds, 300);
        assert_eq!(cfg.server.graceful_drain_seconds, 5);
        assert_eq!(cfg.server.circuit_breaker_failures, 3);
        assert_eq!(cfg.server.circuit_breaker_reset_seconds, 30);
        assert_eq!(cfg.security.auth_failure_limit_per_minute, 60);
        assert!(!cfg.external_providers.enabled);
        assert!(cfg.external_providers.providers.is_empty());
    }

    #[test]
    fn parses_external_provider_env_key_references_without_inline_secrets() {
        let cfg: Config = toml::from_str(
            r#"
[external-providers]
enabled = true

[[external-providers.providers]]
id = "openai"
kind = "open-ai-compatible"
base-url = "https://api.openai.example/v1"
api-key-env = "OPENAI_API_KEY"

[[external-providers.routes]]
model-alias = "gpt-proxy"
provider = "openai"
provider-model = "gpt-4o-mini"

[[models]]
alias = "gpt-proxy"
path = "/models/remote-placeholder"
role = "chat"
"#,
        )
        .expect("parse external provider config");

        assert!(cfg.external_providers.enabled);
        let provider = cfg.external_providers.provider("openai").expect("provider");
        assert_eq!(provider.kind, ExternalProviderKind::OpenAiCompatible);
        assert_eq!(provider.api_key_env, "OPENAI_API_KEY");
        let route = cfg
            .external_providers
            .route_for_model("gpt-proxy")
            .expect("provider route");
        assert_eq!(route.provider, "openai");
        assert_eq!(route.provider_model.as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn parses_native_embedding_runtime_contract() {
        let cfg: Config = toml::from_str(
            r#"
[runtime]
backend = "candle-native"

[runtime.embeddings]
mode = "semantic"
model-alias = "embed-prod"
"#,
        )
        .expect("parse config");

        assert_eq!(cfg.runtime.embeddings.mode, NativeEmbeddingMode::Semantic);
        assert_eq!(
            cfg.runtime.embeddings.model_alias.as_deref(),
            Some("embed-prod")
        );

        let cfg: Config = toml::from_str(
            r#"
[runtime.embeddings]
mode = "dev-fallback"
"#,
        )
        .expect("parse dev fallback config");

        assert_eq!(
            cfg.runtime.embeddings.mode,
            NativeEmbeddingMode::DevFallback
        );
        assert_eq!(cfg.runtime.embeddings.model_alias, None);
    }

    #[test]
    fn parses_server_tls_config() {
        let cfg: Config = toml::from_str(
            r#"
[server.tls]
enabled = true
cert-path = "/etc/llmctl/tls/server.crt"
key-path = "/etc/llmctl/tls/server.key"
require-client-cert = false
"#,
        )
        .expect("parse server tls config");

        assert!(cfg.server.tls.enabled);
        assert_eq!(
            cfg.server.tls.cert_path.as_deref(),
            Some(Path::new("/etc/llmctl/tls/server.crt"))
        );
        assert_eq!(
            cfg.server.tls.key_path.as_deref(),
            Some(Path::new("/etc/llmctl/tls/server.key"))
        );
        assert!(!cfg.server.tls.require_client_cert);
    }

    #[test]
    fn api_key_metadata_defaults_for_legacy_config() {
        let key: ApiKeyConfig = toml::from_str(
            r#"
id = "platform-chat"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
subject = "platform"
team = "infra"
scopes = ["chat"]
"#,
        )
        .expect("parse api key");

        assert_eq!(key.created_at, None);
        assert_eq!(key.expires_at, None);
        assert_eq!(key.rotated_at, None);
        assert_eq!(key.owner, None);
        assert_eq!(key.purpose, None);
        assert_eq!(key.last_four, None);
        assert_eq!(key.fingerprint, None);
        assert_eq!(key.status, "active");
    }

    #[test]
    fn api_key_metadata_accepts_kebab_case_config_fields() {
        let key: ApiKeyConfig = toml::from_str(
            r#"
id = "platform-chat"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
subject = "platform"
team = "infra"
scopes = ["chat"]
created-at = "2026-01-02T03:04:05Z"
expires-at = "2027-01-02T03:04:05Z"
rotated-at = "2026-02-02T03:04:05Z"
owner = "platform"
purpose = "chat serving"
last-four = "cdef"
fingerprint = "sha256:0123456789abcdef"
status = "retiring"
"#,
        )
        .expect("parse api key metadata");

        assert_eq!(
            key.created_at.expect("created_at").to_rfc3339(),
            "2026-01-02T03:04:05+00:00"
        );
        assert_eq!(
            key.expires_at.expect("expires_at").to_rfc3339(),
            "2027-01-02T03:04:05+00:00"
        );
        assert_eq!(
            key.rotated_at.expect("rotated_at").to_rfc3339(),
            "2026-02-02T03:04:05+00:00"
        );
        assert_eq!(key.owner.as_deref(), Some("platform"));
        assert_eq!(key.purpose.as_deref(), Some("chat serving"));
        assert_eq!(key.last_four.as_deref(), Some("cdef"));
        assert_eq!(key.fingerprint.as_deref(), Some("sha256:0123456789abcdef"));
        assert_eq!(key.status, "retiring");
    }
}
