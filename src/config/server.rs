//! HTTP server and server-TLS configuration.
use super::*;

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
