//! Resource budget and storage configuration.
use super::*;

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
