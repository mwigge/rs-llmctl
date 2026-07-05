//! Data-fabric, audit, external-provider, model, and quota configuration.
use super::*;

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
