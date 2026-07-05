//! Model catalog types, install request/plan DTOs, and the built-in registry.
use super::*;

pub(crate) const HF_BASE_URL: &str = "https://huggingface.co";
pub(crate) const MODEL_DOWNLOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ModelSource {
    LocalPath {
        path: PathBuf,
    },
    DirectUrl {
        url: String,
    },
    HuggingFace {
        repo: String,
        filename: String,
        #[serde(default = "default_revision")]
        revision: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInstallRequest {
    pub alias: String,
    pub source: ModelSource,
    pub cache_dir: PathBuf,
    #[serde(default)]
    pub copy_to_cache: bool,
    #[serde(default)]
    pub expected_sha256: Option<String>,
    #[serde(default = "default_role")]
    pub role: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub weight: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledModel {
    pub alias: String,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub source: ModelSource,
    pub source_kind: ModelInstallSourceKind,
    pub verification: ModelInstallVerification,
    pub config: ModelConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelInstallSourceKind {
    Local,
    Offline,
    Download,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInstallVerification {
    pub sha256_required: bool,
    pub expected_sha256: Option<String>,
    pub actual_sha256: Option<String>,
    pub verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInstallPlan {
    pub alias: String,
    pub source_kind: ModelInstallSourceKind,
    pub source_url: Option<String>,
    pub cache_dir: PathBuf,
    pub verification: ModelInstallVerification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineInstallManifest {
    pub models: Vec<OfflineManifestModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineManifestModel {
    pub alias: String,
    pub path: PathBuf,
    #[serde(default = "default_role")]
    pub role: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub weight: u32,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: &'static str,
    pub alias: &'static str,
    pub repo: &'static str,
    pub filename: &'static str,
    pub revision: &'static str,
    pub role: &'static str,
}

fn default_revision() -> String {
    "main".to_string()
}

fn default_role() -> String {
    "chat".to_string()
}

static BUILTIN_CATALOG: LazyLock<Vec<CatalogModel>> = LazyLock::new(|| {
    vec![
        CatalogModel {
            id: "qwen2.5-7b-instruct-q4-k-m",
            alias: "qwen2.5-7b",
            repo: "Qwen/Qwen2.5-7B-Instruct-GGUF",
            filename: "qwen2.5-7b-instruct-q4_k_m.gguf",
            revision: "main",
            role: "chat",
        },
        CatalogModel {
            id: "llama-3.2-3b-instruct-q4-k-m",
            alias: "llama3.2-3b",
            repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            revision: "main",
            role: "chat",
        },
        CatalogModel {
            id: "mistral-7b-instruct-v0.3-q4-k-m",
            alias: "mistral-7b",
            repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            filename: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            revision: "main",
            role: "chat",
        },
    ]
});

pub fn builtin_catalog() -> Vec<CatalogModel> {
    BUILTIN_CATALOG.clone()
}

pub fn catalog_model(id_or_alias: &str) -> Option<CatalogModel> {
    BUILTIN_CATALOG
        .iter()
        .find(|model| model.id == id_or_alias || model.alias == id_or_alias)
        .cloned()
}
