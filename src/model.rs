use crate::config::ModelConfig;
use crate::observability::{
    emit_runtime_telemetry, RuntimeTelemetryEvent, TelemetryEventName, TelemetrySignal,
};
use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use futures_util::StreamExt;
use reqwest::header::{HeaderValue, RANGE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const HF_BASE_URL: &str = "https://huggingface.co";
const MODEL_DOWNLOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

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

pub fn huggingface_download_url(repo: &str, filename: &str, revision: &str) -> Result<String> {
    ensure_relative_component(repo, "repo")?;
    ensure_relative_component(filename, "filename")?;
    ensure_relative_component(revision, "revision")?;
    Ok(format!(
        "{HF_BASE_URL}/{repo}/resolve/{revision}/{filename}?download=true"
    ))
}

pub fn source_url(source: &ModelSource) -> Result<Option<String>> {
    match source {
        ModelSource::LocalPath { .. } => Ok(None),
        ModelSource::DirectUrl { url } => {
            validate_direct_download_url(url)?;
            Ok(Some(url.clone()))
        }
        ModelSource::HuggingFace {
            repo,
            filename,
            revision,
        } => huggingface_download_url(repo, filename, revision).map(Some),
    }
}

pub fn install_plan(req: &ModelInstallRequest) -> Result<ModelInstallPlan> {
    validate_alias(&req.alias)?;
    let source_kind = match &req.source {
        ModelSource::LocalPath { .. } => ModelInstallSourceKind::Local,
        ModelSource::DirectUrl { .. } | ModelSource::HuggingFace { .. } => {
            ModelInstallSourceKind::Download
        }
    };
    let expected_sha256 = req
        .expected_sha256
        .as_ref()
        .map(|expected| normalized_sha256(expected))
        .transpose()?;
    let sha256_required = source_kind == ModelInstallSourceKind::Download;
    if sha256_required && expected_sha256.is_none() {
        bail!("expected_sha256 is required for downloaded model sources");
    }

    Ok(ModelInstallPlan {
        alias: req.alias.clone(),
        source_kind,
        source_url: source_url(&req.source)?,
        cache_dir: req.cache_dir.clone(),
        verification: ModelInstallVerification {
            sha256_required,
            expected_sha256,
            actual_sha256: None,
            verified: false,
        },
    })
}

pub async fn install_model(req: &ModelInstallRequest) -> Result<InstalledModel> {
    let plan = install_plan(req)?;
    fs::create_dir_all(&req.cache_dir)
        .await
        .with_context(|| format!("create model cache {}", req.cache_dir.display()))?;

    let path = match &req.source {
        ModelSource::LocalPath { path } => {
            register_local_model(path, &req.cache_dir, req.copy_to_cache).await?
        }
        ModelSource::DirectUrl { url } => {
            let expected = plan
                .verification
                .expected_sha256
                .as_deref()
                .ok_or_else(|| {
                    anyhow!("download plan for '{}' is missing expected sha256 — refusing unverified download", req.alias)
                })?;
            download_model(url, &req.cache_dir, &req.alias, expected).await?
        }
        ModelSource::HuggingFace {
            repo,
            filename,
            revision,
        } => {
            let url = huggingface_download_url(repo, filename, revision)?;
            let expected = plan
                .verification
                .expected_sha256
                .as_deref()
                .ok_or_else(|| {
                    anyhow!("download plan for '{}' is missing expected sha256 — refusing unverified download", req.alias)
                })?;
            download_model(&url, &req.cache_dir, filename, expected).await?
        }
    };

    ensure_supported_artifact_path(&path).await?;
    let sha256 = sha256_model_artifact(&path).await?;
    if let Some(expected) = &plan.verification.expected_sha256 {
        if sha256 != *expected {
            bail!(
                "sha256 mismatch for {}: expected {expected}, got {sha256}",
                path.display()
            );
        }
    }
    let bytes = model_artifact_len(&path).await?;
    let config = ModelConfig {
        alias: req.alias.clone(),
        path: path.clone(),
        role: req.role.clone(),
        family: req.family.clone(),
        weight: req.weight,
    };
    let verification = ModelInstallVerification {
        actual_sha256: Some(sha256.clone()),
        verified: plan.verification.expected_sha256.is_some(),
        ..plan.verification
    };
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::ModelInstallVerification,
        Utc::now(),
        BTreeMap::from([
            (
                "llmctl.model.alias".to_string(),
                serde_json::json!(req.alias.as_str()),
            ),
            (
                "llmctl.model.verified".to_string(),
                serde_json::json!(verification.verified),
            ),
            ("llmctl.model.bytes".to_string(), serde_json::json!(bytes)),
        ]),
    ));

    Ok(InstalledModel {
        alias: req.alias.clone(),
        path,
        sha256,
        bytes,
        source: req.source.clone(),
        source_kind: plan.source_kind,
        verification,
        config,
    })
}

pub async fn import_offline_manifest(path: &Path) -> Result<Vec<InstalledModel>> {
    let body = fs::read_to_string(path)
        .await
        .with_context(|| format!("read offline install manifest {}", path.display()))?;
    let manifest: OfflineInstallManifest = toml::from_str(&body)
        .with_context(|| format!("parse offline install manifest {}", path.display()))?;
    install_offline_manifest(&manifest, path.parent().unwrap_or_else(|| Path::new("."))).await
}

pub async fn install_offline_manifest(
    manifest: &OfflineInstallManifest,
    base_dir: &Path,
) -> Result<Vec<InstalledModel>> {
    anyhow::ensure!(
        !manifest.models.is_empty(),
        "offline install manifest must include at least one model"
    );

    let mut installed = Vec::with_capacity(manifest.models.len());
    for entry in &manifest.models {
        validate_alias(&entry.alias)?;
        let expected = normalized_sha256(&entry.sha256)?;
        let path = resolve_manifest_path(base_dir, &entry.path);
        ensure_supported_artifact_path(&path).await?;
        let metadata = fs::metadata(&path)
            .await
            .with_context(|| format!("stat offline model {}", path.display()))?;
        anyhow::ensure!(
            metadata.is_file() || metadata.is_dir(),
            "offline model is not a file or native safetensors directory: {}",
            path.display()
        );
        let sha256 = sha256_model_artifact(&path).await?;
        if sha256 != expected {
            bail!(
                "sha256 mismatch for {}: expected {expected}, got {sha256}",
                path.display()
            );
        }
        let source = ModelSource::LocalPath { path: path.clone() };
        let config = ModelConfig {
            alias: entry.alias.clone(),
            path: path.clone(),
            role: entry.role.clone(),
            family: entry.family.clone(),
            weight: entry.weight,
        };
        let bytes = model_artifact_len(&path).await?;
        installed.push(InstalledModel {
            alias: entry.alias.clone(),
            path,
            sha256: sha256.clone(),
            bytes,
            source,
            source_kind: ModelInstallSourceKind::Offline,
            verification: ModelInstallVerification {
                sha256_required: true,
                expected_sha256: Some(expected),
                actual_sha256: Some(sha256),
                verified: true,
            },
            config,
        });
    }
    Ok(installed)
}

pub async fn register_local_model(
    path: &Path,
    cache_dir: &Path,
    copy_to_cache: bool,
) -> Result<PathBuf> {
    ensure_supported_artifact_path(path).await?;
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat local model {}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file() || metadata.is_dir(),
        "local model is not a file or native safetensors directory: {}",
        path.display()
    );

    if !copy_to_cache {
        return Ok(path.to_path_buf());
    }

    fs::create_dir_all(cache_dir)
        .await
        .with_context(|| format!("create model cache {}", cache_dir.display()))?;
    let filename = path
        .file_name()
        .ok_or_else(|| anyhow!("local model has no filename: {}", path.display()))?;
    let destination = unique_destination(cache_dir, filename);
    if metadata.is_dir() {
        copy_dir(path, &destination).await?;
        return Ok(destination);
    }
    if is_safetensors_path(path) {
        let directory_destination = unique_destination(cache_dir, filename).with_extension("");
        fs::create_dir_all(&directory_destination)
            .await
            .with_context(|| format!("create {}", directory_destination.display()))?;
        copy_safetensors_file_layout(path, &directory_destination).await?;
        return Ok(directory_destination);
    }
    fs::copy(path, &destination)
        .await
        .with_context(|| format!("copy {} to {}", path.display(), destination.display()))?;
    Ok(destination)
}

async fn copy_dir(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)
        .await
        .with_context(|| format!("create {}", destination.display()))?;
    let mut entries = fs::read_dir(source)
        .await
        .with_context(|| format!("read {}", source.display()))?;
    while let Some(entry) = entries.next_entry().await? {
        let entry_type = entry.file_type().await?;
        let target = destination.join(entry.file_name());
        if entry_type.is_dir() {
            Box::pin(copy_dir(&entry.path(), &target)).await?;
        } else if entry_type.is_file() {
            fs::copy(entry.path(), &target).await.with_context(|| {
                format!("copy {} to {}", entry.path().display(), target.display())
            })?;
        }
    }
    Ok(())
}

pub async fn download_model(
    url: &str,
    cache_dir: &Path,
    name_hint: &str,
    expected_sha256: &str,
) -> Result<PathBuf> {
    validate_direct_download_url(url)?;
    validate_direct_download_resolves_public(url)?;
    let expected_sha256 = normalized_sha256(expected_sha256)?;
    fs::create_dir_all(cache_dir)
        .await
        .with_context(|| format!("create model cache {}", cache_dir.display()))?;
    let filename = download_filename(url, name_hint)?;
    let destination = unique_destination(cache_dir, filename.as_ref());
    let partial = destination.with_extension("part");

    let partial_len = match fs::metadata(&partial).await {
        Ok(metadata) if metadata.is_file() => metadata.len(),
        _ => 0,
    };
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(MODEL_DOWNLOAD_TIMEOUT)
        .build()
        .context("build model download client")?;
    let mut request = client.get(url);
    if partial_len > 0 {
        request = request.header(
            RANGE,
            HeaderValue::from_str(&format!("bytes={partial_len}-"))
                .context("build resumable download range header")?,
        );
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("download model from {url}"))?
        .error_for_status()
        .with_context(|| format!("download model from {url}"))?;
    let append_partial =
        partial_len > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let mut output = if append_partial {
        fs::OpenOptions::new()
            .append(true)
            .open(&partial)
            .await
            .with_context(|| format!("open partial download {}", partial.display()))?
    } else {
        fs::File::create(&partial)
            .await
            .with_context(|| format!("create partial download {}", partial.display()))?
    };
    let mut stream = response.bytes_stream();
    let write_result: Result<()> = async {
        while let Some(chunk) = stream.next().await {
            output
                .write_all(&chunk.with_context(|| format!("read model download from {url}"))?)
                .await
                .with_context(|| format!("write partial download {}", partial.display()))?;
        }
        output
            .flush()
            .await
            .with_context(|| format!("flush partial download {}", partial.display()))?;
        Ok(())
    }
    .await;
    drop(output);
    if let Err(err) = write_result {
        let _ = fs::remove_file(&partial).await;
        return Err(err);
    }

    verify_downloaded_model(&partial, &destination, &expected_sha256).await?;
    Ok(destination)
}

async fn verify_downloaded_model(
    partial: &Path,
    destination: &Path,
    expected_sha256: &str,
) -> Result<()> {
    let Some(name) = destination.file_name().and_then(|name| name.to_str()) else {
        bail!("model path has no filename: {}", destination.display());
    };
    ensure_supported_model_name(name)?;
    anyhow::ensure!(
        !name.to_ascii_lowercase().ends_with(".safetensors"),
        "direct safetensors downloads are not supported without config.json and tokenizer.json sidecars; use an offline manifest or local safetensors directory"
    );
    let expected_sha256 = normalized_sha256(expected_sha256)?;
    let sha256 = sha256_file(partial).await?;
    if sha256 != expected_sha256 {
        let _ = fs::remove_file(partial).await;
        bail!(
            "sha256 mismatch for {}: expected {expected_sha256}, got {sha256}",
            destination.display()
        );
    }
    fs::rename(partial, destination)
        .await
        .with_context(|| format!("move {} to {}", partial.display(), destination.display()))?;
    Ok(())
}

pub async fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .await
        .with_context(|| format!("open model for checksum {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .with_context(|| format!("read model for checksum {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn download_filename(url: &str, name_hint: &str) -> Result<String> {
    let without_query = url.split('?').next().unwrap_or(url);
    let from_url = without_query.rsplit('/').next().unwrap_or("");
    let candidate = if is_supported_model_filename(from_url) {
        from_url
    } else {
        name_hint
    };
    ensure_supported_model_name(candidate)?;
    Ok(candidate.to_string())
}

fn unique_destination(cache_dir: &Path, filename: &std::ffi::OsStr) -> PathBuf {
    let destination = cache_dir.join(filename);
    if !destination.exists() {
        return destination;
    }
    let path = Path::new(filename);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("gguf");
    for idx in 1.. {
        let candidate = cache_dir.join(format!("{stem}-{idx}.{ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("unbounded destination search should always return")
}

fn validate_alias(alias: &str) -> Result<()> {
    anyhow::ensure!(!alias.trim().is_empty(), "model alias cannot be empty");
    anyhow::ensure!(
        alias
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')),
        "model alias may only contain ASCII letters, numbers, '.', '-', and '_'"
    );
    Ok(())
}

async fn ensure_supported_artifact_path(path: &Path) -> Result<()> {
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat model artifact {}", path.display()))?;
    if metadata.is_dir() {
        ensure_safetensors_layout(path).await?;
        return Ok(());
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        bail!("model path has no filename: {}", path.display());
    };
    ensure_supported_model_name(name)?;
    if is_safetensors_path(path) {
        ensure_safetensors_file_layout(path).await?;
    }
    Ok(())
}

fn ensure_supported_model_name(name: &str) -> Result<()> {
    anyhow::ensure!(
        is_supported_model_filename(name),
        "model artifact must use .gguf or .safetensors extension: {name}"
    );
    Ok(())
}

fn is_supported_model_filename(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".gguf") || lower.ends_with(".safetensors")
}

fn is_safetensors_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".safetensors"))
}

async fn ensure_safetensors_file_layout(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("native safetensors file has no parent directory"))?;
    for required in ["config.json", "tokenizer.json"] {
        let required_path = parent.join(required);
        anyhow::ensure!(
            required_path.is_file(),
            "native safetensors file {} must have sibling {required}",
            path.display()
        );
    }
    Ok(())
}

async fn copy_safetensors_file_layout(source: &Path, destination_dir: &Path) -> Result<()> {
    ensure_safetensors_file_layout(source).await?;
    let filename = source
        .file_name()
        .ok_or_else(|| anyhow!("native safetensors file has no filename"))?;
    fs::copy(source, destination_dir.join(filename))
        .await
        .with_context(|| format!("copy {} to {}", source.display(), destination_dir.display()))?;
    let parent = source
        .parent()
        .ok_or_else(|| anyhow!("native safetensors file has no parent directory"))?;
    for sidecar in ["config.json", "tokenizer.json"] {
        fs::copy(parent.join(sidecar), destination_dir.join(sidecar))
            .await
            .with_context(|| format!("copy safetensors sidecar {sidecar}"))?;
    }
    Ok(())
}

async fn ensure_safetensors_layout(path: &Path) -> Result<()> {
    for required in ["config.json", "tokenizer.json"] {
        let required_path = path.join(required);
        anyhow::ensure!(
            required_path.is_file(),
            "native safetensors directory {} must include {required}",
            path.display()
        );
    }
    let mut entries = fs::read_dir(path)
        .await
        .with_context(|| format!("read native safetensors directory {}", path.display()))?;
    let mut has_weights = false;
    while let Some(entry) = entries.next_entry().await? {
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.to_ascii_lowercase().ends_with(".safetensors"))
        {
            has_weights = true;
            break;
        }
    }
    anyhow::ensure!(
        has_weights,
        "native safetensors directory {} must include at least one .safetensors weight file",
        path.display()
    );
    Ok(())
}

async fn sha256_model_artifact(path: &Path) -> Result<String> {
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat model artifact {}", path.display()))?;
    if metadata.is_file() {
        return sha256_file(path).await;
    }
    ensure_safetensors_layout(path).await?;
    let mut files = Vec::new();
    let mut entries = fs::read_dir(path)
        .await
        .with_context(|| format!("read model artifact directory {}", path.display()))?;
    while let Some(entry) = entries.next_entry().await? {
        let entry_path = entry.path();
        if entry.file_type().await?.is_file() {
            files.push(entry_path);
        }
    }
    files.sort();
    let mut hasher = Sha256::new();
    for file in files {
        let name = file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        hasher.update(name.as_bytes());
        hasher.update(b"\0");
        let digest = sha256_file(&file).await?;
        hasher.update(digest.as_bytes());
        hasher.update(b"\0");
    }
    Ok(hex::encode(hasher.finalize()))
}

async fn model_artifact_len(path: &Path) -> Result<u64> {
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat model artifact {}", path.display()))?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    let mut total = 0_u64;
    let mut entries = fs::read_dir(path)
        .await
        .with_context(|| format!("read model artifact directory {}", path.display()))?;
    while let Some(entry) = entries.next_entry().await? {
        let metadata = entry.metadata().await?;
        if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn validate_direct_download_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("parse model URL {url}"))?;
    anyhow::ensure!(
        parsed.scheme() == "https",
        "direct model downloads require https URLs"
    );
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("direct model download URL must include a host"))?;
    anyhow::ensure!(
        !matches!(host, "localhost" | "metadata.google.internal"),
        "direct model download URL host is not allowed"
    );
    if let Ok(ip) = host.parse::<IpAddr>() {
        anyhow::ensure!(
            !is_blocked_download_ip(ip),
            "direct model download URL must not target local or private addresses"
        );
    }
    Ok(())
}

fn validate_direct_download_resolves_public(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("parse model URL {url}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("direct model download URL must include a host"))?;
    let port = parsed.port_or_known_default().unwrap_or(443);
    let resolved = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolve direct model download host {host}"))?
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !resolved.is_empty(),
        "direct model download URL host did not resolve"
    );
    for addr in resolved {
        anyhow::ensure!(
            !is_blocked_download_ip(addr.ip()),
            "direct model download URL must not resolve to local or private addresses"
        );
    }
    Ok(())
}

fn is_blocked_download_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.octets() == [169, 254, 169, 254]
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
    }
}

fn validate_sha256(value: &str) -> Result<()> {
    anyhow::ensure!(
        value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()),
        "sha256 must be 64 hexadecimal characters"
    );
    Ok(())
}

fn normalized_sha256(value: &str) -> Result<String> {
    let normalized = value.to_ascii_lowercase();
    validate_sha256(&normalized)?;
    Ok(normalized)
}

fn resolve_manifest_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn ensure_relative_component(value: &str, label: &str) -> Result<()> {
    anyhow::ensure!(!value.trim().is_empty(), "{label} cannot be empty");
    anyhow::ensure!(
        !value.contains("..") && !value.starts_with('/') && !value.starts_with('\\'),
        "{label} must be a relative HuggingFace path component"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn registers_local_model_without_copy() {
        let dir = tempdir().unwrap();
        let model = dir.path().join("tiny.gguf");
        fs::write(&model, b"gguf").await.unwrap();
        let installed = install_model(&ModelInstallRequest {
            alias: "tiny".to_string(),
            source: ModelSource::LocalPath {
                path: model.clone(),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: None,
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 7,
        })
        .await
        .unwrap();
        assert_eq!(installed.path, model);
        assert_eq!(installed.bytes, 4);
        assert_eq!(installed.config.alias, "tiny");
        assert_eq!(installed.config.weight, 7);
    }

    #[tokio::test]
    async fn rejects_bare_safetensors_file_without_native_sidecars() {
        let dir = tempdir().unwrap();
        let model = dir.path().join("model.safetensors");
        fs::write(&model, b"safetensors").await.unwrap();
        let err = install_model(&ModelInstallRequest {
            alias: "mistral".to_string(),
            source: ModelSource::LocalPath {
                path: model.clone(),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: None,
            role: "chat".to_string(),
            family: Some("mistral".to_string()),
            weight: 1,
        })
        .await
        .unwrap_err();

        assert!(
            err.to_string().contains("sibling config.json"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn registers_local_safetensors_file_with_sidecars_without_copy() {
        let dir = tempdir().unwrap();
        let model = dir.path().join("model.safetensors");
        fs::write(&model, b"safetensors").await.unwrap();
        fs::write(dir.path().join("config.json"), b"{}")
            .await
            .unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}")
            .await
            .unwrap();
        let installed = install_model(&ModelInstallRequest {
            alias: "mistral".to_string(),
            source: ModelSource::LocalPath {
                path: model.clone(),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: None,
            role: "chat".to_string(),
            family: Some("mistral".to_string()),
            weight: 1,
        })
        .await
        .unwrap();

        assert_eq!(installed.path, model);
        assert_eq!(installed.config.family.as_deref(), Some("mistral"));
    }

    #[tokio::test]
    async fn registers_safetensors_directory_layout() {
        let dir = tempdir().unwrap();
        let model_dir = dir.path().join("deepseek");
        fs::create_dir_all(&model_dir).await.unwrap();
        fs::write(model_dir.join("config.json"), b"{}")
            .await
            .unwrap();
        fs::write(model_dir.join("tokenizer.json"), b"{}")
            .await
            .unwrap();
        fs::write(model_dir.join("model.safetensors"), b"weights")
            .await
            .unwrap();
        let installed = install_model(&ModelInstallRequest {
            alias: "deepseek".to_string(),
            source: ModelSource::LocalPath {
                path: model_dir.clone(),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: None,
            role: "thinking".to_string(),
            family: Some("deepseek".to_string()),
            weight: 1,
        })
        .await
        .unwrap();

        assert_eq!(installed.path, model_dir);
        assert!(installed.bytes > 0);
    }

    #[tokio::test]
    async fn copies_local_model_into_cache_and_checks_sha() {
        let dir = tempdir().unwrap();
        let model = dir.path().join("tiny.gguf");
        fs::write(&model, b"model-bytes").await.unwrap();
        let expected = sha256_file(&model).await.unwrap();
        let installed = install_model(&ModelInstallRequest {
            alias: "tiny".to_string(),
            source: ModelSource::LocalPath { path: model },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: true,
            expected_sha256: Some(expected.clone()),
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 0,
        })
        .await
        .unwrap();
        assert!(installed.path.starts_with(dir.path().join("cache")));
        assert_eq!(installed.sha256, expected);
    }

    #[tokio::test]
    async fn parses_offline_manifest_relative_paths_and_defaults() {
        let dir = tempdir().unwrap();
        let model = dir.path().join("tiny.gguf");
        fs::write(&model, b"manifest-model").await.unwrap();
        let expected = sha256_file(&model).await.unwrap();
        let manifest: OfflineInstallManifest = toml::from_str(&format!(
            r#"
[[models]]
alias = "tiny"
path = "tiny.gguf"
sha256 = "{expected}"
"#
        ))
        .unwrap();

        let installed = install_offline_manifest(&manifest, dir.path())
            .await
            .unwrap();

        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].alias, "tiny");
        assert_eq!(installed[0].path, model);
        assert_eq!(installed[0].config.role, "chat");
        assert_eq!(installed[0].config.weight, 0);
        assert_eq!(installed[0].source_kind, ModelInstallSourceKind::Offline);
        assert_eq!(
            installed[0].verification.expected_sha256.as_deref(),
            Some(expected.as_str())
        );
        assert!(installed[0].verification.verified);
    }

    #[tokio::test]
    async fn rejects_offline_manifest_sha_mismatch() {
        let dir = tempdir().unwrap();
        let model = dir.path().join("tiny.gguf");
        fs::write(&model, b"manifest-model").await.unwrap();
        let manifest = OfflineInstallManifest {
            models: vec![OfflineManifestModel {
                alias: "tiny".to_string(),
                path: PathBuf::from("tiny.gguf"),
                role: "chat".to_string(),
                family: Some("qwen3".to_string()),
                weight: 1,
                sha256: "0".repeat(64),
            }],
        };

        let err = install_offline_manifest(&manifest, dir.path())
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("sha256 mismatch"), "{err}");
    }

    #[tokio::test]
    async fn registers_multiple_models_from_offline_manifest() {
        let dir = tempdir().unwrap();
        let chat = dir.path().join("chat.gguf");
        let embed = dir.path().join("embed.gguf");
        fs::write(&chat, b"chat-model").await.unwrap();
        fs::write(&embed, b"embed-model").await.unwrap();
        let manifest = OfflineInstallManifest {
            models: vec![
                OfflineManifestModel {
                    alias: "chat".to_string(),
                    path: PathBuf::from("chat.gguf"),
                    role: "chat".to_string(),
                    family: Some("qwen3".to_string()),
                    weight: 10,
                    sha256: sha256_file(&chat).await.unwrap(),
                },
                OfflineManifestModel {
                    alias: "embed".to_string(),
                    path: PathBuf::from("embed.gguf"),
                    role: "embedding".to_string(),
                    family: Some("qwen3".to_string()),
                    weight: 2,
                    sha256: sha256_file(&embed).await.unwrap(),
                },
            ],
        };

        let installed = install_offline_manifest(&manifest, dir.path())
            .await
            .unwrap();
        let configs: Vec<_> = installed.into_iter().map(|model| model.config).collect();

        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].alias, "chat");
        assert_eq!(configs[0].weight, 10);
        assert_eq!(configs[1].alias, "embed");
        assert_eq!(configs[1].role, "embedding");
    }

    #[test]
    fn builds_huggingface_download_url() {
        let url = huggingface_download_url("org/repo", "model.gguf", "main").unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/org/repo/resolve/main/model.gguf?download=true"
        );
    }

    #[tokio::test]
    async fn rejects_direct_download_without_expected_sha_before_network() {
        let dir = tempdir().unwrap();
        let err = install_model(&ModelInstallRequest {
            alias: "tiny".to_string(),
            source: ModelSource::DirectUrl {
                url: "http://127.0.0.1:9/tiny.gguf".to_string(),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: None,
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 0,
        })
        .await
        .unwrap_err()
        .to_string();

        assert!(
            err.contains("expected_sha256 is required for downloaded model sources"),
            "{err}"
        );
    }

    #[test]
    fn rejects_unsafe_direct_download_urls_before_network() {
        let dir = tempdir().unwrap();
        let request = |url: &str| ModelInstallRequest {
            alias: "tiny".to_string(),
            source: ModelSource::DirectUrl {
                url: url.to_string(),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: Some("0".repeat(64)),
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 0,
        };

        let http = install_plan(&request("http://models.example/tiny.gguf"))
            .unwrap_err()
            .to_string();
        assert!(http.contains("require https"), "{http}");

        let localhost = install_plan(&request("https://127.0.0.1/tiny.gguf"))
            .unwrap_err()
            .to_string();
        assert!(localhost.contains("local or private"), "{localhost}");

        let metadata = install_plan(&request(
            "https://169.254.169.254/latest/meta-data/tiny.gguf",
        ))
        .unwrap_err()
        .to_string();
        assert!(metadata.contains("local or private"), "{metadata}");
    }

    #[tokio::test]
    async fn rejects_huggingface_download_without_expected_sha_before_network() {
        let dir = tempdir().unwrap();
        let err = install_model(&ModelInstallRequest {
            alias: "tiny".to_string(),
            source: ModelSource::HuggingFace {
                repo: "org/repo".to_string(),
                filename: "tiny.gguf".to_string(),
                revision: "main".to_string(),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: None,
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 0,
        })
        .await
        .unwrap_err()
        .to_string();

        assert!(
            err.contains("expected_sha256 is required for downloaded model sources"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn checksum_mismatch_removes_partial_and_keeps_final_absent() {
        let dir = tempdir().unwrap();
        let final_path = dir.path().join("tiny.gguf");
        let partial = final_path.with_extension("part");
        fs::write(&partial, b"downloaded-bytes").await.unwrap();

        let err = verify_downloaded_model(&partial, &final_path, &"0".repeat(64))
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("sha256 mismatch"), "{err}");
        assert!(!partial.exists(), "partial download should be cleaned up");
        assert!(!final_path.exists(), "final model should not exist");
    }

    #[tokio::test]
    async fn plans_sources_with_distinct_verification_requirements() {
        let dir = tempdir().unwrap();
        let local = install_plan(&ModelInstallRequest {
            alias: "tiny".to_string(),
            source: ModelSource::LocalPath {
                path: dir.path().join("tiny.gguf"),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: None,
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 0,
        })
        .unwrap();
        assert_eq!(local.source_kind, ModelInstallSourceKind::Local);
        assert!(!local.verification.sha256_required);

        let download = install_plan(&ModelInstallRequest {
            alias: "tiny".to_string(),
            source: ModelSource::DirectUrl {
                url: "https://example.com/tiny.gguf".to_string(),
            },
            cache_dir: dir.path().join("cache"),
            copy_to_cache: false,
            expected_sha256: Some("0".repeat(64)),
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 0,
        })
        .unwrap();
        assert_eq!(download.source_kind, ModelInstallSourceKind::Download);
        assert!(download.verification.sha256_required);
    }

    #[test]
    fn has_catalog_basics() {
        let qwen = catalog_model("qwen2.5-7b").unwrap();
        assert_eq!(qwen.repo, "Qwen/Qwen2.5-7B-Instruct-GGUF");
        assert!(builtin_catalog()
            .iter()
            .all(|model| model.filename.ends_with(".gguf")));
    }
}
