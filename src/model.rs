use crate::config::ModelConfig;
use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const HF_BASE_URL: &str = "https://huggingface.co";

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
    pub weight: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledModel {
    pub alias: String,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub source: ModelSource,
    pub config: ModelConfig,
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

pub fn builtin_catalog() -> Vec<CatalogModel> {
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
}

pub fn catalog_model(id_or_alias: &str) -> Option<CatalogModel> {
    builtin_catalog()
        .into_iter()
        .find(|model| model.id == id_or_alias || model.alias == id_or_alias)
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
        ModelSource::DirectUrl { url } => Ok(Some(url.clone())),
        ModelSource::HuggingFace {
            repo,
            filename,
            revision,
        } => huggingface_download_url(repo, filename, revision).map(Some),
    }
}

pub async fn install_model(req: &ModelInstallRequest) -> Result<InstalledModel> {
    validate_alias(&req.alias)?;
    fs::create_dir_all(&req.cache_dir)
        .await
        .with_context(|| format!("create model cache {}", req.cache_dir.display()))?;

    let path = match &req.source {
        ModelSource::LocalPath { path } => {
            register_local_model(path, &req.cache_dir, req.copy_to_cache).await?
        }
        ModelSource::DirectUrl { url } => download_model(url, &req.cache_dir, &req.alias).await?,
        ModelSource::HuggingFace {
            repo,
            filename,
            revision,
        } => {
            let url = huggingface_download_url(repo, filename, revision)?;
            download_model(&url, &req.cache_dir, filename).await?
        }
    };

    ensure_gguf_path(&path)?;
    let sha256 = sha256_file(&path).await?;
    if let Some(expected) = &req.expected_sha256 {
        let expected = expected.to_ascii_lowercase();
        if sha256 != expected {
            bail!(
                "sha256 mismatch for {}: expected {expected}, got {sha256}",
                path.display()
            );
        }
    }
    let bytes = fs::metadata(&path)
        .await
        .with_context(|| format!("stat installed model {}", path.display()))?
        .len();
    let config = ModelConfig {
        alias: req.alias.clone(),
        path: path.clone(),
        role: req.role.clone(),
        weight: req.weight,
    };

    Ok(InstalledModel {
        alias: req.alias.clone(),
        path,
        sha256,
        bytes,
        source: req.source.clone(),
        config,
    })
}

pub async fn register_local_model(
    path: &Path,
    cache_dir: &Path,
    copy_to_cache: bool,
) -> Result<PathBuf> {
    ensure_gguf_path(path)?;
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat local model {}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "local model is not a file: {}",
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
    fs::copy(path, &destination)
        .await
        .with_context(|| format!("copy {} to {}", path.display(), destination.display()))?;
    Ok(destination)
}

pub async fn download_model(url: &str, cache_dir: &Path, name_hint: &str) -> Result<PathBuf> {
    fs::create_dir_all(cache_dir)
        .await
        .with_context(|| format!("create model cache {}", cache_dir.display()))?;
    let filename = download_filename(url, name_hint)?;
    let destination = unique_destination(cache_dir, filename.as_ref());
    let partial = destination.with_extension("part");

    let response = reqwest::get(url)
        .await
        .with_context(|| format!("download model from {url}"))?
        .error_for_status()
        .with_context(|| format!("download model from {url}"))?;
    let mut output = fs::File::create(&partial)
        .await
        .with_context(|| format!("create partial download {}", partial.display()))?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        output
            .write_all(&chunk.with_context(|| format!("read model download from {url}"))?)
            .await
            .with_context(|| format!("write partial download {}", partial.display()))?;
    }
    output.flush().await?;
    drop(output);
    fs::rename(&partial, &destination)
        .await
        .with_context(|| format!("move {} to {}", partial.display(), destination.display()))?;
    ensure_gguf_path(&destination)?;
    Ok(destination)
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
    let candidate = if from_url.ends_with(".gguf") {
        from_url
    } else {
        name_hint
    };
    ensure_gguf_name(candidate)?;
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

fn ensure_gguf_path(path: &Path) -> Result<()> {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        bail!("model path has no filename: {}", path.display());
    };
    ensure_gguf_name(name)
}

fn ensure_gguf_name(name: &str) -> Result<()> {
    anyhow::ensure!(
        name.to_ascii_lowercase().ends_with(".gguf"),
        "model file must use .gguf extension: {name}"
    );
    Ok(())
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
            weight: 0,
        })
        .await
        .unwrap();
        assert!(installed.path.starts_with(dir.path().join("cache")));
        assert_eq!(installed.sha256, expected);
    }

    #[test]
    fn builds_huggingface_download_url() {
        let url = huggingface_download_url("org/repo", "model.gguf", "main").unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/org/repo/resolve/main/model.gguf?download=true"
        );
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
