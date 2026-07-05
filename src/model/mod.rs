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

mod types;
pub use types::*;
mod download;
pub use download::*;

#[cfg(test)]
mod tests;

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
