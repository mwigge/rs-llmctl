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
