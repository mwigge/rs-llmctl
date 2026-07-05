use super::*;
use crate::config::{Config, ModelConfig};
use std::path::PathBuf;

#[test]
fn native_validation_plan_covers_artifacts_hardware_soak_and_benchmarks() {
    let cfg = Config::default();
    let plan = native_validation_plan(
        &cfg,
        NativeRuntimeValidationOptions {
            soak_minutes: 30,
            streaming_concurrency: 5,
            rotation_keys: 2,
            quota_concurrency: 7,
        },
    );

    assert_eq!(plan.status, "planned");
    assert_eq!(plan.mode, "deterministic-offline");
    assert!(!plan.network);
    assert_eq!(plan.real_artifact_smoke_tests.len(), 4);
    assert!(plan
        .real_artifact_smoke_tests
        .iter()
        .all(|test| test.required_format == NativeModelFormat::Safetensors));
    assert!(plan
        .real_artifact_smoke_tests
        .iter()
        .all(|test| test.artifact_validation.status == "planned-missing-local-artifact"));
    assert!(plan
        .hardware_matrix
        .iter()
        .any(|target| target.target == "cpu"));
    assert!(plan
        .hardware_matrix
        .iter()
        .any(|target| target.target == "nvidia-cuda"));
    assert!(plan
        .hardware_matrix
        .iter()
        .any(|target| target.target == "amd-vulkan"));
    assert!(plan
        .hardware_matrix
        .iter()
        .any(|target| target.target == "apple-metal"));
    assert_eq!(plan.soak_tests[0].duration_minutes, 30);
    assert_eq!(plan.soak_tests[0].concurrency, 5);
    assert_eq!(plan.api_key_rotation_and_quota.rotation_keys, 2);
    assert_eq!(plan.api_key_rotation_and_quota.quota_concurrency, 7);
    assert!(plan
        .benchmark
        .metrics
        .contains(&"tokens_per_second".to_string()));
    assert!(plan
        .benchmark
        .memory_fields
        .contains(&"peak_vram_bytes".to_string()));
}

#[test]
fn candle_native_status_declares_starter_roles_and_default_budget() {
    let cfg = Config::default();
    let status = status_from_config(&cfg);

    assert_eq!(status.backend, RuntimeBackend::CandleNative);
    assert!(status.primary);
    assert!(status.implemented);
    assert_eq!(status.resource_policy.budget_fraction, 0.80);
    assert_eq!(status.resource_policy.cpu_fraction, 0.80);
    assert_eq!(status.resource_policy.ram_fraction, 0.80);
    assert_eq!(status.resource_policy.gpu_vram_fraction, 0.80);
    assert_eq!(status.embeddings.mode, NativeEmbeddingMode::Semantic);
    assert_eq!(status.embeddings.semantic_model_alias, None);
    assert_eq!(status.embeddings.semantic_backend, "candle-bert-embeddings");
    assert_eq!(
        status.embeddings.fallback_backend,
        "deterministic-local-fallback"
    );
    assert_eq!(
        status.embeddings.fallback_status,
        "non-semantic-dev-fallback"
    );
    assert!(status.embeddings.fallback_dev_only);
    assert_eq!(
        status
            .starter_roles
            .iter()
            .map(|role| role.role.as_str())
            .collect::<Vec<_>>(),
        vec!["query", "recommendation", "thinking", "coding"]
    );
    assert!(status
        .starter_roles
        .iter()
        .all(|role| role.default_family == "qwen3"));
    assert!(status
        .starter_roles
        .iter()
        .all(|role| role.alternative_families == ["gemma4", "kimi", "mistral"]));
    assert!(status
        .starter_roles
        .iter()
        .all(|role| role.eu_friendly_family.as_deref() == Some("mistral")));
    assert!(status
        .resource_policy
        .gpu_detection
        .iter()
        .any(|entry| entry.contains("nvidia")));
    assert!(status
        .resource_policy
        .gpu_detection
        .iter()
        .any(|entry| entry.contains("amd")));
    assert!(status
        .resource_policy
        .gpu_detection
        .iter()
        .any(|entry| entry.contains("apple")));
}

#[test]
fn runtime_status_reports_gemma4_quarantine_without_evidence() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = Config::default();
    cfg.storage.model_dir = dir.path().to_path_buf();
    cfg.models = vec![ModelConfig {
        alias: "gemma4".to_string(),
        path: PathBuf::from("gemma4.gguf"),
        role: "coding".to_string(),
        family: Some("gemma4".to_string()),
        weight: 1,
    }];

    let status = status_from_config(&cfg);
    assert_eq!(status.model_readiness.len(), 1);
    assert_eq!(status.model_readiness[0].state, ReadinessState::Quarantined);
    assert!(!status.model_readiness[0].evidence_present);
}

#[test]
fn runtime_status_reports_gemma4_qualified_with_persisted_evidence() {
    use crate::readiness::{
        CommandEvidence, Gemma4ReadinessEvidence, LanguageFixtureEvidence, SamplingParameters,
        GEMMA4_READINESS_SCHEMA_VERSION,
    };
    use chrono::{TimeZone, Utc};

    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = Config::default();
    cfg.storage.model_dir = dir.path().to_path_buf();
    cfg.models = vec![ModelConfig {
        alias: "gemma4".to_string(),
        path: PathBuf::from("gemma4.gguf"),
        role: "coding".to_string(),
        family: Some("gemma4".to_string()),
        weight: 1,
    }];

    let evidence_path = readiness::evidence_path(&cfg.storage.model_dir, "gemma4");
    let evidence = Gemma4ReadinessEvidence {
        schema_version: GEMMA4_READINESS_SCHEMA_VERSION.to_string(),
        generated_at: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
        state: ReadinessState::Qualified,
        artifact_path: "gemma4".to_string(),
        artifact_sha256: "0".repeat(64),
        runtime_revision: "rs-llmctl/test".to_string(),
        sampling: SamplingParameters {
            strategy: "greedy".to_string(),
            temperature: "0".to_string(),
            max_tokens: 256,
        },
        expected_output: readiness::CANONICAL_TEN_LINE_OUTPUT.to_string(),
        fixtures: vec![LanguageFixtureEvidence {
            language: readiness::FixtureLanguage::Go,
            prompt: "prompt".to_string(),
            raw_generation: "raw".to_string(),
            generated_source: "source".to_string(),
            toolchain_version: "go1.0".to_string(),
            commands: vec![CommandEvidence {
                program: "go".to_string(),
                args: vec!["run".to_string(), "main.go".to_string()],
                exit_code: Some(0),
                stdout: readiness::CANONICAL_TEN_LINE_OUTPUT.to_string(),
                stderr: String::new(),
                passed: true,
            }],
            output_matches: true,
            passed: true,
        }],
    };
    readiness::write_evidence(&evidence_path, &evidence).expect("write evidence");

    let status = status_from_config(&cfg);
    assert_eq!(status.model_readiness.len(), 1);
    assert_eq!(status.model_readiness[0].state, ReadinessState::Qualified);
    assert!(status.model_readiness[0].evidence_present);
}
