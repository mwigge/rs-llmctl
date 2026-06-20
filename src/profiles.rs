use crate::model;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const PROFILE_SCHEMA: &str = "specialist-model-profile/v1";
pub const ADAPTER_CATALOG_SCHEMA: &str = "backend-adapter-catalog/v1";

/// Lifecycle state of a specialist model profile: newly imported and untrusted (`Candidate`),
/// passed all policy checks (`Qualified`), or failed one or more checks (`Quarantined`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QualificationStatus {
    Candidate,
    Qualified,
    Quarantined,
}

/// Identity and provenance of a model artifact on disk or from a remote source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactProfile {
    pub path: PathBuf,
    pub sha256: Option<String>,
    pub source: String,
    pub source_revision: Option<String>,
}

/// License metadata that must be reviewed and accepted before a profile can qualify.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LicenseProfile {
    pub identifier: Option<String>,
    pub source_url: Option<String>,
    pub accepted: bool,
}

/// Per-language capability and qualification record for a specialist model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageCapability {
    pub language: String,
    pub capabilities: Vec<String>,
    pub qualified: bool,
    pub score: Option<u32>,
}

/// Estimated resource footprint of running a model, used for hardware-fit checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub kv_cache_bytes: u64,
}

/// A named verification command that gates qualification when `required` is true.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationGate {
    pub name: String,
    pub command: Vec<String>,
    pub required: bool,
}

/// Persisted profile describing a specialist model: artifact provenance, license, language
/// capabilities, resource estimates, and current qualification status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialistModelProfile {
    pub schema: String,
    pub alias: String,
    pub artifact: ArtifactProfile,
    pub license: LicenseProfile,
    pub languages: Vec<LanguageCapability>,
    pub format: String,
    pub quantization: Option<String>,
    pub backend: String,
    pub context_tokens: u32,
    pub memory: MemoryEstimate,
    pub prompt_template: String,
    pub output_grammar: Option<String>,
    pub verification_commands: Vec<Vec<String>>,
    pub gates: Vec<QualificationGate>,
    pub security_maturity: String,
    pub qualification: QualificationStatus,
    pub quarantine_reason: Option<String>,
}

/// Describes a backend's capabilities: supported protocols, accelerators, model formats, and
/// feature/maturity flags used by `evaluate_profile` for runtime-compatibility checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendAdapterProfile {
    pub id: String,
    pub protocol: Vec<String>,
    pub accelerators: Vec<String>,
    pub model_formats: Vec<String>,
    pub tool_calls: bool,
    pub structured_output: bool,
    pub health_endpoint: Option<String>,
    pub lifecycle: Vec<String>,
    pub maturity: String,
    pub unattended_default: bool,
}

/// The full set of known backend adapters, as returned by [`backend_catalog`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterCatalog {
    pub schema: String,
    pub adapters: Vec<BackendAdapterProfile>,
}

/// Result of running all qualification checks against a profile: overall pass/fail plus the
/// individual check outcomes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfilePolicyReport {
    pub qualified: bool,
    pub checks: Vec<ProfilePolicyCheck>,
}

/// A single named qualification check result with a human-readable detail message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfilePolicyCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Builds a `Candidate` profile from a local model file or directory at `path`, computing its
/// SHA-256 digest and inferring its artifact format and quantization.
pub async fn import_local_candidate(path: &Path, alias: &str) -> Result<SpecialistModelProfile> {
    let metadata =
        fs::metadata(path).with_context(|| format!("stat model candidate {}", path.display()))?;
    if !metadata.is_file() && !metadata.is_dir() {
        bail!("model candidate must be a file or directory");
    }
    let format = infer_format(path)?;
    let sha256 = model::sha256_file(path).await?;
    Ok(candidate_profile(
        alias,
        ArtifactProfile {
            path: path.to_path_buf(),
            sha256: Some(sha256),
            source: "local-file".to_string(),
            source_revision: None,
        },
        format,
    ))
}

/// Builds a `Candidate` profile from a built-in catalog entry identified by `id_or_alias`. The
/// artifact has no local checksum yet, so the profile remains unqualified until verified.
pub fn import_catalog_candidate(id_or_alias: &str) -> Result<SpecialistModelProfile> {
    let entry = model::catalog_model(id_or_alias)
        .with_context(|| format!("unknown built-in catalog model {id_or_alias}"))?;
    Ok(candidate_profile(
        entry.alias,
        ArtifactProfile {
            path: PathBuf::from(entry.filename),
            sha256: None,
            source: format!("https://huggingface.co/{}", entry.repo),
            source_revision: Some(entry.revision.to_string()),
        },
        "gguf".to_string(),
    ))
}

fn candidate_profile(
    alias: &str,
    artifact: ArtifactProfile,
    format: String,
) -> SpecialistModelProfile {
    let quantization = infer_quantization(&artifact.path);
    SpecialistModelProfile {
        schema: PROFILE_SCHEMA.to_string(),
        alias: alias.to_string(),
        artifact,
        license: LicenseProfile {
            identifier: None,
            source_url: None,
            accepted: false,
        },
        languages: ["go", "rust", "python", "typescript"]
            .into_iter()
            .map(|language| LanguageCapability {
                language: language.to_string(),
                capabilities: vec![
                    "generation".to_string(),
                    "bug-fix".to_string(),
                    "test-creation".to_string(),
                    "constrained-refactor".to_string(),
                ],
                qualified: false,
                score: None,
            })
            .collect(),
        format,
        quantization,
        backend: "rs-llmctl".to_string(),
        context_tokens: 0,
        memory: MemoryEstimate {
            ram_bytes: 0,
            vram_bytes: 0,
            kv_cache_bytes: 0,
        },
        prompt_template: String::new(),
        output_grammar: None,
        verification_commands: Vec::new(),
        gates: Vec::new(),
        security_maturity: "unknown".to_string(),
        qualification: QualificationStatus::Candidate,
        quarantine_reason: None,
    }
}

/// Runs all qualification checks (license, checksum, provenance, hardware fit, runtime
/// compatibility, security maturity, and Gemma 4 artifact constraints) against `profile` and
/// returns the resulting report without mutating the profile.
pub fn evaluate_profile(
    profile: &SpecialistModelProfile,
    available_vram_bytes: Option<u64>,
) -> ProfilePolicyReport {
    let adapter = backend_catalog()
        .adapters
        .into_iter()
        .find(|adapter| adapter.id == profile.backend);
    let checks = vec![
        check(
            "license",
            profile.license.accepted
                && profile
                    .license
                    .identifier
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
            "accepted SPDX or operator-reviewed license metadata is required",
        ),
        check(
            "checksum",
            profile.artifact.sha256.as_deref().is_some_and(is_sha256),
            "verified SHA-256 artifact identity is required",
        ),
        check(
            "source-provenance",
            profile.artifact.source == "local-file"
                || profile.artifact.source.starts_with("https://"),
            "source must be local-file or HTTPS",
        ),
        check(
            "hardware-fit",
            available_vram_bytes
                .map(|available| profile.memory.vram_bytes <= available)
                .unwrap_or(profile.memory.vram_bytes == 0),
            "estimated VRAM must fit the selected host",
        ),
        check(
            "runtime-compatibility",
            adapter.as_ref().is_some_and(|adapter| {
                adapter
                    .model_formats
                    .iter()
                    .any(|format| format == &profile.format)
            }),
            "selected backend must advertise the artifact format",
        ),
        check(
            "security-maturity",
            profile.security_maturity != "unknown"
                && adapter
                    .as_ref()
                    .is_some_and(|adapter| adapter.maturity != "unsupported"),
            "profile and adapter security maturity must be reviewed",
        ),
        check(
            "gemma4-artifact",
            gemma4_compatible(profile),
            "Gemma 4 requires GGUF, compressed-tensors, or native safetensors",
        ),
    ];
    ProfilePolicyReport {
        qualified: checks.iter().all(|check| check.passed),
        checks,
    }
}

/// Evaluates `profile` and updates its `qualification` status and `quarantine_reason`
/// accordingly: `Qualified` when every check passes, otherwise `Quarantined` with the names of
/// the failed checks recorded. Returns the updated profile alongside the evaluation report.
pub fn qualify_profile(
    mut profile: SpecialistModelProfile,
    available_vram_bytes: Option<u64>,
) -> (SpecialistModelProfile, ProfilePolicyReport) {
    let report = evaluate_profile(&profile, available_vram_bytes);
    profile.qualification = if report.qualified {
        QualificationStatus::Qualified
    } else {
        QualificationStatus::Quarantined
    };
    profile.quarantine_reason = (!report.qualified).then(|| {
        report
            .checks
            .iter()
            .filter(|check| !check.passed)
            .map(|check| check.name.as_str())
            .collect::<Vec<_>>()
            .join(",")
    });
    (profile, report)
}

/// Returns whether `profile` satisfies the Gemma 4 artifact-format constraint. Profiles whose
/// alias does not mention "gemma" are unaffected and always return `true`; Gemma profiles must
/// use GGUF, compressed-tensors, or native safetensors.
pub fn gemma4_compatible(profile: &SpecialistModelProfile) -> bool {
    if !profile.alias.to_ascii_lowercase().contains("gemma") {
        return true;
    }
    matches!(
        profile.format.as_str(),
        "gguf" | "compressed-tensors" | "safetensors-native"
    )
}

/// Returns the built-in catalog of known backend adapters and their capabilities.
pub fn backend_catalog() -> AdapterCatalog {
    AdapterCatalog {
        schema: ADAPTER_CATALOG_SCHEMA.to_string(),
        adapters: vec![
            adapter(
                "rs-llmctl",
                &["openai-chat"],
                &["cpu", "metal"],
                &["gguf", "safetensors-native"],
                true,
                "beta",
            ),
            adapter(
                "llama.cpp",
                &["openai-chat"],
                &["cpu", "rocm", "vulkan", "metal", "cuda"],
                &["gguf"],
                true,
                "beta",
            ),
            adapter(
                "llama-swap",
                &["openai-chat"],
                &["cpu", "rocm", "vulkan", "metal", "cuda"],
                &["gguf"],
                true,
                "experimental",
            ),
            adapter(
                "ollama",
                &["openai-chat", "ollama"],
                &["cpu", "rocm", "metal", "cuda"],
                &["gguf"],
                true,
                "experimental",
            ),
            adapter(
                "localai",
                &["openai-chat"],
                &["cpu", "rocm", "vulkan", "metal", "cuda"],
                &["gguf"],
                true,
                "experimental",
            ),
            adapter(
                "vllm",
                &["openai-chat"],
                &["rocm", "cuda"],
                &["safetensors-native", "compressed-tensors"],
                true,
                "experimental",
            ),
            adapter(
                "sglang",
                &["openai-chat"],
                &["rocm", "cuda"],
                &["safetensors-native", "compressed-tensors"],
                true,
                "experimental",
            ),
            adapter(
                "lm-studio",
                &["openai-chat"],
                &["cpu", "rocm", "vulkan", "metal", "cuda"],
                &["gguf"],
                false,
                "experimental",
            ),
            adapter(
                "mistral.rs",
                &["openai-chat"],
                &["cpu", "metal", "cuda"],
                &["gguf", "safetensors-native"],
                true,
                "experimental",
            ),
        ],
    }
}

fn adapter(
    id: &str,
    protocol: &[&str],
    accelerators: &[&str],
    formats: &[&str],
    structured_output: bool,
    maturity: &str,
) -> BackendAdapterProfile {
    BackendAdapterProfile {
        id: id.to_string(),
        protocol: strings(protocol),
        accelerators: strings(accelerators),
        model_formats: strings(formats),
        tool_calls: true,
        structured_output,
        health_endpoint: Some("/health".to_string()),
        lifecycle: vec![
            "start".to_string(),
            "stop".to_string(),
            "health".to_string(),
        ],
        maturity: maturity.to_string(),
        unattended_default: false,
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn check(name: &str, passed: bool, detail: &str) -> ProfilePolicyCheck {
    ProfilePolicyCheck {
        name: name.to_string(),
        passed,
        detail: detail.to_string(),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn infer_format(path: &Path) -> Result<String> {
    if path.is_dir() {
        return Ok("safetensors-native".to_string());
    }
    match path.extension().and_then(|value| value.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("gguf") => Ok("gguf".to_string()),
        Some(extension) if extension.eq_ignore_ascii_case("safetensors") => {
            Ok("safetensors-native".to_string())
        }
        _ => bail!("profile candidate must be GGUF or safetensors"),
    }
}

fn infer_quantization(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy().to_ascii_uppercase();
    ["Q2_K", "Q3_K", "Q4_K_M", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]
        .into_iter()
        .find(|quantization| name.contains(quantization))
        .map(str::to_string)
}

/// Returns the path where a profile for `alias` is stored under `root`.
pub fn profile_path(root: &Path, alias: &str) -> PathBuf {
    root.join("profiles").join(format!("{alias}.json"))
}

/// Serializes `profile` as pretty JSON and writes it under `root`, creating the `profiles`
/// directory if needed. Returns the path the profile was written to.
pub fn write_profile(root: &Path, profile: &SpecialistModelProfile) -> Result<PathBuf> {
    validate_alias(&profile.alias)?;
    let path = profile_path(root, &profile.alias);
    let parent = path
        .parent()
        .with_context(|| format!("profile path {} has no parent directory", path.display()))?;
    fs::create_dir_all(parent)?;
    fs::write(&path, serde_json::to_vec_pretty(profile)?)?;
    Ok(path)
}

/// Reads and parses the profile for `alias` from under `root`.
pub fn read_profile(root: &Path, alias: &str) -> Result<SpecialistModelProfile> {
    validate_alias(alias)?;
    let path = profile_path(root, alias);
    serde_json::from_slice(&fs::read(&path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

/// Returns all profiles stored under `root`, sorted by alias. Returns an empty list if the
/// `profiles` directory does not exist; entries that fail to parse are silently skipped.
pub fn list_profiles(root: &Path) -> Result<Vec<SpecialistModelProfile>> {
    let directory = root.join("profiles");
    let Ok(entries) = fs::read_dir(directory) else {
        return Ok(Vec::new());
    };
    let mut profiles = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("json"))
        .filter_map(|entry| serde_json::from_slice(&fs::read(entry.path()).ok()?).ok())
        .collect::<Vec<SpecialistModelProfile>>();
    profiles.sort_by(|left: &SpecialistModelProfile, right| left.alias.cmp(&right.alias));
    Ok(profiles)
}

/// Removes the persisted profile for `alias` from under `root`.
pub fn remove_profile(root: &Path, alias: &str) -> Result<()> {
    validate_alias(alias)?;
    let path = profile_path(root, alias);
    fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))
}

fn validate_alias(alias: &str) -> Result<()> {
    if alias.is_empty()
        || !alias
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        bail!("profile alias contains unsupported characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_candidates_record_digest_but_remain_untrusted() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model-Q4_K_M.gguf");
        fs::write(&path, b"candidate").unwrap();
        let profile = import_local_candidate(&path, "local-candidate")
            .await
            .unwrap();
        assert_eq!(profile.format, "gguf");
        assert_eq!(profile.quantization.as_deref(), Some("Q4_K_M"));
        assert!(profile.artifact.sha256.as_deref().is_some_and(is_sha256));
        assert_eq!(profile.qualification, QualificationStatus::Candidate);
    }

    #[test]
    fn catalog_candidates_are_not_automatically_trusted() {
        let profile = import_catalog_candidate("qwen2.5-7b").unwrap();
        let report = evaluate_profile(&profile, Some(u64::MAX));
        assert!(!report.qualified);
        assert_eq!(profile.qualification, QualificationStatus::Candidate);
    }

    #[tokio::test]
    async fn qualify_profile_transitions_candidate_to_qualified_when_all_checks_pass() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model-Q4_K_M.gguf");
        fs::write(&path, b"candidate").unwrap();
        let mut profile = import_local_candidate(&path, "local-candidate")
            .await
            .unwrap();
        assert_eq!(profile.qualification, QualificationStatus::Candidate);

        profile.license = LicenseProfile {
            identifier: Some("Apache-2.0".to_string()),
            source_url: Some("https://example.invalid/license".to_string()),
            accepted: true,
        };
        profile.security_maturity = "reviewed".to_string();
        profile.memory.vram_bytes = 1_000;

        let (qualified_profile, report) = qualify_profile(profile, Some(u64::MAX));

        assert!(report.qualified, "{report:?}");
        assert_eq!(
            qualified_profile.qualification,
            QualificationStatus::Qualified
        );
        assert_eq!(qualified_profile.quarantine_reason, None);
    }

    #[test]
    fn adapter_catalog_contains_requested_experimental_services() {
        let ids = backend_catalog()
            .adapters
            .into_iter()
            .map(|adapter| adapter.id)
            .collect::<Vec<_>>();
        for expected in [
            "rs-llmctl",
            "llama.cpp",
            "llama-swap",
            "ollama",
            "localai",
            "vllm",
            "sglang",
            "lm-studio",
            "mistral.rs",
        ] {
            assert!(ids.iter().any(|id| id == expected));
        }
    }
}
