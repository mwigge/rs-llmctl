use crate::model;
use crate::native;
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use uuid::Uuid;

pub const GEMMA4_READINESS_SCHEMA_VERSION: &str = "gemma4-readiness/v1";
pub const CANONICAL_TEN_LINE_OUTPUT: &str = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n";

/// Overall outcome of a Gemma 4 readiness run: `Qualified` when every language fixture passed,
/// `Quarantined` otherwise (or when no evidence has been recorded yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReadinessState {
    Qualified,
    Quarantined,
}

/// One of the four languages exercised by the Gemma 4 readiness fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FixtureLanguage {
    Go,
    Rust,
    Python,
    #[serde(rename = "typescript")]
    TypeScript,
}

impl FixtureLanguage {
    fn file_name(self) -> &'static str {
        match self {
            Self::Go => "main.go",
            Self::Rust => "main.rs",
            Self::Python => "main.py",
            Self::TypeScript => "main.ts",
        }
    }

    fn directory_name(self) -> &'static str {
        match self {
            Self::Go => "go",
            Self::Rust => "rust",
            Self::Python => "python",
            Self::TypeScript => "typescript",
        }
    }

    fn prompt(self) -> String {
        format!(
            "Write one minimal {} program that prints the decimal integers 1 through 10, \
             one integer per line, with no extra output. Return only source code.",
            match self {
                Self::Go => "Go",
                Self::Rust => "Rust",
                Self::Python => "Python",
                Self::TypeScript => "TypeScript",
            }
        )
    }
}

/// Sampling configuration used when generating the readiness fixture sources, persisted as part
/// of the evidence record for reproducibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplingParameters {
    pub strategy: String,
    pub temperature: String,
    pub max_tokens: u32,
}

/// Captured result of running a single command (compile or execute) as part of a fixture check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandEvidence {
    pub program: String,
    pub args: Vec<String>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub passed: bool,
}

/// Evidence for a single language fixture: the prompt used, the generated source, the toolchain
/// that ran it, and whether its output matched the canonical expected output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageFixtureEvidence {
    pub language: FixtureLanguage,
    pub prompt: String,
    pub raw_generation: String,
    pub generated_source: String,
    pub toolchain_version: String,
    pub commands: Vec<CommandEvidence>,
    pub output_matches: bool,
    pub passed: bool,
}

/// Top-level persisted record of a Gemma 4 readiness run, covering the model artifact, sampling
/// configuration, expected output, and per-language fixture evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gemma4ReadinessEvidence {
    pub schema_version: String,
    pub generated_at: DateTime<Utc>,
    pub state: ReadinessState,
    pub artifact_path: String,
    pub artifact_sha256: String,
    pub runtime_revision: String,
    pub sampling: SamplingParameters,
    pub expected_output: String,
    pub fixtures: Vec<LanguageFixtureEvidence>,
}

/// Owns a temporary readiness workspace directory and best-effort removes it (including on
/// panic/unwind) when dropped.
struct WorkspaceGuard(PathBuf);

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Runs the four-language Gemma 4 readiness fixtures against `model_path` and returns the
/// resulting evidence record. `alias` identifies the model under test in the persisted
/// `artifact_path` so the evidence can be traced back to the configured model alias even when
/// `model_path` has no usable file name.
pub async fn run_gemma4_readiness(
    model_path: &Path,
    alias: &str,
) -> Result<Gemma4ReadinessEvidence> {
    let languages = [
        FixtureLanguage::Go,
        FixtureLanguage::Rust,
        FixtureLanguage::Python,
        FixtureLanguage::TypeScript,
    ];
    let prompts = languages
        .iter()
        .map(|language| language.prompt())
        .collect::<Vec<_>>();
    let generations = native::generate_gemma4_sources(model_path, &prompts, 256)?;
    let workspace_path =
        std::env::temp_dir().join(format!("rs-llmctl-readiness-{}", Uuid::new_v4()));
    fs::create_dir_all(&workspace_path)
        .with_context(|| format!("create readiness workspace {}", workspace_path.display()))?;
    // Owns the workspace directory and removes it on drop, including on unwind, so a panic
    // partway through fixture verification cannot leak the temporary workspace.
    let workspace = WorkspaceGuard(workspace_path);

    let fixtures = languages
        .into_iter()
        .zip(prompts)
        .zip(generations)
        .map(|((language, prompt), generation)| {
            verify_fixture(&workspace.0, language, prompt, generation)
        })
        .collect::<Result<Vec<_>>>()?;
    let state = readiness_state_from_fixtures(&fixtures);

    Ok(Gemma4ReadinessEvidence {
        schema_version: GEMMA4_READINESS_SCHEMA_VERSION.to_string(),
        generated_at: Utc::now(),
        state,
        artifact_path: alias.to_string(),
        artifact_sha256: model::sha256_file(model_path).await?,
        runtime_revision: format!("rs-llmctl/{}", env!("CARGO_PKG_VERSION")),
        sampling: SamplingParameters {
            strategy: "greedy".to_string(),
            temperature: "0".to_string(),
            max_tokens: 256,
        },
        expected_output: CANONICAL_TEN_LINE_OUTPUT.to_string(),
        fixtures,
    })
}

/// Derives the overall readiness state from per-language fixture results: `Qualified` only when
/// every fixture passed, otherwise `Quarantined`.
fn readiness_state_from_fixtures(fixtures: &[LanguageFixtureEvidence]) -> ReadinessState {
    if fixtures.iter().all(|fixture| fixture.passed) {
        ReadinessState::Qualified
    } else {
        ReadinessState::Quarantined
    }
}

/// Serializes `evidence` as pretty JSON and writes it to `path`, creating any missing parent
/// directories first.
pub fn write_evidence(path: &Path, evidence: &Gemma4ReadinessEvidence) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create evidence directory {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_vec_pretty(evidence)?)
        .with_context(|| format!("write readiness evidence {}", path.display()))
}

/// Reads and parses the readiness evidence at `path`, returning its recorded state. Returns
/// `Quarantined` if the file is missing, unreadable, fails to parse, or was written by an
/// incompatible schema version.
pub fn read_state(path: &Path) -> ReadinessState {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Gemma4ReadinessEvidence>(&bytes).ok())
        .filter(|evidence| evidence.schema_version == GEMMA4_READINESS_SCHEMA_VERSION)
        .map(|evidence| evidence.state)
        .unwrap_or(ReadinessState::Quarantined)
}

/// Returns the path under `model_dir` where Gemma 4 readiness evidence for `alias` is persisted.
pub fn evidence_path(model_dir: &Path, alias: &str) -> PathBuf {
    model_dir
        .join("readiness")
        .join(format!("{alias}.gemma4.json"))
}

fn verify_fixture(
    workspace: &Path,
    language: FixtureLanguage,
    prompt: String,
    raw_generation: String,
) -> Result<LanguageFixtureEvidence> {
    let source = extract_source(&raw_generation);
    let fixture_dir = workspace.join(language.directory_name());
    fs::create_dir_all(&fixture_dir)?;
    fs::write(fixture_dir.join(language.file_name()), &source)?;

    let toolchain_version = toolchain_version(language);
    let commands = run_fixture_commands(&fixture_dir, language)?;
    let output_matches = commands.last().is_some_and(|command| {
        command.passed
            && command.stdout.lines().collect::<Vec<_>>()
                == CANONICAL_TEN_LINE_OUTPUT.lines().collect::<Vec<_>>()
    });
    let passed = output_matches && commands.iter().all(|command| command.passed);

    Ok(LanguageFixtureEvidence {
        language,
        prompt,
        raw_generation,
        generated_source: source,
        toolchain_version,
        commands,
        output_matches,
        passed,
    })
}

fn extract_source(generation: &str) -> String {
    let trimmed = generation.trim();
    if let Some(after_open) = trimmed.split_once("```").map(|(_, rest)| rest) {
        let after_language = after_open
            .split_once('\n')
            .map_or(after_open, |(_, source)| source);
        if let Some((source, _)) = after_language.split_once("```") {
            return format!("{}\n", source.trim());
        }
    }
    format!("{}\n", trimmed)
}

fn toolchain_version(language: FixtureLanguage) -> String {
    let (program, args): (&str, &[&str]) = match language {
        FixtureLanguage::Go => ("go", &["version"]),
        FixtureLanguage::Rust => ("rustc", &["--version"]),
        FixtureLanguage::Python => ("python3", &["--version"]),
        FixtureLanguage::TypeScript => ("node", &["--version"]),
    };
    Command::new(program)
        .args(args)
        .output()
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            format!("{stdout}{stderr}").trim().to_string()
        })
        .unwrap_or_else(|err| format!("unavailable: {err}"))
}

fn run_fixture_commands(
    fixture_dir: &Path,
    language: FixtureLanguage,
) -> Result<Vec<CommandEvidence>> {
    match language {
        FixtureLanguage::Go => Ok(vec![run_command(fixture_dir, "go", &["run", "main.go"])?]),
        FixtureLanguage::Rust => {
            let compile = run_command(fixture_dir, "rustc", &["main.rs", "-o", "fixture"])?;
            if !compile.passed {
                return Ok(vec![compile]);
            }
            Ok(vec![compile, run_command(fixture_dir, "./fixture", &[])?])
        }
        FixtureLanguage::Python => Ok(vec![run_command(fixture_dir, "python3", &["main.py"])?]),
        FixtureLanguage::TypeScript => Ok(vec![run_command(
            fixture_dir,
            "node",
            &["--experimental-strip-types", "main.ts"],
        )?]),
    }
}

fn run_command(cwd: &Path, program: &str, args: &[&str]) -> Result<CommandEvidence> {
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .output()
        .with_context(|| format!("run {program} {}", args.join(" ")))?;
    Ok(command_evidence(program, args, &output))
}

fn command_evidence(program: &str, args: &[&str], output: &Output) -> CommandEvidence {
    CommandEvidence {
        program: program.to_string(),
        args: args.iter().map(|arg| (*arg).to_string()).collect(),
        exit_code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        passed: output.status.success(),
    }
}

/// Returns an error if `evidence` is not in the `Qualified` state, instructing the caller to
/// inspect the persisted fixture evidence before proceeding.
pub fn ensure_qualified(evidence: &Gemma4ReadinessEvidence) -> Result<()> {
    if evidence.state != ReadinessState::Qualified {
        bail!("Gemma 4 readiness is quarantined; inspect persisted fixture evidence");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_output_is_exactly_ten_lines() {
        assert_eq!(
            CANONICAL_TEN_LINE_OUTPUT.lines().collect::<Vec<_>>(),
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        );
    }

    #[test]
    fn extracts_fenced_source_without_language_marker() {
        assert_eq!(extract_source("```go\npackage main\n```"), "package main\n");
    }

    #[test]
    fn plain_text_without_code_fences_is_returned_newline_terminated() {
        assert_eq!(
            extract_source("  package main\nfunc main() {}  "),
            "package main\nfunc main() {}\n"
        );
    }

    #[test]
    fn unclosed_code_fence_falls_back_to_raw_trimmed_text() {
        let generation = "```go\npackage main\nfunc main() {}";
        assert_eq!(
            extract_source(generation),
            format!("{}\n", generation.trim())
        );
    }

    #[test]
    fn missing_evidence_is_quarantined() {
        assert_eq!(
            read_state(Path::new("/path/that/does/not/exist")),
            ReadinessState::Quarantined
        );
    }

    #[test]
    fn canonical_sources_pass_all_configured_toolchains() -> Result<()> {
        let workspace = tempfile::tempdir()?;
        let fixtures = [
            (
                FixtureLanguage::Go,
                "package main\nimport \"fmt\"\nfunc main() { for i := 1; i <= 10; i++ { fmt.Println(i) } }\n",
            ),
            (
                FixtureLanguage::Rust,
                "fn main() { for i in 1..=10 { println!(\"{i}\"); } }\n",
            ),
            (
                FixtureLanguage::Python,
                "for i in range(1, 11):\n    print(i)\n",
            ),
            (
                FixtureLanguage::TypeScript,
                "for (let i: number = 1; i <= 10; i++) { console.log(i); }\n",
            ),
        ];

        for (language, source) in fixtures {
            let evidence = verify_fixture(
                workspace.path(),
                language,
                language.prompt(),
                source.to_string(),
            )?;
            assert!(evidence.passed, "{language:?}: {evidence:?}");
        }
        Ok(())
    }

    #[test]
    fn fixture_with_a_compile_error_fails_and_quarantines_overall_state() -> Result<()> {
        let workspace = tempfile::tempdir()?;
        let broken_source = "this is not valid rust\n".to_string();
        let broken_fixture = verify_fixture(
            workspace.path(),
            FixtureLanguage::Rust,
            FixtureLanguage::Rust.prompt(),
            broken_source,
        )?;
        assert!(!broken_fixture.passed);
        assert!(!broken_fixture.output_matches);

        let passing_fixture = verify_fixture(
            workspace.path(),
            FixtureLanguage::Python,
            FixtureLanguage::Python.prompt(),
            "for i in range(1, 11):\n    print(i)\n".to_string(),
        )?;
        assert!(passing_fixture.passed);

        assert_eq!(
            readiness_state_from_fixtures(std::slice::from_ref(&broken_fixture)),
            ReadinessState::Quarantined
        );
        assert_eq!(
            readiness_state_from_fixtures(&[broken_fixture, passing_fixture]),
            ReadinessState::Quarantined
        );
        Ok(())
    }

    #[test]
    fn persisted_real_model_evidence_is_qualified() -> Result<()> {
        let evidence: Gemma4ReadinessEvidence = serde_json::from_str(include_str!(
            "../evidence/gemma4-readiness/2026-06-15-gemma-4-12b-it-q4-k-m.json"
        ))?;
        assert_eq!(evidence.schema_version, GEMMA4_READINESS_SCHEMA_VERSION);
        assert_eq!(evidence.state, ReadinessState::Qualified);
        assert!(evidence.fixtures.iter().all(|fixture| fixture.passed));
        Ok(())
    }
}
