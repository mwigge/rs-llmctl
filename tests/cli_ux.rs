use chrono::{Duration, Utc};
use rs_llmctl::audit::{AuditEvent, UsageEvent};
use rs_llmctl::storage::Storage;
use serde_json::json;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use tempfile::TempDir;
use uuid::Uuid;

fn llmctl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_llmctl"))
}

fn write_config(dir: &TempDir) -> std::path::PathBuf {
    let config = dir.path().join("config.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    let body = format!(
        r#"
mode = "single"

[server]
host = "127.0.0.1"
port = 8765
worker_base_port = 18765
llama_server = "127.0.0.1:8080"
context_size = 8192

[security]
production = false
require_auth = false
bind_external = false
api_keys = []

[resources]
budget = 0.8
cpu_only = true
gpu_vendor = "auto"

[storage]
db_path = "{}"
model_dir = "{}"

[observability]
"#,
        db_path.display(),
        model_dir.display()
    );
    fs::write(&config, body).expect("write config");
    config
}

fn assert_success_json(mut command: Command) -> Value {
    let output = command.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("stdout is json")
}

fn assert_json_output(mut command: Command) -> Value {
    let output = command.output().expect("run llmctl");
    assert!(
        !output.stdout.is_empty(),
        "status: {}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("stdout is json")
}

fn read_config(path: &Path) -> String {
    fs::read_to_string(path).expect("read config")
}

fn assert_failure_stderr_contains(mut command: Command, needle: &str) {
    let output = command.output().expect("run llmctl");
    assert!(
        !output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(needle),
        "stderr did not contain {needle:?}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn usage_event(
    model: &str,
    actor: &str,
    team: &str,
    input_tokens: u64,
    output_tokens: u64,
    latency_ms: u64,
) -> UsageEvent {
    UsageEvent {
        id: Uuid::new_v4(),
        request_id: Uuid::new_v4(),
        at: Utc::now(),
        model: model.to_string(),
        actor: actor.to_string(),
        team: team.to_string(),
        input_tokens,
        output_tokens,
        latency_ms,
        status: "ok".to_string(),
    }
}

#[test]
fn swap_set_persists_hot_and_cold_modes_as_json() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut set_hot = llmctl();
    set_hot
        .arg("--config")
        .arg(&config)
        .arg("swap")
        .arg("set")
        .arg("--mode")
        .arg("hot-swap");
    let hot = assert_success_json(set_hot);
    assert_eq!(hot["status"], "set");
    assert_eq!(hot["mode"], "hot-swap");
    assert!(read_config(&config).contains("mode = \"hot-swap\""));

    let mut set_cold = llmctl();
    set_cold
        .arg("--config")
        .arg(&config)
        .arg("swap")
        .arg("set")
        .arg("--mode")
        .arg("cold-swap");
    let cold = assert_success_json(set_cold);
    assert_eq!(cold["mode"], "cold-swap");

    let mut show = llmctl();
    show.arg("--config").arg(&config).arg("swap").arg("show");
    let shown = assert_success_json(show);
    assert_eq!(shown["mode"], "cold-swap");
    assert_eq!(shown["models"], 0);
}

#[test]
fn swap_plan_emits_cold_swap_lifecycle_from_config_mode() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut set_cold = llmctl();
    set_cold
        .arg("--config")
        .arg(&config)
        .arg("swap")
        .arg("set")
        .arg("--mode")
        .arg("cold-swap");
    assert_success_json(set_cold);

    let mut plan = llmctl();
    plan.arg("--config")
        .arg(&config)
        .arg("swap")
        .arg("plan")
        .arg("--active")
        .arg("chat-v1")
        .arg("--replacement")
        .arg("chat-v2");
    let plan = assert_success_json(plan);

    assert_eq!(
        plan["steps"],
        serde_json::json!([
            { "worker_id": "chat-v1", "target": "draining" },
            { "worker_id": "chat-v1", "target": "stopping" },
            { "worker_id": "chat-v1", "target": "stopped" },
            { "worker_id": "chat-v2", "target": "starting" },
            { "worker_id": "chat-v2", "target": "ready" }
        ])
    );
}

#[test]
fn swap_plan_emits_hot_swap_lifecycle_from_config_mode() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut set_hot = llmctl();
    set_hot
        .arg("--config")
        .arg(&config)
        .arg("swap")
        .arg("set")
        .arg("--mode")
        .arg("hot-swap");
    assert_success_json(set_hot);

    let mut plan = llmctl();
    plan.arg("--config")
        .arg(&config)
        .arg("swap")
        .arg("plan")
        .arg("--active")
        .arg("chat-v1")
        .arg("--replacement")
        .arg("chat-v2");
    let plan = assert_success_json(plan);

    assert_eq!(
        plan["steps"],
        serde_json::json!([
            { "worker_id": "chat-v2", "target": "starting" },
            { "worker_id": "chat-v2", "target": "warming" },
            { "worker_id": "chat-v2", "target": "ready" },
            { "worker_id": "chat-v1", "target": "draining" },
            { "worker_id": "chat-v1", "target": "stopping" },
            { "worker_id": "chat-v1", "target": "stopped" }
        ])
    );
}

#[test]
fn swap_plan_rejects_unsupported_modes() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut plan = llmctl();
    plan.arg("--config")
        .arg(&config)
        .arg("swap")
        .arg("plan")
        .arg("--active")
        .arg("chat-v1")
        .arg("--replacement")
        .arg("chat-v2");
    let output = plan.output().expect("run llmctl");

    assert!(
        !output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("swap plan is only supported for cold-swap or hot-swap modes"));
    assert!(stderr.contains("current mode is single"));
}

#[test]
fn server_status_reports_readiness_details_without_secret_config_values() {
    let dir = TempDir::new().expect("tempdir");
    let config = dir.path().join("config.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    let api_key_hash = sha256(b"server-status-token");
    let secret_path = model_dir.join("chat.gguf");
    let body = format!(
        r#"
mode = "hot-swap"

[server]
host = "0.0.0.0"
port = 8765
worker_base_port = 18765
llama_server = "127.0.0.1:8080"
context_size = 8192

[security]
production = false
require_auth = true
bind_external = true
api_keys = [
  {{ id = "operator", sha256 = "{api_key_hash}", subject = "alice", team = "platform", scopes = ["admin"] }}
]

[resources]
budget = 0.8
cpu_only = true
gpu_vendor = "auto"

[storage]
db_path = "{}"
model_dir = "{}"

[observability]

[[models]]
alias = "chat"
path = "{}"
role = "chat"
weight = 1

[[models]]
alias = "embed"
path = "{}"
role = "embedding"
weight = 1
"#,
        db_path.display(),
        model_dir.display(),
        secret_path.display(),
        model_dir.join("embed.gguf").display()
    );
    fs::write(&config, body).expect("write config");

    let mut status = llmctl();
    status
        .arg("--config")
        .arg(&config)
        .arg("server")
        .arg("status");
    let output = status.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let raw = String::from_utf8(output.stdout).expect("stdout utf8");
    let status: Value = serde_json::from_str(&raw).expect("stdout is json");
    assert_eq!(status["status"], "ready");
    assert_eq!(status["mode"], "hot-swap");
    assert_eq!(status["models"]["configured"], 2);
    assert_eq!(
        status["models"]["aliases"],
        serde_json::json!(["chat", "embed"])
    );
    assert_eq!(status["workers"]["planned"], 2);
    assert_eq!(status["storage"]["ready"], true);
    assert_eq!(status["auth"]["required"], true);
    assert_eq!(status["external_bind"]["enabled"], true);
    assert!(!raw.contains("server-status-token"));
    assert!(!raw.contains(&api_key_hash));
    assert!(!raw.contains(&secret_path.display().to_string()));
}

#[test]
fn integration_aqe_contract_exports_openai_governance_contract_without_secrets() {
    let dir = TempDir::new().expect("tempdir");
    let config = dir.path().join("config.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    let api_key_hash = sha256(b"aqe-contract-token");
    let chat_path = model_dir.join("chat-private.gguf");
    let review_path = model_dir.join("review-private.gguf");
    let body = format!(
        r#"
mode = "weighted"

[server]
host = "0.0.0.0"
port = 8765
worker_base_port = 18765
llama_server = "http://upstream.internal:8080"
context_size = 8192

[security]
production = true
require_auth = true
bind_external = true

[[security.api_keys]]
id = "operator"
sha256 = "{api_key_hash}"
subject = "alice"
team = "platform"
scopes = ["admin", "chat", "models.read"]

[resources]
budget = 0.8
cpu_only = true
gpu_vendor = "auto"

[storage]
db_path = "{}"
model_dir = "{}"

[[models]]
alias = "chat"
path = "{}"
role = "chat"
weight = 10

[[models]]
alias = "review"
path = "{}"
role = "code"
weight = 5

[[quotas]]
subject = "alice"
team = "platform"
requests_per_minute = 30
tokens_per_day = 100000
max_concurrency = 4
allowed_models = ["chat", "review"]
"#,
        db_path.display(),
        model_dir.display(),
        chat_path.display(),
        review_path.display()
    );
    fs::write(&config, body).expect("write config");

    let mut contract = llmctl();
    contract
        .arg("--config")
        .arg(&config)
        .arg("integration")
        .arg("aqe-contract");
    let output = contract.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let raw = String::from_utf8(output.stdout).expect("stdout utf8");
    let contract: Value = serde_json::from_str(&raw).expect("stdout is json");
    assert_eq!(contract["kind"], "aqe-openai-governance-contract");
    assert_eq!(contract["endpoint"]["base_url"], "http://0.0.0.0:8765");
    assert_eq!(contract["endpoint"]["openai_paths"]["models"], "/v1/models");
    assert_eq!(
        contract["endpoint"]["openai_paths"]["chat_completions"],
        "/v1/chat/completions"
    );
    assert_eq!(
        contract["auth"]["required_scopes"],
        serde_json::json!(["chat", "models.read"])
    );
    assert_eq!(
        contract["response_headers"],
        serde_json::json!([
            "x-request-id",
            "x-llmctl-model-count",
            "x-llmctl-model",
            "x-llmctl-upstream-model",
            "x-llmctl-quota-decision"
        ])
    );
    assert_eq!(
        contract["quota_reporting"]["fields"],
        serde_json::json!([
            "team",
            "subject",
            "requests_per_minute",
            "tokens_per_day",
            "max_concurrency",
            "allowed_models"
        ])
    );
    assert_eq!(
        contract["team_reporting"]["teams"],
        serde_json::json!(["platform"])
    );
    assert_eq!(
        contract["model_aliases"],
        serde_json::json!([
            { "alias": "chat", "role": "chat", "weight": 10 },
            { "alias": "review", "role": "code", "weight": 5 }
        ])
    );

    assert!(!raw.contains("aqe-contract-token"));
    assert!(!raw.contains(&api_key_hash));
    assert!(!raw.contains(&chat_path.display().to_string()));
    assert!(!raw.contains(&review_path.display().to_string()));
    assert!(!raw.contains("upstream.internal"));
    assert!(!raw.contains("http://upstream.internal:8080"));
}

#[test]
fn server_plan_exports_cpu_startup_plan_with_command_specs_without_secrets() {
    let dir = TempDir::new().expect("tempdir");
    let config = dir.path().join("config.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    let api_key_hash = sha256(b"server-plan-token");
    let chat_path = model_dir.join("chat.gguf");
    let coder_path = model_dir.join("coder.gguf");
    let body = format!(
        r#"
mode = "cold-swap"

[server]
host = "127.0.0.1"
port = 8765
worker_base_port = 19000
llama_server = "/usr/local/bin/llama-server"
context_size = 4096

[security]
production = false
require_auth = true
bind_external = false
api_keys = [
  {{ id = "operator", sha256 = "{api_key_hash}", subject = "alice", team = "platform", scopes = ["admin"] }}
]

[resources]
budget = 0.8
cpu_only = true
gpu_vendor = "nvidia"

[storage]
db_path = "{}"
model_dir = "{}"

[[models]]
alias = "chat"
path = "{}"
role = "chat"
weight = 1

[[models]]
alias = "coder"
path = "{}"
role = "code"
weight = 1
"#,
        db_path.display(),
        model_dir.display(),
        chat_path.display(),
        coder_path.display()
    );
    fs::write(&config, body).expect("write config");

    let mut plan = llmctl();
    plan.arg("--config").arg(&config).arg("server").arg("plan");
    let output = plan.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let raw = String::from_utf8(output.stdout).expect("stdout utf8");
    let plan: Value = serde_json::from_str(&raw).expect("stdout is startup plan json");
    assert_eq!(plan["workers"].as_array().expect("workers").len(), 2);
    assert_eq!(plan["workers"][0]["worker"]["id"], "chat");
    assert_eq!(plan["workers"][0]["worker"]["port"], 19000);
    assert_eq!(plan["workers"][0]["worker"]["backend"]["type"], "cpu");
    assert_eq!(
        plan["workers"][0]["command"]["program"],
        "/usr/local/bin/llama-server"
    );
    assert_eq!(
        plan["workers"][0]["command"]["args"],
        serde_json::json!([
            "--host",
            "127.0.0.1",
            "--port",
            "19000",
            "--model",
            chat_path.display().to_string(),
            "--ctx-size",
            "4096",
            "--n-gpu-layers",
            "0"
        ])
    );
    assert_eq!(plan["workers"][0]["command"]["env"], serde_json::json!([]));
    assert_eq!(plan["workers"][1]["worker"]["id"], "coder");
    assert_eq!(plan["workers"][1]["worker"]["port"], 19001);
    assert!(!raw.contains("server-plan-token"));
    assert!(!raw.contains(&api_key_hash));
}

#[test]
fn server_plan_exports_gpu_startup_plan_with_command_specs_without_secrets() {
    let dir = TempDir::new().expect("tempdir");
    let config = dir.path().join("config.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    let api_key_hash = sha256(b"gpu-server-plan-token");
    let model_path = model_dir.join("gpu-chat.gguf");
    let body = format!(
        r#"
mode = "hot-swap"

[server]
host = "127.0.0.1"
port = 8765
worker_base_port = 19100
llama_server = "llama-server"
context_size = 8192

[security]
production = false
require_auth = true
bind_external = false

[[security.api_keys]]
id = "operator"
sha256 = "{api_key_hash}"
subject = "bob"
team = "platform"
scopes = ["admin"]

[resources]
budget = 0.8
cpu_only = false
gpu_vendor = "nvidia"

[storage]
db_path = "{}"
model_dir = "{}"

[[models]]
alias = "gpu-chat"
path = "{}"
role = "chat"
weight = 1
"#,
        db_path.display(),
        model_dir.display(),
        model_path.display()
    );
    fs::write(&config, body).expect("write config");

    let mut plan = llmctl();
    plan.arg("--config").arg(&config).arg("server").arg("plan");
    let output = plan.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let raw = String::from_utf8(output.stdout).expect("stdout utf8");
    let plan: Value = serde_json::from_str(&raw).expect("stdout is startup plan json");
    assert_eq!(plan["workers"].as_array().expect("workers").len(), 1);
    assert_eq!(plan["workers"][0]["worker"]["id"], "gpu-chat");
    assert_eq!(
        plan["workers"][0]["worker"]["backend"],
        serde_json::json!({ "type": "nvidia", "gpu_layers": 99 })
    );
    assert_eq!(plan["workers"][0]["command"]["program"], "llama-server");
    assert_eq!(
        plan["workers"][0]["command"]["env"],
        serde_json::json!([["GGML_CUDA_VISIBLE_DEVICES", "0"]])
    );
    assert_eq!(
        plan["workers"][0]["command"]["args"],
        serde_json::json!([
            "--host",
            "127.0.0.1",
            "--port",
            "19100",
            "--model",
            model_path.display().to_string(),
            "--ctx-size",
            "8192",
            "--n-gpu-layers",
            "99"
        ])
    );
    assert!(!raw.contains("gpu-server-plan-token"));
    assert!(!raw.contains(&api_key_hash));
}

#[test]
fn server_plan_diff_reports_alias_changes_without_echoing_command_secrets() {
    let dir = TempDir::new().expect("tempdir");
    let old_plan = dir.path().join("old-plan.json");
    let new_plan = dir.path().join("new-plan.json");
    fs::write(
        &old_plan,
        serde_json::to_string_pretty(&json!({
            "workers": [
                {
                    "worker": {
                        "id": "chat",
                        "model": { "alias": "chat", "path": "/models/chat-v1.gguf", "role": "chat", "weight": 1 },
                        "bind_host": "127.0.0.1",
                        "port": 19000,
                        "context_size": 4096,
                        "backend": { "type": "cpu" }
                    },
                    "command": {
                        "program": "llama-server",
                        "args": ["--model", "/models/chat-v1.gguf", "--api-key", "old-command-secret"],
                        "env": [["LLAMA_TOKEN", "old-env-secret"]]
                    }
                },
                {
                    "worker": {
                        "id": "embed",
                        "model": { "alias": "embed", "path": "/models/embed.gguf", "role": "embedding", "weight": 1 },
                        "bind_host": "127.0.0.1",
                        "port": 19001,
                        "context_size": 4096,
                        "backend": { "type": "cpu" }
                    },
                    "command": {
                        "program": "llama-server",
                        "args": ["--model", "/models/embed.gguf"],
                        "env": []
                    }
                }
            ]
        }))
        .expect("serialize old plan"),
    )
    .expect("write old plan");
    fs::write(
        &new_plan,
        serde_json::to_string_pretty(&json!({
            "workers": [
                {
                    "worker": {
                        "id": "chat",
                        "model": { "alias": "chat", "path": "/models/chat-v2.gguf", "role": "chat", "weight": 1 },
                        "bind_host": "127.0.0.1",
                        "port": 19000,
                        "context_size": 4096,
                        "backend": { "type": "cpu" }
                    },
                    "command": {
                        "program": "llama-server",
                        "args": ["--model", "/models/chat-v2.gguf", "--api-key", "new-command-secret"],
                        "env": [["LLAMA_TOKEN", "new-env-secret"]]
                    }
                },
                {
                    "worker": {
                        "id": "coder",
                        "model": { "alias": "coder", "path": "/models/coder.gguf", "role": "code", "weight": 1 },
                        "bind_host": "127.0.0.1",
                        "port": 19001,
                        "context_size": 4096,
                        "backend": { "type": "cpu" }
                    },
                    "command": {
                        "program": "llama-server",
                        "args": ["--model", "/models/coder.gguf"],
                        "env": []
                    }
                }
            ]
        }))
        .expect("serialize new plan"),
    )
    .expect("write new plan");

    let mut plan_diff = llmctl();
    plan_diff
        .arg("server")
        .arg("plan-diff")
        .arg(&old_plan)
        .arg(&new_plan);
    let output = plan_diff.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let raw = String::from_utf8(output.stdout).expect("stdout utf8");
    let diff: Value = serde_json::from_str(&raw).expect("stdout is plan diff json");
    assert_eq!(diff["added"], json!(["coder"]));
    assert_eq!(diff["removed"], json!(["embed"]));
    assert_eq!(diff["changed"], json!(["chat"]));
    assert_eq!(diff["counts"]["added"], 1);
    assert_eq!(diff["counts"]["removed"], 1);
    assert_eq!(diff["counts"]["changed"], 1);
    assert!(!raw.contains("old-command-secret"));
    assert!(!raw.contains("new-command-secret"));
    assert!(!raw.contains("old-env-secret"));
    assert!(!raw.contains("new-env-secret"));
}

#[test]
fn data_export_and_audit_monthly_are_scriptable_json_reports() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut audit = llmctl();
    audit
        .arg("--config")
        .arg(&config)
        .arg("audit")
        .arg("request")
        .arg("--actor")
        .arg("operator")
        .arg("--action")
        .arg("model.swap.review")
        .arg("--resource")
        .arg("models");
    let event = assert_success_json(audit);
    assert_eq!(event["actor"], "operator");

    let mut export = llmctl();
    export
        .arg("--config")
        .arg(&config)
        .arg("data")
        .arg("export")
        .arg("--hours")
        .arg("1");
    let data = assert_success_json(export);
    assert_eq!(
        data["audit_events"].as_array().expect("audit_events").len(),
        1
    );
    assert!(data["usage_events"].is_array());
    assert!(data["observation_events"].is_array());
    assert!(data["quota_decisions"].is_array());
    assert!(data["models"].is_array());

    let mut monthly = llmctl();
    monthly
        .arg("--config")
        .arg(&config)
        .arg("audit")
        .arg("report")
        .arg("monthly")
        .arg("--year")
        .arg("2026")
        .arg("--month")
        .arg("5");
    let report = assert_success_json(monthly);
    assert_eq!(report["year"], 2026);
    assert_eq!(report["month"], 5);
    assert!(report["audit_events"].is_array());
    assert!(report["usage_summary"].is_object());
}

#[test]
fn audit_retention_plan_reports_counts_without_deleting_events() {
    let dir = TempDir::new().expect("tempdir");
    let config = dir.path().join("config.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    fs::write(
        &config,
        format!(
            r#"
mode = "single"

[server]
host = "127.0.0.1"
port = 8765
worker_base_port = 18765
llama_server = "127.0.0.1:8080"
context_size = 8192

[security]
production = false
require_auth = false
bind_external = false
api_keys = []

[resources]
budget = 0.8
cpu_only = true
gpu_vendor = "auto"

[storage]
db_path = "{}"
model_dir = "{}"

[observability]

[audit]
retention-days = 30
"#,
            db_path.display(),
            model_dir.display()
        ),
    )
    .expect("write config");

    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let storage = Storage::connect(&db_path).await.expect("storage");
        let mut old = AuditEvent::new(
            None,
            "operator",
            "platform",
            "old.audit",
            "audit",
            "ok",
            json!({ "source": "test" }),
        );
        old.at = Utc::now() - Duration::days(45);
        storage.insert_audit_event(&old).await.expect("insert old");

        let mut current = AuditEvent::new(
            None,
            "operator",
            "platform",
            "current.audit",
            "audit",
            "ok",
            json!({ "source": "test" }),
        );
        current.at = Utc::now() - Duration::days(7);
        storage
            .insert_audit_event(&current)
            .await
            .expect("insert current");
    });

    let mut plan = llmctl();
    plan.arg("--config")
        .arg(&config)
        .arg("audit")
        .arg("retention")
        .arg("plan");
    let plan = assert_success_json(plan);

    assert_eq!(plan["status"], "planned");
    assert_eq!(plan["operation"], "audit_retention");
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["deletes"], false);
    assert_eq!(plan["retention"]["days"], 30);
    assert!(plan["retention"]["cutoff"].is_string());
    assert_eq!(plan["counts"]["total"], 2);
    assert_eq!(plan["counts"]["in_retention_window"], 1);
    assert_eq!(plan["counts"]["outside_retention_window"], 1);

    let mut export = llmctl();
    export
        .arg("--config")
        .arg(&config)
        .arg("data")
        .arg("export")
        .arg("--hours")
        .arg("2000");
    let export = assert_success_json(export);
    assert_eq!(
        export["audit_events"]
            .as_array()
            .expect("audit_events")
            .len(),
        2
    );

    let raw = serde_json::to_string(&plan).expect("serialize plan");
    assert!(!raw.contains("secret"));
    assert!(!raw.contains("sha256"));
}

#[test]
fn audit_retention_plan_envelope_wraps_payload_with_hash_metadata() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let envelope_path = dir.path().join("retention-plan-envelope.json");

    let mut plan = llmctl();
    plan.arg("--config")
        .arg(&config)
        .arg("audit")
        .arg("retention")
        .arg("plan")
        .arg("--envelope");
    let envelope = assert_success_json(plan);

    assert_eq!(envelope["metadata"]["report_kind"], "retention_plan");
    assert!(
        envelope["metadata"]["sha256"]
            .as_str()
            .expect("sha256")
            .len()
            == 64
    );
    assert_eq!(envelope["payload"]["status"], "planned");
    assert_eq!(envelope["payload"]["operation"], "audit_retention");
    assert_eq!(envelope["payload"]["dry_run"], true);
    assert_eq!(envelope["payload"]["deletes"], false);

    fs::write(
        &envelope_path,
        serde_json::to_vec_pretty(&envelope).expect("serialize envelope"),
    )
    .expect("write envelope");

    let mut verify = llmctl();
    verify
        .arg("data")
        .arg("verify-envelope")
        .arg(&envelope_path);
    let verified = assert_success_json(verify);

    assert_eq!(verified["status"], "valid");
    assert_eq!(verified["valid"], true);
    assert_eq!(verified["expected_sha256"], envelope["metadata"]["sha256"]);
    assert_eq!(verified["actual_sha256"], envelope["metadata"]["sha256"]);
}

#[test]
fn report_envelopes_keep_payloads_and_include_metadata_hashes() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut audit = llmctl();
    audit
        .arg("--config")
        .arg(&config)
        .arg("audit")
        .arg("request")
        .arg("--actor")
        .arg("operator")
        .arg("--action")
        .arg("model.swap.review")
        .arg("--resource")
        .arg("models");
    let event = assert_success_json(audit);
    let request_id = event["id"].as_str().expect("event id");

    let mut monthly = llmctl();
    monthly
        .arg("--config")
        .arg(&config)
        .arg("audit")
        .arg("report")
        .arg("monthly")
        .arg("--year")
        .arg("2026")
        .arg("--month")
        .arg("5")
        .arg("--envelope");
    let monthly = assert_success_json(monthly);
    assert_eq!(monthly["metadata"]["report_kind"], "monthly_audit");
    assert!(
        monthly["metadata"]["sha256"]
            .as_str()
            .expect("sha256")
            .len()
            == 64
    );
    assert!(monthly["payload"]["audit_events"].is_array());

    let mut request = llmctl();
    request
        .arg("--config")
        .arg(&config)
        .arg("audit")
        .arg("report")
        .arg("request")
        .arg(request_id)
        .arg("--envelope");
    let request = assert_success_json(request);
    assert_eq!(request["metadata"]["report_kind"], "per_request_audit");
    assert!(request["metadata"]["sha256"].is_string());
    assert_eq!(request["payload"]["request_id"], request_id);

    let mut export = llmctl();
    export
        .arg("--config")
        .arg(&config)
        .arg("data")
        .arg("export")
        .arg("--hours")
        .arg("1")
        .arg("--envelope");
    let export = assert_success_json(export);
    assert_eq!(export["metadata"]["report_kind"], "data_export");
    assert!(export["metadata"]["sha256"].is_string());
    assert!(export["payload"]["audit_events"].is_array());
}

#[test]
fn data_verify_envelope_accepts_valid_offline_file() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let envelope_path = dir.path().join("export-envelope.json");

    let mut export = llmctl();
    export
        .arg("--config")
        .arg(&config)
        .arg("data")
        .arg("export")
        .arg("--hours")
        .arg("1")
        .arg("--envelope");
    let envelope = assert_success_json(export);
    fs::write(
        &envelope_path,
        serde_json::to_vec_pretty(&envelope).expect("serialize envelope"),
    )
    .expect("write envelope");

    let mut verify = llmctl();
    verify
        .arg("data")
        .arg("verify-envelope")
        .arg(&envelope_path);
    let verified = assert_success_json(verify);

    assert_eq!(verified["status"], "valid");
    assert_eq!(verified["valid"], true);
    assert_eq!(
        verified["path"].as_str().expect("path"),
        envelope_path.to_string_lossy()
    );
    assert_eq!(verified["expected_sha256"], envelope["metadata"]["sha256"]);
    assert_eq!(verified["actual_sha256"], envelope["metadata"]["sha256"]);
}

#[test]
fn data_verify_envelope_reports_tampered_payload_hash_mismatch() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let envelope_path = dir.path().join("tampered-envelope.json");

    let mut export = llmctl();
    export
        .arg("--config")
        .arg(&config)
        .arg("data")
        .arg("export")
        .arg("--hours")
        .arg("1")
        .arg("--envelope");
    let mut envelope = assert_success_json(export);
    envelope["payload"]["usage_summary"]["request_count"] = Value::from(99);
    fs::write(
        &envelope_path,
        serde_json::to_vec_pretty(&envelope).expect("serialize envelope"),
    )
    .expect("write envelope");

    let mut verify = llmctl();
    verify
        .arg("data")
        .arg("verify-envelope")
        .arg(&envelope_path);
    let verified = assert_json_output(verify);

    assert_eq!(verified["status"], "invalid");
    assert_eq!(verified["valid"], false);
    assert_eq!(verified["expected_sha256"], envelope["metadata"]["sha256"]);
    assert_ne!(verified["actual_sha256"], envelope["metadata"]["sha256"]);
}

#[test]
fn model_import_manifest_registers_multiple_offline_models() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let bundle = dir.path().join("bundle");
    fs::create_dir(&bundle).expect("create bundle");
    let chat_bytes = b"chat-model";
    let review_bytes = b"review-model";
    fs::write(bundle.join("chat.gguf"), chat_bytes).expect("write chat model");
    fs::write(bundle.join("review.gguf"), review_bytes).expect("write review model");
    let manifest = bundle.join("manifest.toml");
    fs::write(
        &manifest,
        format!(
            r#"
[[models]]
alias = "chat"
path = "chat.gguf"
role = "chat"
weight = 10
sha256 = "{}"

[[models]]
alias = "review"
path = "review.gguf"
role = "review"
weight = 3
sha256 = "{}"
"#,
            sha256(chat_bytes),
            sha256(review_bytes)
        ),
    )
    .expect("write manifest");

    let mut import = llmctl();
    import
        .arg("--config")
        .arg(&config)
        .arg("model")
        .arg("import-manifest")
        .arg(&manifest);
    let imported = assert_success_json(import);

    assert_eq!(imported["status"], "imported");
    assert_eq!(imported["imported"].as_array().expect("imported").len(), 2);
    assert_eq!(imported["models"][0]["alias"], "chat");
    assert_eq!(imported["models"][1]["alias"], "review");
    let saved = read_config(&config);
    assert!(saved.contains("alias = \"chat\""));
    assert!(saved.contains("alias = \"review\""));
}

#[test]
fn model_inventory_reports_configured_models_without_full_paths() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let bundle = dir.path().join("bundle");
    fs::create_dir(&bundle).expect("create bundle");
    fs::write(bundle.join("chat.gguf"), b"chat-model").expect("write chat model");
    fs::write(bundle.join("review.gguf"), b"review-model").expect("write review model");
    let manifest = bundle.join("manifest.toml");
    fs::write(
        &manifest,
        format!(
            r#"
[[models]]
alias = "chat"
path = "chat.gguf"
role = "chat"
weight = 10
sha256 = "{}"

[[models]]
alias = "review"
path = "review.gguf"
role = "review"
weight = 3
sha256 = "{}"
"#,
            sha256(b"chat-model"),
            sha256(b"review-model")
        ),
    )
    .expect("write manifest");

    let mut import = llmctl();
    import
        .arg("--config")
        .arg(&config)
        .arg("model")
        .arg("import-manifest")
        .arg(&manifest);
    assert_success_json(import);

    let mut inventory = llmctl();
    inventory
        .arg("--config")
        .arg(&config)
        .arg("model")
        .arg("inventory");
    let output = inventory.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let raw = String::from_utf8(output.stdout).expect("stdout utf8");
    let inventory: Value = serde_json::from_str(&raw).expect("stdout is json");

    assert_eq!(inventory["configured"], 2);
    assert_eq!(inventory["models"][0]["alias"], "chat");
    assert_eq!(inventory["models"][0]["role"], "chat");
    assert_eq!(inventory["models"][0]["weight"], 10);
    assert_eq!(inventory["models"][0]["path"], "chat.gguf");
    assert!(inventory["models"][0]["updated_at"].is_string());
    assert_eq!(inventory["models"][1]["alias"], "review");
    assert_eq!(inventory["models"][1]["path"], "review.gguf");
    assert!(inventory["models"][1]["updated_at"].is_string());
    assert!(!raw.contains(&dir.path().display().to_string()));
    assert!(!raw.contains(&bundle.display().to_string()));
}

#[test]
fn quota_status_and_report_are_scriptable_json() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut set = llmctl();
    set.arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("set")
        .arg("--subject")
        .arg("alice")
        .arg("--team")
        .arg("platform")
        .arg("--requests-per-minute")
        .arg("2")
        .arg("--tokens-per-day")
        .arg("100")
        .arg("--model")
        .arg("llama");
    assert_success_json(set);

    let mut status = llmctl();
    status
        .arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("status")
        .arg("--subject")
        .arg("alice")
        .arg("--model")
        .arg("llama");
    let status = assert_success_json(status);
    assert_eq!(status["subject"], "alice");
    assert_eq!(status["team"], "platform");
    assert_eq!(status["model"], "llama");
    assert_eq!(status["allowed"], true);
    assert_eq!(status["usage"]["requests_last_minute"], 0);
    assert_eq!(status["usage"]["tokens_today"], 0);
    assert_eq!(status["policy"]["requests_per_minute"], 2);

    let mut report = llmctl();
    report
        .arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("report")
        .arg("--hours")
        .arg("1");
    let report = assert_success_json(report);
    assert_eq!(report["hours"], 1);
    assert_eq!(report["policies"].as_array().expect("policies").len(), 1);
    assert_eq!(report["policy_summary"]["policy_count"], 1);
    assert_eq!(report["policy_summary"]["total_requests_per_minute"], 2);
    assert_eq!(report["policy_summary"]["total_tokens_per_day"], 100);
    assert_eq!(report["policy_summary"]["by_team"][0]["team"], "platform");
    assert_eq!(
        report["policy_summary"]["by_team"][0]["subjects"],
        json!(["alice"])
    );
    assert_eq!(
        report["policy_summary"]["by_team"][0]["allowed_models"],
        json!(["llama"])
    );
    assert!(report["decisions"].is_array());
    assert!(report["usage_summary"].is_object());
}

#[test]
fn quota_export_import_round_trips_json_and_preserves_config_fields() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut set = llmctl();
    set.arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("set")
        .arg("--subject")
        .arg("alice")
        .arg("--team")
        .arg("platform")
        .arg("--requests-per-minute")
        .arg("3")
        .arg("--tokens-per-day")
        .arg("1000")
        .arg("--model")
        .arg("llama")
        .arg("--model")
        .arg("mistral");
    assert_success_json(set);

    let mut export = llmctl();
    export
        .arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("export");
    let exported = assert_success_json(export);
    assert_eq!(exported["status"], "exported");
    assert_eq!(exported["format"], "json");
    assert_eq!(exported["count"], 1);
    assert_eq!(exported["quotas"][0]["subject"], "alice");

    let import_path = dir.path().join("quotas.json");
    fs::write(
        &import_path,
        serde_json::to_vec_pretty(&exported).expect("json"),
    )
    .expect("write");

    let before = read_config(&config);
    assert!(before.contains("[server]"));
    assert!(before.contains("worker_base_port = 18765"));

    let mut replace = llmctl();
    replace
        .arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("set")
        .arg("--subject")
        .arg("bob")
        .arg("--requests-per-minute")
        .arg("9");
    assert_success_json(replace);

    let mut import = llmctl();
    import
        .arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("import")
        .arg(&import_path);
    let imported = assert_success_json(import);
    assert_eq!(imported["status"], "imported");
    assert_eq!(imported["format"], "json");
    assert_eq!(imported["count"], 1);
    assert_eq!(imported["quotas"][0]["subject"], "alice");

    let saved = read_config(&config);
    assert!(saved.contains("[server]"));
    assert!(saved.contains("worker_base_port = 18765"));
    assert!(saved.contains("subject = \"alice\""));
    assert!(!saved.contains("subject = \"bob\""));
}

#[test]
fn quota_import_accepts_toml_policy_file_and_validates_limits() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let policy = dir.path().join("quotas.toml");
    fs::write(
        &policy,
        r#"
[[quotas]]
subject = "team-default"
team = "platform"
requests_per_minute = 10
tokens_per_day = 2000
max_concurrency = 2
allowed_models = ["llama"]
"#,
    )
    .expect("write policy");

    let mut import = llmctl();
    import
        .arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("import")
        .arg(&policy);
    let imported = assert_success_json(import);
    assert_eq!(imported["status"], "imported");
    assert_eq!(imported["format"], "toml");
    assert_eq!(imported["quotas"][0]["subject"], "team-default");

    let invalid_policy = dir.path().join("invalid-quotas.toml");
    fs::write(
        &invalid_policy,
        r#"
[[quotas]]
subject = "broken"
team = "platform"
requests_per_minute = 0
tokens_per_day = 2000
max_concurrency = 1
allowed_models = ["llama"]
"#,
    )
    .expect("write invalid policy");

    let mut invalid = llmctl();
    invalid
        .arg("--config")
        .arg(&config)
        .arg("quota")
        .arg("import")
        .arg(&invalid_policy);
    let output = invalid.output().expect("run llmctl");
    assert!(
        !output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("requests_per_minute"));

    let saved = read_config(&config);
    assert!(saved.contains("subject = \"team-default\""));
    assert!(!saved.contains("subject = \"broken\""));
}

#[test]
fn quota_import_rejects_invalid_json_policy_cases() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let cases = [
        (
            "duplicate-subject.json",
            r#"
{
  "quotas": [
    { "subject": "alice", "team": "platform", "requests_per_minute": 1, "tokens_per_day": 100, "max_concurrency": 1, "allowed_models": ["chat"] },
    { "subject": "alice", "team": "research", "requests_per_minute": 2, "tokens_per_day": 200, "max_concurrency": 1, "allowed_models": ["chat"] }
  ]
}
"#,
            "duplicate subject",
        ),
        (
            "duplicate-team.json",
            r#"
{
  "quotas": [
    { "subject": "alice", "team": "platform", "requests_per_minute": 1, "tokens_per_day": 100, "max_concurrency": 1, "allowed_models": ["chat"] },
    { "subject": "bob", "team": "platform", "requests_per_minute": 2, "tokens_per_day": 200, "max_concurrency": 1, "allowed_models": ["chat"] }
  ]
}
"#,
            "duplicate team",
        ),
        (
            "zero-limit.json",
            r#"
[
  { "subject": "alice", "team": "platform", "requests_per_minute": 0, "tokens_per_day": 100, "max_concurrency": 1, "allowed_models": ["chat"] }
]
"#,
            "requests_per_minute",
        ),
        (
            "negative-limit.json",
            r#"
[
  { "subject": "alice", "team": "platform", "requests_per_minute": -1, "tokens_per_day": 100, "max_concurrency": 1, "allowed_models": ["chat"] }
]
"#,
            "expected u32",
        ),
        (
            "overflow-limit.json",
            r#"
[
  { "subject": "alice", "team": "platform", "requests_per_minute": 1, "tokens_per_day": 18446744073709551616, "max_concurrency": 1, "allowed_models": ["chat"] }
]
"#,
            "expected u64",
        ),
        (
            "whitespace-alias.json",
            r#"
[
  { "subject": "alice", "team": "platform", "requests_per_minute": 1, "tokens_per_day": 100, "max_concurrency": 1, "allowed_models": ["   "] }
]
"#,
            "allowed_models",
        ),
    ];

    for (name, body, needle) in cases {
        let policy = dir.path().join(name);
        fs::write(&policy, body).expect("write policy");

        let mut import = llmctl();
        import
            .arg("--config")
            .arg(&config)
            .arg("quota")
            .arg("import")
            .arg(&policy);
        assert_failure_stderr_contains(import, needle);
    }
}

#[test]
fn quota_import_rejects_invalid_toml_policy_cases() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let cases = [
        (
            "duplicate-subject.toml",
            r#"
[[quotas]]
subject = "alice"
team = "platform"
requests_per_minute = 1
tokens_per_day = 100
max_concurrency = 1
allowed_models = ["chat"]

[[quotas]]
subject = "alice"
team = "research"
requests_per_minute = 2
tokens_per_day = 200
max_concurrency = 1
allowed_models = ["chat"]
"#,
            "duplicate subject",
        ),
        (
            "duplicate-team.toml",
            r#"
[[quotas]]
subject = "alice"
team = "platform"
requests_per_minute = 1
tokens_per_day = 100
max_concurrency = 1
allowed_models = ["chat"]

[[quotas]]
subject = "bob"
team = "platform"
requests_per_minute = 2
tokens_per_day = 200
max_concurrency = 1
allowed_models = ["chat"]
"#,
            "duplicate team",
        ),
        (
            "zero-limit.toml",
            r#"
[[quotas]]
subject = "alice"
team = "platform"
requests_per_minute = 1
tokens_per_day = 0
max_concurrency = 1
allowed_models = ["chat"]
"#,
            "tokens_per_day",
        ),
        (
            "negative-limit.toml",
            r#"
[[quotas]]
subject = "alice"
team = "platform"
requests_per_minute = -1
tokens_per_day = 100
max_concurrency = 1
allowed_models = ["chat"]
"#,
            "expected u32",
        ),
        (
            "overflow-limit.toml",
            r#"
[[quotas]]
subject = "alice"
team = "platform"
requests_per_minute = 1
tokens_per_day = 18446744073709551616
max_concurrency = 1
allowed_models = ["chat"]
"#,
            "number too large",
        ),
        (
            "whitespace-alias.toml",
            r#"
[[quotas]]
subject = "alice"
team = "platform"
requests_per_minute = 1
tokens_per_day = 100
max_concurrency = 1
allowed_models = ["   "]
"#,
            "allowed_models",
        ),
    ];

    for (name, body, needle) in cases {
        let policy = dir.path().join(name);
        fs::write(&policy, body).expect("write policy");

        let mut import = llmctl();
        import
            .arg("--config")
            .arg(&config)
            .arg("quota")
            .arg("import")
            .arg(&policy);
        assert_failure_stderr_contains(import, needle);
    }
}

#[tokio::test]
async fn usage_chargeback_reports_json_with_team_and_actor_filters_without_secrets() {
    let dir = TempDir::new().expect("tempdir");
    let config = dir.path().join("config.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    let api_key_hash = sha256(b"chargeback-secret-token");
    fs::write(
        &config,
        format!(
            r#"
mode = "single"

[server]
host = "127.0.0.1"
port = 8765
worker_base_port = 18765
llama_server = "127.0.0.1:8080"
context_size = 8192

[security]
production = false
require_auth = true
bind_external = false

[[security.api_keys]]
id = "operator"
sha256 = "{api_key_hash}"
subject = "alice"
team = "platform"
scopes = ["admin"]

[resources]
budget = 0.8
cpu_only = true
gpu_vendor = "auto"

[storage]
db_path = "{}"
model_dir = "{}"

[observability]
"#,
            db_path.display(),
            model_dir.display()
        ),
    )
    .expect("write config");

    let storage = Storage::connect(&db_path).await.expect("connect storage");
    storage
        .insert_usage_event(&usage_event("llama", "alice", "platform", 10, 20, 100))
        .await
        .expect("insert alice usage");
    storage
        .insert_usage_event(&usage_event("llama", "bob", "platform", 1, 2, 50))
        .await
        .expect("insert bob usage");
    storage
        .insert_usage_event(&usage_event("mistral", "alice", "research", 4, 5, 150))
        .await
        .expect("insert research usage");
    let mut old_usage = usage_event("llama", "alice", "platform", 100, 100, 500);
    old_usage.at = Utc::now() - Duration::hours(3);
    storage
        .insert_usage_event(&old_usage)
        .await
        .expect("insert old usage");

    let mut chargeback = llmctl();
    chargeback
        .arg("--config")
        .arg(&config)
        .arg("usage")
        .arg("chargeback")
        .arg("--hours")
        .arg("1")
        .arg("--team")
        .arg("platform")
        .arg("--actor")
        .arg("alice");
    let output = chargeback.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let raw = String::from_utf8(output.stdout).expect("stdout utf8");
    let report: Value = serde_json::from_str(&raw).expect("stdout is json");

    assert_eq!(report["hours"], 1);
    assert!(report["from"].is_string());
    assert!(report["to"].is_string());
    assert!(report["generated_at"].is_string());
    assert_eq!(report["filters"]["team"], "platform");
    assert_eq!(report["filters"]["actor"], "alice");
    assert_eq!(report["usage_summary"]["request_count"], 1);
    assert_eq!(report["usage_summary"]["input_tokens"], 10);
    assert_eq!(report["usage_summary"]["output_tokens"], 20);
    assert_eq!(report["usage_summary"]["total_tokens"], 30);
    assert_eq!(report["usage_summary"]["by_team"][0]["key"], "platform");
    assert_eq!(report["usage_summary"]["by_actor"][0]["key"], "alice");
    assert_eq!(report["usage_summary"]["by_model"][0]["key"], "llama");
    assert!(report.get("usage_events").is_none());
    assert!(!raw.contains("chargeback-secret-token"));
    assert!(!raw.contains(&api_key_hash));
}

#[test]
fn security_check_and_observe_plan_are_top_level_json_commands() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let mut security = llmctl();
    security
        .arg("--config")
        .arg(&config)
        .arg("security")
        .arg("check");
    let security = assert_success_json(security);
    assert_eq!(security["status"], "ok");
    assert_eq!(security["require_auth"], false);
    assert_eq!(security["api_keys"], 0);

    let mut plan = llmctl();
    plan.arg("--config").arg(&config).arg("observe").arg("plan");
    let plan = assert_success_json(plan);
    assert_eq!(plan["service_name"], "rs-llmctl");
    assert_eq!(plan["traces_enabled"], true);
    assert_eq!(plan["metrics_enabled"], true);
    assert_eq!(plan["logs_enabled"], true);
    assert_eq!(plan["exporter"]["type"], "none");
}

#[test]
fn compliance_evidence_reports_cra_pci_and_release_integrity() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = write_config(&dir);

    let mut evidence = llmctl();
    evidence
        .arg("--config")
        .arg(&config)
        .arg("compliance")
        .arg("evidence");
    let evidence = assert_success_json(evidence);

    assert_eq!(
        evidence["cra_article_14"]["regulatory_start_date"],
        "2026-09-11"
    );
    assert_eq!(
        evidence["pci_dss"]["baseline"],
        "pci_dss_v4_0_1_aligned_controls"
    );
    assert_eq!(
        evidence["release_integrity"]["sbom"],
        "packaging/generate-sbom.sh"
    );
    assert_eq!(
        evidence["release_integrity"]["signing"],
        "packaging/sign-release.sh"
    );
}

#[test]
fn security_hash_key_outputs_sha256_metadata_without_plaintext() {
    let secret = "sk-test-super-secret-0123456789abcd";
    let mut hash = llmctl();
    hash.arg("security")
        .arg("hash-key")
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = hash.spawn().expect("spawn llmctl");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(secret.as_bytes())
        .expect("write stdin");
    let output = child.wait_with_output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stdout.contains(secret));
    assert!(!stderr.contains(secret));

    let report: Value = serde_json::from_slice(&output.stdout).expect("stdout is json");
    assert_eq!(report["sha256"], sha256(secret.as_bytes()));
    assert_eq!(report["metadata"]["algorithm"], "sha256");
    assert_eq!(report["metadata"]["encoding"], "hex");
    assert_eq!(report["metadata"]["purpose"], "api-key");
    assert!(report.get("secret").is_none());
    assert!(report["metadata"].get("secret").is_none());
}

#[test]
fn security_hash_key_does_not_require_config_file() {
    let dir = TempDir::new().expect("tempdir");
    let missing_config = dir.path().join("missing.toml");
    let secret = "standalone-admin-secret-0123456789abcd";

    let mut hash = llmctl();
    hash.arg("--config")
        .arg(&missing_config)
        .arg("security")
        .arg("hash-key")
        .arg("--env")
        .arg("LLMCTL_TEST_HASH_KEY_SECRET")
        .env("LLMCTL_TEST_HASH_KEY_SECRET", secret);

    let report = assert_success_json(hash);
    assert_eq!(report["sha256"], sha256(secret.as_bytes()));
    assert_eq!(report["metadata"]["input"], "env");
}

#[test]
fn security_add_key_inserts_and_updates_hashed_api_key_without_leaking_digest() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let digest = sha256(b"admin-token");
    let replacement_digest = sha256(b"replacement-admin-token");

    let mut add = llmctl();
    add.arg("--config")
        .arg(&config)
        .arg("security")
        .arg("add-key")
        .arg("--id")
        .arg("operator")
        .arg("--sha256")
        .arg(&digest)
        .arg("--subject")
        .arg("alice")
        .arg("--team")
        .arg("platform")
        .arg("--scope")
        .arg("admin")
        .arg("--scope")
        .arg("models.read");
    let output = add.output().expect("run llmctl");
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stdout.contains(&digest));
    assert!(!stderr.contains(&digest));
    let report: Value = serde_json::from_slice(&output.stdout).expect("stdout is json");
    assert_eq!(report["status"], "saved");
    assert_eq!(report["action"], "inserted");
    assert_eq!(report["key"]["id"], "operator");
    assert_eq!(report["key"]["subject"], "alice");
    assert_eq!(report["key"]["team"], "platform");
    assert_eq!(
        report["key"]["scopes"],
        serde_json::json!(["admin", "models.read"])
    );
    assert!(report["key"].get("sha256").is_none());

    let saved = read_config(&config);
    assert!(saved.contains("[[security.api-keys]]"));
    let saved_toml: toml::Value = toml::from_str(&saved).expect("saved config is toml");
    let keys = saved_toml["security"]["api-keys"]
        .as_array()
        .expect("api keys");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0]["id"].as_str(), Some("operator"));
    assert_eq!(keys[0]["sha256"].as_str(), Some(digest.as_str()));
    assert_eq!(
        keys[0]["scopes"]
            .as_array()
            .expect("scopes")
            .iter()
            .map(|scope| scope.as_str().expect("scope").to_string())
            .collect::<Vec<_>>(),
        vec!["admin".to_string(), "models.read".to_string()]
    );

    let mut update = llmctl();
    update
        .arg("--config")
        .arg(&config)
        .arg("security")
        .arg("add-key")
        .arg("--id")
        .arg("operator")
        .arg("--sha256")
        .arg(&replacement_digest)
        .arg("--subject")
        .arg("alice")
        .arg("--team")
        .arg("ml")
        .arg("--scope")
        .arg("chat");
    let updated = assert_success_json(update);
    assert_eq!(updated["action"], "updated");
    assert_eq!(updated["api_keys"], 1);

    let saved = read_config(&config);
    assert!(!saved.contains(&format!("sha256 = \"{digest}\"")));
    assert!(saved.contains(&format!("sha256 = \"{replacement_digest}\"")));
    let saved_toml: toml::Value = toml::from_str(&saved).expect("saved config is toml");
    assert_eq!(
        saved_toml["security"]["api-keys"]
            .as_array()
            .expect("api keys")
            .len(),
        1
    );
}

#[test]
fn security_add_key_rejects_invalid_sha256_without_saving() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let before = read_config(&config);

    let mut add = llmctl();
    add.arg("--config")
        .arg(&config)
        .arg("security")
        .arg("add-key")
        .arg("--id")
        .arg("operator")
        .arg("--sha256")
        .arg("plain-secret")
        .arg("--subject")
        .arg("alice")
        .arg("--team")
        .arg("platform")
        .arg("--scope")
        .arg("admin");
    let output = add.output().expect("run llmctl");
    assert!(
        !output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stdout.contains("plain-secret"));
    assert!(!stderr.contains("plain-secret"));
    assert!(stderr.contains("sha256 must be 64 hexadecimal characters"));
    assert_eq!(read_config(&config), before);
}

#[test]
fn security_audit_config_reports_scriptable_posture_without_secrets() {
    let dir = TempDir::new().expect("tempdir");
    let config = dir.path().join("prod.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    let unit_path = dir.path().join("llmctld.service");
    fs::write(
        &unit_path,
        r#"
[Unit]
Description=rs-llmctl

[Service]
ExecStart=/usr/local/bin/llmctld --config /etc/rs-llmctl/config.toml
"#,
    )
    .expect("write unit");
    fs::write(
        &config,
        format!(
            r#"
mode = "single"

[server]
host = "0.0.0.0"
port = 8765
worker_base_port = 18765
llama_server = "127.0.0.1:8080"
context_size = 8192

[security]
production = true
require_auth = true
bind_external = true

[[security.api_keys]]
id = "operator"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
subject = "operator"
team = "platform"
scopes = ["admin"]

[resources]
budget = 0.8
cpu_only = true
gpu_vendor = "auto"

[storage]
db_path = "{}"
model_dir = "{}"

[observability]

[observability.exporter]
endpoint = "https://otel.example.test/v1/traces"
headers = {{ authorization = "env:OTEL_AUTH", "x-tenant" = "platform" }}

[audit]
retention-days = 90
report-directory = "/var/log/rs-llmctl/audit"
report-formats = ["json"]
monthly-reports = true
"#,
            db_path.display(),
            model_dir.display()
        ),
    )
    .expect("write config");

    let mut audit = llmctl();
    audit
        .arg("--config")
        .arg(&config)
        .arg("security")
        .arg("audit-config")
        .arg("--systemd-unit")
        .arg(&unit_path);
    let report = assert_success_json(audit);

    assert_eq!(report["status"], "ok");
    assert_eq!(
        report["config"].as_str().expect("config path"),
        config.to_string_lossy()
    );
    assert_eq!(report["external_bind"]["enabled"], true);
    assert_eq!(report["external_bind"]["host"], "0.0.0.0");
    assert_eq!(report["auth"]["require_auth"], true);
    assert_eq!(report["auth"]["api_key_count"], 1);
    assert_eq!(report["auth"]["hashed_api_keys"], true);
    assert_eq!(report["auth"]["keys"][0]["id"], "operator");
    assert_eq!(report["auth"]["keys"][0]["sha256_present"], true);
    assert!(report["auth"]["keys"][0].get("sha256").is_none());
    assert_eq!(
        report["observability"]["secret_headers"][0]["value_source"],
        "env"
    );
    assert_eq!(
        report["observability"]["secret_headers"][0]["reference"],
        "env:OTEL_AUTH"
    );
    assert_eq!(report["audit"]["retention_days"], 90);
    assert_eq!(report["audit"]["monthly_reports"], true);
    assert_eq!(report["systemd"]["checked"], true);
    assert_eq!(report["systemd"]["present"], true);
    assert_eq!(report["systemd"]["has_exec_start"], true);

    let output = serde_json::to_string(&report).expect("serialize report");
    assert!(!output.contains("0123456789abcdef"));
}

#[test]
fn security_audit_config_marks_invalid_scriptable_posture() {
    let dir = TempDir::new().expect("tempdir");
    let config = dir.path().join("bad.toml");
    let db_path = dir.path().join("llmctl.db");
    let model_dir = dir.path().join("models");
    fs::write(
        &config,
        format!(
            r#"
mode = "single"

[server]
host = "0.0.0.0"
port = 8765
worker_base_port = 18765
llama_server = "127.0.0.1:8080"
context_size = 8192

[security]
production = true
require_auth = false
bind_external = true

[[security.api_keys]]
id = "plain"
sha256 = "not-a-hash"
subject = "operator"
team = "platform"
scopes = []

[resources]
budget = 0.8
cpu_only = true
gpu_vendor = "auto"

[storage]
db_path = "{}"
model_dir = "{}"

[observability]

[observability.exporter]
headers = {{ authorization = "Bearer plaintext" }}

[audit]
retention-days = 0
"#,
            db_path.display(),
            model_dir.display()
        ),
    )
    .expect("write config");

    let mut audit = llmctl();
    audit
        .arg("--config")
        .arg(&config)
        .arg("security")
        .arg("audit-config");
    let report = assert_success_json(audit);

    assert_eq!(report["status"], "warning");
    assert_eq!(report["external_bind"]["enabled"], true);
    assert_eq!(report["auth"]["require_auth"], false);
    assert_eq!(report["auth"]["hashed_api_keys"], false);
    assert_eq!(
        report["observability"]["secret_headers"][0]["value_source"],
        "plaintext"
    );
    assert_eq!(report["audit"]["retention_days"], 0);
    assert_eq!(report["systemd"]["checked"], false);
    assert!(report["findings"]
        .as_array()
        .expect("findings")
        .iter()
        .any(|finding| finding == "external/production serving requires authentication"));

    let output = serde_json::to_string(&report).expect("serialize report");
    assert!(!output.contains("Bearer plaintext"));
    assert!(!output.contains("not-a-hash"));
}
