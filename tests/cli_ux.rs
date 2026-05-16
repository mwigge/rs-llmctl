use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

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

fn read_config(path: &Path) -> String {
    fs::read_to_string(path).expect("read config")
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
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
    assert!(report["decisions"].is_array());
    assert!(report["usage_summary"].is_object());
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
