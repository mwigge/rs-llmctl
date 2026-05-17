use serde_json::Value;
use std::fs;
use std::process::Command;
use tempfile::TempDir;

fn llmctl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_llmctl"))
}

fn write_config(dir: &TempDir) -> std::path::PathBuf {
    let config = dir.path().join("config.toml");
    let db_path = dir.path().join("state").join("llmctl.db");
    let model_dir = dir.path().join("models");
    let model_path = model_dir.join("chat.gguf");
    let body = format!(
        r#"
mode = "single"

[server]
host = "127.0.0.1"
port = 8765
worker_base_port = 18765
context_size = 4096

[runtime]
backend = "candle-native"

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

[[models]]
alias = "chat"
path = "{}"
role = "chat"
weight = 10
"#,
        db_path.display(),
        model_dir.display(),
        model_path.display(),
    );
    fs::write(&config, body).expect("write config");
    config
}

#[test]
fn server_check_validates_storage_and_server_plan_prints_startup_plan_json() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);

    let check = llmctl()
        .arg("--config")
        .arg(&config)
        .arg("server")
        .arg("check")
        .output()
        .expect("run llmctl server check");
    assert!(
        check.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        check.status,
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(dir.path().join("state").join("llmctl.db").exists());
    assert!(dir.path().join("models").exists());

    let output = llmctl()
        .arg("--config")
        .arg(&config)
        .arg("server")
        .arg("plan")
        .output()
        .expect("run llmctl server plan");

    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let plan: Value = serde_json::from_slice(&output.stdout).expect("stdout is startup plan json");
    assert_eq!(plan["workers"].as_array().expect("workers").len(), 1);
    assert_eq!(plan["workers"][0]["worker"]["id"], "chat");
    assert_eq!(plan["workers"][0]["worker"]["port"], 18765);
    assert_eq!(plan["workers"][0]["worker"]["context_size"], 4096);
    assert_eq!(
        plan["workers"][0]["command"]["program"],
        "<in-process:candle-native>"
    );
    assert_eq!(plan["workers"][0]["command"]["args"][0], "--model");
}

#[test]
fn server_security_check_rejects_insecure_external_bind() {
    let dir = TempDir::new().expect("tempdir");
    let config = write_config(&dir);
    let body = fs::read_to_string(&config)
        .expect("read config")
        .replace("host = \"127.0.0.1\"", "host = \"0.0.0.0\"");
    fs::write(&config, body).expect("write insecure config");

    let output = llmctl()
        .arg("--config")
        .arg(&config)
        .arg("server")
        .arg("security-check")
        .output()
        .expect("run llmctl server security-check");

    assert!(
        !output.status.success(),
        "dry-run unexpectedly succeeded:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("requires authentication"),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
