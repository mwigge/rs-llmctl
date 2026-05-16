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

#[test]
fn security_hash_key_outputs_sha256_metadata_without_plaintext() {
    let secret = "sk-test-super-secret";
    let mut hash = llmctl();
    hash.arg("security").arg("hash-key").arg(secret);

    let output = hash.output().expect("run llmctl");
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
    let secret = "standalone-admin-secret";

    let mut hash = llmctl();
    hash.arg("--config")
        .arg(&missing_config)
        .arg("security")
        .arg("hash-key")
        .arg(secret);

    let report = assert_success_json(hash);
    assert_eq!(report["sha256"], sha256(secret.as_bytes()));
    assert_eq!(report["metadata"]["input"], "argument");
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
