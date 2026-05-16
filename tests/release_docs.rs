use std::fs;

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("read {path}: {err}"))
}

#[test]
fn ci_workflow_enforces_core_rust_gates() {
    let workflow = read(".github/workflows/ci.yml");

    for gate in [
        "cargo fmt --all -- --check",
        "cargo clippy --all-targets --all-features -- -D warnings",
        "cargo test --all-targets --all-features",
        "cargo build --release --bins",
    ] {
        assert!(workflow.contains(gate), "CI workflow should run `{gate}`");
    }
}

#[test]
fn package_manifest_publishes_expected_binaries() {
    let manifest = read("Cargo.toml");

    for expected in [
        r#"name = "rs-llmctl""#,
        r#"name = "llmctl""#,
        r#"path = "src/bin/llmctl.rs""#,
        r#"name = "llmctld""#,
        r#"path = "src/bin/llmctld.rs""#,
    ] {
        assert!(
            manifest.contains(expected),
            "Cargo.toml should declare `{expected}`"
        );
    }
}

#[test]
fn docs_cover_tdd_lints_and_enterprise_security_posture() {
    let docs = format!("{}\n{}", read("README.md"), read("llmctl-to-rust.md")).to_lowercase();

    for topic in [
        "tdd",
        "cargo fmt",
        "cargo clippy",
        "cargo test",
        "cargo build --release",
        "target/release/llmctl",
        "target/release/llmctld",
        "pci dss",
        "external bind",
        "offline install",
        "offline install manifest",
        "model import-manifest",
        "[[models]]",
        "sha256",
        "resource budget",
        "quota",
        "audit.retention-days",
        "observability.exporter.endpoint",
        "security.require-auth",
        "security.bind-external",
        "env:",
        "audit",
        "usage report",
        "aqe",
        "openai_base_url",
        "/usr/local/bin/llmctl",
        "/usr/local/bin/llmctld",
        "/etc/rs-llmctl/config.toml",
        "/var/lib/rs-llmctl/models",
        "llmctl --config /etc/rs-llmctl/config.toml server check",
        "llmctl --config /etc/rs-llmctl/config.toml security check",
        "llmctl --config /etc/rs-llmctl/config.toml observe plan",
        "systemd",
        "llmctld.service",
    ] {
        assert!(docs.contains(topic), "docs should cover `{topic}`");
    }
}

#[test]
fn systemd_template_documents_server_deployment_controls() {
    let unit = read("packaging/systemd/llmctld.service").to_lowercase();

    for required in [
        "[unit]",
        "after=network-online.target",
        "[service]",
        "type=simple",
        "user=llmctl",
        "group=llmctl",
        "environment=llmctl_config=/etc/rs-llmctl/config.toml",
        "execstart=/usr/local/bin/llmctld --config ${llmctl_config}",
        "nonewprivileges=true",
        "privatetmp=true",
        "protectsystem=strict",
        "protecthome=true",
        "readwritepaths=/var/lib/rs-llmctl /var/log/rs-llmctl",
        "restart=on-failure",
        "[install]",
        "wantedby=multi-user.target",
    ] {
        assert!(
            unit.contains(required),
            "systemd unit should include `{required}`"
        );
    }
}
