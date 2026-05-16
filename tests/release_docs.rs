use std::fs;
use std::os::unix::fs::PermissionsExt;

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
fn docs_cover_ordered_deployment_operations() {
    let docs = format!("{}\n{}", read("README.md"), read("llmctl-to-rust.md")).to_lowercase();

    let ordered_steps = [
        "1. import the offline install manifest",
        "2. run the dry-run validation gate",
        "3. run the security audit",
        "4. run readiness checks",
        "5. start the daemon under systemd",
        "6. verify aqe/openai client access",
        "7. export the audit envelope",
    ];

    let mut previous = 0;
    for step in ordered_steps {
        let index = docs
            .find(step)
            .unwrap_or_else(|| panic!("docs should include ordered deployment step `{step}`"));
        assert!(
            index >= previous,
            "deployment step `{step}` should appear after the previous step"
        );
        previous = index;
    }

    for command in [
        "model import-manifest ./manifest.toml",
        "server check",
        "security check",
        "observe plan",
        "systemctl enable --now llmctld.service",
        "openai_base_url=http://host:8765/v1",
        "data export --hours 24",
    ] {
        assert!(
            docs.contains(command),
            "ordered deployment docs should include `{command}`"
        );
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

#[test]
fn install_validation_artifact_is_safe_offline_and_pins_release_checks() {
    let script_path = "packaging/validate-install.sh";
    let script = read(script_path);
    let mode = fs::metadata(script_path)
        .unwrap_or_else(|err| panic!("metadata {script_path}: {err}"))
        .permissions()
        .mode();

    assert_ne!(mode & 0o111, 0, "{script_path} should be executable");

    for required in [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "CONFIG=${CONFIG:-/etc/rs-llmctl/config.toml}",
        "UNIT=${UNIT:-/etc/systemd/system/llmctld.service}",
        "LLMCTL=${LLMCTL:-llmctl}",
        "LLMCTLD=${LLMCTLD:-llmctld}",
        "\"${LLMCTLD}\" --config \"${CONFIG}\" --dry-run",
        "\"${LLMCTL}\" --config \"${CONFIG}\" security check",
        "\"${LLMCTL}\" --config \"${CONFIG}\" observe plan",
        "command -v systemd-analyze",
        "systemd-analyze verify \"${UNIT}\"",
        "systemd-analyze not found; skipping unit verification",
    ] {
        assert!(
            script.contains(required),
            "{script_path} should include `{required}`"
        );
    }

    for forbidden in [
        "curl ",
        "wget ",
        "apt ",
        "dnf ",
        "yum ",
        "pacman ",
        "systemctl start",
        "systemctl enable",
        "cargo install",
    ] {
        assert!(
            !script.contains(forbidden),
            "{script_path} should stay offline/passive and not include `{forbidden}`"
        );
    }

    let readme = read("README.md");
    assert!(
        readme.contains("packaging/validate-install.sh"),
        "README should document the install validation script"
    );
}
