use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("read {path}: {err}"))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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
fn docs_pin_policy_operation_runbook_commands_without_plaintext_key_guidance() {
    let readme = read("README.md");
    let rust_plan = read("llmctl-to-rust.md");
    let runbook = read("examples/policy-operations-runbook.md");
    let docs = format!("{readme}\n{rust_plan}\n{runbook}");
    let docs_lower = docs.to_lowercase();

    for required in [
        "llmctl --config /etc/rs-llmctl/config.toml quota export > quotas.json",
        "llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.json",
        "llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.toml",
        "llmctl security hash-key \"$LLMCTL_NEW_API_KEY\"",
        "[[security.api_keys]]",
        "sha256 = \"<sha256-from-hash-key>\"",
        "llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.json",
        "server plan export",
    ] {
        assert!(
            docs.contains(required),
            "policy operation docs should include `{required}`"
        );
    }

    for forbidden in [
        "sha256 = \"sk-",
        "api_key = \"sk-",
        "api-key = \"sk-",
        "secret = \"sk-",
        "password = \"sk-",
        "bearer sk-",
        "authorization = \"bearer",
    ] {
        assert!(
            !docs_lower.contains(forbidden),
            "policy operation docs should not advise plaintext key pattern `{forbidden}`"
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

#[test]
fn config_staging_artifact_requires_operator_review_and_pins_examples() {
    let script_path = "packaging/stage-config.sh";
    let script = read(script_path);
    let mode = fs::metadata(script_path)
        .unwrap_or_else(|err| panic!("metadata {script_path}: {err}"))
        .permissions()
        .mode();

    assert_ne!(mode & 0o111, 0, "{script_path} should be executable");

    for required in [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "TARGET=${TARGET:-/etc/rs-llmctl/config.toml}",
        "EXAMPLES_DIR=${EXAMPLES_DIR:-examples}",
        "case \"${profile}\" in",
        "cpu-only|gpu-amd|gpu-auto|gpu-metal|gpu-nvidia|local-dev|production-external-bind)",
        "printf 'Review %s before installing to %s",
        "read -r confirmation",
        "\"COPY\"",
        "install -D -m 0640 \"${source}\" \"${TARGET}\"",
        "No service has been started or enabled.",
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
        "systemctl restart",
        "cargo install",
        "llmctld ",
        "llmctl ",
    ] {
        assert!(
            !script.contains(forbidden),
            "{script_path} should stay offline/passive and not include `{forbidden}`"
        );
    }

    let readme = read("README.md");
    for required in [
        "packaging/stage-config.sh production-external-bind",
        "TARGET=/etc/rs-llmctl/config.toml",
        "type `COPY`",
        "does not start or enable services",
        "cpu-only",
        "gpu-amd",
        "gpu-auto",
        "gpu-metal",
        "gpu-nvidia",
        "local-dev",
        "production-external-bind",
    ] {
        assert!(
            readme.contains(required),
            "README should document config staging detail `{required}`"
        );
    }
}

#[test]
fn release_checksum_artifact_generation_is_pinned() {
    let workflow = read(".github/workflows/ci.yml");

    for required in [
        "packaging/generate-checksums.sh",
        "SHA256SUMS",
        "actions/upload-artifact@v4",
        "release-checksums",
    ] {
        assert!(
            workflow.contains(required),
            "CI workflow should include `{required}`"
        );
    }

    let script_path = "packaging/generate-checksums.sh";
    let script = read(script_path);
    let mode = fs::metadata(script_path)
        .unwrap_or_else(|err| panic!("metadata {script_path}: {err}"))
        .permissions()
        .mode();

    assert_ne!(mode & 0o111, 0, "{script_path} should be executable");

    for required in [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "sha256sum target/release/llmctl target/release/llmctld > SHA256SUMS",
        "test -x target/release/llmctl",
        "test -x target/release/llmctld",
    ] {
        assert!(
            script.contains(required),
            "{script_path} should include `{required}`"
        );
    }

    for forbidden in ["curl ", "wget ", "apt ", "dnf ", "yum ", "pacman "] {
        assert!(
            !script.contains(forbidden),
            "{script_path} should stay offline and not include `{forbidden}`"
        );
    }

    let readme = read("README.md");
    for required in [
        "packaging/generate-checksums.sh",
        "sha256sum target/release/llmctl target/release/llmctld > SHA256SUMS",
        "SHA256SUMS",
    ] {
        assert!(
            readme.contains(required),
            "README should document `{required}`"
        );
    }
}

#[test]
fn hardened_example_configs_parse_and_do_not_embed_plaintext_secrets() {
    let examples_dir = Path::new("examples");
    let entries = fs::read_dir(examples_dir)
        .unwrap_or_else(|err| panic!("read {}: {err}", examples_dir.display()));

    let mut config_paths = Vec::new();
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("read examples entry: {err}"))
            .path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("toml")
            && path.file_name().and_then(|name| name.to_str())
                != Some("offline-model-manifest.toml")
        {
            config_paths.push(path);
        }
    }
    config_paths.sort();

    let expected = [
        "cpu-only.toml",
        "gpu-amd.toml",
        "gpu-auto.toml",
        "gpu-metal.toml",
        "gpu-nvidia.toml",
        "local-dev.toml",
        "production-external-bind.toml",
    ];
    assert_eq!(
        config_paths
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap())
            .collect::<Vec<_>>(),
        expected,
        "examples should cover the hardened deployment profiles"
    );

    for path in config_paths {
        let body = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        let cfg: rs_llmctl::config::Config =
            toml::from_str(&body).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()));

        assert!(
            body.contains("offline-model-manifest.toml"),
            "{} should reference the offline model manifest",
            path.display()
        );
        assert!(
            !body.contains("sk-") && !body.contains("plaintext") && !body.contains("changeme"),
            "{} should not contain plaintext secret placeholders",
            path.display()
        );
        assert!(
            !cfg.security.api_keys.is_empty(),
            "{} should include hashed API key placeholders",
            path.display()
        );
        for key in &cfg.security.api_keys {
            assert!(
                is_sha256_hex(&key.sha256),
                "{} API key `{}` should be a sha256 hex digest placeholder",
                path.display(),
                key.id
            );
        }
        for (name, value) in &cfg.observability.exporter.headers {
            let sensitive = ["authorization", "api-key", "apikey", "token", "secret"]
                .iter()
                .any(|needle| name.to_ascii_lowercase().contains(needle));
            if sensitive {
                assert!(
                    value.starts_with("env:"),
                    "{} observability header `{name}` should use env:NAME",
                    path.display()
                );
            }
        }

        rs_llmctl::config::validate_production_security(&cfg)
            .unwrap_or_else(|err| panic!("validate {}: {err}", path.display()));
    }
}
