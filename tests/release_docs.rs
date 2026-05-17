use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("read {path}: {err}"))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn normalize_doc_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn product_docs() -> String {
    [
        "README.md",
        "docs/operations.md",
        "docs/client-sdk.md",
        "docs/ai-developer-workflows.md",
        "docs/configuration.md",
        "docs/security.md",
        "docs/observability-reporting.md",
        "docs/storage.md",
        "docs/blog-local-model-operations.md",
        "examples/policy-operations-runbook.md",
    ]
    .into_iter()
    .map(read)
    .collect::<Vec<_>>()
    .join("\n")
}

#[test]
fn ci_workflow_enforces_core_rust_gates() {
    let workflow = read(".github/workflows/ci.yml");

    for gate in [
        "cargo fmt --all -- --check",
        "cargo clippy --all-targets --all-features -- -D warnings",
        "cargo test --all-targets --all-features",
        "cargo build --release --bin llmctl",
        "tags:",
        "'v*'",
        "tests/smoke/smoke_native_release.sh",
        "runs-on: [self-hosted, linux, x64, llmctl-native-smoke]",
        "LLMCTL_NATIVE_SMOKE_CONFIG: ${{ secrets.LLMCTL_NATIVE_SMOKE_CONFIG }}",
        "LLMCTL_NATIVE_SMOKE_CONFIG_TOML: ${{ secrets.LLMCTL_NATIVE_SMOKE_CONFIG_TOML }}",
        "LLMCTL_NATIVE_SMOKE_MODEL_PATH: ${{ secrets.LLMCTL_NATIVE_SMOKE_MODEL_PATH }}",
        "sigstore/cosign-installer@v3",
        "packaging/sign-release.sh dist",
        "gh release create",
    ] {
        assert!(workflow.contains(gate), "CI workflow should run `{gate}`");
    }
}

#[test]
fn package_manifest_publishes_default_binary() {
    let manifest = read("Cargo.toml");

    for expected in [
        r#"name = "rs-llmctl""#,
        r#"name = "llmctl""#,
        r#"path = "src/bin/llmctl.rs""#,
    ] {
        assert!(
            manifest.contains(expected),
            "Cargo.toml should declare default binary detail `{expected}`"
        );
    }
}

#[test]
fn docs_cover_tdd_lints_and_enterprise_security_posture() {
    let docs = product_docs().to_lowercase();

    for topic in [
        "tdd",
        "cargo fmt",
        "cargo clippy",
        "cargo test",
        "cargo build --release",
        "target/release/llmctl",
        "one rust binary",
        "default release package publishes one rust binary",
        "candle-native",
        "candle-native serving first",
        "single `llmctl` service entrypoint",
        "upstream-reported token counts",
        "native tokenizer metering",
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
        "/etc/rs-llmctl/config.toml",
        "/var/lib/rs-llmctl/models",
        "llmctl --config /etc/rs-llmctl/config.toml server check",
        "llmctl --config /etc/rs-llmctl/config.toml server status",
        "llmctl --config /etc/rs-llmctl/config.toml server plan",
        "llmctl --config /etc/rs-llmctl/config.toml security check",
        "llmctl --config /etc/rs-llmctl/config.toml audit retention plan",
        "llmctl --config /etc/rs-llmctl/config.toml observe plan",
        "systemd",
        "llmctld.service",
        "sudo systemctl status llmctld.service",
        "sudo systemctl restart llmctld.service",
        "sudo systemctl stop llmctld.service",
        "sudo systemctl start llmctld.service",
    ] {
        assert!(docs.contains(topic), "docs should cover `{topic}`");
    }

    let removed_legacy_phrase = ["llama-server", "compatibility", "and", "fallback"].join(" ");
    assert!(
        !docs.contains(&removed_legacy_phrase),
        "docs should not keep legacy external-worker compatibility wording"
    );
}

#[test]
fn docs_cover_ordered_deployment_operations() {
    let docs = product_docs().to_lowercase();

    let ordered_steps = [
        "1. import the offline install manifest",
        "2. run the dry-run validation gate",
        "3. run the security audit",
        "4. run readiness checks",
        "5. hand off service activation",
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
fn docs_cover_safe_first_run_operator_path() {
    let docs = product_docs().to_lowercase();
    let docs_flat = normalize_doc_text(&docs);

    for required in [
        "first-run",
        "dry-run by default",
        "json-friendly",
        "--apply",
        "--secret-output",
        "stores only the sha-256 digest",
        "does not download a model by default",
        "--starter-model-path",
        "ask_question",
        "/v1/chat/completions",
    ] {
        assert!(
            docs_flat.contains(required),
            "docs should cover first-run detail `{required}`"
        );
    }
}

#[test]
fn docs_cover_model_lifecycle_operations() {
    let docs = product_docs().to_lowercase();

    for required in [
        "model install",
        "model import-manifest",
        "model inventory",
        "model list",
        "model stop",
        "model start",
        "model update",
        "model upgrade",
        "model downgrade",
        "--new-alias",
        "restart is required",
        "without leaking full paths",
    ] {
        assert!(
            docs.contains(required),
            "docs should cover model lifecycle detail `{required}`"
        );
    }
}

#[test]
fn docs_pin_policy_operation_runbook_commands_without_plaintext_key_guidance() {
    let docs = product_docs();
    let docs_lower = docs.to_lowercase();

    for required in [
        "llmctl --config /etc/rs-llmctl/config.toml quota export > quotas.json",
        "llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.json",
        "llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.toml",
        "llmctl --config /etc/rs-llmctl/config.toml quota list",
        "printf '%s' \"$LLMCTL_NEW_API_KEY\" | llmctl security hash-key --stdin",
        "[[security.api_keys]]",
        "sha256 = \"<sha256-from-hash-key>\"",
        "llmctl --json --config /etc/rs-llmctl/config.toml server plan > server-plan.json",
        "server plan export",
        "llmctl --config /etc/rs-llmctl/config.toml audit retention plan --envelope > retention-plan-envelope.json",
        "llmctl --config /etc/rs-llmctl/config.toml data verify-envelope retention-plan-envelope.json",
        "llmctl server plan-diff server-plan.before.json server-plan.after.json",
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
fn policy_runbook_documents_strict_passive_review_expectations() {
    let docs = read("examples/policy-operations-runbook.md");
    let docs_lower = docs.to_lowercase();

    for required in [
        "quota import rejects policies with blank subjects or teams",
        "requests_per_minute",
        "tokens_per_day",
        "max_concurrency",
        "greater than zero",
        "allowed_models",
        "empty model",
        "aliases",
        "retention-plan-envelope.json",
        "metadata sha256",
        "deletes",
        "server-plan.before.json",
        "server-plan.after.json",
        "review the diff",
        "approving",
        "change record",
    ] {
        assert!(
            docs_lower.contains(required),
            "policy runbook docs should cover `{required}`"
        );
    }

    for forbidden in [
        "systemctl start",
        "systemctl enable",
        "systemctl restart",
        "systemctl reload",
        "systemctl stop",
        "service llmctld start",
        "service llmctld restart",
    ] {
        assert!(
            !docs_lower.contains(forbidden),
            "policy runbook docs should stay passive and not include `{forbidden}`"
        );
    }
}

#[test]
fn docs_pin_enterprise_reporting_and_client_safe_metadata() {
    let docs = format!(
        "{}\n{}",
        read("README.md"),
        read("examples/policy-operations-runbook.md")
    );
    let docs_lower = docs.to_lowercase();
    let docs_flat = normalize_doc_text(&docs_lower);

    for required in [
        "data/audit summaries",
        "quota/team governance summaries",
        "external client non-secret response metadata",
        "safe for aqe/openai-compatible clients",
        "aqe/openai-compatible clients can consume these summaries",
        "without exposing secrets",
    ] {
        assert!(
            docs_flat.contains(required),
            "enterprise docs should cover `{required}`"
        );
    }

    for required in [
        "usage totals",
        "audit event counts",
        "retention windows",
        "quota limits",
        "team attribution",
        "request identifiers",
        "model aliases",
        "policy status",
    ] {
        assert!(
            docs_lower.contains(required),
            "enterprise reporting docs should mention `{required}`"
        );
    }
}

#[test]
fn docs_pin_server_storage_router_and_aqe_contract_controls() {
    let docs = format!(
        "{}\n{}",
        read("README.md"),
        read("examples/policy-operations-runbook.md")
    );
    let docs_lower = docs.to_lowercase();
    let docs_flat = normalize_doc_text(&docs_lower);

    for required in [
        "external database storage with postgres",
        "database url",
        "redacted",
        "migration plan",
        "admission/backpressure limits",
        "upstream timeout budgets",
        "non-secret failure responses",
        "stable 429/504 errors",
        "integration aqe-contract",
        "openai paths",
        "required auth scopes",
        "safe response headers",
        "quota/team reporting fields",
        "model aliases",
    ] {
        assert!(
            docs_flat.contains(required),
            "server storage/router/AQE docs should cover `{required}`"
        );
    }

    for forbidden in [
        "database passwords",
        "raw connection secrets",
        "upstream urls, prompts, file paths, api keys, or bearer tokens",
    ] {
        assert!(
            docs_flat.contains(forbidden),
            "docs should explicitly forbid `{forbidden}`"
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
        "execstart=/usr/local/bin/llmctl --config ${llmctl_config} server run",
        "nonewprivileges=true",
        "privatetmp=true",
        "protectsystem=strict",
        "protecthome=true",
        "readwritepaths=/var/lib/rs-llmctl /var/log/rs-llmctl",
        "cpuaccounting=true",
        "memoryaccounting=true",
        "systemd-run --property=cpuaccounting=true",
        "cpuquota=<server-plan.resource_limits.systemd.cpuquota>",
        "memorymax=<server-plan.resource_limits.systemd.memorymax>",
        "--property=cpuquota=<server-plan.resource_limits.systemd.cpuquota>",
        "--property=memorymax=<server-plan.resource_limits.systemd.memorymax>",
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
fn resource_docs_cover_systemd_enforcement_without_gpu_vram_overclaim() {
    let readme = read("README.md");
    let readme_lower = readme.to_lowercase();
    let docs_flat = normalize_doc_text(&readme_lower);

    for required in [
        "resource_limits.systemd",
        "cpuquota",
        "memorymax",
        "unit_properties",
        "systemd_run_args",
        "systemd drop-in",
        "systemd-run",
        "default runtime policy budgets 80% of cpu, ram, and detected gpu vram",
        "gpu vram budgets are exported as `metadata-only` planning evidence",
        "does not claim hard gpu vram enforcement",
    ] {
        assert!(
            docs_flat.contains(required),
            "resource docs should cover `{required}`"
        );
    }

    for forbidden in [
        "hard gpu vram enforcement is supported",
        "gpu vram is hard enforced",
        "enforces gpu vram with cgroups",
    ] {
        assert!(
            !docs_flat.contains(forbidden),
            "resource docs should not overclaim `{forbidden}`"
        );
    }
}

#[test]
fn docs_pin_native_scheduler_as_fifo_runtime_with_metadata_only_future_controls() {
    let readme = read("README.md");
    let docs_flat = normalize_doc_text(&readme.to_lowercase());

    for required in [
        "native scheduler contract",
        "implemented fifo queue",
        "bounded per-engine concurrency",
        "queue/admission wait metadata",
        "admission/backpressure",
        "continuous batching",
        "kv cache budget metadata",
        "cancellation token metadata",
        "implemented=false",
    ] {
        assert!(
            docs_flat.contains(required),
            "README should document scheduler contract detail `{required}`"
        );
    }
}

#[test]
fn docs_cover_client_sdk_tool_sessions_tls_and_runtime_caveats() {
    let docs = product_docs();
    let docs_lower = docs.to_lowercase();
    let docs_flat = normalize_doc_text(&docs_lower);

    for required in [
        "separate `rs-llmctl-client` crate",
        "rs-llmctl-client = \"1.2\"",
        "client-managed sessions",
        "metadata.session_id",
        "client-side tool loops",
        "local-first provider abstraction",
        "contract-only provider metadata",
        "routes_external_provider_traffic = false",
        "rs-llmctl` audits, routes, and meters",
        "does not execute tools",
        "does not keep hidden conversation state",
        "security.tls-termination",
        "rustls-backed clients",
        "[server.tls]",
        "native rustls",
        "server certificates only",
        "kimi is tracked as a product target but fails closed",
        "continuous batching",
        "schedulercontract::fifo_runtime",
        "routes_external_provider_traffic",
        "implemented=false",
    ] {
        assert!(
            docs_flat.contains(required),
            "client/runtime docs should cover `{required}`"
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
        "CONFIG=${CONFIG:-${LLMCTL_CONFIG:-/etc/rs-llmctl/config.toml}}",
        "UNIT=${UNIT:-/etc/systemd/system/${SERVICE_NAME}.service}",
        "LLMCTL=${LLMCTL:-${BIN_DIR}/llmctl}",
        "STATE_DIR=${LLMCTL_STATE_DIR:-/var/lib/rs-llmctl}",
        "LOG_DIR=${LLMCTL_LOG_DIR:-/var/log/rs-llmctl}",
        "require_dir \"${STATE_DIR}/models\"",
        "require_dir \"${STATE_DIR}/reports\"",
        "require_dir \"${LOG_DIR}\"",
        "require_executable \"${LLMCTL}\"",
        "\"${LLMCTL}\" --config \"${CONFIG}\" security check",
        "\"${LLMCTL}\" --config \"${CONFIG}\" server status",
        "\"${LLMCTL}\" --config \"${CONFIG}\" server plan",
        "\"${LLMCTL}\" --config \"${CONFIG}\" audit retention plan",
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
        "systemctl restart",
        "systemctl reload",
        "service llmctld",
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
fn installer_docs_cover_systemd_defaults_and_overrides() {
    let script = read("install.sh");
    let readme = read("README.md");
    let readme_lower = readme.to_lowercase();

    for required in [
        "install_systemd=\"${LLMCTL_INSTALL_SYSTEMD:-auto}\"",
        "config_dir=\"${LLMCTL_CONFIG_DIR:-/etc/rs-llmctl}\"",
        "config_file=\"${LLMCTL_CONFIG:-${config_dir}/config.toml}\"",
        "state_dir=\"${LLMCTL_STATE_DIR:-/var/lib/rs-llmctl}\"",
        "log_dir=\"${LLMCTL_LOG_DIR:-/var/log/rs-llmctl}\"",
        "service_name=\"${LLMCTL_SERVICE_NAME:-llmctld}\"",
        "start_service=\"${LLMCTL_START_SERVICE:-0}\"",
        "enable_audit_timer=\"${LLMCTL_ENABLE_AUDIT_TIMER:-0}\"",
        "system service installs cannot use a home-directory PREFIX",
        "useradd --system --gid llmctl",
        "Run first-run with a model and API key before starting the service",
        "enable --now ${service_name}.service",
        "cpu_quota_percent=$(( $(nproc) * 80 ))",
        "systemctl enable --now llmctl-monthly-audit.timer",
        "http://127.0.0.1:8765/v1",
        "release archive includes legacy llmctld",
        "default install uses llmctl only",
    ] {
        assert!(
            script.contains(required),
            "install.sh should include installer behavior `{required}`"
        );
    }

    for required in [
        "single default `llmctl` binary",
        "creates the `llmctl` system user",
        "`/var/lib/rs-llmctl/models`",
        "`/var/lib/rs-llmctl/reports`",
        "`/var/log/rs-llmctl`",
        "installs `llmctld.service` without starting it",
        "LLMCTL_START_SERVICE=1",
        "LLMCTL_ENABLE_AUDIT_TIMER=1",
        "first-run --apply",
        "SHA256SUMS.sig",
        "sudo systemctl status llmctld.service",
        "sudo systemctl restart llmctld.service",
        "sudo systemctl stop llmctld.service",
        "sudo systemctl start llmctld.service",
        "http://127.0.0.1:8765/v1",
        "stable `llmctld.service` unit name",
        "llmctl --config /etc/rs-llmctl/config.toml server run",
        "LLMCTL_INSTALL_SYSTEMD=0",
        "binary-only install",
        "LLMCTL_CONFIG_DIR",
        "LLMCTL_CONFIG",
        "LLMCTL_STATE_DIR",
        "LLMCTL_LOG_DIR",
        "LLMCTL_SERVICE_NAME",
        "system service installs",
        "home-directory prefix",
    ] {
        let documented = readme.contains(required) || readme_lower.contains(required);
        assert!(
            documented,
            "README should document installer behavior `{required}`"
        );
    }

    let removed_legacy_phrase = ["llama-server", "compatibility", "and", "fallback"].join(" ");
    assert!(
        !readme_lower.contains(&removed_legacy_phrase),
        "README should describe native-first operation without legacy compatibility wording"
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
        "release-artifacts",
        "dist/rs-llmctl-*.tar.gz",
        "dist/SHA256SUMS",
        "packaging/generate-sbom.sh dist",
        "packaging/sign-release.sh dist",
        "dist/rs-llmctl.sbom-fallback.json",
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
        "artifact=\"rs-llmctl-${OS}-${ARCH}\"",
        "tarball=\"${DIST_DIR}/${artifact}.tar.gz\"",
        "install -m 0755 target/release/llmctl \"${stage}/llmctl\"",
        "install -m 0644 CHANGELOG.md \"${stage}/CHANGELOG.md\"",
        "if [[ \"${OS}\" == \"linux\" && -f packaging/systemd/llmctld.service ]]; then",
        "sha256sum rs-llmctl-*.tar.gz | sort -k2 > SHA256SUMS",
        "test -x target/release/llmctl",
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
        "dist/rs-llmctl-<os>-<arch>.tar.gz",
        "SHA256SUMS",
        "CHANGELOG.md",
        "llmctld.service",
    ] {
        assert!(
            readme.contains(required),
            "README should document `{required}`"
        );
    }
}

#[test]
fn changelog_documents_native_release_artifacts_and_service_name() {
    let changelog = read("CHANGELOG.md").to_lowercase();
    let readme = read("README.md").to_lowercase();

    for required in [
        "native-first packaging",
        "single `llmctl`",
        "stable service name",
        "llmctld.service",
        "execstart",
        "packaging/validate-install.sh",
        "passive and offline",
        "dist/rs-llmctl-<os>-<arch>.tar.gz",
        "dist/sha256sums",
        "packaging/sign-release.sh dist",
    ] {
        assert!(
            changelog.contains(required),
            "CHANGELOG.md should document `{required}`"
        );
    }

    assert!(
        readme.contains("release notes live in [changelog.md](changelog.md)")
            && readme.contains("`readme.md`, `changelog.md`,")
            && readme.contains("systemd unit template"),
        "README should point operators to changelog-backed release artifact notes"
    );
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
