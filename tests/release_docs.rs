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
    ] {
        assert!(workflow.contains(gate), "CI workflow should run `{gate}`");
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
        "pci dss",
        "external bind",
        "offline install",
        "resource budget",
        "audit",
        "usage report",
        "aqe",
        "openai_base_url",
    ] {
        assert!(docs.contains(topic), "docs should cover `{topic}`");
    }
}
