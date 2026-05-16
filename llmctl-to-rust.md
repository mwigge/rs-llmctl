# llmctl to Rust Plan

`rs-llmctl` is a full Rust rewrite of `llmctl` for enterprise-grade server-side
model delivery. There is no Go compatibility layer.

## Product Split

- Milliways: developer-machine client and workflow UX.
- rs-llmctl: production model-serving platform.

## Required Capabilities

- OpenAI-compatible `/v1/models` and `/v1/chat/completions`.
- CPU-only serving.
- NVIDIA/AMD GPU-aware resource detection and 80% resource budgeting.
- Hot and cold model swap modes.
- Offline local model registration, direct URL download, and bundle-oriented lifecycle.
- External clients such as Agentic QE through `OPENAI_BASE_URL=http://host:8765/v1`.
- API key auth, scopes, quotas, audit trails, and reporting.
- OTel-compatible operations and AI observability.
- PCI DSS v4.0.1-aligned controls as the baseline posture.

## Implementation Notes

The first implementation provides the Rust workspace, daemon, CLI, SQLite
storage, audit/usage/observation persistence, quota checks, resource snapshots,
and an OpenAI-compatible proxy to supervised `llama-server` workers. The system
is intentionally structured so enterprise hardening can deepen without changing
the public CLI/API contract.

Development follows TDD for release-owned surfaces: write focused tests first,
implement the smallest passing change, then run `cargo fmt`, `cargo clippy`,
and `cargo test`. The minimal CI gate mirrors those local checks with
`cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test --all-targets --all-features`, and `cargo build --release --bins`.

Release packaging keeps the binary names stable. `Cargo.toml` declares
`llmctl` for the operator CLI and `llmctld` for daemon startup, and release
artifacts are expected to include `target/release/llmctl` and
`target/release/llmctld`. CI should block changes that break a release build or
rename those entry points without an explicit migration plan.

Installer-owned deployment artifacts are limited to stable placement and
operator runbooks: install `target/release/llmctl` as `/usr/local/bin/llmctl`,
install `target/release/llmctld` as `/usr/local/bin/llmctld`, place production
config at `/etc/rs-llmctl/config.toml`, and provide
`packaging/systemd/llmctld.service` as the systemd template. The service uses
the dedicated `llmctl` account and starts `/usr/local/bin/llmctld --config
${LLMCTL_CONFIG}` with `LLMCTL_CONFIG=/etc/rs-llmctl/config.toml`.

Package validation is dry-run oriented and must not require starting the daemon:
`llmctl --config /etc/rs-llmctl/config.toml server check`, `llmctl --config
/etc/rs-llmctl/config.toml security check`, `llmctl --config
/etc/rs-llmctl/config.toml observe plan`, and `systemd-analyze verify
/etc/systemd/system/llmctld.service` cover config/storage initialization,
external-bind security prerequisites, observability exporter shape, and unit
syntax.

The enterprise security baseline is PCI DSS v4.0.1-aligned. Production service
exposure requires authenticated external bind, scoped API keys, quota policy,
audit persistence, usage report generation, and resource budget controls.
Offline install remains a first-class deployment path through local model
registration and bundle-oriented operations, while AQE/OpenAI endpoint usage is
supported through `OPENAI_BASE_URL=http://host:8765/v1` and recorded in audit
and usage reporting.

Offline package runbooks should prefer an offline install manifest for
air-gapped sites:

```toml
[[models]]
alias = "qwen"
path = "models/qwen.gguf"
role = "chat"
weight = 1
sha256 = "hex-encoded-sha256"
```

Operators register the bundle with `llmctl model import-manifest
./manifest.toml`, or with `llmctl --config /etc/rs-llmctl/config.toml model
import-manifest ./manifest.toml` during server install. Relative model paths
resolve from the manifest directory, and `sha256` pins the bytes shipped in the
approved bundle. Server bundles should stage model files under
`/var/lib/rs-llmctl/models` before importing the offline install manifest.

Enterprise runtime controls are release-blocking documentation requirements:
`security.require-auth`, `security.bind-external`, quota policy, resource budget
limits, `audit.retention-days`, usage report retention, and
`observability.exporter.endpoint` must be documented for production operators.
Runtime secrets should be represented as `env:` references so offline manifests,
archives, and config examples do not embed credentials.
External bind requires the full production baseline before enablement:
`security.production = true`, `security.require-auth = true`, hashed
`[[security.api-keys]]`, scoped access, quota policy, audit retention/reporting,
usage reporting, resource budget limits, and the approved observability
collector endpoint.

## CLI Contract

The Rust CLI keeps the operational command groups stable while the library
internals mature:

- `init`
- `server run`, `server check`, `server security-check`
- `model install`, `model list`
- `swap set`, `swap show`
- `quota set`, `quota list`
- `observe snapshot`, `observe drift`, `observe usage`, `observe show`
- `audit report monthly`, `audit report request`, `audit request`
- `usage report`
- `data export`

`llmctld` performs the daemon startup sequence expected by production
deployments: load config, validate production/external-bind security, initialize
storage, and serve the API.
