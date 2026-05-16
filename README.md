# rs-llmctl

Rust implementation of `llmctl` for enterprise model delivery.

`rs-llmctl` owns the server-side data plane and control plane: OpenAI-compatible
serving, model lifecycle, hot/cold swap routing, resource budgeting, quotas,
audit trails, reporting, and observability. Milliways remains the developer
client.

```bash
cargo run --bin llmctl -- init
cargo run --bin llmctl -- model install /models/qwen.gguf --alias qwen
cargo run --bin llmctl -- quota set --subject team-a --team platform --model qwen
cargo run --bin llmctl -- server check
cargo run --bin llmctld -- --config ~/.config/rs-llmctl/config.toml
```

Default production posture requires authentication before binding externally.
Use dev mode for local unauthenticated experiments only.

Production configs store API keys as SHA-256 digests only. Plaintext secret
fields in the security section are rejected, and sensitive observability
headers such as authorization, API key, token, or secret headers must use an
`env:NAME` reference instead of embedding the value in TOML.

## Development Gates

Changes are expected to be test-driven: add or update the focused test first,
watch it fail for the intended reason, implement the smallest change, then run
the relevant gate locally before opening a review. CI enforces the baseline Rust
gates and verifies that production binaries compile in release mode:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --bins
```

For narrow documentation or release changes, add focused tests that pin the
release surface and still run the full CI gate before merging.

## Release Package Readiness

The package publishes two binary entry points from `Cargo.toml`:

- `llmctl`, built from `src/bin/llmctl.rs`, is the operator CLI.
- `llmctld`, built from `src/bin/llmctld.rs`, is the production daemon.

Before cutting an archive or installer, run `cargo build --release --bins` and
package the resulting `target/release/llmctl` and `target/release/llmctld`
binaries together with this README, the release notes, license metadata, and an
example production config. Generate and publish checksums for the release
binaries:

```bash
packaging/generate-checksums.sh
sha256sum target/release/llmctl target/release/llmctld > SHA256SUMS
```

CI runs the checksum script after the release build and uploads `SHA256SUMS` as
the `release-checksums` artifact. Package checks should fail if either binary
name changes because downstream automation, systemd units, and air-gapped
install runbooks address those names directly.

Server packages should install the release binaries at stable paths:

```bash
install -D -m 0755 target/release/llmctl /usr/local/bin/llmctl
install -D -m 0755 target/release/llmctld /usr/local/bin/llmctld
install -D -m 0644 packaging/systemd/llmctld.service /etc/systemd/system/llmctld.service
install -d -m 0750 -o llmctl -g llmctl /etc/rs-llmctl /var/lib/rs-llmctl/models /var/log/rs-llmctl
```

Stage a reviewed starter config explicitly before validation. The helper prints
the selected example and requires the operator to type `COPY` before it writes
`TARGET=/etc/rs-llmctl/config.toml`:

```bash
packaging/stage-config.sh production-external-bind
TARGET=/etc/rs-llmctl/config.toml packaging/stage-config.sh cpu-only
```

Available profiles are `cpu-only`, `gpu-amd`, `gpu-auto`, `gpu-metal`,
`gpu-nvidia`, `local-dev`, and `production-external-bind`. This config staging
flow is offline and passive: it copies only the reviewed TOML file.
It does not start or enable services. Run it with privileges appropriate for writing
`/etc/rs-llmctl/config.toml`, or set `TARGET` to a package-root path for
review in CI or image builds.

The production daemon template is `packaging/systemd/llmctld.service`. It
starts `/usr/local/bin/llmctld --config ${LLMCTL_CONFIG}` with
`LLMCTL_CONFIG=/etc/rs-llmctl/config.toml`, runs as the dedicated `llmctl`
user/group, and grants write access only to `/var/lib/rs-llmctl` and
`/var/log/rs-llmctl`.

Dry-run validation for a staged server should run before enabling systemd:

```bash
packaging/validate-install.sh
```

The validation script is safe to run offline: it performs the daemon dry-run
startup plan, runs `security check`, prints `server status`, `server plan`,
`audit retention plan`, and `observe plan`, and verifies the systemd unit only
when `systemd-analyze` is available. It does not start or enable services,
install packages, download models, or contact remote endpoints. Override staged
paths or binary locations with `CONFIG=...`, `UNIT=...`, `LLMCTL=...`, or
`LLMCTLD=...` when validating a package root.

External bind is a release-blocking deployment control, not a packaging default.
Before setting `server.host = "0.0.0.0"` or `security.bind-external = true`,
the staged config must also set `security.production = true`,
`security.require-auth = true`, at least one hashed `[[security.api-keys]]`
entry with scopes, quota policy for served subjects, audit retention/reporting,
and an approved `observability.exporter.endpoint`. Store model data under
`/var/lib/rs-llmctl/models` and keep runtime secrets in `env:` references.

Offline deployments should ship an offline install manifest next to the staged
model files. Operators import it with:

```bash
llmctl --config /etc/rs-llmctl/config.toml model import-manifest ./manifest.toml
```

Minimal manifest shape:

```toml
[[models]]
alias = "qwen"
path = "models/qwen.gguf"
role = "chat"
weight = 1
sha256 = "hex-encoded-sha256"
```

Relative paths resolve from the manifest directory. Include `sha256` whenever a
bundle is copied across trust boundaries so the install rejects unexpected model
bytes before registration.

Hardened starter configs live under `examples/`: `local-dev.toml`,
`production-external-bind.toml`, `cpu-only.toml`, `gpu-auto.toml`,
`gpu-nvidia.toml`, `gpu-amd.toml`, and `gpu-metal.toml`. Each profile references
`examples/offline-model-manifest.toml`, uses SHA-256 API key digest
placeholders, and keeps observability authorization in an `env:` reference.

## Ordered Deployment Operations

Production deployments should follow this order so every gate is captured before
traffic reaches the daemon:

1. Import the offline install manifest after staging the approved bundle under
   `/var/lib/rs-llmctl/models`:

   ```bash
   llmctl --config /etc/rs-llmctl/config.toml model import-manifest ./manifest.toml
   ```

2. Run the dry-run validation gate without starting the daemon:

   ```bash
   llmctl --config /etc/rs-llmctl/config.toml server check
   llmctl --config /etc/rs-llmctl/config.toml server status
   llmctl --config /etc/rs-llmctl/config.toml server plan
   ```

3. Run the security audit for production/external-bind controls:

   ```bash
   llmctl --config /etc/rs-llmctl/config.toml security check
   llmctl --config /etc/rs-llmctl/config.toml audit retention plan
   ```

4. Run readiness checks for observability and the systemd unit:

   ```bash
   llmctl --config /etc/rs-llmctl/config.toml observe plan
   systemd-analyze verify /etc/systemd/system/llmctld.service
   ```

5. Hand off service activation only after the dry-run, security audit, and
   readiness checks pass. Keep the release package documentation passive: the
   operator change record should reference the approved activation procedure
   instead of embedding service-control commands here.

6. Verify AQE/OpenAI client access against the OpenAI-compatible endpoint with
   `OPENAI_BASE_URL=http://host:8765/v1` and the production API key scope
   intended for the client.

7. Export the audit envelope for the deployment window:

   ```bash
   llmctl --config /etc/rs-llmctl/config.toml data export --hours 24
   ```

## Policy Operations Runbook

Quota policy can be moved between environments without editing unrelated server
settings. Export the reviewed policy set from the source environment, then
import it into the staged target config:

```bash
llmctl --config /etc/rs-llmctl/config.toml quota export > quotas.json
llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.json
llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.toml
```

Quota import rejects policies with blank subjects or teams,
`requests_per_minute`, `tokens_per_day`, or `max_concurrency` values that are
not greater than zero, or `allowed_models` entries that contain empty model
aliases. Treat those failures as policy review findings, not as runtime
overrides. After import, capture the effective policy list:

```bash
llmctl --config /etc/rs-llmctl/config.toml quota list
```

The add-key workflow hashes the operator-provided API key first and stores only
the digest in config:

```bash
llmctl security hash-key "$LLMCTL_NEW_API_KEY"
```

Add the returned digest to the reviewed config as a scoped key entry:

```toml
[[security.api_keys]]
id = "ops-admin"
sha256 = "<sha256-from-hash-key>"
subject = "ops-admin"
team = "platform"
scopes = ["admin"]
```

Do not place raw API key material in TOML. The `sha256` field is for the digest
printed by `security hash-key`.

Capture a server plan export before any production service activation change:

```bash
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.json
```

The plan artifact records the worker commands the daemon would launch and can be
attached to the same change record as the quota import and key digest review.
For policy changes, keep a before/after pair and review the diff before
approving the rollout:

```bash
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.before.json
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.after.json
llmctl server plan-diff server-plan.before.json server-plan.after.json
```

Retention review should use the signed envelope form so the payload can be
verified offline and attached to the change record:

```bash
llmctl --config /etc/rs-llmctl/config.toml audit retention plan --envelope > retention-plan-envelope.json
llmctl --config /etc/rs-llmctl/config.toml data verify-envelope retention-plan-envelope.json
```

Review the envelope `metadata sha256` against the verification output and
confirm the payload keeps `dry_run` set to true and `deletes` set to false.
See `examples/policy-operations-runbook.md` for a compact operator checklist.

## Enterprise Security Posture

`rs-llmctl` uses a PCI DSS v4.0.1-aligned baseline for enterprise deployments:
external bind and production mode require API key authentication, scoped keys,
quotas, audit events, usage reports, and resource budget enforcement. Binding
to `0.0.0.0` or setting `security.bind-external = true` must be paired with
`security.require-auth = true` and at least one configured API key.

Observability is configured as an OTel-oriented exporter plan. Set
`observability.service-name`, enable or disable traces, metrics, and logs, and
configure `observability.exporter.endpoint` with `http-protobuf` or `grpc`
protocols for an OTLP collector. Collector authentication belongs in
environment-backed header references, for example `env:OTEL_EXPORTER_OTLP_HEADERS`,
not plaintext config.

Audit retention and report generation are explicit config: `audit.retention-days`
defaults to 365, `audit.report-directory` can point at an operator-managed
artifact path, `audit.report-formats` defaults to JSON, and
`audit.monthly-reports` controls scheduled monthly report generation.

Model lifecycle is designed for offline install paths as well as direct URL
downloads: operators can pre-stage approved model bundles and register local
files with `llmctl model install`. Runtime planning defaults to an 80% resource
budget, with CPU-only and GPU-aware detection available through the resource
configuration and observation commands.

Enterprise runtime controls belong in config, not release packaging scripts:
`security.require-auth = true` for production, `security.bind-external = true`
only when API keys are configured, quota policies for every served subject,
`audit.retention-days` and report output for compliance review, and
`observability.exporter.endpoint` pointed at the approved collector. Secrets and
collector headers should use `env:` references so archives and manifests never
carry plaintext runtime credentials.

The serving API is OpenAI-compatible for enterprise clients, including Agentic
QE (AQE), by pointing `OPENAI_BASE_URL` at `http://host:8765/v1`. AQE/OpenAI
endpoint usage is captured through audit trails, usage reports, quota checks,
and per-request reporting so regulated deployments can review who used which
model, when, and under which policy.

## CLI

`llmctl init` writes the default TOML config, creates the model directory, and
initializes SQLite storage.

Server commands:

- `llmctl server run` validates startup state and runs the serving API.
- `llmctl server check` verifies config and storage initialization.
- `llmctl server security-check` enforces the production/external-bind auth
  policy without starting the daemon.

Model commands:

- `llmctl model install <path-url-or-catalog-id> --alias <name>` registers a
  local model, downloads a URL, or installs a built-in catalog model into the
  configured model directory.
- `llmctl model list` prints configured models.

Swap commands:

- `llmctl swap set --mode hot-swap` enables hot model swapping in config.
- `llmctl swap set --mode cold-swap` enables cold model swapping in config.
- `llmctl swap show` prints the configured swap mode and model aliases.

Quota commands:

- `llmctl quota set --subject <id> --team <team> --model <alias>` upserts a
  quota policy in config.
- `llmctl quota export` prints configured quota policies as a portable JSON
  document.
- `llmctl quota import <path>` replaces configured quota policies from a JSON
  export or a TOML policy file.
- `llmctl quota list` prints configured quota policies.

Operations commands:

- `llmctl observe snapshot` records a local resource snapshot.
- `llmctl observe drift --hours 24` summarizes drift observations.
- `llmctl observe usage --hours 24` summarizes usage events.
- `llmctl observe show --limit 20` prints recent observations.
- `llmctl audit report monthly --year 2026 --month 5` emits a monthly audit
  report; omitted bounds default to the current month.
- `llmctl audit report request <request-id>` emits a per-request audit report.
- `llmctl audit request --actor <id> --action <name> --resource <target>`
  records a manual audit request.
- `llmctl usage report --hours 24` summarizes usage events.
- `llmctl data export --hours 24` exports audit, usage, observation, quota, and
  model inventory records for the time window.
