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

## Development Gates

Changes are expected to be test-driven: add or update the focused test first,
watch it fail for the intended reason, implement the smallest change, then run
the relevant gate locally before opening a review. CI enforces the baseline Rust
gates:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

For narrow documentation or release changes, add focused tests that pin the
release surface and still run the full CI gate before merging.

## Enterprise Security Posture

`rs-llmctl` uses a PCI DSS v4.0.1-aligned baseline for enterprise deployments:
external bind and production mode require API key authentication, scoped keys,
quotas, audit events, usage reports, and resource budget enforcement. Binding
to `0.0.0.0` or setting `security.bind-external = true` must be paired with
`security.require-auth = true` and at least one configured API key.

Model lifecycle is designed for offline install paths as well as direct URL
downloads: operators can pre-stage approved model bundles and register local
files with `llmctl model install`. Runtime planning defaults to an 80% resource
budget, with CPU-only and GPU-aware detection available through the resource
configuration and observation commands.

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
