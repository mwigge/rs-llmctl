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

Quota commands:

- `llmctl quota set --subject <id> --team <team> --model <alias>` upserts a
  quota policy in config.
- `llmctl quota list` prints configured quota policies.

Operations commands:

- `llmctl observe snapshot` records a local resource snapshot.
- `llmctl observe drift --hours 24` summarizes drift observations.
- `llmctl observe usage --hours 24` summarizes usage events.
- `llmctl observe show --limit 20` prints recent observations.
- `llmctl audit report monthly` emits this month's audit report.
- `llmctl audit report request <request-id>` emits a per-request audit report.
- `llmctl audit request --actor <id> --action <name> --resource <target>`
  records a manual audit request.
- `llmctl usage report --hours 24` summarizes usage events.
