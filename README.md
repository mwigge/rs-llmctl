# rs-llmctl

`rs-llmctl` is a standalone operations tool for running local and private LLM
models like real server infrastructure. It gives operators a small Rust control
plane, an OpenAI-compatible serving endpoint, model lifecycle commands,
resource budgeting, quotas, audit trails, usage reporting, and OTel-ready
observability.

![rs-llmctl operations overview](docs/images/rs-llmctl-operations-hero.png)

The goal is simple: make model delivery boring in the best way. You should be
able to stage a model, verify it, budget CPU/RAM/VRAM, serve it to internal
clients, swap it safely, and keep useful evidence about what happened.

## What It Does

- Serves OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints.
- Runs CPU-only servers and GPU-backed workers for NVIDIA, AMD/Vulkan, and
  Apple Metal style deployments.
- Supports offline model import from local manifests and verified local model
  files, plus controlled direct downloads when networking is allowed.
- Plans and runs hot-swap, cold-swap, weighted, fallback, and single-model
  serving modes.
- Applies an 80% default resource budget so the host keeps room for the OS and
  neighboring services.
- Enforces API key auth, scopes, team/user quotas, admission/backpressure
  limits, and upstream timeout budgets.
- Records audit events, usage events, quota decisions, resource observations,
  data exports, and monthly/per-request reporting.
- Keeps production CORS explicit so browser-based clients only work from
  approved origins.

## Quick Start

```bash
cargo run --bin llmctl -- init
cargo run --bin llmctl -- model install /models/qwen.gguf --alias qwen
cargo run --bin llmctl -- quota set --subject team-a --team platform --model qwen
cargo run --bin llmctl -- server check
cargo run --bin llmctld -- --config ~/.config/rs-llmctl/config.toml
```

For an OpenAI-compatible client:

```bash
export OPENAI_BASE_URL=http://host:8765/v1
export OPENAI_API_KEY=<your-rs-llmctl-api-key>
```

External bind is intentionally strict. Production configs must enable
`security.require-auth` and `security.bind-external`, define hashed API keys,
document TLS termination or mTLS evidence, set quota policy, and use reviewed
CORS origins for browser clients.

## Model Operations

![rs-llmctl model lifecycle](docs/images/rs-llmctl-lifecycle.png)

The normal production flow is offline-first:

1. Put approved `.gguf` files under a staged model bundle.
2. Ship an offline install manifest with SHA-256 hashes.
3. Import the manifest into the reviewed config.
4. Run dry-run and security checks.
5. Start `llmctld` through the deployment system.
6. Export audit/data evidence for the deployment window.

```bash
llmctl --config /etc/rs-llmctl/config.toml model import-manifest ./manifest.toml
llmctl --config /etc/rs-llmctl/config.toml server check
llmctl --config /etc/rs-llmctl/config.toml server status
llmctl --config /etc/rs-llmctl/config.toml server plan
llmctl --config /etc/rs-llmctl/config.toml security check
llmctl --config /etc/rs-llmctl/config.toml audit retention plan
llmctl --config /etc/rs-llmctl/config.toml observe plan
llmctl --config /etc/rs-llmctl/config.toml data export --hours 24
```

Minimal manifest:

```toml
[[models]]
alias = "qwen"
path = "models/qwen.gguf"
role = "chat"
weight = 1
sha256 = "hex-encoded-sha256"
```

Relative paths resolve from the manifest directory. Direct URL installs require
HTTPS, SHA-256 verification, redirect blocking, and network timeouts.

## Production Shape

The release package publishes two binaries:

- `llmctl`: operator CLI for config, model, quota, audit, usage, data, and
  validation work.
- `llmctld`: daemon that starts planned workers and serves the API.

Build and package:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --bins
packaging/generate-checksums.sh
sha256sum target/release/llmctl target/release/llmctld > SHA256SUMS
```

Development is TDD-oriented: add or update the focused test first, make the
smallest implementation change, then run the Rust gates before review.

Typical install paths:

```bash
install -D -m 0755 target/release/llmctl /usr/local/bin/llmctl
install -D -m 0755 target/release/llmctld /usr/local/bin/llmctld
install -D -m 0644 packaging/systemd/llmctld.service /etc/systemd/system/llmctld.service
install -d -m 0750 -o llmctl -g llmctl /etc/rs-llmctl /var/lib/rs-llmctl/models /var/log/rs-llmctl
```

Stage a reviewed config:

```bash
packaging/stage-config.sh production-external-bind
TARGET=/etc/rs-llmctl/config.toml packaging/stage-config.sh cpu-only
```

The script asks you to type `COPY`, writes only the selected TOML profile, and
does not start or enable services. Available profiles are `cpu-only`,
`gpu-amd`, `gpu-auto`, `gpu-metal`, `gpu-nvidia`, `local-dev`, and
`production-external-bind`.

Validate a staged package offline:

```bash
packaging/validate-install.sh
```

The validation script runs daemon dry-run planning, `security check`,
`server status`, `server plan`, `audit retention plan`, `observe plan`, and
`systemd-analyze verify` when available.

## Security Baseline

`rs-llmctl` uses a PCI DSS v4.0.1-aligned baseline for production posture:

- external bind requires authentication and scoped API keys;
- raw API keys are never stored in config;
- external production bind requires documented TLS termination or mTLS evidence;
- sensitive exporter headers must use `env:` references;
- response headers expose only safe metadata such as request IDs, model aliases,
  quota state, and policy status;
- audit, usage, quota, and data export records are available for review;
- production CORS origins are explicit, not wildcard.

Hash a new key without putting it in process arguments:

```bash
printf '%s' "$LLMCTL_NEW_API_KEY" | llmctl security hash-key --stdin
```

Then add only the digest:

```toml
[[security.api_keys]]
id = "ops-admin"
sha256 = "<sha256-from-hash-key>"
subject = "ops-admin"
team = "platform"
scopes = ["admin"]
```

For production external bind, record where TLS is terminated and where the
operator evidence lives:

```toml
[security.tls-termination]
enabled = true
provider = "envoy-edge"
evidence = "change-record-or-runbook-url"
m-tls = true
```

## Policy And Reporting

Quota policy moves between environments as reviewed data:

```bash
llmctl --config /etc/rs-llmctl/config.toml quota export > quotas.json
llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.json
llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.toml
llmctl --config /etc/rs-llmctl/config.toml quota list
```

Quota import rejects policies with blank subjects or teams,
`requests_per_minute`, `tokens_per_day`, or `max_concurrency` values that are
not greater than zero, and `allowed_models` entries with empty model aliases.

Useful evidence commands:

```bash
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.json
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.before.json
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.after.json
llmctl server plan-diff server-plan.before.json server-plan.after.json
llmctl --config /etc/rs-llmctl/config.toml audit retention plan --envelope > retention-plan-envelope.json
llmctl --config /etc/rs-llmctl/config.toml data verify-envelope retention-plan-envelope.json
```

Review the envelope `metadata sha256`, confirm `dry_run`, confirm `deletes`,
and attach the result to the change record. `audit.retention-days` controls the
review window. `audit retention apply --yes` exists for deliberate pruning after
review.

Enterprise reporting covers data/audit summaries, quota/team governance
summaries, usage totals, audit event counts, retention windows, quota limits,
team attribution, request identifiers, model aliases, and policy status. The
AQE/OpenAI-compatible contract is available without secrets:

```bash
llmctl --config /etc/rs-llmctl/config.toml integration aqe-contract
```

It lists OpenAI paths, required auth scopes, safe response headers,
quota/team reporting fields, and model aliases.

## Storage And Observability

SQLite is the default runtime store. External database storage with Postgres is
available through `storage.database-url`; the database URL is redacted in
connection plans and migrations render dialect-specific DDL.

Router controls include admission/backpressure limits, upstream timeout budgets,
non-secret failure responses, and stable 429/504 errors that do not expose
database passwords, raw connection secrets, upstream URLs, prompts, file paths,
API keys, or bearer tokens.

OTel-oriented observability is configured through
`observability.exporter.endpoint`, traces/metrics/logs switches, and
environment-backed headers. Use `env:` for collector credentials.

Compliance evidence is available through `llmctl compliance evidence`, with
focused CRA Article 14, PCI DSS, release integrity, SBOM, and signing views.

## Deeper Docs

- [Operations guide](docs/operations.md)
- [Security model](docs/security.md)
- [Compliance evidence](docs/compliance.md)
- [Observability and reporting](docs/observability-reporting.md)
- [Storage notes](docs/storage.md)
- [Blog: Running Local Models Like Real Infrastructure](docs/blog-local-model-operations.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
