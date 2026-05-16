# rs-llmctl

`rs-llmctl` is a standalone operations tool for running local and private LLM
models like real server infrastructure. It gives operators one default Rust
binary, an OpenAI-compatible serving endpoint, model lifecycle commands,
resource budget planning and reporting, quotas, audit trails, usage reporting,
and OTel-ready observability.

![rs-llmctl operations overview](docs/images/rs-llmctl-operations-hero.png)

The goal is simple: make model delivery boring in the best way. You should be
able to stage a model, verify it, plan CPU/RAM/VRAM budgets, serve it to
internal clients, swap it safely, and keep useful evidence about what happened.
The product direction is candle-native serving first. `llama-server`
compatibility and fallback remain available for existing deployments during
migration; in plain terms, llama-server compatibility and fallback are
secondary to the native path.

The MVP runtime target is intentionally small: one query model, one
recommendation model, one thinking model, and one coding model, starting with
Qwen3-family GGUF or safetensors artifacts. The Candle contract also tracks
Gemma 4, Kimi, and Mistral-family options; Mistral is the EU-friendly default
alternative in the runtime status contract. The default runtime policy budgets
80% of CPU, RAM, and detected GPU VRAM so the rest of the server has headroom.
Hardware selection is automatic for NVIDIA/CUDA, AMD/ROCm, Apple/Metal targets,
and CPU fallback. Inspect the native runtime contract with:

```bash
llmctl --config /etc/rs-llmctl/config.toml runtime status
llmctl --config /etc/rs-llmctl/config.toml runtime heartbeat
llmctl --config /etc/rs-llmctl/config.toml runtime placement
llmctl --config /etc/rs-llmctl/config.toml runtime validate
llmctl --config /etc/rs-llmctl/config.toml runtime route --role coding
```

For a small two-server layout, set `cluster.nodes` so one node owns the
`thinking` and `recommendation` roles while another owns `coding`. Placement
output reports node IDs, base URLs, roles, and model aliases without exposing
local model paths.
Heartbeat output uses the same contract for a laptop, one server, or a cluster:
it reports node ID, runtime backend, routing mode, placement health, assigned
model counts, unassigned model aliases, and the configured budget fraction.
When `llmctl server run` is active it emits the same heartbeat as
`llmctl.runtime.heartbeat` every `runtime.heartbeat-interval-seconds` seconds
(default `30`; set `0` to disable periodic runtime heartbeat emission).
The native serving path is alias-keyed: every configured non-zero-weight model
gets its own in-process Candle engine, and OpenAI-compatible requests route to
the resolved alias without a sidecar process. Qwen3, Gemma-family, and
Mistral-family models use native Candle loaders where Candle exposes the
architecture; Kimi remains blocked until Candle exposes a reviewed Kimi module
or rs-llmctl vendors one. The native scheduler contract is explicit:
scheduler metadata, admission/backpressure keys, KV cache budgets, and
cancellation metadata are serialized in load plans from the start so the engine
surface is observable as it grows.
The lower-level scheduler fields remain metadata-only where they describe
future queue discipline, continuous batching, KV cache budget metadata, and
cancellation token metadata; those specific scheduler knobs are reported with
`implemented=false` until they are backed by runtime behavior.

## What It Does

- Serves OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints.
- Runs CPU-only servers and GPU-backed workers for NVIDIA, AMD/Vulkan, and
  Apple Metal style deployments.
- Supports offline model import from local manifests and verified local model
  files, plus controlled direct downloads when networking is allowed.
- Plans and runs hot-swap, cold-swap, weighted, fallback, and single-model
  serving modes.
- Plans and enforces an 80% default CPU/RAM budget in the packaged Linux
  systemd service, and reports detected GPU VRAM budgets as runtime planning
  evidence because portable GPU VRAM cgroup enforcement is not available.
- Enforces API key auth, scoped brute-force throttling, team/user quotas,
  admission/backpressure limits, upstream timeout budgets, circuit breakers,
  and drain-aware readiness for graceful shutdown.
- Records audit events, usage events, quota decisions, resource observations,
  data exports, and monthly/per-request reporting.
- Gives AI-developer workflows local search and local recommendation endpoints
  for code, docs, and private operational material.
- Keeps production CORS explicit so browser-based clients only work from
  approved origins.

## Install In 5 Steps

One-line install from GitHub Releases:

```bash
curl -fsSL https://raw.githubusercontent.com/mwigge/rs-llmctl/main/install.sh | sh
```

On Linux with systemd, the installer asks for `sudo` when needed, verifies the
release archive against `SHA256SUMS`, safely extracts the single default `llmctl` binary
to `/usr/local/bin`, creates the `llmctl` system user, stages
`/etc/rs-llmctl/config.toml`, creates `/var/lib/rs-llmctl`,
`/var/lib/rs-llmctl/models`, `/var/lib/rs-llmctl/reports`, and
`/var/log/rs-llmctl`, installs `llmctld.service`, then enables and starts it.
The service keeps the stable `llmctld.service` unit name for operators, but its
default `ExecStart` runs `llmctl --config /etc/rs-llmctl/config.toml server
run`. The default service binds to `http://127.0.0.1:8765/v1` without requiring
API keys; switch to a production profile before exposing it externally.

Set `PREFIX=/some/path` to choose another binary prefix. System service installs
must not use a home-directory prefix; use `LLMCTL_INSTALL_SYSTEMD=0` for a
binary-only install. `LLMCTL_CONFIG_DIR`, `LLMCTL_CONFIG`,
`LLMCTL_STATE_DIR`, `LLMCTL_LOG_DIR`, and `LLMCTL_SERVICE_NAME` override the
default system paths and service name.

For a pinned version or another fork:

```bash
curl -fsSL https://raw.githubusercontent.com/mwigge/rs-llmctl/main/install.sh | RS_LLMCTL_VERSION=v1.1.0 RS_LLMCTL_REPO=your-org/rs-llmctl sh
```

After install:

```bash
sudo systemctl status llmctld.service
llmctl --config /etc/rs-llmctl/config.toml server status
```

Service lifecycle commands:

```bash
sudo systemctl status llmctld.service
sudo journalctl -u llmctld.service -f
sudo systemctl restart llmctld.service
sudo systemctl stop llmctld.service
sudo systemctl start llmctld.service
```

For production, add at least one hashed API key with `security hash-key`,
configure TLS termination evidence, import or install a verified model, then run
`server check`, `security check`, `observe plan`, and `compliance evidence`
before binding externally.

## Operate In 10 Steps

1. Stage `/etc/rs-llmctl/config.toml` with the `production-aiops` profile.
2. Set `observability.exporter.endpoint` to your OTel collector and keep
   traces, metrics, and logs enabled.
3. Put API keys, OTel tokens, and policy signing secrets in your secret store,
   not in process arguments or plaintext config.
4. Import a SHA-256 verified model manifest with `model import-manifest`.
5. Run `server plan`, `server check`, `security check`, and `security
   audit-config`.
6. Start `llmctld.service` and test `/v1/models` and `/v1/chat/completions`.
7. Export `aiops slo-plan --format prometheus` into Prometheus/Alertmanager
   and `aiops slo-plan --format grafana` into Grafana.
8. Use lineage headers such as `x-llmctl-lineage-id: corpus:ops-v1` so requests
   can be tied back to prompts, corpora, models, and releases.
9. Run eval suites with `eval run-suite`, export data with Arrow/Parquet, and
   create monthly audit envelopes.
10. Sign policy changes with Ed25519 and append them to the transparency log
    before promotion.

## Quick Start

```bash
cargo run --bin llmctl -- init
cargo run --bin llmctl -- model install /models/qwen.gguf --alias qwen
cargo run --bin llmctl -- quota set --subject team-a --team platform --model qwen
cargo run --bin llmctl -- server check
cargo run --bin llmctl -- --config ~/.config/rs-llmctl/config.toml server run
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
5. Start `llmctld.service` through the deployment system.
6. Export audit/data evidence for the deployment window.

```bash
llmctl --config /etc/rs-llmctl/config.toml model import-manifest ./manifest.toml
llmctl --config /etc/rs-llmctl/config.toml model inventory
llmctl --config /etc/rs-llmctl/config.toml model status qwen
llmctl --config /etc/rs-llmctl/config.toml model stop qwen --dry-run > model-stop-plan.json
llmctl --config /etc/rs-llmctl/config.toml model stop qwen
llmctl --config /etc/rs-llmctl/config.toml model start qwen --weight 1 --dry-run > model-start-plan.json
llmctl --config /etc/rs-llmctl/config.toml model start qwen --weight 1
llmctl --config /etc/rs-llmctl/config.toml model upgrade ./models/qwen-v2.gguf --alias qwen --new-alias qwen-v2 --sha256 <sha256> --dry-run > model-upgrade-plan.json
llmctl --config /etc/rs-llmctl/config.toml model upgrade ./models/qwen-v2.gguf --alias qwen --new-alias qwen-v2 --sha256 <sha256>
llmctl service status --dry-run > service-status-plan.json
llmctl service restart --dry-run > service-restart-plan.json
llmctl --config /etc/rs-llmctl/config.toml server check
llmctl --config /etc/rs-llmctl/config.toml server status
llmctl --config /etc/rs-llmctl/config.toml server plan
llmctl --config /etc/rs-llmctl/config.toml security check
llmctl --config /etc/rs-llmctl/config.toml audit retention plan
llmctl --config /etc/rs-llmctl/config.toml observe plan
llmctl --config /etc/rs-llmctl/config.toml data export --hours 24
```

For a production AIOps-style starting point, use the config wizard profile:

```bash
llmctl --config /etc/rs-llmctl/config.toml init \
  --profile production-aiops \
  --bind 0.0.0.0 \
  --otel-endpoint https://otel-collector.example/v1/traces \
  --log-format json \
  --event-format jsonl \
  --data-format arrow-json \
  --tls-provider envoy-edge \
  --tls-evidence change-record-123 \
  --mtls
```

That writes the typed config for SSE streaming, OTel trace/metric/log export,
JSON logging, schema-versioned events, the data fabric, CRA-style monthly
audit reporting, and external-bind security controls. The wizard does not add
API keys for you; create hashed keys with `security hash-key` and review them
like any other production secret material.

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

Model lifecycle commands are config-first: `model install` and
`model import-manifest` register verified artifacts, `model inventory` shows
configured models without leaking full paths, `model list` emits raw configured
model entries, `model stop` sets a model weight to zero, and `model start`
restores a positive weight. `model update`, `model upgrade`, and
`model downgrade` replace a configured model from a verified local file or
allowed remote source; use `--new-alias` when the replacement should be staged
beside the previous alias. Start, stop, and replacement changes report that a
service restart is required before routing changes take effect. Use `--dry-run`
on `model start`, `model stop`, `model update`, `model upgrade`, or
`model downgrade` to emit the JSON lifecycle plan without editing the config or
copying model artifacts.

Lifecycle outputs are script-friendly JSON by default. Model and service plans
include `runtime_backend` and `entrypoint` fields; on the default native path
they identify `candle-native` and the single service entrypoint
`llmctl server run`. `llmctl service status/start/stop/restart/upgrade/downgrade`
wraps the systemd lifecycle around the stable `llmctld.service` unit name while
keeping the installed runtime binary as `llmctl`.

## Production Shape

The default release package publishes one Rust binary:

- `llmctl`: operator CLI and service entrypoint for config, serving, model,
  quota, audit, usage, data, and validation work.

Legacy source builds may still include `llmctld`, and older release archives
may still carry it, but new default packaging installs only `llmctl`. The
systemd unit remains named `llmctld.service` so existing service lifecycle
habits and monitoring labels do not need to change immediately.

Build and package:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --bin llmctl
packaging/generate-checksums.sh
# writes dist/rs-llmctl-<os>-<arch>.tar.gz and dist/SHA256SUMS
```

Development is TDD-oriented: add or update the focused test first, make the
smallest implementation change, then run the Rust gates before review.

Typical install paths:

```bash
install -D -m 0755 target/release/llmctl /usr/local/bin/llmctl
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

The validation script checks the installed `llmctl` binary, systemd unit,
`security check`, `server status`, `server plan`, `audit retention plan`,
`observe plan`, and `systemd-analyze verify` when available.

## Security Baseline

`rs-llmctl` uses a PCI DSS v4.0.1-aligned baseline for production posture:

- external bind requires authentication and scoped API keys;
- raw API keys are never stored in config;
- external production bind requires documented TLS termination or mTLS evidence;
- CRA Article 14 is treated as active, so production external bind requires
  monthly audit reports, audit retention, and an OTel exporter endpoint;
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
llmctl --json --config /etc/rs-llmctl/config.toml server plan > server-plan.json
llmctl --json --config /etc/rs-llmctl/config.toml server plan > server-plan.before.json
llmctl --json --config /etc/rs-llmctl/config.toml server plan > server-plan.after.json
llmctl server plan-diff server-plan.before.json server-plan.after.json
llmctl --config /etc/rs-llmctl/config.toml audit retention plan --envelope > retention-plan-envelope.json
llmctl --config /etc/rs-llmctl/config.toml data verify-envelope retention-plan-envelope.json
```

Review the envelope `metadata sha256`, confirm the plan metadata, confirm
`deletes`, and attach the result to the change record. `audit.retention-days`
controls the review window. `audit retention apply --yes` exists for deliberate
pruning after review.

Enterprise reporting covers data/audit summaries, quota/team governance
summaries, usage totals, audit event counts, retention windows, quota limits,
team attribution, request identifiers, model aliases, and policy status. The
current upstream usage totals use upstream-reported token counts; exact native
tokenizer accounting is used on the Candle-native path when the model tokenizer
loads; native tokenizer metering is exact when tokenizer metadata is available,
with estimated accounting explicitly labeled otherwise.
Resource budget reports now include enforceable Linux systemd property values
for CPU and memory: `CPUQuota` and `MemoryMax` appear under
`resource_limits.systemd` in `server plan` output. The same plan also exports
`unit_properties` lines for systemd unit/drop-in files and `systemd_run_args`
for transient `systemd-run` tests, for example `--property=CPUQuota=640%` and
`--property=MemoryMax=8589934592`. The packaged systemd unit applies
`CPUQuota=80%` and `MemoryMax=80%` by default; apply a generated systemd drop-in when a
specific host plan should override that baseline. The default runtime policy
budgets 80% of CPU, RAM, and detected GPU VRAM so first-time plans have
concrete headroom. GPU VRAM budgets are exported as `metadata-only`
planning evidence with `hard_enforced=false` and no systemd property;
`rs-llmctl` does not claim hard GPU VRAM enforcement because common GPU runtimes
do not expose a portable cgroup property equivalent to `MemoryMax`. The
AQE/OpenAI-compatible contract is available without secrets:

```bash
llmctl --config /etc/rs-llmctl/config.toml integration aqe-contract
```

It lists OpenAI paths, required auth scopes, safe response headers,
quota/team reporting fields, and model aliases.

Local AI-developer workflows can also use `/v1/local/search` and
`/v1/local/recommendations` with caller-provided documents. That keeps private
code and local material under the caller's control while still returning ranked
context and recommendation metadata for an assistant or AQE workflow.

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
environment-backed headers. Use `env:` for collector credentials. Runtime
events emit OTel-friendly signals for request routing, audit, quota, usage,
resource snapshots, native runtime status, and drift observations.

Data movement is explicit and schema-versioned:

```bash
llmctl --config /etc/rs-llmctl/config.toml data contracts
llmctl --config /etc/rs-llmctl/config.toml data contracts --dataset finops
llmctl --config /etc/rs-llmctl/config.toml data export --dataset security --format json
llmctl --config /etc/rs-llmctl/config.toml data export --dataset observability --format jsonl
llmctl --config /etc/rs-llmctl/config.toml data export --dataset finops --format arrow-json
llmctl --config /etc/rs-llmctl/config.toml data export --dataset finops --format arrow-ipc --output finops.arrow
llmctl --config /etc/rs-llmctl/config.toml data export --dataset finops --format parquet --output finops.parquet
llmctl --config /etc/rs-llmctl/config.toml eval run --model qwen --suite golden-code --score 0.91 --baseline 0.85
llmctl --config /etc/rs-llmctl/config.toml lineage record --kind model --id qwen --parent corpus:internal-docs
llmctl aiops slo-plan
llmctl aiops slo-plan --format prometheus --output llmctl-slo-rules.yaml
llmctl aiops slo-plan --format grafana --output llmctl-slo-dashboard.json
llmctl aiops incident-template --severity high --team platform
llmctl aiops gaps
```

The data fabric gives operators one contract surface for security,
observability, user, finops, model, drift, and audit data. JSON and JSONL are
available for simple scripts, `arrow-json` exposes the schema and rows, and
native Arrow IPC/Parquet writers produce files for analytics systems.

Lineage is how you explain where a model answer came from. A client can send
`x-llmctl-lineage-id` or `metadata.lineage_ids` on chat, search, and
recommendation requests. `rs-llmctl` records those joins with the request ID,
model, corpus, and source endpoint so audits can connect a response back to a
prompt template, document corpus, embedding index, model, or release.

The dashboard path is intentionally simple. `llmctl aiops slo-plan --format
prometheus` emits Alertmanager-compatible rules, and `--format grafana` emits a
Grafana dashboard JSON file. Import those files with your normal monitoring
provisioning workflow.

Policy bundles can still be signed and verified with HMAC-SHA256 key material
from the environment:

```bash
llmctl policy bundle --name platform --input policy.json --output policy-bundle.json --signing-key-env LLMCTL_POLICY_KEY
llmctl policy verify-bundle policy-bundle.json --signing-key-env LLMCTL_POLICY_KEY
```

For reviewed promotion workflows, use Ed25519 keys and publish the policy
artifact to the local append-only transparency log:

```bash
llmctl policy keygen --private-key policy-ed25519.private.json --public-key policy-ed25519.public.json
llmctl policy sign --input policy.json --signature policy-signature.json --private-key policy-ed25519.private.json
llmctl policy verify --input policy.json --signature policy-signature.json --public-key policy-ed25519.public.json
llmctl policy log append --log policy-transparency.jsonl --artifact policy.json --signature policy-signature.json
llmctl policy log verify --log policy-transparency.jsonl
llmctl policy legal-hold-plan --dataset audit --case-id case-123 --reason "regulatory review"
```

Sigstore and Rekor are public supply-chain transparency tools. Sigstore is a
keyless signing ecosystem, and Rekor is an append-only transparency log for
signed artifact metadata. `rs-llmctl` includes local Ed25519 signing and a local
hash-chained transparency log today; Sigstore/Rekor integration is the external
publication path to add when your organization wants public or shared
transparency-log evidence.

Compliance evidence is available through `llmctl compliance evidence`, with
focused CRA Article 14, PCI DSS, release integrity, SBOM, and signing views.

## Deeper Docs

- [Operations guide](docs/operations.md)
- [AI developer workflows](docs/ai-developer-workflows.md)
- [Security model](docs/security.md)
- [Compliance evidence](docs/compliance.md)
- [Observability and reporting](docs/observability-reporting.md)
- [Configuration](docs/configuration.md)
- [Data contracts](docs/data-contracts.md)
- [AIOps/MLOps platform](docs/aiops-mlops-platform.md)
- [Storage notes](docs/storage.md)
- [Final acceptance review](docs/reviews/final-acceptance-review.md)
- [Blog: Running Local Models Like Real Infrastructure](docs/blog-local-model-operations.md)
- [Blog: Rust Native Model Operations](docs/blog-rust-native-model-ops.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
