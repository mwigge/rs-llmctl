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
The product direction is Candle-native serving first. New deployments should
plan around the in-process runtime and the single `llmctl` service entrypoint.
Older external-worker deployments can be migrated deliberately, but the README,
installer, and release package now describe the native path as the default
operating model.

The MVP runtime target is intentionally small: one query model, one
recommendation model, one thinking model, and one coding model, starting with
Qwen3-family GGUF or safetensors artifacts. The Candle contract also tracks
Gemma-family, Mistral, DeepSeek, Kimi, and MiniMax options; Mistral is the
EU-friendly default alternative in the runtime status contract. DeepSeek is
wired for Candle safetensors through DeepSeekV2; DeepSeek GGUF, Kimi, and
MiniMax fail closed until reviewed Candle-compatible decoders exist. The default
runtime policy budgets 80% of CPU, RAM, and detected GPU VRAM so the rest of
the server has headroom. Native decoding is CPU-backed in this release; explicit
NVIDIA/CUDA, AMD/ROCm, and Apple/Metal execution requests fail closed until the
Candle device path is wired and validated. Inspect the native runtime contract
with:

```bash
llmctl --config /etc/rs-llmctl/config.toml runtime status
llmctl --config /etc/rs-llmctl/config.toml runtime heartbeat
llmctl --config /etc/rs-llmctl/config.toml runtime placement
llmctl --config /etc/rs-llmctl/config.toml runtime validation-plan
llmctl --config /etc/rs-llmctl/config.toml runtime validate
llmctl --config /etc/rs-llmctl/config.toml runtime validation-run \
  --evidence-output ./native-validation.json
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
the resolved alias without a sidecar process. Qwen3, Gemma-family, Mistral
safetensors, and DeepSeek safetensors models use native Candle loaders where
Candle exposes the architecture and artifact format; DeepSeek GGUF, Kimi, and
MiniMax remain blocked until reviewed native decoders are wired. The native scheduler contract is explicit:
the in-process runtime now applies an implemented FIFO queue with bounded
per-engine concurrency and observable queue/admission wait metadata. It also
stamps deterministic prefill/decode phase scheduling metadata so operators can
separate admission timing from the request-local decode loop. Continuous
batching, KV cache budget metadata, KV cache key metadata, and cancellation
token metadata remain serialized as contract fields with `implemented=false`;
cross-request KV reuse and token-level decode cancellation are unsupported until
those specific behaviors are backed by runtime execution.

The native validation track starts offline and becomes executable on the target
host. `runtime validation-plan` emits JSON for real Qwen/Gemma/Mistral/DeepSeek
safetensors smoke tests, CPU/NVIDIA/AMD-Vulkan/Apple Metal coverage, long
streaming soak tests, graceful drain during active streams, circuit breaker and
heartbeat checks under load, API-key rotation with quota concurrency, and
benchmark fields for latency, tokens/sec, RSS memory, and VRAM. `runtime
validation-run` validates the configured positive-weight artifacts and writes
pass/fail evidence without downloading models.

## What It Does

- Serves OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints.
- Ships a static, build-step-free chat page at `/playground` for poking at any
  routed model from a browser with a model picker and a bearer API-key field.
- Screens inbound chat messages with regex/phrase-based guardrails — PII
  detection/redaction and prompt-injection ("jailbreak") phrase filtering —
  with flag/redact/block actions and audit-trail findings.
- Documents a separate `rs-llmctl-client` Rust SDK crate for application code,
  including client-managed sessions, client-side tool loops, streaming, and
  non-secret metadata.
- Runs CPU-only native serving now and reports GPU placement/budget evidence for
  NVIDIA, AMD/Vulkan, and Apple Metal style deployments without pretending those
  accelerators are active native decode devices yet.
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
`/var/log/rs-llmctl`, and installs `llmctld.service` without starting it.
The service intentionally keeps the stable `llmctld.service` unit name for
operator runbooks, monitoring labels, and upgrade habits, while its default
`ExecStart` runs `llmctl --config /etc/rs-llmctl/config.toml server run`. The
default installer flow expects `first-run --apply` with a model and generated
API key before the service is enabled. Set `LLMCTL_START_SERVICE=1` only when
the config is already complete and intentionally ready to run. The staged
default config binds the API to `http://127.0.0.1:8765/v1`.

Set `PREFIX=/some/path` to choose another binary prefix. System service installs
must not use a home-directory prefix; use `LLMCTL_INSTALL_SYSTEMD=0` for a
binary-only install. `LLMCTL_CONFIG_DIR`, `LLMCTL_CONFIG`,
`LLMCTL_STATE_DIR`, `LLMCTL_LOG_DIR`, and `LLMCTL_SERVICE_NAME` override the
default system paths and service name. The monthly audit timer is installed
when packaged units are present but stays disabled by default; set
`LLMCTL_ENABLE_AUDIT_TIMER=1` only after enabling monthly reports in config.

`install.sh` verifies archive integrity with `SHA256SUMS`. Tagged CI releases
also publish `SHA256SUMS.sig` or `SHA256SUMS.minisig`; verify that release
signature with `cosign` or `minisign` before production installation when your
policy requires publisher authentication.

For a pinned version or another fork:

```bash
curl -fsSL https://raw.githubusercontent.com/mwigge/rs-llmctl/main/install.sh | RS_LLMCTL_VERSION=v1.2.1 RS_LLMCTL_REPO=your-org/rs-llmctl sh
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

## First-Run Operator Path

Use `first-run` when bringing up a new host or local operator sandbox. It is
dry-run by default, emits JSON, does not download a model, and does not write
plaintext secrets to config:

```bash
llmctl --config ./config.toml first-run \
  --secret-output ./operator.secret
```

Apply only after reviewing the plan. `--apply` requires `--secret-output`; the
raw key is written once to that file with `0600` permissions on Unix, while the
config stores only the SHA-256 digest and non-secret metadata. Starter model
configuration stays offline unless you provide a local file:

```bash
llmctl --config ./config.toml first-run --apply \
  --secret-output ./operator.secret \
  --starter-model-path /models/qwen.gguf \
  --starter-model-alias qwen
```

The output includes an `ask_question` smoke plan for `rs-llmctl-client` and an
OpenAI-compatible `/v1/chat/completions` request plan using the generated key
from your secret store, without printing the secret.

The release smoke path does not require Docker. `tests/smoke/smoke_native_release.sh`
requires `LLMCTL_NATIVE_SMOKE_MODEL_PATH` to point at one real local model
artifact, installs the packaged tarball into a temporary prefix, runs
`first-run --apply` with one generated API key, starts `llmctl server run`, and
queries the model through the `rs-llmctl-client` example. A successful run ends
with `ok release smoke passed for <model>`. Use a VM or privileged systemd test
host only when the test must validate systemd activation itself.

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

For the Rust `rs-llmctl-client` crate:

```bash
export LLMCTL_BASE_URL=http://host:8765
export LLMCTL_API_KEY=<your-rs-llmctl-api-key>
```

Rust applications should depend on the separate `rs-llmctl-client` crate, not
the server crate. The SDK is intentionally a client-side wrapper around the
OpenAI-compatible API: it keeps conversation sessions in the application,
resends the full message history, attaches stable `metadata.session_id` and
lineage metadata, and runs tool calls in the caller process before submitting
tool results back to `/v1/chat/completions`. `rs-llmctl` authorizes, routes,
audits, meters, and records metadata for those requests; it does not execute
tools and does not run tool side effects on behalf of clients.

Small Rust applications can use the `ask_question` helper for the first call:

```rust
use rs_llmctl_client::{AskConfig, LlmctlClient, Question};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = LlmctlClient::from_env()?;
    let answer = client
        .ask_question(
            AskConfig::new("qwen").system("Answer as the internal platform assistant."),
            Question::new("Which model is serving coding requests?"),
        )
        .await?;
    println!("{answer}");
    Ok(())
}
```

`LlmctlClient::from_env()` accepts `LLMCTL_BASE_URL`/`LLMCTL_API_KEY`,
or `RS_LLMCTL_BASE_URL`/`RS_LLMCTL_API_KEY`. It intentionally ignores
`OPENAI_BASE_URL`/`OPENAI_API_KEY`; those names are for generic
OpenAI-compatible clients pointed at `/v1`.
The SDK also exposes `/v1/embeddings` through `EmbeddingRequest` for local
search, recommendation, and RAG workflows.
On Candle-native deployments, production embeddings use the semantic native
embedding contract:

```toml
[runtime.embeddings]
mode = "semantic"
model-alias = "embed"
```

The configured alias should point to a BERT-style safetensors embedding model
with `tokenizer.json` and `config.json`. The deterministic native vectorizer is
kept only for local development with `mode = "dev-fallback"` and labels
responses as `non-semantic-dev-fallback`.

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
family = "qwen3"
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
`llmctl service ...` targets the installed system unit by default; pass `--user`
only when you intentionally installed a user-scoped service.

Lifecycle outputs are script-friendly JSON by default. Model and service plans
include `runtime_backend` and `entrypoint` fields; on the default native path
they identify `candle-native` and the single service entrypoint
`llmctl server run`. `llmctl service status/start/stop/restart` wraps the
systemd lifecycle around the stable `llmctld.service` unit name while keeping
the installed runtime binary as `llmctl`. `service upgrade` and
`service downgrade` are guarded planning commands; use the verified release
installer or system package manager to change binary versions, then restart the
service.

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
# writes dist/rs-llmctl-<os>-<arch>.tar.gz, dist/SHA256SUMS,
# and stages README.md, CHANGELOG.md, LICENSE, and llmctld.service
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

Release notes live in [CHANGELOG.md](CHANGELOG.md). Each release entry should
name the native runtime posture, service-unit decision, packaging artifacts,
and verification commands used for the cut. The default binary archive contains
the runtime binary plus the operator-facing `README.md`, `CHANGELOG.md`,
`LICENSE`, and systemd unit template; checksums and optional signatures remain
separate artifacts in `dist/`.

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

Generate a new API key when you want `rs-llmctl` to mint strong random key
material for a client. Prefer writing the raw secret directly to a secret-store
staging file; the file is created with `0600` permissions on Unix and the CLI
does not print the secret when `--output` is used. Put the raw value in your
secret manager, systemd credential, Vault/SOPS/1Password item, Kubernetes
Secret, or equivalent. Keep only the SHA-256 digest and non-secret metadata in
config:

```bash
llmctl security generate-key --prefix llmctl-prod --output ./ops-admin.secret
llmctl --config /etc/rs-llmctl/config.toml security add-key \
  --id ops-admin-2026-q2 \
  --sha256 <sha256-from-generate-key> \
  --subject ops-admin \
  --team platform \
  --scope admin \
  --owner platform-sre \
  --purpose operations-admin \
  --last-four <last-four-from-generate-key>
```

If your secret store generates the raw key, hash it without putting it in
process arguments:

```bash
printf '%s' "$LLMCTL_NEW_API_KEY" | llmctl security hash-key --stdin
```

Then add or review only the digest:

```toml
[[security.api_keys]]
id = "ops-admin-2026-q2"
sha256 = "<sha256-from-hash-key>"
subject = "ops-admin"
team = "platform"
scopes = ["admin"]
owner = "platform-sre"
purpose = "operations-admin"
last-four = "<last-four>"
status = "active"
```

Keep key IDs stable and non-secret. Use IDs that encode owner and rotation
window, such as `platform-chat-2026-q2`, then track them without exposing
digests:

```bash
llmctl --config /etc/rs-llmctl/config.toml security list-keys
llmctl --config /etc/rs-llmctl/config.toml security key-usage --id platform-chat-2026-q2 --hours 168
```

Rotate with an overlap window: add a new active key ID, mark the old key as
`retiring`, restart the service, move clients, confirm usage has moved, then
revoke the stale ID. `--replace` is still available for emergency in-place
replacement, but overlap rotation is the default operational path:

```bash
llmctl --config /etc/rs-llmctl/config.toml security rotate-key \
  --id platform-chat-2026-q2 \
  --new-id platform-chat-2026-q3 \
  --sha256 <new-sha256> \
  --last-four <new-last-four> \
  --reason "quarterly rotation"
sudo systemctl restart llmctld.service
llmctl --config /etc/rs-llmctl/config.toml security key-usage --hours 24
llmctl --config /etc/rs-llmctl/config.toml security revoke-key \
  --id platform-chat-2026-q2 \
  --reason "clients moved to q3 key"
sudo systemctl restart llmctld.service
```

Every authenticated request is audited with actor, team, action, model/resource,
outcome, request ID, and non-secret API-key metadata. `security key-usage`
joins audit rows to usage rows by request ID so reviewers can see request
counts, token totals, latency totals, models, actors, teams, and rotation or
revocation events without storing or exporting raw secrets. Developer clients
stay simple:

```bash
export OPENAI_BASE_URL=http://host:8765/v1
export OPENAI_API_KEY="$LLMCTL_CLIENT_API_KEY"
curl -sS "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}]}'
```

For production external bind, record where TLS is terminated and where the
operator evidence lives:

```toml
[security]
trusted-proxies = ["127.0.0.1"]

[security.tls-termination]
enabled = true
provider = "envoy-edge"
evidence = "change-record-or-runbook-url"
m-tls = true
```

This is the production HTTPS control for the service edge. The Rust binary uses
Rustls-backed HTTP clients for outbound HTTPS model downloads, OTel export, and
Postgres TLS, and it can also serve inbound HTTPS when `[server.tls]` has a
certificate and key. The default listener is still plain HTTP on the configured
bind address. For production, either put the service behind Envoy, NGINX,
HAProxy, a cloud load balancer, ingress, or service mesh, or use native Rustls
server-certificate TLS and document the certificate source, rotation owner, and
evidence URL. Client-certificate mTLS remains an edge/service-mesh control and
should be recorded in `security.tls-termination`.

### Guardrails

Inbound chat messages can be screened by lightweight, regex/phrase-based
guardrails before they reach a model — no external moderation service, zero
added network calls:

```toml
[guardrails.pii]
action = "redact"          # off | flag | redact | block
categories = ["email", "credit-card", "ssn", "phone", "api-key"]

[guardrails.jailbreak]
action = "flag"            # off | flag | block
phrases = ["additional phrase to match"]
```

PII detection covers email addresses, phone numbers, credit-card numbers,
SSNs, and common API-key shapes (`sk-…`, `ghp_…`, `xox…`, etc.); jailbreak
detection matches a built-in list of prompt-injection phrases plus any
operator-supplied additions. `redact` replaces matched spans with
`[REDACTED:<CATEGORY>]` markers in place before the request is forwarded;
`block` rejects the request with HTTP 400 and records a `denied` audit event;
`flag` records a `flagged` audit event and lets the request through unchanged.
Findings are always written to the audit trail with category and match counts
— never the matched text itself.

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
`--property=MemoryMax=8589934592`. The packaged Linux installer computes
`CPUQuota=(nproc * 80)%` and applies `MemoryMax=80%` by default; apply a
generated systemd drop-in when a specific host plan should override that
baseline. The default runtime policy budgets 80% of CPU, RAM, and detected GPU
VRAM so first-time plans have concrete headroom. GPU VRAM budgets are exported
as `metadata-only`
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
resource snapshots, native runtime status, and drift observations. Usage spans
carry GenAI semantic-convention attributes (`gen_ai.system`,
`gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.response.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) alongside the
existing `llmctl.*` attributes, so generic GenAI-aware OTel consumers can
group and chart `rs-llmctl` spans without custom mapping.

Two ecosystem integrations build on that foundation:

```toml
# Send OTel data straight to a Langfuse project — derives the OTLP/HTTP
# endpoint and HTTP Basic auth header from project keys when no explicit
# observability.exporter.endpoint is set.
[observability.langfuse]
enabled = true
host = "https://cloud.langfuse.com"
public-key = "env:LANGFUSE_PUBLIC_KEY"
secret-key = "env:LANGFUSE_SECRET_KEY"

# Fire a fire-and-forget HTTP callback after every completion, carrying the
# same usage/lineage metadata as the audit trail — for ecosystems (chat ops,
# dashboards, ticketing) that consume callbacks rather than OTLP.
[observability.webhook]
enabled = true
url = "https://hooks.example.internal/llmctl-usage"
headers = { "Authorization" = "env:LLMCTL_WEBHOOK_TOKEN" }
timeout-ms = 5000
```

The webhook never blocks or fails the in-flight request — delivery runs on a
background task and failures are logged at `warn`, not surfaced to the caller.

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

Runtime caveats are part of the product contract. Qwen3, Gemma-family, Mistral
safetensors, and DeepSeek safetensors are the native runnable families where
Candle exposes the needed model APIs. DeepSeek GGUF remains closed because
Candle 0.10.2 does not expose quantized DeepSeek2 weights. Kimi and MiniMax
remain closed until Candle exposes reviewed architecture modules or rs-llmctl
vendors maintained implementations. FIFO queue discipline is implemented for
native chat requests with bounded per-engine concurrency and wait-time metadata.
Prefill/decode phase scheduling metadata is emitted for every admitted native
request, but continuous batching and low-level KV-cache scheduler controls are
still serialized as metadata-only contract fields with `implemented=false`;
size latency and capacity plans against observed single-request behavior until
those runtime behaviors are wired. Cancellation has an admission-time cancelled
metadata check, while cancellation token metadata and token-level decode loop
cancellation remain explicit unsupported scheduler boundaries.

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
- [Client SDK and tool loops](docs/client-sdk.md)
- [Security model](docs/security.md)
- [Compliance evidence](docs/compliance.md)
- [Observability and reporting](docs/observability-reporting.md)
- [Configuration](docs/configuration.md)
- [Data contracts](docs/data-contracts.md)
- [AIOps/MLOps platform](docs/aiops-mlops-platform.md)
- [Storage notes](docs/storage.md)
- [Final acceptance review](docs/reviews/final-acceptance-review.md)
- [Changelog](CHANGELOG.md)
- [Blog: Running Local Models Like Real Infrastructure](docs/blog-local-model-operations.md)
- [Blog: Rust Native Model Operations](docs/blog-rust-native-model-ops.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
