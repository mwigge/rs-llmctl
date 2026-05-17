# Configuration

`rs-llmctl` keeps production behavior in one typed TOML config. The goal is
that an operator can see how serving, security, streaming, logs, telemetry,
events, and data exports are intended to behave before the daemon starts.

## Wizard Profiles

Create a local development config:

```bash
llmctl init --profile local-dev
```

For a safer first-run operator journey, start with the plan-only command. It is
JSON-friendly, dry-run by default, does not download starter models, and keeps
plaintext API keys out of config:

```bash
llmctl --config ./config.toml first-run \
  --secret-output ./operator.secret
```

To apply the reviewed plan, provide `--apply`. A generated API key is written
once to `--secret-output`, the config stores only the SHA-256 digest, and a
starter model is configured only from a local path supplied by the operator:

```bash
llmctl --config ./config.toml first-run --apply \
  --secret-output ./operator.secret \
  --starter-model-path /models/qwen.gguf \
  --starter-model-alias qwen
```

The JSON response includes an `ask_question` smoke plan and an
OpenAI-compatible `/v1/chat/completions` smoke request plan. Run those after the
daemon is started and the secret has been moved into the operator's secret
store or exported through `LLMCTL_API_KEY`.

The packaged smoke script automates that default path without Docker when a
local model artifact is available:

```bash
LLMCTL_NATIVE_SMOKE_MODEL_PATH=/models/qwen.gguf \
  tests/smoke/smoke_native_release.sh
```

It installs the release tarball into a temporary prefix, applies first-run with
one generated API key, starts the server, and calls the local chat endpoint.

Create a CPU-only host config:

```bash
llmctl init --profile cpu-only
```

Create a production AIOps-oriented config:

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

The production profile enables auth-required external bind, monthly audit
reports, OTel traces/metrics/logs, JSON daemon logs, schema-versioned event
output, and the data fabric. It still leaves API key creation as an explicit
operator action.

The TLS flags populate `security.tls-termination` evidence for the production
edge. The default listener remains HTTP; put production external bind behind
Envoy, NGINX, HAProxy, a cloud load balancer, ingress, or service mesh for
HTTPS/mTLS. Outbound HTTPS from the Rust binary uses Rustls-backed clients for
verified model downloads, OTel export, and Postgres TLS. External provider
routing is disabled in this native-only release.

## Relevant Blocks

```toml
[server]
upstream-timeout-seconds = 300
graceful-drain-seconds = 5
circuit-breaker-failures = 3
circuit-breaker-reset-seconds = 30

[runtime.embeddings]
mode = "semantic"
model-alias = "embed"

[security]
auth-failure-limit-per-minute = 60
trusted-proxies = ["127.0.0.1"]

[security.tls-termination]
enabled = true
provider = "envoy-edge"
evidence = "change-record-or-runbook-url"
m-tls = true

[storage]
max-connections = 5

[sse]
enabled = true
heartbeat-seconds = 15
max-stream-seconds = 3600

[log]
format = "json"

[events]
format = "jsonl"
schema-version = 1

[observability]
traces-enabled = true
metrics-enabled = true
logs-enabled = true

[observability.exporter]
endpoint = "https://otel-collector.example/v1/traces"
protocol = "http-protobuf"
timeout-ms = 5000

[data-fabric]
enabled = true
format = "arrow-json"
schema-version = 1

[data-fabric.datasets]
security = true
observability = true
usage = true
user = true
finops = true
models = true
drift = true
audit = true
```

`llmctld` uses `[log].format = "json"` unless the operator overrides it with
`--json-logs`. OTel collector secrets should be referenced through environment
variables in exporter headers, for example `authorization = "env:OTEL_TOKEN"`.
`storage.max-connections` controls the SQLx pool size for SQLite/Postgres.
`runtime.embeddings.mode = "semantic"` is the production native embedding
contract; the alias must identify a loaded semantic embedding model. Use
`mode = "dev-fallback"` only for local development because it emits
deterministic, non-semantic vectors.
`server.graceful-drain-seconds` flips readiness to `draining` before process
shutdown, and the circuit-breaker settings protect internal runtime/provider
boundaries from repeated failing retries.

## Validation

Use these checks before starting a production service:

```bash
llmctl --config /etc/rs-llmctl/config.toml server check
llmctl --config /etc/rs-llmctl/config.toml security check
llmctl --config /etc/rs-llmctl/config.toml security audit-config
llmctl --config /etc/rs-llmctl/config.toml observe plan
llmctl --config /etc/rs-llmctl/config.toml compliance evidence
```
