# Configuration

`rs-llmctl` keeps production behavior in one typed TOML config. The goal is
that an operator can see how serving, security, streaming, logs, telemetry,
events, and data exports are intended to behave before the daemon starts.

## Wizard Profiles

Create a local development config:

```bash
llmctl init --profile local-dev
```

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

## Relevant Blocks

```toml
[server]
upstream-timeout-seconds = 300
graceful-drain-seconds = 5
circuit-breaker-failures = 3
circuit-breaker-reset-seconds = 30

[security]
auth-failure-limit-per-minute = 60

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
`server.graceful-drain-seconds` flips readiness to `draining` before process
shutdown, and the circuit-breaker settings protect compatibility upstreams from
repeated failing retries.

## Validation

Use these checks before starting a production service:

```bash
llmctl --config /etc/rs-llmctl/config.toml server check
llmctl --config /etc/rs-llmctl/config.toml security check
llmctl --config /etc/rs-llmctl/config.toml security audit-config
llmctl --config /etc/rs-llmctl/config.toml observe plan
llmctl --config /etc/rs-llmctl/config.toml compliance evidence
```
