# Observability And Reporting

`rs-llmctl` records what operators usually need after the fact: who used the
model, which model was routed, whether quota allowed it, how many tokens were
reported, and which request ID ties the evidence together.

## Observability

The config exposes an OTel-oriented exporter plan:

- `observability.exporter.endpoint`
- protocol selection for OTLP-style collectors
- traces, metrics, and logs switches
- environment-backed headers with `env:NAME`

Runtime events include audit rows, usage rows, quota decisions, model inventory,
and resource observations.

The daemon emits RED-style metrics for SLO dashboards and alerts:

- `llmctl_requests_total` with `endpoint`, `model`, `team`, and `status`;
- `llmctl_request_errors_total` for non-OK request outcomes;
- `llmctl_request_latency_ms` as an end-to-end latency histogram;
- `llmctl_upstream_requests_total`, `llmctl_upstream_errors_total`, and
  `llmctl_upstream_latency_ms` for compatibility worker calls;
- `llmctl_admission_rejections_total` for saturated global or scoped admission
  limits;
- `llmctl.tokens.input` and `llmctl.tokens.output` for token throughput.

`llmctl aiops slo-plan --format prometheus` uses these metric names.

The production config also has explicit controls for the local event surface:

- `sse.enabled`, `sse.heartbeat-seconds`, and `sse.max-stream-seconds`;
- `log.format`, normally `json` in production;
- `events.format`, one of `json`, `jsonl`, or `cloud-events`;
- `events.schema-version`;
- `data-fabric.enabled`, `data-fabric.format`, and dataset switches.

Because CRA Article 14 is treated as active, production external-bind configs
must keep traces, metrics, and logs enabled and configure an OTLP exporter
endpoint. `security check` rejects production external bind without that
exporter.

When an OTLP endpoint is configured, the daemon installs trace, metric, and log
providers and bridges `tracing` events into OpenTelemetry logs. Request IDs,
model aliases, quota decisions, usage totals, and lifecycle events are the
correlation keys operators should expect to search by.

Server-side request routing, audit events, quota decisions, usage records, drift
observations, and resource snapshots emit OTel-friendly runtime event names:

- `llmctl.request.routing`
- `llmctl.quota.decision`
- `llmctl.worker.lifecycle`
- `llmctl.resource.snapshot`
- `llmctl.drift.observation`
- `llmctl.model.install.verification`
- `llmctl.runtime.heartbeat`

`llmctl server run` emits `llmctl.runtime.heartbeat` at startup and then every
`runtime.heartbeat-interval-seconds` seconds. Set that value to `0` to disable
the background heartbeat loop.

The HTTP serving path also emits production SLI metrics:

- `llmctl_requests_total`, `llmctl_request_errors_total`, and
  `llmctl_request_latency_ms`;
- `llmctl_upstream_requests_total`, `llmctl_upstream_errors_total`, and
  `llmctl_upstream_latency_ms`;
- `llmctl_upstream_circuit_state_total` and
  `llmctl_upstream_circuit_consecutive_failures`;
- `llmctl_admission_rejections_total` and `llmctl_auth_failures_total`.
- `llmctl_slo_violations_total` for requests that miss the default success or
  latency SLO.

Circuit breakers count transport errors, 5xx responses, and upstream 429
responses as backend health failures. Other client-facing 4xx responses are
not allowed to poison the circuit. Half-open probes are single-flight so a
recovering backend does not receive a burst of simultaneous probes.

During shutdown, readiness flips to `draining` before the configured graceful
drain window elapses, so load balancers can stop sending new traffic while
in-flight requests finish.

Sensitive attributes are redacted before emission. Prompts, messages, bearer
tokens, API keys, passwords, collector authorization headers, and local file
paths are not exported as span/log attributes.

## Reports

Reports focus on:

- data/audit summaries;
- quota/team governance summaries;
- usage totals;
- audit event counts;
- retention windows;
- quota limits;
- team attribution;
- request identifiers;
- model aliases;
- policy status.

Use envelopes when attaching evidence to a change record. The envelope metadata
contains a SHA-256 payload hash that can be verified later:

```bash
llmctl --config /etc/rs-llmctl/config.toml audit retention plan --envelope > retention-plan-envelope.json
llmctl --config /etc/rs-llmctl/config.toml data verify-envelope retention-plan-envelope.json
```

Retention planning is dry-run by default. Retention pruning requires the
explicit `audit retention apply --yes` command.

## Data Fabric Exports

Data contracts and filtered exports are available through `llmctl data`:

```bash
llmctl data contracts
llmctl data export --dataset security --format json
llmctl data export --dataset observability --format jsonl
llmctl data export --dataset finops --format arrow-json
```

Use `docs/data-contracts.md` as the operator reference for schemas and export
formats.
