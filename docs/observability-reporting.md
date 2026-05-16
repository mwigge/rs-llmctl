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
