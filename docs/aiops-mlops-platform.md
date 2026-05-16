# AIOps And MLOps Platform Surface

`rs-llmctl` is not just a model launcher. It is moving toward a small server-side
AIOps/MLOps platform for effective private model operation: serve models,
control access, account for usage, capture evidence, export data, and keep
enough telemetry to operate the service like production infrastructure.

## Delivered Platform Capabilities

- Config wizard profiles for local development, CPU-only hosts, and
  production AIOps deployments.
- OpenAI-compatible serving paths for models and chat completions.
- SSE streaming passthrough with typed config.
- Worker lifecycle planning and supervision for model backends.
- Hot/cold swap planning, weighted routing, fallback behavior, quotas, and
  admission control.
- Offline model import and verified direct-download install paths.
- OTel trace/metric/log configuration with JSON daemon logs.
- Audit, usage, quota, observation, drift, and data export records.
- CRA Article 14 active-control evidence, PCI DSS aligned evidence, release
  integrity checks, and report envelopes.
- Local search and recommendation endpoints for AI-developer workflows.
- Schema-versioned data contracts and domain filtered JSON, JSONL,
  Arrow-schema JSON, native Arrow IPC, and Parquet exports.
- Eval run persistence and model quality baseline reports.
- Lineage records for prompts, corpora, embedding indexes, models, and
  releases.
- Generated SLO plans, alert rule templates, and incident evidence templates.
- HMAC-signed policy bundles and legal-hold retention plans per dataset.

## Current Gaps

Run the machine-readable gap report:

```bash
llmctl aiops gaps
```

The tracked gaps are now narrower:

- Built-in eval runners for golden prompts; today operators record eval scores
  and baselines through `llmctl eval run`.
- Runtime request-to-lineage joins; today lineage is an operator-controlled
  record stream through `llmctl lineage record`.
- Prometheus/Alertmanager and Grafana-specific renderers; today `aiops
  slo-plan` emits portable alert templates.
- Asymmetric signing and transparency log publication for policy bundles;
  today bundles use HMAC-SHA256 with key material supplied through an
  environment variable.

## Practical Next Step

For a small internal platform, the next high-value path is:

1. Add optional built-in eval execution against configured model endpoints.
2. Attach lineage IDs directly to serving, RAG, model import, and release
   events.
3. Add Prometheus/Alertmanager and Grafana renderers.
4. Add Sigstore or minisign support for policy bundles.
5. Publish policy-bundle and incident evidence hashes to an append-only log.
