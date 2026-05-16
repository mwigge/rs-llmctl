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
- Schema-versioned data contracts and domain filtered exports.

## Current Gaps

Run the machine-readable gap report:

```bash
llmctl aiops gaps
```

The tracked gaps are:

- Native Arrow IPC and Parquet writers for the data fabric.
- First-class model eval suites, golden prompts, and baseline comparison
  history.
- Stronger lineage for prompts, RAG corpora, embeddings, model manifests, and
  releases.
- Generated SLOs, alert rules, and incident evidence envelopes.
- Signed policy-as-code bundles and legal-hold retention scopes per dataset.

## Practical Next Step

For a small internal platform, the next high-value path is:

1. Add native Arrow/Parquet export behind an optional feature.
2. Add `llmctl eval run/list/report` and store model quality baselines.
3. Add lineage records for model, prompt, corpus, and release provenance.
4. Generate SLO and alert templates from config.
5. Add signed policy bundles that gate production rollout.
