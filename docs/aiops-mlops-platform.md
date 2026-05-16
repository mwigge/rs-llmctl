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
- Eval run persistence, model quality baseline reports, and manifest-driven
  golden prompt execution against OpenAI-compatible endpoints.
- Lineage records for prompts, corpora, embedding indexes, models, and
  releases, plus runtime request-to-lineage joins for chat, local search, and
  local recommendations.
- Generated SLO plans, Prometheus/Alertmanager alert rules, Grafana
  dashboards, and incident evidence templates.
- HMAC-signed policy bundles, Ed25519 signatures, hash-chained transparency log
  entries, and legal-hold retention plans per dataset.

## Remaining Extensions

Run the machine-readable gap report:

```bash
llmctl aiops gaps
```

The remaining work is extension-level, not a blocker for a small internal
deployment:

- Advanced eval judges and rubric scoring. Golden prompts already run through
  `llmctl eval run-suite`; judge models and custom rubrics are the next layer.
- Automatic lineage inference from managed model manifests and RAG indexes.
  Runtime joins are recorded when clients provide lineage IDs.
- Deployment sync for SLO artifacts; today `aiops slo-plan --format
  prometheus` and `--format grafana` render files for operator-managed
  Prometheus and Grafana provisioning.
- External Sigstore/Rekor publication. Ed25519 signatures and local
  transparency logs are available now; public transparency-log publication is
  optional organization-level integration.

## Practical Next Step

For a small internal platform, the next high-value path is:

1. Add optional LLM-as-judge and rubric evaluators for eval suites.
2. Infer lineage IDs from managed model manifests and RAG indexes.
3. Add optional apply/push helpers for Prometheus rule files and Grafana
   dashboard provisioning.
4. Add optional managed transparency-log publication for policy-bundle and
   incident evidence hashes.
5. Add Sigstore integration when external keyless signing is required.

## Junior SRE Mental Model

- **Lineage** answers “what did this response depend on?” Use request headers
  or body metadata to attach prompt, corpus, embedding index, model, and release
  IDs to a request.
- **Dashboard** answers “is the service healthy?” Render Prometheus rules and a
  Grafana dashboard from `llmctl aiops slo-plan`.
- **Policy signing** answers “who approved this config?” HMAC is useful for
  environment-key workflows; Ed25519 gives reviewed keypairs and signatures.
- **Transparency log** answers “was this artifact changed later?” Each log entry
  includes the previous hash, creating a local hash chain.
- **Sigstore/Rekor** are external supply-chain tools for public or shared
  transparency evidence. They are not required for a private first deployment,
  but they are the natural next integration for enterprise release governance.
