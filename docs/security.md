# Security Model

`rs-llmctl` treats production model serving as a controlled internal service,
not a casual development endpoint.

## Baseline

The production posture is PCI DSS v4.0.1-aligned:

- authentication is required for external bind;
- external bind requires documented TLS termination or mTLS evidence;
- CRA Article 14 is treated as an active control, requiring monthly audit
  reports, audit retention, and an OTel exporter endpoint for production
  external bind;
- API keys are stored as SHA-256 digests only;
- scopes control access to chat, model listing, and admin operations;
- quotas and admission limits protect shared capacity;
- audit events and usage records are written for operational review;
- response metadata is intentionally non-secret;
- production CORS uses approved origins, not wildcard origins.

Run `llmctl compliance pci-dss` when you need an evidence-oriented view for
reviews or monthly control reporting. It does not replace a formal assessor
report, but it keeps the operational evidence in one predictable shape.

## Secrets

Use stdin or environment variables when creating key digests:

```bash
printf '%s' "$LLMCTL_NEW_API_KEY" | llmctl security hash-key --stdin
```

Config must not contain raw API keys, bearer tokens, database passwords, raw
connection secrets, or plaintext collector credentials. Sensitive exporter
headers use `env:NAME` references.

## External Clients

AQE/OpenAI-compatible clients can use the API through
`OPENAI_BASE_URL=http://host:8765/v1`. Safe response headers include request
identifiers, model aliases, quota state, and policy status. They do not include
upstream URLs, prompts, file paths, API keys, or bearer tokens.

Production external bind should normally be published as HTTPS through a load
balancer, ingress, reverse proxy, or service mesh. The runtime validates that
the deployment config documents that control:

```toml
[security.tls-termination]
enabled = true
provider = "envoy-edge"
evidence = "change-record-or-runbook-url"
m-tls = true
```

`llmctl security check`, `llmctl security audit-config`, and
`llmctl compliance evidence` include the same TLS evidence fields so monthly
reviews can prove the control was present at deployment time.
