# Security Model

`rs-llmctl` treats production model serving as a controlled internal service,
not a casual development endpoint.

## Baseline

The production posture is PCI DSS v4.0.1-aligned:

- authentication is required for external bind;
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
