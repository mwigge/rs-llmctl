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

Generate keys with the CLI when you want rs-llmctl to mint strong random
client material:

```bash
llmctl security generate-key --prefix llmctl-prod --output ./platform-chat.secret
llmctl --config /etc/rs-llmctl/config.toml security add-key \
  --id platform-chat-2026-q2 \
  --sha256 <sha256-from-generate-key> \
  --subject alice \
  --team platform \
  --scope chat \
  --owner platform-sre \
  --purpose chat-api \
  --last-four <last-four-from-generate-key>
```

With `--output`, the raw secret is written once to a new file with `0600`
permissions on Unix and is not printed to stdout. Move that raw value into the
client secret store and keep only the digest plus non-secret metadata in config.
Suitable raw-secret stores include a password manager, Vault, SOPS, 1Password,
a systemd credential, Kubernetes Secret, or your platform secret manager. Use
stdin or environment variables when an external secret store generates the key
material:

```bash
printf '%s' "$LLMCTL_NEW_API_KEY" | llmctl security hash-key --stdin
```

Config must not contain raw API keys, bearer tokens, database passwords, raw
connection secrets, or plaintext collector credentials. Sensitive exporter
headers use `env:NAME` references.

Track the non-secret inventory and rotation state through the config-backed
commands:

```bash
llmctl --config /etc/rs-llmctl/config.toml security list-keys
llmctl --config /etc/rs-llmctl/config.toml security rotate-key \
  --id platform-chat-2026-q2 \
  --new-id platform-chat-2026-q3 \
  --sha256 <new-sha256> \
  --last-four <new-last-four> \
  --reason "quarterly rotation"
llmctl --config /etc/rs-llmctl/config.toml security revoke-key \
  --id platform-chat-2026-q2 \
  --reason "clients moved to q3 key"
```

Overlap rotation marks the old key `retiring` and adds the new key as `active`.
Revocation removes the key from active config and writes a revocation audit
event. Both operations require a service restart so the running daemon reloads
its in-memory key set. Emergency in-place replacement is available with
`rotate-key --replace`, but overlap rotation is easier to audit.

## API Key Usage Audit

Each successful authentication attaches the configured key ID and non-secret
metadata to the request principal. Audit rows then include `api_key_id` in
`detail_json` alongside the actor, team, action, model/resource, outcome, and
request ID. Review usage by key without exposing SHA-256 digests or raw keys:

```bash
llmctl --config /etc/rs-llmctl/config.toml security key-usage --hours 24
llmctl --config /etc/rs-llmctl/config.toml security key-usage --id platform-chat-2026-q2 --hours 168
```

Use this report during rotation windows to confirm traffic has moved off an old
key before revoking it. The report joins audit rows to usage rows by request ID,
so it includes request counts, token totals, latency totals, models, actors,
teams, and lifecycle events.

## External Clients

AQE/OpenAI-compatible clients can use the API through
`OPENAI_BASE_URL=http://host:8765/v1`. Safe response headers include request
identifiers, model aliases, quota state, and policy status. They do not include
upstream URLs, prompts, file paths, API keys, or bearer tokens.

Production external bind can be published as HTTPS through a load balancer,
ingress, reverse proxy, or service mesh. The runtime validates that the
deployment config documents that control:

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

The `llmctl` binary uses Rustls-backed clients for outbound HTTPS model
downloads, compatibility upstream calls, OTel export, and Postgres TLS. It can
also serve inbound HTTPS directly with native Rustls:

```toml
[server.tls]
enabled = true
cert-path = "/etc/rs-llmctl/tls/server.crt"
key-path = "/etc/rs-llmctl/tls/server.key"
require-client-cert = false
```

Native TLS currently supports server certificates only. Keep mTLS
client-certificate validation at the platform edge or service mesh. Either way,
external bind still requires auth, reviewed CORS origins for browser clients,
and non-secret response metadata only.
