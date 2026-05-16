# Policy Operations Runbook

Use this checklist for reviewed policy changes after the base server package is
installed and before production traffic is shifted.

## Quota Export And Import

Export quota policy from the source environment:

```bash
llmctl --config /etc/rs-llmctl/config.toml quota export > quotas.json
```

Import the reviewed JSON export into the target environment:

```bash
llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.json
```

For hand-authored policy review, import a TOML policy file with the same quota
fields used in config:

```bash
llmctl --config /etc/rs-llmctl/config.toml quota import ./quotas.toml
```

Quota import rejects policies with blank subjects or teams,
`requests_per_minute`, `tokens_per_day`, or `max_concurrency` values that are
not greater than zero, or `allowed_models` entries that contain empty model
aliases. Fix the policy file and repeat review instead of carrying invalid
limits forward.

Run `llmctl --config /etc/rs-llmctl/config.toml quota list` after import and
attach the output to the change record.

## Policy Signing And Transparency Log

Preserve HMAC bundle verification for environments that already use secret
manager-backed HMAC keys:

```bash
llmctl policy bundle --name platform --input policy.json --output policy-bundle.json --signing-key-env LLMCTL_POLICY_KEY
llmctl policy verify-bundle policy-bundle.json --signing-key-env LLMCTL_POLICY_KEY
```

For asymmetric review, generate an Ed25519 keypair once for the approved
promotion lane, sign the reviewed policy artifact, verify it before rollout,
and append the artifact hash to the local transparency log:

```bash
llmctl policy keygen --private-key policy-ed25519.private.json --public-key policy-ed25519.public.json
llmctl policy sign --input policy.json --signature policy-signature.json --private-key policy-ed25519.private.json
llmctl policy verify --input policy.json --signature policy-signature.json --public-key policy-ed25519.public.json
llmctl policy log append --log policy-transparency.jsonl --artifact policy.json --signature policy-signature.json
llmctl policy log verify --log policy-transparency.jsonl
```

Attach the signature JSON and transparency-log verification output to the
change record. A log entry records the artifact hash, optional signature hash,
previous entry hash, and entry hash; verification must stay valid before any
new entry is appended.

## Add-Key Workflow

Hash the new API key outside the config file:

```bash
llmctl security hash-key "$LLMCTL_NEW_API_KEY"
```

Add only the returned digest to the reviewed config:

```toml
[[security.api_keys]]
id = "ops-admin"
sha256 = "<sha256-from-hash-key>"
subject = "ops-admin"
team = "platform"
scopes = ["admin"]
```

The `sha256` value must come from `security hash-key`. Keep the original key in
the operator secret manager and distribute it through the approved channel.

## Server Plan Export

Capture a server plan export before any production service activation change:

```bash
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.json
```

Review `server-plan.json` for the planned worker count, model aliases, ports,
program path, arguments, and environment before approving the rollout.

For policy-only changes, keep before/after plan artifacts and review the diff
before approving the change record:

```bash
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.before.json
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.after.json
llmctl server plan-diff server-plan.before.json server-plan.after.json
```

## Retention Envelope Review

Capture the retention plan as a verifiable envelope:

```bash
llmctl --config /etc/rs-llmctl/config.toml audit retention plan --envelope > retention-plan-envelope.json
llmctl --config /etc/rs-llmctl/config.toml data verify-envelope retention-plan-envelope.json
```

Attach `retention-plan-envelope.json` and the verification output to the change
record. Review the envelope `metadata sha256`, confirm the verification hashes
match, and confirm the payload keeps `dry_run` set to true and `deletes` set to
false.

## Enterprise Reporting Metadata Review

Attach data/audit summaries and quota/team governance summaries to the same
change record. Data/audit summaries should show usage totals, audit event
counts, retention windows, and envelope hashes. Quota/team governance summaries
should show quota limits, team attribution, model aliases, and policy status.

External client non-secret response metadata may be shared with approved
AQE/OpenAI-compatible clients when it is limited to request identifiers, model
aliases, policy status, quota state, and audit correlation fields. These fields
are safe for AQE/OpenAI-compatible clients because AQE/OpenAI-compatible clients
can consume these summaries without exposing secrets.

## Server Storage And Router Contract Review

For server deployments that use an external database, attach the redacted
Postgres storage plan and migration plan to the change record before connecting
live infrastructure. The storage review must not include database passwords or
raw connection secrets.

Review router maturity controls with the same deployment record: admission and
backpressure limits, upstream timeout budgets, and non-secret failure responses
for saturated or slow model workers. Failure responses should preserve request
correlation without exposing upstream URLs, prompts, file paths, API keys, or
bearer tokens.

Export the AQE/OpenAI-compatible governance contract for approved external
clients:

```bash
llmctl --config /etc/rs-llmctl/config.toml integration aqe-contract
```

Attach the contract output with the quota/team governance summary so external
clients can validate paths, auth scopes, safe response headers, quota fields,
team fields, and model aliases.
