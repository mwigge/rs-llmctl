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
