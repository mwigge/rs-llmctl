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

Capture a server plan export before the service is started or restarted:

```bash
llmctld --config /etc/rs-llmctl/config.toml --dry-run > server-plan.json
```

Review `server-plan.json` for the planned worker count, model aliases, ports,
program path, arguments, and environment before approving the rollout.
