# Operations Guide

`rs-llmctl` is built around a quiet operating loop: stage, verify, plan, serve,
observe, and report.

## Ordered Deployment Operations

1. Import the offline install manifest after staging the approved bundle under
   `/var/lib/rs-llmctl/models`.
2. Run the dry-run validation gate with `server check`, `server status`, and
   `server plan`.
3. Run the security audit with `security check` and `audit retention plan`.
4. Run readiness checks with `observe plan` and `systemd-analyze verify`.
5. Hand off service activation only after dry-run, security audit, and readiness
   checks pass.
6. Verify AQE/OpenAI client access with
   `OPENAI_BASE_URL=http://host:8765/v1`.
7. Export the audit envelope with `data export --hours 24`.

## Models

Offline manifests keep model delivery repeatable. Relative paths resolve from
the manifest directory, and SHA-256 hashes pin the bytes accepted into the
inventory.

Direct downloads are treated as controlled operations: they require HTTPS,
expected SHA-256, public-address DNS resolution, no redirects, and a bounded
network timeout.

## Runtime Modes

- `single`: serve the configured model directly.
- `cold-swap`: route the requested alias to its worker and stop old workers as
  part of the swap plan.
- `hot-swap`: warm the replacement before draining the active worker.
- `weighted`: route to the highest weighted model.
- `fallback`: prefer the requested model when it has weight, otherwise use the
  preferred weighted model.

The daemon exposes live swap orchestration through the authenticated admin
endpoint:

```bash
curl -sS \
  -H "Authorization: Bearer $LLMCTL_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"active":"qwen-prod","replacement":"qwen-canary","mode":"hot"}' \
  http://127.0.0.1:8765/v1/admin/swap
```

The caller needs the `admin` scope. Hot swap starts and probes the replacement
before draining the active worker. Cold swap drains and stops the active worker
before starting the replacement. Each execution is written to the audit trail
with request ID, planned steps, worker statuses, and success state.

## Resource Budget

The default resource budget is 80%. That applies to CPU/RAM/VRAM planning so
the model service does not assume it owns the whole host.

The packaged Linux systemd unit applies `CPUQuota=80%` and `MemoryMax=80%` as
the default cgroup guard. Generated `server plan` output also includes
host-specific `CPUQuota` and `MemoryMax` properties for reviewed drop-ins.
Detected GPU VRAM remains planning evidence because there is no portable
systemd cgroup property for hard GPU VRAM enforcement.

## Package Validation

`packaging/validate-install.sh` is intentionally passive and offline. It does
not download models, install packages, or start services.
