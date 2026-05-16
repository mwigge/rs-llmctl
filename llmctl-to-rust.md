# llmctl to Rust Plan

`rs-llmctl` is a full Rust rewrite of `llmctl` for enterprise-grade server-side
model delivery. There is no Go compatibility layer.

## Product Split

- Milliways: developer-machine client and workflow UX.
- rs-llmctl: production model-serving platform.

## Required Capabilities

- OpenAI-compatible `/v1/models` and `/v1/chat/completions`.
- CPU-only serving.
- NVIDIA/AMD GPU-aware resource detection and 80% resource budgeting.
- Hot and cold model swap modes.
- Offline local model registration, direct URL download, and bundle-oriented lifecycle.
- External clients such as Agentic QE through `OPENAI_BASE_URL=http://host:8765/v1`.
- API key auth, scopes, quotas, audit trails, and reporting.
- OTel-compatible operations and AI observability.
- PCI DSS v4.0.1-aligned controls as the baseline posture.

## Implementation Notes

The first implementation provides the Rust workspace, daemon, CLI, SQLite
storage, audit/usage/observation persistence, quota checks, resource snapshots,
and an OpenAI-compatible proxy to supervised `llama-server` workers. The system
is intentionally structured so enterprise hardening can deepen without changing
the public CLI/API contract.

## CLI Contract

The Rust CLI keeps the operational command groups stable while the library
internals mature:

- `init`
- `server run`, `server check`, `server security-check`
- `model install`, `model list`
- `quota set`, `quota list`
- `observe snapshot`, `observe drift`, `observe usage`, `observe show`
- `audit report monthly`, `audit report request`, `audit request`
- `usage report`

`llmctld` performs the daemon startup sequence expected by production
deployments: load config, validate production/external-bind security, initialize
storage, and serve the API.
