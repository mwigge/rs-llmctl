# Changelog

All notable release-facing changes are recorded here. Keep entries focused on
operator behavior, packaging contents, service lifecycle, and verification.

## 1.6.2 - 2026-06-16

- Native generation: Gemma4 GGUF tokenizer now loads successfully. Previously
  the engine failed at startup with `unsupported tokenizer model 'gemma4'`
  because Candle's built-in GGUF tokenizer only handles GPT-2 style BPE.
  Gemma4 GGUF stores BPE merges like GPT-2 but uses Metaspace (▁) whitespace
  escaping instead of byte-level encoding. When Candle rejects the tokenizer
  model identifier, the engine now falls back to building the tokenizer
  directly from the GGUF vocab and merges with Metaspace as both pre-tokenizer
  and decoder. This matches the fix merged into llama.cpp
  (ggml-org/llama.cpp#21343). No configuration changes required.

  **Known limitation**: Gemma4 GGUF inference (forward pass) is not yet
  functional with Candle 0.10.2. The `quantized_gemma3` model uses a single
  `attention.key_length` for all layers, but Gemma4 has per-layer variable
  head dimensions — global attention layers use head_dim=512 and sliding-window
  attention layers use head_dim=256 (`attention.key_length_swa`). Requests to
  a Gemma4 GGUF model will return a 503 error. Safetensors Gemma4 via the
  non-quantized path is unaffected.

## 1.5.0 - 2026-06-11

- Storage: Postgres deployments now work end-to-end. Query placeholders are
  rewritten from `?` to `$1, $2, ...` for the Postgres dialect, and id/time/
  JSON columns use `TEXT` on both dialects to match the values the
  application binds (previously Postgres used native `UUID`/`TIMESTAMPTZ`/
  `JSONB`, which rejected the application's string bindings). Schema patches
  (`observation_events.request_id`) are now applied on Postgres as well as
  SQLite, via `ADD COLUMN IF NOT EXISTS`.
- Quota admission locking is now per-scope (per team, or per subject for
  teamless principals) instead of one global lock, so unrelated
  teams/subjects no longer serialize on the same admission check. The
  Postgres advisory lock key is now derived per scope. Lock acquisition and
  release are wrapped in a single `with_quota_admission` helper with a
  `Drop`-based safety net and a metric for any release that doesn't go
  through the normal path.
- `chat.completions` and `embeddings` request handling: the repeated
  audit-then-error-response blocks and the authentication/scope-check
  prologue are now shared helpers (`audit_reject`, `audit_reject_response`,
  `authenticate_with_chat_scope`), removing several hundred lines of
  duplicated control flow with no behavior change.
- `llmctl model install` no longer panics if a download plan is missing its
  expected SHA-256; it now returns an error refusing the unverified
  download.
- Trusted-proxy CIDR matching now normalizes IPv4-mapped IPv6 addresses
  (`::ffff:10.0.0.1`) to their IPv4 form before comparing against configured
  trusted proxies/networks, so dual-stack listeners match IPv4 trusted-proxy
  entries correctly.
- Repo hygiene: removed committed build artifacts (`dist/`, including the
  compiled `llmctl` binary and tarball) from version control; `dist/` is now
  gitignored and produced only by the packaging scripts/CI.

## 1.3.0 - 2026-06-07

- Guardrails: regex/phrase-based PII redaction and prompt-injection
  ("jailbreak") phrase filtering on inbound chat messages, configurable per
  category with flag/redact/block actions and audit-trail findings
  (`[guardrails.pii]`, `[guardrails.jailbreak]`).
- Web playground: a static, build-step-free HTML+JS chat page served at
  `/playground` with a model picker against `/v1/models`, chat against
  `/v1/chat/completions`, and a bearer API-key field.
- Observability: usage spans now carry GenAI semantic-convention attributes
  (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`,
  `gen_ai.response.model`) alongside the existing `gen_ai.usage.*` and
  `llmctl.*` attributes.
- Observability: Langfuse integration derives an OTLP/HTTP exporter (endpoint
  and HTTP Basic auth header) from project keys (`[observability.langfuse]`)
  when no explicit exporter endpoint is configured.
- Observability: fire-and-forget webhook/callback exporter
  (`[observability.webhook]`) delivers usage/lineage metadata to an HTTP
  endpoint after every completion, for ecosystems that consume callbacks
  rather than OTLP.

## 1.2.1 - 2026-05-17

- Release smoke now installs the packaged archive into a temporary prefix,
  runs first-run with one generated API key and one real local model artifact,
  starts the installed server, and proves the Rust client can complete a chat
  request before reporting release readiness.
- Native GGUF generation now selects the last-token logits without reshaping
  scalar argmax output, fixing the one-token path exercised by the release
  smoke.

## 1.2.0 - 2026-05-17

- Native-first packaging: the default archive publishes the single `llmctl`
  runtime binary, README, changelog, license, and `llmctld.service` systemd
  template.
- Stable service name: Linux installs continue to use `llmctld.service` for
  runbooks and monitoring, while `ExecStart` runs `llmctl --config
  /etc/rs-llmctl/config.toml server run`.
- Install validation: `packaging/validate-install.sh` remains passive and offline,
  checking the installed binary, config, state/log directories,
  service unit, CLI readiness commands, and `systemd-analyze verify` when
  available.
- Release integrity: `packaging/generate-checksums.sh` writes
  `dist/rs-llmctl-<os>-<arch>.tar.gz` and `dist/SHA256SUMS`; use
  `packaging/sign-release.sh dist` for optional `cosign` or `minisign`
  signatures.
- Release publishing: tagged CI runs now publish GitHub Release assets and run
  a self-hosted native smoke job against the packaged tarball.
- Installer hardening: systemd installs leave the daemon stopped until
  `first-run --apply`; monthly audit timers are generated with the selected
  install paths and are opt-in with `LLMCTL_ENABLE_AUDIT_TIMER=1`.
- Native routing hardening: non-loopback bind addresses trigger production
  security gates, external provider routing fails closed for the native-only
  release, readiness reports only active locally routed models, and cluster
  role placement is honored by the serving router.
