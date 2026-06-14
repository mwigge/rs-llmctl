# Changelog

All notable release-facing changes are recorded here. Keep entries focused on
operator behavior, packaging contents, service lifecycle, and verification.

## 1.3.1 - 2026-06-14

- Native: `gemma4` GGUF models (e.g. `gemma-4-12b-it`) now load through a new
  per-layer-aware quantized model loader and a SentencePiece-metaspace BPE
  tokenizer built from `tokenizer.ggml.model == "gemma4"` GGUF metadata.
- Native: generation prompts for `gemma4` models now use the
  `<start_of_turn>{role}\n...<end_of_turn>\n` chat template (with `system`
  content folded into the first `user` turn), and the configured `<bos>`
  token is prepended to the prompt's `input_ids` when the GGUF metadata
  requests it (`tokenizer.ggml.add_bos_token = true`).

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
