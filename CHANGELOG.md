# Changelog

All notable release-facing changes are recorded here. Keep entries focused on
operator behavior, packaging contents, service lifecycle, and verification.

## 1.6.5 - 2026-06-18

- Native generation: new `mistral-rs` cargo feature wires the `mistralrs`
  0.8.1 crate as an opt-in second inference backend. The feature is off
  by default; when enabled together with `gpu-metal` or `gpu-cuda`, the
  matching mistralrs backend feature propagates automatically via
  `mistralrs?/metal` / `mistralrs?/cuda`. Stage-1 spike confirmed the
  backend loads Devstral 24B's GGUF in 26 s — handling the non-canonical
  head_dim that breaks candle 0.10.2's `quantized_llama`. macOS Metal
  runs require full Xcode (mistralrs-paged-attn invokes `xcrun metal`
  in build.rs); Linux NVIDIA / AMD ROCm users build cleanly with just
  the standard CUDA toolchain.
- Native generation: Mistral-family GGUF support added through candle's
  existing `quantized_llama` path. `CandleModelFamily::Mistral` GGUF
  loading is now wired (previously bailed); the `RealCandleModel`
  enum gains a `MistralGguf(quantized_llama::ModelWeights)` variant
  and `format_native_chat_prompt(Mistral, _)` emits the classic
  `<s>[INST] ... [/INST]` template. Works for Llama-arch GGUFs that
  follow canonical head_dim (Mistral 7B v0.3, Llama 3.1 8B Instruct,
  CodeLlama 13B). Non-canonical Mistrals (Devstral 24B, Mistral Small
  3.1 24B) fail at the first forward pass — documented under "Variant
  decision log" in `docs/native-gguf-internals.md`.
- Native generation: new `"mistral-instruct"` `tool_protocol`
  identifier returned by `CandleModelFamily::Mistral::tool_protocol()`
  and advertised under `/v1/models` `capabilities.tool_protocol`. The
  `tool_protocol` enum in `docs/openapi.yaml` extended accordingly.
- Tier detection: AMD VRAM is now probed via Linux DRM sysfs
  (`/sys/class/drm/cardN/device/mem_info_vram_total`) with ROCm SMI
  fallback, alongside the existing NVIDIA path. Probe-failure default
  remains `Tier2Nv12` (advisory only).
- Documentation:
  - `docs/native-gguf-internals.md` — new "Variant decision log" section
    explains why Gemma 4 E4B is no longer the daily driver, why
    Qwen3-Coder MoE isn't the Mac premium, why Devstral 24B isn't the
    Mac premium, and lists the net per-tier deployment recommendation
    pointing macOS premium users at `llama-server` (Homebrew install,
    Metal works without Xcode).
  - `docs/native-operating-modes.md` — Mode A (hybrid cloud-planner +
    local-executor), Mode B (offline-only), Mode C (educational
    verbose-observability), with the orchestrator setup recipes per
    mode.
  - `docs/blog-qwen3-learns-to-trace-itself.md` — narrative blog post
    on the Qwen3 14B local Metal setup and the agentic
    chaostooling-otel demo where the model read three real OTel
    instrumentation patterns and rewrote its own counter program with
    tracing on each iteration.
  - `examples/qwen3-tier1.toml` and `examples/qwen3-tier3.toml` —
    opinionated single-binary serve configs for the 6 GB NVIDIA tier
    and the 16-18 GB Mac / discrete-GPU tier.

## 1.3.1 - 2026-06-14

- Native: `gemma4` GGUF models (e.g. `gemma-4-12b-it`) now load through a new
  per-layer-aware quantized model loader and a SentencePiece-metaspace BPE
  tokenizer built from `tokenizer.ggml.model == "gemma4"` GGUF metadata.
- Native: generation prompts for `gemma4` models now use the
  `<start_of_turn>{role}\n...<end_of_turn>\n` chat template (with `system`
  content folded into the first `user` turn), and the configured `<bos>`
  token is prepended to the prompt's `input_ids` when the GGUF metadata
  requests it (`tokenizer.ggml.add_bos_token = true`).

## 1.6.4 - 2026-06-17

- Native generation: Qwen3 14B Q4_K_M is the new daily-driver tool-capable
  model, replacing Gemma 4 E4B as the recommended Tier 3 model for
  16-18 GB-usable hardware (Apple M-series 24 GB unified, AMD 16 GB VRAM).
  Uses candle's existing `quantized_qwen3` path — no new architecture
  wiring required.
- A new `qwen3_runtime_python_counting_program` integration test exercises
  both forward (`range(1, 11)`) and reverse (`range(10, 0, -1)`) iteration
  on Metal to verify the model understands prompt direction rather than
  memorising a single canned example.
- Measured M1 baseline on Apple M-series with Metal (recorded as regression
  baseline in `openspec/changes/add-tool-capable-tiered-runtime/tasks.md`):
  - Model load: 7.3 s (no PLE dequant; contrast Gemma 4 E4B at 74 s)
  - Prefill (~30 tokens): 53-236 ms
  - Generation: ~19 tok/s
  - Working-set / swap delta during full test: under 2 GB growth
- This milestone (M1 of `add-tool-capable-tiered-runtime`) ships Qwen3 as
  the primary native runtime. Gemma 4 E4B remains supported for users who
  explicitly select it. Tier 1 (NV6) and Tier 2 (NV12) deployments,
  Qwen3-Coder MoE wiring, capability advertisement on /v1/models, and
  full observability spans are planned in milestones M2-M6.

## 1.6.3 - 2026-06-17

- Native generation: GPU acceleration is now available via two opt-in cargo
  features:
  - `gpu-metal`: Apple Metal on macOS (Apple Silicon).
  - `gpu-cuda`: NVIDIA CUDA on Linux/Windows. Also covers AMD GPUs on Linux
    when built against ROCm/HIP's CUDA shim (`HIP_PLATFORM=amd`).
  At runtime, a new `best_device()` helper probes the compiled-in backends
  in order (Metal → CUDA → CPU) and picks the first one that initialises.
  Build commands:
  ```
  cargo build --release --features native-candle,native-tokenizers,gpu-metal   # macOS
  cargo build --release --features native-candle,native-tokenizers,gpu-cuda    # NVIDIA / AMD ROCm
  cargo build --release --features native-candle,native-tokenizers             # CPU only
  ```
  The default build remains CPU-only so the release binary stays portable.
  Measured Metal speedup on Gemma 4 E4B (M-series 24 GB): prefill of 13 tokens
  drops from 118 s on CPU to 115 ms on Metal (~1000×). Model load stays ~75 s
  because the ~10.7 GB PLE table dequantisation is CPU-bound.
  See `docs/native-gguf-internals.md` for the per-VRAM-tier model
  recommendation matrix (6 GB / 12 GB / 24 GB targets) and the known
  limitation that prevents `dequantize_f16` from being used to halve PLE
  memory on Metal (Q4_K_M→F32→F16 cast collapses argmax onto punctuation).
- Native generation: Gemma4 GGUF forward pass is now functional. A new
  `src/gemma4_gguf.rs` module (gated behind `native-candle` +
  `native-tokenizers`) re-implements the Gemma4 transformer in Candle with
  every Gemma4-specific feature that `quantized_gemma3` lacks:
  - per-layer variable head_dim (256 for SWA layers, 512 for global) derived
    from the actual `attn_q` weight shape
  - cross-layer KV sharing (`shared_kv_layers = 18`): the last 18 layers
    skip K/V projection and reuse the cache from layer 22 (SWA source) or
    layer 23 (Global source)
  - Per-Layer Embedding (PLE) — `[262144, 10752]` lookup table dequantised to
    F32 (~10.7 GB) at load time, projected through `1/sqrt(embedding_length)`
    and combined with the input embedding scaled by `1/sqrt(2)` and
    `sqrt(per_layer_dim)`
  - per-layer `layer_output_scale` scalar applied to the complete layer
    output (post-PLE, post-residual)
  - `final_logit_softcapping = 30` applied after the LM head
  - V tensor RMS-normalised without learnable weights before attention
  - attention scaling = 1.0 (q_norm absorbs the `1/sqrt(head_dim)` factor)
- Native generation: Gemma4 GGUF tokenizer (introduced in 1.6.2) is now
  verified end-to-end with the new forward pass. The
  `gemma4_gguf_forward_pass_produces_coherent_tokens` integration test
  loads the E4B Q4_K_M model and asserts that prompting with "Say hello world"
  produces decoded output containing ASCII alphabetic content.

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
