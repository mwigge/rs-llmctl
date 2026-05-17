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

## First Run

`llmctl first-run` gives operators a scriptable bootstrap path before the
daemon is started. The default mode is a dry-run JSON plan:

```bash
llmctl --config ./config.toml first-run \
  --secret-output ./operator.secret
```

The plan shows the generated-key action, digest-only config storage,
offline-only starter model recommendation, and both smoke checks: an
`ask_question` helper plan for `rs-llmctl-client` and an OpenAI-compatible
`/v1/chat/completions` request plan.

Apply mode is intentionally explicit:

```bash
llmctl --config ./config.toml first-run --apply \
  --secret-output ./operator.secret \
  --starter-model-path /models/qwen.gguf \
  --starter-model-alias qwen
```

`--apply` writes config, creates local storage paths, enables auth, writes the
raw API key once to `--secret-output`, stores only the SHA-256 digest in config,
and configures the starter model only from the provided local path. It does not
download a model by default.

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

## Native Runtime Validation

Use `runtime validation-plan` before a hardware-backed soak. It is deterministic
and offline, so it can run in CI or on a laptop without model downloads:

```bash
llmctl --config ./config.toml --json runtime validation-plan \
  --soak-minutes 240 \
  --streaming-concurrency 8 \
  --rotation-keys 3 \
  --quota-concurrency 16
```

The JSON plan covers:

- Real artifact smoke-test evidence for Qwen, Gemma, Mistral, and DeepSeek
  safetensors layouts: `*.safetensors`, `tokenizer.json`, and `config.json`.
- Hardware targets for CPU, NVIDIA CUDA, AMD-Vulkan, and Apple Metal. Missing
  accelerators are represented as planned/skipped evidence, not failures.
- Long streaming soak scenarios and scheduler-under-load assertions.
- Graceful drain behavior while streams are active.
- Circuit breaker and heartbeat checks while the scheduler is saturated.
- API-key rotation overlap and quota concurrency assertions.
- Benchmark JSON/JSONL fields for latency, first-token latency, tokens/sec,
  input/output tokens, RSS memory, peak RSS, VRAM, and peak VRAM.

Use `runtime validation-run` on the target host as the executable gate. It
validates cluster placement and all positive-weight native model artifacts,
writes optional evidence, and exits non-zero when a required artifact or model
family contract is missing:

```bash
llmctl --config ./config.toml --json runtime validation-run \
  --evidence-output ./artifacts/native-validation.json
```

When artifacts are absent, the command emits `planned-missing-local-artifact`.
When a configured path is present but incomplete, it emits actionable blocked
evidence from the Candle artifact validator. It does not perform network access
or start serving.

## Package Validation

`packaging/validate-install.sh` is intentionally passive and offline. It does
not download models, install packages, or start services.
