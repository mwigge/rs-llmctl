# Native Operating Modes

rs-llmctl's tool-capable native runtime is designed to work in three distinct
operating modes, each tuned to a different combination of hardware budget,
connectivity, and user role. The same binary, the same Qwen3 model family,
and the same `qwen3-native` tool-call protocol cover all three — only the
surrounding orchestration changes.

The mode the runtime is in is **emergent from configuration**, not an
explicit setting on the rs-llmctl side. The orchestrator (typically the
milliways sommelier) decides how to route prompts; rs-llmctl just serves
whichever request lands on it.

---

## Mode A — Hybrid: cloud planner + local executor

```
                ┌─────────────────────────────────────────────┐
   User prompt  │  Cloud planner (Claude / Codex / MiniMax)   │
   ───────────▶ │   • holds the big-picture repo context      │
                │   • decomposes into focused micro-tasks     │
                │   • dispatches each as a tool call          │
                └─────────────────────────────────────────────┘
                                      │
                  micro-task: "apply  │  small input context
                  this patch to X"    ▼  (~1 k tokens)
                ┌─────────────────────────────────────────────┐
                │  Local executor (rs-llmctl + Qwen3 14B)     │
                │   • bash, edit, glob, write, read           │
                │   • tool turns at ~19 tok/s on Metal        │
                │   • costs $0 — keeps token budget for plans │
                └─────────────────────────────────────────────┘
                                      │
                          result      ▼
                ┌─────────────────────────────────────────────┐
                │  Cloud planner reviews + dispatches next    │
                └─────────────────────────────────────────────┘
```

**When this mode wins**

- The user has API budget for the cloud planner but wants to keep the
  per-task cost low.
- Tasks span a repo too large for the local model's KV cache to hold
  end-to-end.
- The local box is mid-tier (8–14 GB VRAM) — strong enough to execute
  reliably but not strong enough to plan strategically.

**Setup**

- Cloud kitchens registered in milliways' `internal/maitre/config.go`
  default carte (`claude`, `codex`, `minimax`).
- Local kitchen `local-qwen` registered pointing at
  `http://127.0.0.1:8765/v1` (rs-llmctl's default bind).
- Sommelier routing keywords:
    - `think` / `plan` / `explore` / `review` / `sign-off` → `claude`
    - `explain` / `design` / `spec` → `codex`
    - `reason` / `analyze` / `write` → `minimax`
    - `code` / `edit` / `bash` / `glob` / `write-file` / `read` / `test`
      / `implement` / `fix` / `build` → `local-qwen`
- `Routing.BudgetFallback = "local-qwen"` — cloud-budget exhaustion
  falls back onto the free worker rather than failing the request.

**Quality bar for the local executor**

The cloud planner carries the cognitive load, so the local model only
needs to be a reliable tool caller:

- Emit parseable `<tool_call>...</tool_call>` JSON consistently.
- Not hallucinate file paths.
- Respect diff format when applying patches.
- Know when to give up and escalate.

Qwen3 14B Q4_K_M satisfies all four. Qwen3-Coder-30B-A3B (MoE) is a
quality upgrade when the local box has 16+ GB unified memory.

---

## Mode B — Offline: local-only with graceful degradation

```
                ┌─────────────────────────────────────────────┐
   User prompt  │  rs-llmctl + Qwen3 (whatever fits the box)  │
   ───────────▶ │   • plans AND executes                      │
                │   • no cloud roundtrips                     │
                │   • context capped by the local KV budget   │
                └─────────────────────────────────────────────┘
```

**When this mode wins**

- No internet (plane, train, ship, restricted environment).
- Sensitive code that must not leave the box.
- Air-gapped corporate deployments.
- Cost-zero student / hobbyist use.

**Setup**

- Only the `local-qwen` kitchen is enabled (or the milliways sommelier
  is started with `MILLIWAYS_LOCAL_ONLY=1`).
- All routing keywords funnel into the local kitchen regardless of the
  task category.
- The KV cache budget caps the effective context window per tier:
    - Tier 1 (6 GB VRAM, INT8 KV cache): ~32 k tokens
    - Tier 2 (12 GB VRAM, FP16 KV cache): ~128 k tokens
    - Tier 3 (16+ GB or unified): ~128 k tokens
- User behaviour shifts: narrow each prompt to a single file or a
  bounded subdirectory rather than expecting whole-repo reasoning.

**Quality bar for the local model**

Higher than Mode A — the local model is now solo. Qwen3 14B is the
floor for "feels like Claude Code"; Qwen3 8B works for everyday edits;
Qwen3 4B works for narrow tasks (single file, single function).

---

## Mode C — Educational: verbose-observability, no cloud

```
                ┌─────────────────────────────────────────────┐
   User prompt  │  rs-llmctl + Qwen3 with all OTel spans on   │
   ───────────▶ │   • live tool-call inspection               │
                │   • token-by-token sampling viz             │
                │   • "this would have cost $X with Claude"   │
                │   • replay mode for saved sessions          │
                └─────────────────────────────────────────────┘
```

**When this mode wins**

- Teaching AI engineering / agentic coding to students.
- Demoing the tool-call protocol to a stakeholder.
- Security research that needs reproducible model behaviour without an
  API key leaking the prompt.
- Self-learning — understanding what the model is actually doing.

**Setup**

- Identical to Mode B for the inference path.
- OTel exporter pointed at a local collector (Jaeger / Tempo / SigNoz)
  so spans are visible in real-time rather than batched to a cloud
  backend.
- The three native runtime spans (`native.model.load`,
  `native.model.prefill`, `native.model.generation`) and their
  histograms (`native.model.load.duration_ms`,
  `native.model.tokens_per_second`, `native.model.peak_resident_mb`)
  surface every model interaction.
- The `/v1/models` capability advertisement (see
  `docs/openapi.yaml`) shows the runtime's self-reported tier,
  GPU backend, tool protocol, and context window for the configured
  model — useful to compare what the model claims vs how it behaves.

**Why this mode matters**

It maps directly onto consumer-laptop hardware (the 6 GB tier is a
realistic student/teacher budget) and produces zero API cost while
giving a complete view of the orchestration loop. The educational
audience is a real one — it's the lowest-friction onboarding path into
the agentic-coding pattern.

---

## Mode is not a config flag

There is no `mode = "hybrid" | "offline" | "educational"` setting in
`rs-llmctl.toml`. Mode is determined by:

1. Which orchestrator is in front of rs-llmctl (cloud-aware sommelier
   vs local-only sommelier vs none).
2. Whether the OTel exporter is pointed at a local or cloud collector.
3. Which model is loaded into rs-llmctl (the tier matrix in
   `docs/native-gguf-internals.md` covers the model selection).

This is intentional: the runtime stays oblivious to its operating
context, and the orchestrator decides the policy. If a future
deployment needs an explicit mode hint, it should be added to the
sommelier's configuration rather than to rs-llmctl's.

---

## Related

- `docs/native-gguf-internals.md` — model details and the per-tier
  recommendation matrix.
- `docs/observability-reporting.md` — the OTel instruments referenced
  by Mode C.
- `docs/openapi.yaml` `/v1/models` schema — the capability metadata
  external orchestrators consume.
