# Teaching a Local Model to Trace Itself

There's a moment in this experiment that's worth describing first, because
everything else is just the path that led to it.

A 14-billion-parameter language model running entirely on a single Mac, no
internet involved, no cloud API key, no Python in the loop. We gave it three
real OpenTelemetry instrumentation patterns copied verbatim from a working
`chaostooling-otel` repository it had never seen, plus a small counter program
the same model had written earlier in the session. We asked: "Pick the right
pattern, explain why, and rewrite the program."

It thought for a while — visibly, since we left thinking-mode on. Then it
said:

> Pattern 1 (decorators) doesn't fit — the decorator example requires a
> function with parameters, which the loop isn't.
>
> Pattern 3 (database helpers) doesn't fit — the user's code isn't
> interacting with a database.
>
> Pattern 2 (manual span creation) is the right choice. Each iteration needs
> its own span, and manual creation gives that control.

And it produced this:

```python
from chaosotel import ensure_initialized, get_tracer, flush

ensure_initialized()
tracer = get_tracer()

for i in range(1, 11):
    with tracer.start_as_current_span(f"loop_iteration_{i}") as span:
        span.set_attribute("iteration_number", i)
        print(i)

flush()
```

Look at what's there. The exact `ensure_initialized` / `get_tracer` / `flush`
API surface from the chaostooling-otel README. The right Python context-manager
shape for a span. A sensibly-named per-iteration span (`loop_iteration_{i}`)
that nobody asked it to add. An `iteration_number` attribute on the span —
also not in the prompt. The `flush()` call at the end so spans actually
reach a backend.

That's not autocomplete. That's reading, reasoning, and applying.

Total wall time: 76 seconds, including a 7-second model load. Cost to the
operator: zero.

The rest of this post is about how we got there.

## The Daily Driver

The local model is Qwen3 14B Instruct quantised to Q4_K_M (about 9 GB of
GGUF on disk). It runs on Apple Silicon Metal via [candle][candle], which
rs-llmctl already integrates. The startup looks like this:

```
$ cargo build --release --features native-candle,native-tokenizers,gpu-metal
$ ~/.local/bin/llmctl serve --config examples/qwen3-tier3.toml
INFO detected hardware tier=tier3-mac family=qwen3 params_b=14
     recommended_quant=Q4_K_M context_window=131072
INFO using GPU for Candle inference backend=metal
INFO loading Qwen3 14B GGUF (Q4_K_M, 8.9 GB)
INFO Pipeline input modalities are [📝 Text]
INFO native model load completed   load_ms=7302  model.family=qwen3
                                   gpu.backend=metal
```

Seven seconds. For a 14-billion-parameter model. On a Mac.

The integration test that exercises this path lives in
`src/native.rs::qwen3_runtime_python_counting_program`. It loads the GGUF,
asks the model to write a Python program that counts from 1 to 10, extracts
the fenced code block from the response, writes it to `/tmp/rs_llmctl_count_forward.py`,
and runs it with the real `python3` interpreter. Then it does the same with
a reverse-counting prompt to confirm the model actually understands what
"from 10 down to 1" means.

The output the model produced for the forward prompt — a `.py` file you
could run yourself:

```python
for i in range(1, 11):
    print(i)
```

And the reverse:

```python
# Countdown from 10 to 1
for i in range(10, 0, -1):
    print(i)
```

Both files execute cleanly in a real Python interpreter. The forward one
prints `1, 2, 3, ..., 10`; the reverse prints `10, 9, 8, ..., 1`.

Neither is impressive on its own. What matters is that the test
*asserts* the stdout sequence — the model isn't just generating
Python-shaped text, it's generating Python that does the thing the prompt
asked. That's the bar before any of the more interesting tests are
meaningful.

## Why It's Qwen3 14B and Not Something Else

This wasn't the first model we tried. The proposal that became this blog
post started life targeting Gemma 4 E4B, vendoring the Per-Layer Embedding,
shared KV layers, layer output scale, and final logit softcap into a Rust
module in `src/gemma4_gguf.rs`. That work shipped — Gemma 4 E4B GGUF
forward pass produces coherent text on Metal, and the vendored loader stays
in the codebase for anyone who wants it.

What Gemma 4 lost on was the operational envelope. The PLE tensor
dequantises to roughly 10.7 GB of F32 at load time on a 24 GB Mac, leaving
6-8 GB for the OS. Repeated test runs in the same session trigger swap
thrashing. Cold start is around 74 seconds. We tried the obvious
optimisation — F16 PLE — and it broke on Metal: the F32-to-F16 cast in
candle 0.10.2 loses enough fidelity in the per-layer embedding magnitudes
that the model's argmax collapses onto punctuation tokens. It says `()` instead
of `Hello`.

Qwen3 14B has no PLE table. It loads in 7 seconds, holds a steady ~11 GB
working set, and ships the same `qwen3-native` tool-call protocol that the
larger Qwen3-Coder MoE uses on Linux. For a Mac daily driver doing
agentic dev work, it's the right shape.

We documented every alternative we tried and rejected in
[`docs/native-gguf-internals.md`][internals] under the "Variant decision
log" section. The short version:

- **Gemma 4 E4B** — works, but the PLE memory profile makes it a poor
  default on shared-memory Macs.
- **Gemma 4 E2B** — different architecture (AltUp + Laurel + per-layer
  activation sparsity); would have required another full vendoring
  session for no operational benefit over Qwen3 4B.
- **Qwen3-Coder-30B-A3B (MoE)** — candle 0.10.2's `moe_gemm_gguf` kernel
  is CUDA-only. Bails on Metal and CPU. Linux users with NVIDIA / AMD
  ROCm can use it; macOS users can't.
- **Devstral Small 24B (Mistral)** — candle's `quantized_llama` derives
  head_dim from `embedding_length / head_count` and ignores the
  `llama.attention.key_length` metadata field. Devstral has non-canonical
  head_dim (128 ≠ 5120/32). Reshape mismatch at the first forward pass.
- **mistralrs (the alternative Rust engine)** — handles Devstral's head_dim
  correctly (we verified — its loader read the file cleanly in 26 s and
  extracted the OpenHands-style chat template), but its `metal` feature
  pulls in `mistralrs-paged-attn` whose `build.rs` invokes `xcrun metal`
  to compile Metal shaders. That tool ships only with full Xcode.app,
  not with Xcode Command Line Tools. On machines where corporate
  policy restricts Xcode, the build fails before we get to inference.
  The feature flag stays — Linux CUDA users still benefit.

The pragmatic shape that emerged: candle for the dense daily driver
(Qwen3 14B), `llama-server` subprocess for premium Mac cases that need
Devstral or MoE (Homebrew already ships `llama-server` with working Metal,
no Xcode required), and mistralrs for Linux GPU users who want pure-Rust
agentic-coder-grade inference without a subprocess.

That's the boring infrastructure under what comes next.

## The Agentic Showcase

The counter test proves the model can generate runnable Python. The
chaostooling-otel test asks something harder: *can it read documentation
it's never seen, decide which pattern applies, and rewrite its own code
correctly?*

The setup is in `src/native.rs::qwen3_runtime_adds_chaosotel_tracing_to_counter_program`.
We:

1. Read three real instrumentation patterns from the
   `chaostooling-oss/chaostooling-otel` repo's README — the
   `@instrument_action` decorator, manual `tracer.start_as_current_span`,
   and the `instrument_db_span` helper.
2. Read the counter program the model wrote in the previous test.
3. Compose a prompt that includes both, plus the question: "Pick the
   most appropriate pattern, explain why, and rewrite the program."
4. Send it to Qwen3 14B with thinking mode on (no `/no_think`).
5. Split the response into the `<think>...</think>` reasoning and the
   user-visible answer.
6. Extract the Python code block from the answer and write it to
   `/tmp/rs_llmctl_count_traced.py`.

The full pipeline:

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  Three chaostooling-otel patterns (verbatim from README)        │
   │  + Counter program (from earlier test) + the question           │
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼     363 input tokens
   ┌─────────────────────────────────────────────────────────────────┐
   │  Qwen3 14B Q4_K_M on Metal                                      │
   │  Prefill 308 ms · Generation 17.3 tok/s · 1138 tokens total    │
   └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  <think>                                                        │
   │    Reject Pattern 1 (no function to decorate)                   │
   │    Reject Pattern 3 (no database operation)                     │
   │    Pick Pattern 2 (manual span — each iteration needs one)      │
   │  </think>                                                       │
   │                                                                 │
   │  User-visible answer:                                           │
   │    "Pattern 2: Manual Span Creation" with rationale             │
   │    Rewritten program as a fenced ```python``` block             │
   │    Summary section                                              │
   └─────────────────────────────────────────────────────────────────┘
```

We tested for three things:

1. **The model's user-visible answer mentions which pattern it picked
   and why.** Not just generating instrumented code blind — explaining
   the choice to the human.
2. **The extracted Python contains one of the chaosotel API markers**
   (`get_tracer`, `start_as_current_span`, `instrument_action`, or
   `instrument_db_span`). The model has to use the real API surface,
   not invent a similar-looking one.
3. **The traced program still contains the original counter loop**
   (`range(...)` + `print`). Adding tracing shouldn't destroy the
   semantics of the underlying program.

All three assertions passed. The model produced exactly the kind of
patch a thoughtful developer would write.

```python
from chaosotel import ensure_initialized, get_tracer, flush

ensure_initialized()
tracer = get_tracer()

for i in range(1, 11):
    with tracer.start_as_current_span(f"loop_iteration_{i}") as span:
        span.set_attribute("iteration_number", i)
        print(i)

flush()
```

This is the output that makes the case for local agentic dev. The cloud
hasn't been touched. The model size is small enough to fit a laptop. The
runtime is one binary plus a GGUF file. And the output is something you'd
ship — not "almost right" or "good enough to start from," but the exact
pattern from the real codebase, applied to the right place.

## What This Setup Costs

The measured numbers from the run:

| Metric | Value | Notes |
|---|---|---|
| Model file on disk | 9 GB | Qwen3-14B-Q4_K_M.gguf |
| Cold load | 7.3 s | First-run; subsequent loads similar |
| Working set during inference | ~11 GB | Plus KV cache for context |
| Prefill (30-token chat prompt) | 53-236 ms | Depends on RoPE / context warmth |
| Prefill (363-token chaosotel context) | 308 ms | The harder agentic prompt |
| Generation throughput | ~17-19 tok/s | Steady-state, /no_think disabled |
| Per-task cost | $0 | No cloud, no API key |
| Hardware required | Any Apple Silicon 16 GB+ | Tested on 24 GB unified |

The 19 tok/s figure is for greedy decode on Metal. For comparison, the same
Mac CPU-only would land around 2-3 tok/s. The ~10× speedup is the entire
point of the `gpu-metal` cargo feature work — without it, this experience
isn't viable on a developer laptop.

For deployment outside the Mac, the same rs-llmctl binary with
`gpu-cuda` covers NVIDIA and AMD-via-ROCm on Linux. Tier matrix and
per-tier example configs live in `examples/qwen3-tier1.toml` (6 GB
NVIDIA) and `examples/qwen3-tier3.toml` (16+ GB and Apple unified).

## What's Next

The roadmap from here is small:

- **`llama-server` subprocess integration** so Mac users can opt into
  Devstral / MoE without switching tools. The plumbing is mostly
  documented; the missing piece is rs-llmctl spawning and proxying to
  the Homebrew binary.
- **Upstream candle issues** for the two limits we hit: Metal MoE
  kernel and `quantized_llama` reading `key_length`. Either patch
  collapses an entire row of the variant decision table.
- **Real-traffic observability validation.** The OTel spans for
  `native.model.load`, `native.model.prefill`, and `native.model.generation`
  are wired into the production request path but haven't been
  end-to-end captured in a test yet. The agentic chaosotel demo above
  shows the model can teach itself OTel; we should hold ourselves to
  the same standard with the runtime.

## Try It

The full agentic chaosotel demo is reproducible from this repository:

```
$ cargo test --features native-candle,native-tokenizers,gpu-metal \
    qwen3_runtime_adds_chaosotel_tracing_to_counter_program \
    -- --nocapture --ignored
```

It expects `~/.local/share/milliways/models/Qwen3-14B-Instruct-Q4_K_M.gguf`.
The download is one `curl` command from
[unsloth/Qwen3-14B-GGUF][unsloth-qwen3].

The test writes the generated program to
`/tmp/rs_llmctl_count_traced.py`. Read it after the run. It's the
artifact of a small model reasoning about an OTel pattern it had never
seen, applying it to code it had written itself, and showing its work.

That's enough to make local agentic dev feel real.

[candle]: https://github.com/huggingface/candle
[internals]: ./native-gguf-internals.md
[unsloth-qwen3]: https://huggingface.co/unsloth/Qwen3-14B-GGUF
