# ADR-0001: AMD GPU (ROCm/Vulkan) Acceleration for Native Inference

## Status

Proposed

## Context

rs-llmctl's `native` feature runs inference in-process via `candle-core`
0.10.2 (`src/native.rs`). Today this path is **CPU-only and fail-closed**:

- `load_real_candle_decoder`, `RealCandleModel::forward_next`, and the GGUF
  loaders hardcode `candle_core::Device::Cpu` (`src/native.rs:835, 888, 1795,
  1967`).
- `NativeCandleEngineLoader::load` explicitly rejects any
  `NativeAcceleration` other than `Cpu`/`Auto` with: *"native Candle decoding
  currently supports CPU execution only"* (`src/native.rs:1593-1599`).
- `candle-core` 0.10.2 ships exactly three device backends: CPU, CUDA
  (NVIDIA, via `cudarc`), and Metal (Apple). **There is no ROCm, HIP, or
  Vulkan backend in this version of candle.**

Despite this, the config/resource-planning layers already model AMD GPUs as
a first-class target:

- `NativeAcceleration::AmdRocm` enum variant (`src/native.rs:1045`), mapped
  from `gpu_vendor` via `from_resources` (`src/native.rs:1056-...`).
- `hardware_matrix()` in `src/runtime.rs:319` lists `("amd-vulkan",
  NativeAcceleration::AmdRocm, false)` as an optional validation target.
- `WorkerBackend::AmdVulkan { gpu_layers }` in `src/worker.rs:39,53-55`,
  selected when `gpu_vendor` is `"amd"`/`"vulkan"`/`"amd-vulkan"`.
- `src/resources.rs` already does AMD GPU discovery via sysfs
  (`amd_sysfs_gpus`, vendor ID `0x1002`) and `rocm-smi --showmeminfo vram`
  (`src/resources.rs:267-282`), populating `GpuBudget` with AMD VRAM
  figures.

So the **resource-planning and config layers anticipate AMD GPU execution**,
but the **execution layer (candle-native) cannot deliver it** — any request
for AMD acceleration fails closed today. On the current dev box, ROCm is not
installed (`/opt/rocm` absent, no `rocm-*` packages, `rocm-smi` not found);
`vulkaninfo` is present, suggesting a Vulkan-capable graphics stack but no
ROCm/HIP compute stack.

The trigger for this ADR: the user has an AMD RX 9060 XT (RDNA4, Navi 44)
and wants GPU-accelerated local inference, citing llama.cpp's mature
`-DGGML_HIP=ON` AMD support and AMD's day-0 ROCm support for Gemma models.

## Options Considered

### (a) Wait for candle's ROCm backend upstream

candle's only ROCm work is huggingface/candle#3424: an unmerged, explicitly
experimental WIP using the `rocm-rs` crate with AOT-compiled HIP kernels.
It currently supports **BERT only**; the PR description states "many unsafe
implementations remain, not all features complete." It does not cover
quantized GGUF model execution, which is what `quantized_gemma4` /
`quantized_qwen3` and the rest of `src/native.rs` depend on.

| Pros | Cons |
|---|---|
| Zero engineering effort | No ETA — PR has been WIP with no merge timeline |
| No new dependencies, no build complexity | Doesn't cover GGUF/quantized models even if merged today |
| Stays within the existing single-backend (candle) architecture | Blocks AMD GPU acceleration indefinitely |

### (b) Add a llama.cpp-based runtime backend for AMD (HIP/hipBLAS)

Introduce a second runtime path — alongside, not replacing, candle-native —
that runs a HIP-enabled llama.cpp (`-DGGML_HIP=ON`) for GGUF inference when
`gpu_vendor` resolves to AMD. Two integration shapes:

- **FFI bindings** (e.g. `llama-cpp-2` / `llama-cpp-sys-2`): link a
  HIP-built `libllama`/`libggml` into the `llmctl` binary, add a new
  `NativeAcceleration`-driven code path that constructs and drives a
  llama.cpp context instead of a candle one.
- **Subprocess/server model**: build `llama-server` with HIP support as a
  separate binary, and have rs-llmctl spawn it as a worker process. Notably,
  `WorkerSpec`/`CommandSpec` (`src/worker.rs:120-160, 153-...`) already model
  an external-process worker with a `program: PathBuf` field — the in-process
  candle path is itself represented as a sentinel
  `program: "<in-process:candle-native>"`. This means the **worker
  abstraction already has a seam for "external program" backends**; a
  llama.cpp subprocess backend would slot into an existing extension point
  rather than requiring a new abstraction.

| Pros | Cons |
|---|---|
| llama.cpp HIP/hipBLAS AMD support is mature and production-used | New dependency (FFI crate or vendored llama.cpp build) |
| Subprocess shape reuses the existing `WorkerSpec`/`CommandSpec` "external program" seam — smallest architectural delta | Build pipeline must compile/link a HIP-enabled llama.cpp — needs ROCm dev headers + correct GPU target (gfx-series for RDNA4) in CI and on target hosts |
| Decoupled from candle's version/feature churn | FFI shape requires a new in-process code path parallel to `native.rs` (new `RuntimeBackend` variant, GGUF compat testing, tokenizer/chat-template parity with existing candle loaders) |
| Gets GGUF + RDNA4 working with a stack AMD actively supports | Two execution paths to maintain, test, and keep behaviourally consistent (sampling, stop sequences, streaming) |
| `rocm-smi`/sysfs VRAM discovery in `resources.rs` already works for budget planning | Requires ROCm runtime installed on the host (currently absent) |

### (c) Contribute to candle's ROCm PR upstream

Help land huggingface/candle#3424 and extend it to quantized GGUF models.
Technically the "correct" long-term fix (single backend, no new
dependencies). **Not recommended for this team** — this is a multi-month
upstream effort spanning unsafe FFI kernel work in a project we don't
control, with no guarantee of timeline or eventual GGUF coverage. Mentioned
for completeness only.

### (d) CPU-only now; reconcile the AMD config plumbing

Keep candle-native CPU-only and either:
- **Remove** `NativeAcceleration::AmdRocm` / `gpu_vendor="amd-vulkan"` /
  `WorkerBackend::AmdVulkan` plumbing, since it currently implies AMD GPU
  support that fail-closes at runtime, or
- **Keep it as a documented future hook** — annotate it clearly as
  "resource-planning models this target; no execution backend implements it
  yet" so it's ready to wire up when (b) lands, without misleading operators
  who set `gpu_vendor = "amd"` today.

Recommend **keep as documented hook**, not remove — `resources.rs`'s AMD GPU
discovery (sysfs + `rocm-smi`) is independently useful for reporting/capacity
planning even before an AMD execution backend exists, and removing the enum
variant would just have to be re-added for (b).

### Other options considered and discarded

- **ONNX Runtime + ROCm execution provider** — would require exporting/
  converting GGUF models to ONNX, a separate model-prep pipeline, and a third
  inference runtime. Discarded: more invasive than (b) for less mature AMD
  GPU coverage of the quantized models we actually run.
- **Vulkan compute backend** (e.g. via `ggml`'s Vulkan backend in llama.cpp,
  which doesn't require ROCm at all) — worth noting as a *variant* of (b):
  llama.cpp also supports `-DGGML_VULKAN=ON`, which works on AMD without
  installing ROCm (uses the Vulkan stack already present on this machine, per
  `vulkaninfo`). This could be a **lower-friction first cut of (b)** — same
  architectural seam (subprocess `llama-server`), but `GGML_VULKAN` avoids
  the ROCm install/build dependency entirely, at some performance cost vs
  HIP/hipBLAS. Recommend evaluating Vulkan-backend llama.cpp first if (b) is
  pursued, before investing in a HIP build pipeline.

## Decision

1. **Now**: Adopt **(d)** — keep the existing `NativeAcceleration::AmdRocm`
   / `gpu_vendor` / `WorkerBackend::AmdVulkan` plumbing, but document it
   inline (doc comments) as a resource-planning hook with no execution
   backend yet, so `from_resources` selecting `AmdRocm`/`AmdVulkan` is
   understood to still fail-close via `native.rs:1593-1599`. No code
   behaviour change required — this is a documentation/comment correction
   to avoid misleading future readers (including agents).

2. **Track but do not block on (a)**: periodically check
   huggingface/candle#3424 for GGUF/quantized model support landing. Revisit
   this ADR if/when it merges and covers `quantized_gemma4`/`quantized_qwen3`
   equivalents.

3. **If AMD GPU acceleration becomes a priority**, pursue **(b)**, starting
   with the **Vulkan-backend llama.cpp variant** (subprocess `llama-server`
   built with `-DGGML_VULKAN=ON`) as the lowest-friction proof of concept,
   using the existing `WorkerSpec`/`CommandSpec` external-program seam. Only
   invest in a HIP/hipBLAS build pipeline (which requires installing the
   ROCm dev stack — currently absent on this machine) if Vulkan performance
   is insufficient for RDNA4.

4. **(c)** is explicitly out of scope.

### Effort sizing

| Option | Effort | Outcome |
|---|---|---|
| (a) | Zero now | No ETA; blocks AMD GPU indefinitely; doesn't cover GGUF even if merged |
| (b) — Vulkan llama.cpp subprocess | ~1-2 weeks: vendor/build `llama-server` with `GGML_VULKAN`, new `WorkerBackend`/`CommandSpec` wiring, GGUF/chat-template parity testing, OTel spans for the new worker type | Working AMD GPU inference via existing process-worker seam, no ROCm install required |
| (b) — HIP/hipBLAS FFI or subprocess | Multi-week: FFI crate evaluation (`llama-cpp-2`) or HIP build pipeline, ROCm dev stack install/CI, new parallel in-process runtime path if FFI, GGUF compat + sampling/streaming parity testing | Best AMD performance, highest build/maintenance cost |
| (c) | Multi-month, upstream-controlled | Not recommended |
| (d) | Trivial (doc comments only) | Removes misleading "AMD GPU supported" implication; preserves the planning hook |

## Consequences

- No immediate code change to execution behaviour; `gpu_vendor=amd*` still
  fail-closes to CPU via the existing error path in `native.rs`.
- Doc comments added near `NativeAcceleration::AmdRocm`,
  `WorkerBackend::AmdVulkan`, and the `hardware_matrix()` `"amd-vulkan"`
  entry, pointing at this ADR.
- Future AMD GPU work has a documented starting point (Vulkan llama.cpp
  subprocess via the existing `WorkerSpec` external-program seam) rather
  than starting from candle.
- If (b) is later pursued, it introduces a second runtime backend
  (`RuntimeBackend`-style split) that must be designed separately — this ADR
  does not size that design, only the decision to pursue it.

## Out of Scope

- Detailed design of the llama.cpp subprocess backend (new ADR if/when (b)
  is greenlit).
- Any change to `candle-core` version pinning.
- ROCm installation/runtime setup on any host.
