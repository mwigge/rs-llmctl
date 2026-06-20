#!/usr/bin/env bash
# install-amd-hip.sh — standalone AMD GPU setup for rs-llmctl.
#
# Builds llama-server with ROCm HIP support for the detected GPU architecture,
# downloads a GGUF model, writes a standalone launcher and systemd unit, and
# optionally patches rs-llmctl.toml so the managed (LlamaServerSubprocess)
# path also works.
#
# No milliways dependency. Only requires: ROCm (hipcc), cmake, git, curl.
#
# Usage:
#   bash scripts/install-amd-hip.sh               # defaults below
#   MODEL_REPO=unsloth/Qwen3-14B-GGUF bash scripts/install-amd-hip.sh
#   DRY_RUN=1 bash scripts/install-amd-hip.sh     # print plan, build nothing

set -euo pipefail

# ── tunables ──────────────────────────────────────────────────────────────────
BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3-14B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3-14B-Q4_K_M.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3-14b}"
CTX_SIZE="${CTX_SIZE:-32768}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_PARALLEL="${N_PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
MODEL_TEMP="${MODEL_TEMP:-0.60}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-24bba7b98ea1544cc89352c7a573baedcb831a64}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-$HOME/.local/bin}"
LLAMA_LIB_DIR="${LLAMA_LIB_DIR:-$HOME/.local/lib/llmctl}"
MODEL_DIR="${MODEL_DIR:-$HOME/.local/share/rs-llmctl/models}"
LAUNCHER="${LAUNCHER:-$HOME/.local/bin/llmctl-local-server}"
LOG_DIR="${LOG_DIR:-$HOME/.local/share/llmctl/local}"
LLMCTL_CONFIG="${LLMCTL_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/rs-llmctl/config.toml}"
OTLP_ENDPOINT="${OTLP_ENDPOINT:-http://127.0.0.1:4318}"
DRY_RUN="${DRY_RUN:-0}"
# ──────────────────────────────────────────────────────────────────────────────

export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export HIP_PLATFORM="${HIP_PLATFORM:-amd}"

color() { printf '\033[1;%sm%s\033[0m\n' "$1" "$2"; }
info()  { color 36 "==> $*"; }
ok()    { color 32 "[ok] $*"; }
warn()  { color 33 "[!]  $*"; }
fail()  { color 31 "[x]  $*"; exit 1; }
dry()   { color 33 "[dry-run] $*"; }

# ── GPU architecture detection ─────────────────────────────────────────────────
detect_amdgpu_targets() {
  local gfx
  gfx=$(rocminfo 2>/dev/null | grep -Eo 'gfx[0-9]+' | grep -v '^gfx0' | head -1)
  [ -n "$gfx" ] && { echo "$gfx"; return; }
  gfx=$(vulkaninfo 2>/dev/null | grep -Eoi 'gfx[0-9a-f]+' | head -1 | tr '[:upper:]' '[:lower:]')
  [ -n "$gfx" ] && { echo "$gfx"; return; }
  echo "gfx1200"
}
AMDGPU_TARGETS="${AMDGPU_TARGETS:-$(detect_amdgpu_targets)}"

# ── prerequisites ──────────────────────────────────────────────────────────────
check_prerequisites() {
  command -v hipcc  >/dev/null 2>&1 || fail "hipcc not found — install ROCm and ensure /opt/rocm/bin is on PATH"
  command -v cmake  >/dev/null 2>&1 || fail "cmake not found"
  command -v git    >/dev/null 2>&1 || fail "git not found"
  command -v curl   >/dev/null 2>&1 || fail "curl not found"

  local missing=()
  id -Gn 2>/dev/null | grep -qw render || missing+=("render")
  id -Gn 2>/dev/null | grep -qw video  || missing+=("video")
  if [ ${#missing[@]} -gt 0 ]; then
    warn "User not in groups: ${missing[*]}"
    warn "Run: sudo usermod -aG ${missing[*]} \$USER  then re-login (or: newgrp render)"
    warn "Without these groups ROCm silently falls back to CPU"
  fi
  [ -c /dev/kfd ] || warn "/dev/kfd not found — amdgpu kernel driver may not be loaded (lsmod | grep amdgpu)"
}

# ── llama-server HIP build ────────────────────────────────────────────────────
llama_server_is_hip() {
  local bin="$1"
  local bin_dir; bin_dir="$(dirname "$bin")"
  [ -f "$bin_dir/libggml-hip.so" ] && return 0
  [ -f "$LLAMA_LIB_DIR/libggml-hip.so" ] && return 0
  LD_LIBRARY_PATH="$LLAMA_LIB_DIR:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}" \
    ldd "$bin" 2>/dev/null | grep -Eq 'libamdhip64|libhipblas|librocblas'
}

install_llama_shared_libs() {
  local src_dir="$1"
  mkdir -p "$LLAMA_LIB_DIR"
  cp -a "$src_dir"/*.so* "$LLAMA_LIB_DIR"/ 2>/dev/null || true
  if command -v readelf >/dev/null 2>&1; then
    local lib soname base
    for lib in "$LLAMA_LIB_DIR"/lib*.so.*.*; do
      [ -e "$lib" ] || continue
      soname="$(readelf -d "$lib" 2>/dev/null | sed -n 's/.*Library soname: \[\([^]]*\)\].*/\1/p' | head -1)"
      [ -n "$soname" ] || continue
      base="${soname%%.so*}.so"
      ln -sfn "$(basename "$lib")" "$LLAMA_LIB_DIR/$soname"
      ln -sfn "$soname" "$LLAMA_LIB_DIR/$base"
    done
  fi
}

build_llama_server() {
  mkdir -p "$LLAMA_BIN_DIR" "$LLAMA_LIB_DIR"

  if [ -x "$LLAMA_BIN_DIR/llama-server" ] && llama_server_is_hip "$LLAMA_BIN_DIR/llama-server"; then
    ok "HIP-enabled llama-server already installed: $LLAMA_BIN_DIR/llama-server"
    return
  fi

  if [ "$DRY_RUN" = "1" ]; then
    dry "Would build llama.cpp @ $LLAMA_CPP_REF with -DGGML_HIP=ON -DAMDGPU_TARGETS=$AMDGPU_TARGETS"
    return
  fi

  info "Building llama.cpp with HIP backend (ref=$LLAMA_CPP_REF, target=$AMDGPU_TARGETS)..."
  local tmp; tmp="$(mktemp -d)"
  trap "rm -rf '$tmp'" EXIT
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$tmp/llama.cpp"
  (cd "$tmp/llama.cpp" && git fetch --depth 1 origin "$LLAMA_CPP_REF" && git checkout --detach FETCH_HEAD)
  cmake -S "$tmp/llama.cpp" -B "$tmp/llama.cpp/build" \
    -DGGML_HIP=ON \
    -DGGML_NATIVE=OFF \
    -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
  cmake --build "$tmp/llama.cpp/build" --config Release --target llama-server -j
  install -m 0755 "$tmp/llama.cpp/build/bin/llama-server" "$LLAMA_BIN_DIR/llama-server"
  install -m 0755 "$tmp/llama.cpp/build/bin/llama-cli" "$LLAMA_BIN_DIR/llama-cli" 2>/dev/null || true
  install_llama_shared_libs "$tmp/llama.cpp/build/bin"
  rm -rf "$tmp"
  trap - EXIT
  ok "llama-server installed: $LLAMA_BIN_DIR/llama-server"
}

# ── model download ─────────────────────────────────────────────────────────────
fetch_model() {
  local url="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
  local dest="$MODEL_DIR/$MODEL_FILE"
  mkdir -p "$MODEL_DIR"
  if [ -s "$dest" ]; then
    ok "model already cached: $dest"
    MODEL_PATH="$dest"
    return
  fi
  if [ "$DRY_RUN" = "1" ]; then
    dry "Would download $url -> $dest"
    MODEL_PATH="$dest"
    return
  fi
  info "Downloading $MODEL_REPO/$MODEL_FILE..."
  if ! curl -fL -C - --retry 3 --retry-delay 5 -o "$dest" "$url"; then
    rm -f "$dest"
    fail "Download failed. Try manually: curl -fL -o '$dest' '$url'"
  fi
  ok "model cached: $dest"
  MODEL_PATH="$dest"
}

# ── standalone launcher + systemd unit ────────────────────────────────────────
write_launcher() {
  mkdir -p "$LOG_DIR" "$(dirname "$LAUNCHER")"
  if [ "$DRY_RUN" = "1" ]; then
    dry "Would write standalone launcher: $LAUNCHER"
    dry "Would write systemd unit: $HOME/.config/systemd/user/llmctl-local.service"
    return
  fi
  cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:\$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:\${PATH:-}"
export LD_LIBRARY_PATH="$LLAMA_LIB_DIR:/opt/rocm/lib:/opt/rocm/lib64:\${LD_LIBRARY_PATH:-}"
export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
exec "$LLAMA_BIN_DIR/llama-server" \\
  -m "$MODEL_PATH" \\
  --alias "$MODEL_ALIAS" \\
  --host "$BIND_HOST" \\
  --port "$PORT" \\
  --ctx-size "$CTX_SIZE" \\
  --parallel "$N_PARALLEL" \\
  --batch-size "$BATCH_SIZE" \\
  --ubatch-size "$UBATCH_SIZE" \\
  --n-gpu-layers "$N_GPU_LAYERS" \\
  --temp "$MODEL_TEMP" \\
  --cache-type-k "$CACHE_TYPE_K" \\
  --cache-type-v "$CACHE_TYPE_V" \\
  --jinja \\
  --metrics \\
  --flash-attn off
EOF
  chmod +x "$LAUNCHER"
  ok "wrote standalone launcher: $LAUNCHER"

  local unit="$HOME/.config/systemd/user/llmctl-local.service"
  mkdir -p "$(dirname "$unit")"
  cat > "$unit" <<EOF
[Unit]
Description=llmctl AMD HIP local model server (llama.cpp)

[Service]
ExecStart=$LAUNCHER
Restart=on-failure
StandardOutput=append:$LOG_DIR/server.log
StandardError=append:$LOG_DIR/server.err

[Install]
WantedBy=default.target
EOF
  ok "wrote systemd unit: $unit"
  info "To enable at login: systemctl --user enable --now llmctl-local.service"
}

# ── optional: patch rs-llmctl.toml for managed path ──────────────────────────
patch_llmctl_config() {
  [ -f "$LLMCTL_CONFIG" ] || return 0
  if [ "$DRY_RUN" = "1" ]; then
    dry "Would patch $LLMCTL_CONFIG: gpu_vendor=amd, llama_server_bin=$LLAMA_BIN_DIR/llama-server, otlp=$OTLP_ENDPOINT"
    return
  fi
  local tmp; tmp="$(mktemp)"

  # Patch [resources]: set gpu_vendor, cpu_only, llama_server_bin
  awk -v bin="$LLAMA_BIN_DIR/llama-server" '
    /^\[resources\]/ { print; in_res=1; next }
    in_res && /^gpu_vendor/       { print "gpu_vendor = \"amd\""; next }
    in_res && /^cpu_only/         { print "cpu_only = false"; next }
    in_res && /^llama_server_bin/ { next }
    in_res && /^\[/ { print "llama_server_bin = \"" bin "\""; in_res=0 }
    { print }
    END { if (in_res) print "llama_server_bin = \"" bin "\"" }
  ' "$LLMCTL_CONFIG" > "$tmp" && mv "$tmp" "$LLMCTL_CONFIG"

  # Patch [observability] exporter.endpoint for SigNoz/OTLP.
  # The [observability.exporter] section exists but has no endpoint key — insert it.
  tmp="$(mktemp)"
  awk -v ep="$OTLP_ENDPOINT" '
    /^\[observability\.exporter\]/ { print; print "endpoint = \"" ep "\""; in_exp=1; next }
    in_exp && /^endpoint/  { next }
    in_exp && /^\[/        { in_exp=0 }
    { print }
  ' "$LLMCTL_CONFIG" > "$tmp" && mv "$tmp" "$LLMCTL_CONFIG"

  ok "patched $LLMCTL_CONFIG (AMD + OTLP → $OTLP_ENDPOINT)"
}

# ── main ───────────────────────────────────────────────────────────────────────
main() {
  info "rs-llmctl AMD HIP installer"
  info "GPU target:  $AMDGPU_TARGETS"
  info "Model:       $MODEL_REPO / $MODEL_FILE  (alias: $MODEL_ALIAS)"
  info "Endpoint:    http://${BIND_HOST}:${PORT}/v1"
  info "Context:     $CTX_SIZE tokens"
  info "KV cache:    k=$CACHE_TYPE_K  v=$CACHE_TYPE_V"
  info "OTel OTLP:   $OTLP_ENDPOINT"
  [ "$DRY_RUN" = "1" ] && warn "DRY RUN — no files will be written or downloaded"
  echo

  check_prerequisites
  build_llama_server
  fetch_model
  write_launcher
  patch_llmctl_config

  echo
  ok "All done."
  info "Start standalone:  $LAUNCHER"
  info "Or via systemd:    systemctl --user start llmctl-local.service"
  info "OpenAI endpoint:   http://${BIND_HOST}:${PORT}/v1"
  info "Model alias:       $MODEL_ALIAS"
  if [ -f "$LLMCTL_CONFIG" ]; then
    info "Managed path:      llmctl server run  (uses $LLMCTL_CONFIG)"
  fi
  info "OTel collector:    cd docker/signoz && docker compose up -d"
  info "SigNoz UI:         http://localhost:3301"
  info "OTLP endpoint:     $OTLP_ENDPOINT  (set in config)"
}

main "$@"
