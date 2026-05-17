#!/usr/bin/env bash
set -euo pipefail

CONFIG="${LLMCTL_NATIVE_SMOKE_CONFIG:-}"
CONFIG_TOML="${LLMCTL_NATIVE_SMOKE_CONFIG_TOML:-}"
API_KEY="${LLMCTL_NATIVE_SMOKE_API_KEY:-}"
MODEL="${LLMCTL_NATIVE_SMOKE_MODEL:-qwen}"
MODEL_PATH="${LLMCTL_NATIVE_SMOKE_MODEL_PATH:-}"
BASE_URL="${LLMCTL_NATIVE_SMOKE_BASE_URL:-http://127.0.0.1:8765}"
TMP_CONFIG=""
TMP_INSTALL=""
TMP_DATA=""

if [[ "${LLMCTL_NATIVE_SMOKE_INSTALL_ARTIFACT:-1}" == "1" ]]; then
  DIST_DIR="${LLMCTL_NATIVE_SMOKE_DIST:-dist}"
  TARBALL="${LLMCTL_NATIVE_SMOKE_TARBALL:-}"
  if [[ -z "${TARBALL}" ]]; then
    TARBALL="$(find "${DIST_DIR}" -maxdepth 1 -type f -name 'rs-llmctl-*.tar.gz' | sort | head -n 1)"
  fi
  if [[ -z "${TARBALL}" || ! -f "${TARBALL}" ]]; then
    echo "missing release tarball; run packaging/generate-checksums.sh or set LLMCTL_NATIVE_SMOKE_TARBALL" >&2
    exit 1
  fi
  TMP_INSTALL="$(mktemp -d)"
  PREFIX="${TMP_INSTALL}/prefix" \
    LLMCTL_INSTALL_SYSTEMD=0 \
    RS_LLMCTL_TARBALL="${TARBALL}" \
    RS_LLMCTL_SHA256SUMS="${DIST_DIR}/SHA256SUMS" \
    ./install.sh >/dev/null
  LLMCTL_BIN="${TMP_INSTALL}/prefix/bin/llmctl"
else
  cargo build --release --bin llmctl --features native
  LLMCTL_BIN="target/release/llmctl"
fi

if [[ -z "${CONFIG}" && -n "${CONFIG_TOML}" ]]; then
  TMP_CONFIG="$(mktemp)"
  printf '%s\n' "${CONFIG_TOML}" > "${TMP_CONFIG}"
  CONFIG="${TMP_CONFIG}"
fi

if [[ -z "${CONFIG}" && -n "${MODEL_PATH}" ]]; then
  TMP_DATA="$(mktemp -d)"
  TMP_CONFIG="${TMP_DATA}/config.toml"
  SECRET_FILE="${TMP_DATA}/api-key.txt"
  "${LLMCTL_BIN}" --config "${TMP_CONFIG}" first-run --apply \
    --secret-output "${SECRET_FILE}" \
    --data-dir "${TMP_DATA}/state" \
    --starter-model-path "${MODEL_PATH}" \
    --starter-model-alias "${MODEL}" \
    --base-url "${BASE_URL}" >/dev/null
  CONFIG="${TMP_CONFIG}"
  if [[ -z "${API_KEY}" ]]; then
    API_KEY="$(cat "${SECRET_FILE}")"
  fi
fi

if [[ -z "${CONFIG}" || ! -f "${CONFIG}" ]]; then
  echo "set LLMCTL_NATIVE_SMOKE_MODEL_PATH for default one-key first-run smoke, or provide LLMCTL_NATIVE_SMOKE_CONFIG/LLMCTL_NATIVE_SMOKE_CONFIG_TOML" >&2
  exit 1
fi

"${LLMCTL_BIN}" --config "${CONFIG}" server check >/dev/null

"${LLMCTL_BIN}" --config "${CONFIG}" server run &
pid=$!
trap 'kill "${pid}" >/dev/null 2>&1 || true; if [[ -n "${TMP_CONFIG}" && -z "${TMP_DATA}" ]]; then rm -f "${TMP_CONFIG}"; fi; if [[ -n "${TMP_DATA}" ]]; then rm -rf "${TMP_DATA}"; fi; if [[ -n "${TMP_INSTALL}" ]]; then rm -rf "${TMP_INSTALL}"; fi' EXIT

for _ in $(seq 1 120); do
  if curl -fsS "${BASE_URL}/readyz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

auth_args=()
if [[ -n "${API_KEY}" ]]; then
  auth_args=(-H "Authorization: Bearer ${API_KEY}")
fi

curl -fsS "${auth_args[@]}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return one short word.\"}],\"max_tokens\":1}" \
  "${BASE_URL}/v1/chat/completions" >/dev/null

curl -fsS "${auth_args[@]}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return one short word.\"}],\"max_tokens\":1,\"stream\":true}" \
  "${BASE_URL}/v1/chat/completions" | grep -q 'data:'
