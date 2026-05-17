#!/usr/bin/env bash
set -euo pipefail

MODEL="${LLMCTL_NATIVE_SMOKE_MODEL:-qwen}"
MODEL_PATH="${LLMCTL_NATIVE_SMOKE_MODEL_PATH:-}"
PORT="${LLMCTL_NATIVE_SMOKE_PORT:-$((20000 + RANDOM % 20000))}"
WORKER_BASE_PORT="${LLMCTL_NATIVE_SMOKE_WORKER_BASE_PORT:-$((PORT + 10000))}"
BASE_URL="${LLMCTL_NATIVE_SMOKE_BASE_URL:-http://127.0.0.1:${PORT}}"
QUESTION="${LLMCTL_NATIVE_SMOKE_QUESTION:-Return one short word.}"
TMP_CONFIG=""
TMP_INSTALL=""
TMP_DATA=""
SERVER_LOG=""

DIST_DIR="${LLMCTL_NATIVE_SMOKE_DIST:-dist}"
TARBALL="${LLMCTL_NATIVE_SMOKE_TARBALL:-}"
if [[ -z "${TARBALL}" ]]; then
  TARBALL="$(find "${DIST_DIR}" -maxdepth 1 -type f -name 'rs-llmctl-*.tar.gz' | sort | head -n 1)"
fi
if [[ -z "${TARBALL}" || ! -f "${TARBALL}" ]]; then
  echo "missing release tarball; run packaging/generate-checksums.sh or set LLMCTL_NATIVE_SMOKE_TARBALL" >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "LLMCTL_NATIVE_SMOKE_MODEL_PATH must point at one real local GGUF or safetensors model artifact" >&2
  exit 1
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "LLMCTL_NATIVE_SMOKE_MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

TMP_INSTALL="$(mktemp -d)"
TMP_DATA="$(mktemp -d)"
SERVER_LOG="${TMP_DATA}/server.log"
trap 'kill "${pid:-}" >/dev/null 2>&1 || true; rm -rf "${TMP_DATA}" "${TMP_INSTALL}"' EXIT

PREFIX="${TMP_INSTALL}/prefix" \
  LLMCTL_INSTALL_SYSTEMD=0 \
  RS_LLMCTL_TARBALL="${TARBALL}" \
  RS_LLMCTL_SHA256SUMS="${DIST_DIR}/SHA256SUMS" \
  ./install.sh >/dev/null
LLMCTL_BIN="${TMP_INSTALL}/prefix/bin/llmctl"
"${LLMCTL_BIN}" --version

TMP_CONFIG="${TMP_DATA}/config.toml"
SECRET_FILE="${TMP_DATA}/api-key.txt"
"${LLMCTL_BIN}" --config "${TMP_CONFIG}" first-run --apply \
  --secret-output "${SECRET_FILE}" \
  --data-dir "${TMP_DATA}/state" \
  --starter-model-path "${MODEL_PATH}" \
  --starter-model-alias "${MODEL}" \
  --base-url "${BASE_URL}" >/dev/null
sed -i \
  -e "s/^port = .*/port = ${PORT}/" \
  -e "s/^worker_base_port = .*/worker_base_port = ${WORKER_BASE_PORT}/" \
  "${TMP_CONFIG}"
API_KEY="$(cat "${SECRET_FILE}")"

"${LLMCTL_BIN}" --config "${TMP_CONFIG}" server check >/dev/null

"${LLMCTL_BIN}" --config "${TMP_CONFIG}" server run >"${SERVER_LOG}" 2>&1 &
pid=$!

ready=0
for _ in $(seq 1 120); do
  if curl -fsS "${BASE_URL}/readyz" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "server did not become ready at ${BASE_URL}/readyz" >&2
  tail -100 "${SERVER_LOG}" >&2 || true
  exit 1
fi

auth_args=()
if [[ -n "${API_KEY}" ]]; then
  auth_args=(-H "Authorization: Bearer ${API_KEY}")
fi

CHAT_RESPONSE="${TMP_DATA}/chat-response.json"
if ! curl -fsS "${auth_args[@]}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${QUESTION}\"}],\"max_tokens\":8}" \
  "${BASE_URL}/v1/chat/completions" >"${CHAT_RESPONSE}"; then
  echo "direct chat completion smoke failed" >&2
  cat "${CHAT_RESPONSE}" >&2 || true
  tail -100 "${SERVER_LOG}" >&2 || true
  exit 1
fi

CLIENT_LOG="${TMP_DATA}/client.log"
if ! client_output="$(
  LLMCTL_BASE_URL="${BASE_URL}" \
    LLMCTL_API_KEY="${API_KEY}" \
    LLMCTL_SMOKE_MODEL="${MODEL}" \
    LLMCTL_SMOKE_QUESTION="${QUESTION}" \
    cargo run -q -p rs-llmctl-client --example chat 2>"${CLIENT_LOG}"
)"; then
  echo "rs-llmctl-client smoke query failed" >&2
  cat "${CLIENT_LOG}" >&2 || true
  tail -100 "${SERVER_LOG}" >&2 || true
  exit 1
fi
if [[ -z "${client_output//[[:space:]]/}" ]]; then
  echo "rs-llmctl-client smoke query returned an empty response" >&2
  tail -100 "${SERVER_LOG}" >&2 || true
  exit 1
fi

echo "ok release smoke passed for ${MODEL}"
