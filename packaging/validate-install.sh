#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-${LLMCTL_CONFIG:-/etc/rs-llmctl/config.toml}}
SERVICE_NAME=${LLMCTL_SERVICE_NAME:-llmctld}
UNIT=${UNIT:-/etc/systemd/system/${SERVICE_NAME}.service}
PREFIX=${PREFIX:-/usr/local}
BIN_DIR=${BIN_DIR:-${PREFIX}/bin}
LLMCTL=${LLMCTL:-${BIN_DIR}/llmctl}
STATE_DIR=${LLMCTL_STATE_DIR:-/var/lib/rs-llmctl}
LOG_DIR=${LLMCTL_LOG_DIR:-/var/log/rs-llmctl}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'missing file: %s\n' "$1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    printf 'missing directory: %s\n' "$1" >&2
    exit 1
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    printf 'missing executable: %s\n' "$1" >&2
    exit 1
  fi
}

require_contains() {
  local file=$1
  local needle=$2
  if ! grep -Fq "$needle" "$file"; then
    printf 'expected %s to contain: %s\n' "$file" "$needle" >&2
    exit 1
  fi
}

require_file "${CONFIG}"
require_file "${UNIT}"
require_dir "${STATE_DIR}"
require_dir "${STATE_DIR}/models"
require_dir "${STATE_DIR}/reports"
require_dir "${LOG_DIR}"
require_executable "${LLMCTL}"
require_contains "${UNIT}" "Environment=LLMCTL_CONFIG=${CONFIG}"
require_contains "${UNIT}" "ExecStart=${LLMCTL} --config \${LLMCTL_CONFIG} server run"
require_contains "${UNIT}" "ReadWritePaths=${STATE_DIR} ${LOG_DIR}"

run "${LLMCTL}" --config "${CONFIG}" security check
run "${LLMCTL}" --config "${CONFIG}" server status
run "${LLMCTL}" --config "${CONFIG}" server plan
run "${LLMCTL}" --config "${CONFIG}" audit retention plan
run "${LLMCTL}" --config "${CONFIG}" observe plan

if command -v systemd-analyze >/dev/null 2>&1; then
  run systemd-analyze verify "${UNIT}"
else
  printf 'systemd-analyze not found; skipping unit verification\n'
fi
