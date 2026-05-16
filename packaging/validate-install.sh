#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-/etc/rs-llmctl/config.toml}
UNIT=${UNIT:-/etc/systemd/system/llmctld.service}
LLMCTL=${LLMCTL:-llmctl}
LLMCTLD=${LLMCTLD:-llmctld}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run "${LLMCTLD}" --config "${CONFIG}" --dry-run
run "${LLMCTL}" --config "${CONFIG}" security check
run "${LLMCTL}" --config "${CONFIG}" observe plan

if command -v systemd-analyze >/dev/null 2>&1; then
  run systemd-analyze verify "${UNIT}"
else
  printf 'systemd-analyze not found; skipping unit verification\n'
fi
