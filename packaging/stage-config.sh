#!/usr/bin/env bash
set -euo pipefail

TARGET=${TARGET:-/etc/rs-llmctl/config.toml}
EXAMPLES_DIR=${EXAMPLES_DIR:-examples}

usage() {
  cat <<'USAGE'
Usage: packaging/stage-config.sh <profile>

Profiles:
  cpu-only
  gpu-amd
  gpu-auto
  gpu-metal
  gpu-nvidia
  local-dev
  production-external-bind

Set TARGET=/path/to/config.toml to stage somewhere other than
/etc/rs-llmctl/config.toml.
USAGE
}

profile=${1:-}
case "${profile}" in
  cpu-only|gpu-amd|gpu-auto|gpu-metal|gpu-nvidia|local-dev|production-external-bind)
    ;;
  ""|-h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

source="${EXAMPLES_DIR}/${profile}.toml"
if [[ ! -f "${source}" ]]; then
  printf 'Example config not found: %s\n' "${source}" >&2
  exit 1
fi

printf 'Review %s before installing to %s:\n\n' "${source}" "${TARGET}"
sed -n '1,240p' "${source}"
printf '\nType COPY to install this reviewed config, or anything else to abort: '
read -r confirmation

if [[ "${confirmation}" != "COPY" ]]; then
  printf 'Aborted; no files changed.\n'
  exit 0
fi

install -D -m 0640 "${source}" "${TARGET}"
printf 'Installed %s from %s.\n' "${TARGET}" "${source}"
printf 'No service has been started or enabled.\n'
