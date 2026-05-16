#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${1:-dist}"
CHECKSUMS="${DIST_DIR}/SHA256SUMS"

if [[ ! -f "${CHECKSUMS}" ]]; then
  echo "missing ${CHECKSUMS}; run packaging/generate-checksums.sh first" >&2
  exit 1
fi

if command -v cosign >/dev/null 2>&1; then
  cosign sign-blob --yes --output-signature "${CHECKSUMS}.sig" "${CHECKSUMS}"
  echo "[ok] wrote ${CHECKSUMS}.sig with cosign"
elif command -v minisign >/dev/null 2>&1; then
  minisign -Sm "${CHECKSUMS}"
  echo "[ok] wrote ${CHECKSUMS}.minisig with minisign"
else
  echo "missing signing tool: install cosign or minisign" >&2
  exit 1
fi
