#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-dist}"
mkdir -p "${OUT_DIR}"

if [[ "${LLMCTL_CYCLONEDX:-0}" == "1" ]] && command -v cargo-cyclonedx >/dev/null 2>&1; then
  set +e
  cargo cyclonedx --format json --output-cdx --output-prefix "${OUT_DIR}/rs-llmctl"
  status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    echo "[ok] wrote ${OUT_DIR}/rs-llmctl.cdx.json"
    exit 0
  fi
  echo "[warn] cargo-cyclonedx failed; writing dependency metadata fallback" >&2
fi

cp Cargo.lock "${OUT_DIR}/Cargo.lock"
cat > "${OUT_DIR}/rs-llmctl.sbom-fallback.json" <<JSON
{
  "format": "cargo-lock-fallback",
  "component": "rs-llmctl",
  "lockfile": "Cargo.lock",
  "note": "CycloneDX output requires pre-cached cargo-cyclonedx. This fallback preserves the locked dependency graph for offline evidence."
}
JSON
echo "[warn] wrote Cargo.lock dependency evidence fallback"
echo "[warn] set LLMCTL_CYCLONEDX=1 with pre-cached cargo-cyclonedx for CycloneDX output"
