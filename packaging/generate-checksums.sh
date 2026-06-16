#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${1:-dist}"
OS="${OS:-$(uname -s | tr '[:upper:]' '[:lower:]')}"
ARCH="${ARCH:-$(uname -m)}"

case "${OS}" in
  linux|darwin) ;;
  *)
    printf 'unsupported OS: %s\n' "${OS}" >&2
    exit 1
    ;;
esac

case "${ARCH}" in
  x86_64|amd64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *)
    printf 'unsupported architecture: %s\n' "${ARCH}" >&2
    exit 1
    ;;
esac

artifact="rs-llmctl-${OS}-${ARCH}"
stage="${DIST_DIR}/${artifact}"
tarball="${DIST_DIR}/${artifact}.tar.gz"

test -x target/release/llmctl
rm -rf "${stage}"
mkdir -p "${stage}"
install -m 0755 target/release/llmctl "${stage}/llmctl"

if [[ -f README.md ]]; then
  install -m 0644 README.md "${stage}/README.md"
fi

if [[ -f LICENSE ]]; then
  install -m 0644 LICENSE "${stage}/LICENSE"
fi

if [[ -f CHANGELOG.md ]]; then
  install -m 0644 CHANGELOG.md "${stage}/CHANGELOG.md"
fi

if [[ "${OS}" == "linux" && -f packaging/systemd/llmctld.service ]]; then
  install -D -m 0644 packaging/systemd/llmctld.service "${stage}/packaging/systemd/llmctld.service"
fi
if [[ "${OS}" == "linux" && -f packaging/systemd/llmctl-monthly-audit.service ]]; then
  install -D -m 0644 packaging/systemd/llmctl-monthly-audit.service "${stage}/packaging/systemd/llmctl-monthly-audit.service"
  install -D -m 0644 packaging/systemd/llmctl-monthly-audit.timer "${stage}/packaging/systemd/llmctl-monthly-audit.timer"
fi

tar -C "${stage}" -czf "${tarball}" .
(
  cd "${DIST_DIR}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum rs-llmctl-*.tar.gz | sort -k2 > SHA256SUMS
  else
    shasum -a 256 rs-llmctl-*.tar.gz | sort -k2 > SHA256SUMS
  fi
)
