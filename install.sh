#!/usr/bin/env bash
set -euo pipefail

repo="${RS_LLMCTL_REPO:-mwigge/rs-llmctl}"
version="${RS_LLMCTL_VERSION:-latest}"
prefix="${PREFIX:-/usr/local}"
os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

case "${arch}" in
  x86_64|amd64) arch="x86_64" ;;
  aarch64|arm64) arch="aarch64" ;;
  *) echo "unsupported architecture: ${arch}" >&2; exit 1 ;;
esac

case "${os}" in
  linux|darwin) ;;
  *) echo "unsupported OS: ${os}" >&2; exit 1 ;;
esac

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

if [[ "${version}" == "latest" ]]; then
  url="https://github.com/${repo}/releases/latest/download/rs-llmctl-${os}-${arch}.tar.gz"
else
  url="https://github.com/${repo}/releases/download/${version}/rs-llmctl-${os}-${arch}.tar.gz"
fi

echo "Downloading ${url}"
curl -fsSL "${url}" -o "${tmp}/rs-llmctl.tar.gz"
tar -xzf "${tmp}/rs-llmctl.tar.gz" -C "${tmp}"
install -D -m 0755 "${tmp}/llmctl" "${prefix}/bin/llmctl"
install -D -m 0755 "${tmp}/llmctld" "${prefix}/bin/llmctld"

echo "Installed llmctl and llmctld to ${prefix}/bin"
echo "Next: llmctl init --profile production-aiops"
