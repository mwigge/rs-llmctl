#!/usr/bin/env bash
set -euo pipefail

test -x target/release/llmctl
test -x target/release/llmctld

sha256sum target/release/llmctl target/release/llmctld > SHA256SUMS
