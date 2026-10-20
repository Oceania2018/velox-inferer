#!/usr/bin/env bash
# Runs a command inside the velox-dev container (CUDA 13.1 + Rust) with the GPU and this repo.
#   scripts/dev.sh cargo test --workspace --features velox-cli/cuda
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || pwd)"
export MSYS_NO_PATHCONV=1
exec docker run --rm --gpus all -v "$repo":/work -v velox-cargo:/opt/cargo/registry \
  -e CARGO_TARGET_DIR=/work/target/linux -w /work ${VELOX_DOCKER_ARGS:-} velox-dev "$@"
