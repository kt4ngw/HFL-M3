#!/usr/bin/env bash
# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
if [[ -z "$CMAKE_BIN" || ! -x "$CMAKE_BIN" ]]; then
  echo "CMake is missing. Activate the HEonGPU build environment described in README.md." >&2
  exit 1
fi
NVCC_BIN="$(command -v nvcc || true)"
CUDA_ROOT="${CUDA_ROOT:-${NVCC_BIN%/bin/nvcc}}"
HEONGPU_ROOT="${HEONGPU_ROOT:-$HOME/.local/heongpu-v1.1.3}"
NTL_ROOT="${NTL_ROOT:-${CONDA_PREFIX:-}}"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build/heongpu}"
for dependency in CUDA_ROOT HEONGPU_ROOT NTL_ROOT; do
  value="${!dependency}"
  if [[ -z "$value" || "$value" == /path/to/* || ! -d "$value" ]]; then
    echo "$dependency must point to an existing installation (got '$value'). See README.md." >&2
    exit 1
  fi
done
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "CUDA compiler missing: $CUDA_ROOT/bin/nvcc. PyTorch CUDA runtime packages are not a compiler toolkit." >&2
  exit 1
fi
if [[ ! -f "$NTL_ROOT/include/NTL/ZZ.h" ]]; then
  echo "NTL development headers missing under $NTL_ROOT/include/NTL." >&2
  exit 1
fi
if [[ -z "${CUDA_ARCH:-}" ]]; then
  CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.[:space:]')"
fi
if [[ ! "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
  echo "Set CUDA_ARCH to your GPU architecture (RTX 4070 Ti: 89)." >&2
  exit 1
fi
export PATH="$CUDA_ROOT/bin:$(dirname "$CMAKE_BIN"):$PATH"
export LD_LIBRARY_PATH="$HEONGPU_ROOT/lib:$NTL_ROOT/lib:${LD_LIBRARY_PATH:-}"

"$CMAKE_BIN" \
  -S "$PROJECT_ROOT/native/heongpu" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_PREFIX_PATH="$HEONGPU_ROOT;$NTL_ROOT" \
  -DCMAKE_BUILD_RPATH="$HEONGPU_ROOT/lib;$NTL_ROOT/lib" \
  -DNTL_ROOT="$NTL_ROOT"

"$CMAKE_BIN" --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-4}"
echo "Built $BUILD_DIR/private_gram_gpu"
