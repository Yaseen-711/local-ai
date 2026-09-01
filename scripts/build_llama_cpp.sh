#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT/adapters/llama_cpp"
BUILD_DIR="$LLAMA_DIR/build"
CUDA_COMPILER="/usr/local/cuda-12.8/bin/nvcc"

if [ ! -x "$CUDA_COMPILER" ]; then
    echo "ERROR: CUDA compiler not found:"
    echo "  $CUDA_COMPILER"
    exit 1
fi

if [ ! -d "$LLAMA_DIR" ]; then
    echo "ERROR: llama.cpp source directory not found:"
    echo "  $LLAMA_DIR"
    exit 1
fi

if ! git -C "$LLAMA_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: llama.cpp is not a valid Git working tree:"
    echo "  $LLAMA_DIR"
    exit 1
fi

cmake -S "$LLAMA_DIR" \
    -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DGGML_CUDA=ON

cmake --build "$BUILD_DIR" --config Release

echo
echo "llama.cpp build completed successfully."
