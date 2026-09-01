#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="$ROOT/models/gguf/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA="$ROOT/adapters/llama_cpp/build/bin/llama-cli"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found:"
    echo "  $MODEL"
    exit 1
fi

if [ ! -x "$LLAMA" ]; then
    echo "ERROR: llama-cli not found:"
    echo "  $LLAMA"
    echo
    echo "Run:"
    echo "  ./scripts/build_llama_cpp.sh"
    exit 1
fi

"$LLAMA" \
    -m "$MODEL" \
    --gpu-layers auto \
    --ctx-size 4096 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --batch-size 512 \
    --ubatch-size 256 \
    --no-warmup \
    --reasoning off \
    --single-turn \
    -n 64 \
    -p "Reply with exactly one short sentence: Local inference is working."
