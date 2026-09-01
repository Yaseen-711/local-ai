#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="$ROOT/models/gguf/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_SERVER="$ROOT/adapters/llama_cpp/build/bin/llama-server"

HOST="127.0.0.1"
PORT="8080"
MODEL_ALIAS="qwen3.5-9b"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found:"
    echo "  $MODEL"
    exit 1
fi

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found:"
    echo "  $LLAMA_SERVER"
    echo
    echo "Run:"
    echo "  ./scripts/build_llama_cpp.sh"
    exit 1
fi

echo "Starting llama-server..."
echo "Model: $MODEL_ALIAS"
echo "Host:  $HOST"
echo "Port:  $PORT"
echo

exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --alias "$MODEL_ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-layers auto \
    --ctx-size 4096 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --batch-size 512 \
    --ubatch-size 256 \
    --no-warmup \
    --reasoning off
