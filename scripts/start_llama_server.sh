#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PRESET_CONFIG="$ROOT/configs/llama_models.ini"
LLAMA_SERVER="$ROOT/adapters/llama_cpp/build/bin/llama-server"

HOST="127.0.0.1"
PORT="8080"
MODELS_MAX="2"

if [ ! -f "$PRESET_CONFIG" ]; then
    echo "ERROR: Presets config not found:"
    echo "  $PRESET_CONFIG"
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

echo "Starting llama-server in router mode..."
echo "Presets:    $PRESET_CONFIG"
echo "Models Max: $MODELS_MAX"
echo "Host:       $HOST"
echo "Port:       $PORT"
echo

cd "$ROOT"

exec "$LLAMA_SERVER" \
    --models-preset "$PRESET_CONFIG" \
    --models-max "$MODELS_MAX" \
    --host "$HOST" \
    --port "$PORT"
