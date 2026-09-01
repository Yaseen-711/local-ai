# Local AI Foundation

A modular, self-hosted AI infrastructure foundation designed to run locally and be reused across future applications and projects.

The goal is not to build a chatbot or replace an existing AI product.

The goal is to provide a stable local AI infrastructure layer so applications can use local models without being tightly coupled to a specific model format or inference runtime.

## Current architecture

Application
    ↓
AI Foundation
    ↓
Inference Adapter
    ↓
Inference Runtime
    ↓
Local Model
    ↓
GPU / CPU

The first implemented inference runtime is llama.cpp using GGUF models.

## Current verified stack

- Host OS: Ubuntu
- GPU: NVIDIA GeForce RTX 5070
- VRAM: 12 GB
- NVIDIA Driver: 570.211.01
- CUDA Toolkit: 12.8
- CUDA compiler: /usr/local/cuda-12.8/bin/nvcc
- Runtime: llama.cpp
- llama.cpp revision: 8887a48f050554f0ee59f56753860c061836b02d
- Build system: CMake + Ninja
- Build type: Release
- CUDA backend: enabled
- Initial model format: GGUF
- Verified model: Qwen3.5-9B-Q4_K_M

## Verified inference

The following configuration was successfully tested:

- GPU layer offloading: automatic
- Context size: 4096
- KV cache K: Q8_0
- KV cache V: Q8_0
- Batch size: 512
- Micro-batch size: 256
- Reasoning: disabled

Verified output:

Local inference is working.

## Repository layout

core/
    config/
    inference/
    models/

adapters/
    llama_cpp/

api/
configs/
docs/
examples/
models/
scripts/
tests/

## Documentation

- docs/SETUP.md - environment requirements and setup
- docs/REPRODUCE.md - rebuilding the verified runtime
- docs/MODELS.md - model storage and formats
- docs/INFERENCE.md - verified inference configuration

## Design principle

Applications should eventually interact with the AI Foundation through stable interfaces.

Inference runtimes and model formats should be replaceable.

The current implementation intentionally starts with a single verified path before introducing abstraction layers.
