# Models

Models are intentionally stored separately from source code.

Large model files must not be committed to Git.

## Current storage structure

models/
└── gguf/

## Current verified model

models/gguf/Qwen3.5-9B-Q4_K_M.gguf

File size at verification:

5.3 GB

GGUF metadata verification reported:

Architecture: qwen35
Quantization: Q4_K Medium
Tensor count: 427

## Model policy

The AI Foundation should eventually support multiple model formats.

The model directory is organized by format rather than by inference runtime.

Possible future structure:

models/
├── gguf/
├── safetensors/
├── onnx/
└── other/

A model format should not automatically determine the public interface used by applications.

The adapter and runtime layer should handle model-specific loading.
