# Verified Inference Configuration

## Runtime

llama.cpp

## Executable

adapters/llama_cpp/build/bin/llama-cli

## Model

models/gguf/Qwen3.5-9B-Q4_K_M.gguf

## Verified command configuration

GPU layers: automatic
Context size: 4096
K cache: Q8_0
V cache: Q8_0
Batch size: 512
Micro-batch size: 256
Warmup: disabled
Reasoning: disabled

## Reasoning

The Qwen3.5 model was observed producing a visible thinking trace when reasoning was automatically enabled.

The verified configuration disables reasoning with:

--reasoning off

This allows the model to respond directly without generating the visible reasoning process.

Reasoning behavior should eventually be exposed as a configurable runtime option rather than being hardcoded into applications.

## Performance observed during smoke test

Approximate observed values:

Prompt processing: 353 tokens/second
Generation: 80 tokens/second

These values are machine- and configuration-dependent and should not be treated as universal benchmarks.
