# Reproducing the Verified Local Inference Runtime

This document records the verified native inference setup.

The objective is to reproduce the working llama.cpp + CUDA + GGUF inference path on a compatible machine.

## Verified llama.cpp source

Upstream repository:

https://github.com/ggml-org/llama.cpp.git

Verified commit:

8887a48f050554f0ee59f56753860c061836b02d

Repository description at verification:

b10736-7-g8887a48f0

## Required environment

### GPU

NVIDIA GeForce RTX 5070
12 GB VRAM

### NVIDIA software

Driver: 570.211.01
CUDA Toolkit: 12.8
CUDA compiler: /usr/local/cuda-12.8/bin/nvcc

### Build tools

- Git
- GCC
- G++
- Python 3
- CMake
- Ninja

## Clone llama.cpp

From the project root:

git clone https://github.com/ggml-org/llama.cpp.git adapters/llama_cpp

Pin the verified revision:

git -C adapters/llama_cpp checkout 8887a48f050554f0ee59f56753860c061836b02d

## Configure

cmake -S adapters/llama_cpp \
  -B adapters/llama_cpp/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DGGML_CUDA=ON

## Build

cmake --build adapters/llama_cpp/build --config Release

## Verified build characteristics

CMAKE_BUILD_TYPE = Release
CMAKE_CUDA_COMPILER = /usr/local/cuda-12.8/bin/nvcc
CMAKE_GENERATOR = Ninja
GGML_CUDA = ON
GGML_CUDA_GRAPHS = ON
GGML_CUDA_FA = ON

## Smoke test

Place a GGUF model in:

models/gguf/

Then run:

./scripts/run_smoke_test.sh

A successful test should produce:

Local inference is working.
