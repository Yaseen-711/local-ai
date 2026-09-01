# Local AI Foundation — Setup

## Project location

The project repository is located at:

    ~/Projects/local-ai

## Purpose

This project is a reusable local AI foundation.

It provides a stable layer between applications and AI infrastructure so that models and inference runtimes can be replaced without requiring major changes to consumer applications.

The initial architecture is:

    Application
        ↓
    AI Foundation
        ↓
    Inference Runtime
        ↓
    Local Model
        ↓
    GPU / CPU

The first verified runtime is llama.cpp using GGUF models.

---

# Project structure

Current intended structure:

    local-ai/
    ├── core/
    │   ├── config/
    │   ├── inference/
    │   └── models/
    │
    ├── adapters/
    │   └── llama_cpp/
    │
    ├── api/
    ├── configs/
    ├── docs/
    ├── examples/
    │   └── simple_chat/
    ├── models/
    │   └── gguf/
    ├── scripts/
    └── tests/

The `core` directory will contain the project's own reusable abstractions.

The `adapters` directory contains integrations with external inference runtimes.

---

# Containerization

Docker is intentionally not used at the beginning.

The first goal is to establish and understand a working native inference path on the Ubuntu host:

    Ubuntu
        ↓
    NVIDIA Driver
        ↓
    CUDA Runtime
        ↓
    llama.cpp
        ↓
    GGUF Model
        ↓
    GPU Inference

This native path has now been successfully verified.

Containerization can later be introduced around the service layer.

Large model files should normally remain outside container images and be mounted or otherwise provided separately.

---

# Reproduction strategy

Every important environment-specific installation and configuration change should be documented in this repository.

The goal is that the repository can later be cloned onto another compatible machine and rebuilt from the documented instructions.

The source code, runtime revision, build configuration, scripts, and documentation should be version controlled.

Large model files should not be committed to Git.

---

# Verified native build environment

## Development tools

Verified versions:

    Git: 2.43.0
    GCC: 13.3.0
    G++: 13.3.0
    Python: 3.12.3
    CMake: 3.28.3
    Ninja: 1.11.1

---

# CUDA

The NVIDIA CUDA 12.8 compiler used for project builds is:

    /usr/local/cuda-12.8/bin/nvcc

Verified version:

    CUDA compilation tools, release 12.8, V12.8.93

The system also contains an older Ubuntu-packaged CUDA compiler at:

    /usr/bin/nvcc

This compiler was not modified during the setup.

CUDA projects in this repository should explicitly use the CUDA 12.8 compiler when configuring CMake builds.

---

# NVIDIA GPU

Verified hardware:

    GPU: NVIDIA GeForce RTX 5070
    VRAM: 12 GB

Verified NVIDIA software:

    Driver: 570.211.01
    CUDA version reported by nvidia-smi: 12.8

CUDA Toolkit 12.8 is installed at:

    /usr/local/cuda-12.8

---

# llama.cpp runtime

The first verified inference runtime is llama.cpp.

Source location:

    adapters/llama_cpp

Upstream repository:

    https://github.com/ggml-org/llama.cpp.git

Verified commit:

    8887a48f050554f0ee59f56753860c061836b02d

Verified repository description:

    b10736-7-g8887a48f0

The executable used for local inference is:

    adapters/llama_cpp/build/bin/llama-cli

---

# llama.cpp CUDA build configuration

The verified build configuration includes:

    Build type: Release
    Generator: Ninja
    CUDA enabled: ON
    CUDA compiler:
        /usr/local/cuda-12.8/bin/nvcc

Verified relevant CMake configuration:

    CMAKE_BUILD_TYPE=Release
    CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
    GGML_CUDA=ON
    GGML_CUDA_FA=ON
    GGML_CUDA_GRAPHS=ON

The official llama.cpp build documentation uses CMake and supports CUDA builds using:

    -DGGML_CUDA=ON

The exact compiler path is explicitly configured in this project because the system also contains an older CUDA compiler.

---

# Verified model

The currently verified local model is:

    models/gguf/Qwen3.5-9B-Q4_K_M.gguf

Verified file size:

    approximately 5.3 GB

Model format:

    GGUF

Quantization:

    Q4_K Medium

The initial model storage structure is:

    models/
    └── gguf/

Future model formats may include:

    models/
    ├── gguf/
    ├── safetensors/
    ├── onnx/
    └── other/

The model format should not determine the public application interface.

The inference adapter/runtime layer should handle format-specific loading.

---

# Verified local inference

The following inference configuration was successfully tested.

    GPU layers: automatic
    Context size: 4096
    KV cache K: Q8_0
    KV cache V: Q8_0
    Batch size: 512
    Micro-batch size: 256
    Warmup: disabled
    Reasoning: disabled

The verified smoke test command used:

    llama-cli
        model: Qwen3.5-9B-Q4_K_M.gguf
        GPU layers: automatic
        context: 4096
        reasoning: off

The model successfully returned:

    Local inference is working.

Observed approximate performance:

    Prompt processing: 353 tokens/second
    Generation: 80 tokens/second

These measurements are specific to this machine, model, runtime revision, and configuration.

They should not be treated as universal benchmarks.

---

# Reasoning configuration

The model was initially observed producing a visible thinking trace when reasoning was automatically enabled.

llama.cpp provides reasoning controls including:

    --reasoning on
    --reasoning off
    --reasoning auto

The verified smoke test uses:

    --reasoning off

This configuration causes the model to respond directly rather than generating a visible thinking process.

Reasoning should eventually become a configurable runtime option exposed by the AI Foundation.

Applications should not need to know how a particular model implements reasoning.

---

# Current verified inference path

The currently working path is:

    User Prompt
        ↓
    llama-cli
        ↓
    llama.cpp
        ↓
    CUDA backend
        ↓
    RTX 5070
        ↓
    GGUF model
        ↓
    Generated response

This confirms that the following components are working together:

    Ubuntu host
    NVIDIA driver
    CUDA 12.8
    llama.cpp CUDA build
    RTX 5070 GPU
    GGUF model loading
    GPU-accelerated local inference

---

# Current project status

The first native local inference path is now verified.

Completed:

    [x] Ubuntu development environment
    [x] NVIDIA driver verification
    [x] CUDA 12.8 verification
    [x] CUDA compiler verification
    [x] llama.cpp source checkout
    [x] CUDA-enabled llama.cpp build
    [x] GGUF model placement
    [x] GPU inference smoke test
    [x] Q8_0 KV cache configuration
    [x] Reasoning disabled configuration
    [x] Performance observation

Next:

    [ ] Create reproducible project scripts
    [ ] Complete repository documentation
    [ ] Configure Git tracking and ignores
    [ ] Decide how llama.cpp is tracked by the parent repository
    [ ] Create the first reproducible project commit
    [ ] Verify rebuilding from documented commands
    [ ] Introduce llama-server
    [ ] Create a stable local API
    [ ] Design the AI Foundation interface
    [ ] Add additional runtimes and model formats behind adapters

---

# Important design rule

The current llama.cpp + GGUF setup is the first verified implementation.

It is not intended to become the permanent architecture of the entire project.

The eventual goal is:

    Application
        ↓
    Stable AI Foundation Interface
        ↓
    Runtime / Model Adapter
        ↓
    llama.cpp / other runtime
        ↓
    GGUF / SafeTensors / other format
        ↓
    GPU / CPU / future hardware

Applications should eventually depend on the AI Foundation interface rather than directly depending on llama.cpp or a specific model format.

---

# Reproducible project state

The project has now been converted from an exploratory local setup into a reproducible Git repository structure.

The parent repository tracks llama.cpp as a Git submodule.

Verified llama.cpp commit:

    8887a48f050554f0ee59f56753860c061836b02d

The upstream repository is:

    https://github.com/ggml-org/llama.cpp.git

The submodule is intentionally pinned to the verified commit so that the same source revision can be reproduced later.

The llama.cpp working tree is therefore not copied into the Local AI Foundation repository as ordinary source files.

Instead:

    Local AI Foundation repository
            ↓
    Git submodule reference
            ↓
    llama.cpp upstream repository
            ↓
    pinned verified commit

## Reproducible scripts

Two project scripts are available.

### Build llama.cpp

    ./scripts/build_llama_cpp.sh

This configures and builds llama.cpp using:

    Build type: Release
    Generator: Ninja
    CUDA compiler: /usr/local/cuda-12.8/bin/nvcc
    CUDA backend: enabled

### Run the smoke test

    ./scripts/run_smoke_test.sh

This runs the verified inference configuration using:

    GPU layers: automatic
    Context size: 4096
    K cache: Q8_0
    V cache: Q8_0
    Batch size: 512
    Micro-batch size: 256
    Warmup: disabled
    Reasoning: disabled

Expected output:

    Local inference is working.

## Model storage policy

Large model files are not committed to Git.

The verified model is stored locally at:

    models/gguf/Qwen3.5-9B-Q4_K_M.gguf

The Git repository preserves the model directory structure using:

    models/.gitkeep
    models/gguf/.gitkeep

Model formats are ignored through .gitignore.

## Current project status

Completed:

    [x] Ubuntu development environment
    [x] NVIDIA driver verification
    [x] CUDA 12.8 verification
    [x] CUDA compiler verification
    [x] llama.cpp source verification
    [x] llama.cpp pinned Git submodule
    [x] CUDA-enabled llama.cpp build
    [x] GGUF model placement
    [x] GPU inference smoke test
    [x] Q8_0 KV cache configuration
    [x] Reasoning disabled configuration
    [x] Performance observation
    [x] Reproducible build script
    [x] Reproducible inference smoke test script
    [x] Repository documentation
    [x] Model Git ignore policy
    [x] llama.cpp source revision pinning

Next:

    [ ] Verify clean rebuild from project scripts
    [ ] Create first reproducible Git commit
    [ ] Introduce llama-server
    [ ] Verify persistent local inference service
    [ ] Create a stable local API
    [ ] Design the AI Foundation interface
    [ ] Add additional runtimes behind adapters
    [ ] Add additional model formats behind adapters


---

# Reproducible project checkpoint

The first verified local inference implementation has now been converted into a reproducible project checkpoint.

## Repository tracking

The parent project repository tracks:

- Project documentation
- Build and verification scripts
- Directory structure
- llama.cpp as a Git submodule

The llama.cpp submodule is pinned to:

    8887a48f050554f0ee59f56753860c061836b02d

Upstream repository:

    https://github.com/ggml-org/llama.cpp.git

## Model tracking policy

The verified model is intentionally not tracked by Git.

Verified model location:

    models/gguf/Qwen3.5-9B-Q4_K_M.gguf

Large model files are ignored through `.gitignore`.

The directory structure is preserved using `.gitkeep` files.

## Reproducibility scripts

The project currently provides:

    scripts/build_llama_cpp.sh

for configuring and building llama.cpp with CUDA support.

The project also provides:

    scripts/run_smoke_test.sh

for verifying the documented local inference path.

## Verified scripted smoke test

The smoke test was successfully executed through:

    ./scripts/run_smoke_test.sh

The verified response was:

    Local inference is working.

Observed performance during the scripted verification:

    Prompt processing: approximately 338.6 tokens/second
    Generation: approximately 81.5 tokens/second

These measurements are environment-dependent and are recorded only as an observation of this verified configuration.

## Current checkpoint

At this checkpoint, the following path is verified:

    User Prompt
        ↓
    Project smoke test script
        ↓
    llama-cli
        ↓
    llama.cpp
        ↓
    CUDA backend
        ↓
    NVIDIA RTX 5070
        ↓
    GGUF model
        ↓
    Generated response

The first reproducible local inference foundation is now complete.

The next stage is not additional model testing.

The next stage is to introduce llama-server and establish a persistent local inference service before building the higher-level AI Foundation interface.

