# Sovereign Industrial AI Workbench — System Setup & Configuration

This guide provides complete instructions for provisioning, configuring, and verifying the host environment for the Sovereign Industrial AI Workbench.

---

## 1. System Requirements

### Hardware Specifications

| Component | Minimum Specification | Verified Production Baseline |
|---|---|---|
| **GPU** | NVIDIA GPU with $\ge$ 12 GB VRAM | NVIDIA GeForce RTX 5070 (12 GB VRAM) |
| **System RAM** | 32 GB DDR4 / DDR5 | 32 GB DDR5 |
| **CPU** | 8 Cores (x86_64) | Intel Core i7 / AMD Ryzen 7 (16 vCPUs) |
| **Storage** | 100 GB NVMe SSD | Fast PCIe Gen4 NVMe |
| **Network** | Air-gapped / Localhost only | Zero external internet access |

### Operating System & Driver Stack

* **Host OS**: Ubuntu 24.04 LTS / 22.04 LTS
* **NVIDIA Driver**: 570.211.01 or later
* **CUDA Toolkit**: 12.8 (`/usr/local/cuda-12.8/bin/nvcc`)
* **Container Runtime**: Docker Engine 24.0+
* **Python Runtime**: Python 3.12.3

---

## 2. Host Toolchain & Driver Setup

### 2.1 NVIDIA Driver & CUDA 12.8
Confirm the GPU driver and CUDA 12.8 compiler:
```bash
nvidia-smi
/usr/local/cuda-12.8/bin/nvcc --version
```
Ensure `/usr/local/cuda-12.8/bin` is present in your `PATH` or explicitly referenced by CMake during builds.

### 2.2 Host Package Installation
Install standard compilation, build, and development packages:
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    docker.io \
    postgresql-client \
    curl \
    jq
```

### 2.3 Docker Engine Permissions
Ensure the active user can interact with the Docker daemon for sandboxed code execution without requiring `sudo`:
```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker info
```

Pull the verified sandbox image:
```bash
docker pull python:3.12-slim
```

---

## 3. Python Virtual Environment Setup

From the repository root (`~/Projects/local-ai`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e ".[dev]"
```

---

## 4. Configuration Files

The workbench is managed through declarative configuration files:

### 4.1 `configs/settings.toml`
Defines application, database, parsing, and sandbox limits:
```toml
[foundation]
environment = "development"
models_dir = "models"
configs_dir = "configs/models"

[providers.llama_cpp]
base_url = "http://127.0.0.1:8080"
timeout_seconds = 60
default_alias = "qwen3.5-9b"

[database]
url = "postgresql+psycopg://postgres:postgres@localhost:5432/local_ai"
pool_size = 5
max_overflow = 10
echo = false

[document]
default_parser = "docling"
enable_ocr = true
ocr_engine = "rapidocr"
enable_tables = true
enable_figures = true
enable_formulae = true

[artifact]
output_dir = "artifacts"
enable_xlsx = true
enable_docx = true
enable_pdf = true

[workspace]
default_executor = "docker"
docker_image = "python:3.12-slim"
cpu_limit = 2.0
mem_limit = "2g"
network_mode = "none"
default_timeout_seconds = 60.0
base_workspaces_dir = ".workspaces"
```

### 4.2 `configs/llama_models.ini`
Configures multi-model routing for `llama-server`:
```ini
[*]
ctx-size = 4096
batch-size = 512
ubatch-size = 256
gpu-layers = auto
cache-type-k = q8_0
cache-type-v = q8_0
no-warmup = true
reasoning = off

[qwen3.5-9b]
model = models/gguf/Qwen3.5-9B-Q4_K_M.gguf
mmproj = models/gguf/Qwen3.5-9B.mmproj-q8_0.gguf

[qwen3.5-0.8b]
model = models/gguf/Qwen3.5-0.8B-Q4_0.gguf
```

---

## 5. PostgreSQL & pgvector Database Setup

The workbench strictly separates orchestration state from RAG vector storage.

### 5.1 Launch Database Container
```bash
docker run -d \
  --name local-ai-pg \
  --restart unless-stopped \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  pgvector/pgvector:pg16
```

### 5.2 Create Databases & Enable Extensions
```bash
createdb -h localhost -U postgres local_ai
createdb -h localhost -U postgres local_ai_rag
psql -h localhost -U postgres -d local_ai_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5.3 Apply Schema Migrations
```bash
.venv/bin/alembic upgrade head
```

---

## 6. Air-Gap & Offline Configuration

To guarantee air-gap compliance and eliminate network egress, set the following environment variables in your shell profile or deployment unit:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAG_OFFLINE_MODE=true
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/local_ai"
export RAG_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/local_ai_rag"
```

---

## 7. Operational Health Checks

Verify each subsystem sequentially:

### 1. Check Inference Engine
```bash
./scripts/start_llama_server.sh &
curl -s http://127.0.0.1:8080/health
```

### 2. Verify Smoke Test
```bash
./scripts/run_smoke_test.sh
```
Expected output: `Local inference is working.`

### 3. Verify PostgreSQL Connectivity
```bash
psql -h localhost -U postgres -d local_ai -c "SELECT 1;"
psql -h localhost -U postgres -d local_ai_rag -c "SELECT 1;"
```

### 4. Verify Docker Sandbox Isolation
```bash
docker run --rm --network none --memory 2g python:3.12-slim python3 -c "print('Sandbox functional')"
```

### 5. Start Workbench API & Check Telemetry
```bash
.venv/bin/uvicorn apps.api.app:app --host 0.0.0.0 --port 8000 &
curl -s http://localhost:8000/telemetry/health | jq .
```
Expected response:
```json
{
  "status": "healthy",
  "inference_server": "online",
  "database": "connected",
  "rag_database": "connected",
  "models_available": ["qwen3.5-9b", "qwen3.5-0.8b"]
}
```

---

## 8. Reproducing on macOS / Apple Silicon (MacBook)

The Sovereign Industrial AI Workbench can be reproduced on modern Apple Silicon MacBooks (M1 / M2 / M3 / M4) using Apple's unified memory and Metal acceleration instead of NVIDIA CUDA.

### 8.1 Hardware & Memory Guidelines
* **Architecture**: Apple Silicon (`arm64` - M1, M2, M3, or M4 Pro/Max/Ultra recommended).
* **Unified Memory (RAM)**:
  - **Minimum**: 16 GB unified memory (can host `qwen3.5-9b` with small context, but memory pressure will be tight during concurrent RAG indexing).
  - **Recommended**: 32 GB or 36 GB+ unified memory. Apple Silicon shares system memory dynamically between CPU and GPU cores, easily accommodating `qwen3.5-9b` (~5.3 GB), `qwen3.5-0.8b` (~560 MB), vision projection, and embeddings simultaneously with zero PCIe bandwidth bottleneck.

### 8.2 Host Prerequisites via Homebrew
Install the required development toolchain using Homebrew:
```bash
# 1. Install Xcode command line developer tools
xcode-select --install

# 2. Install build tools, Python 3.12, and PostgreSQL client
brew install cmake ninja python@3.12 libpq
```

Make sure Python 3.12 and `libpq` binaries are available in your path:
```bash
export PATH="/opt/homebrew/opt/libpq/bin:/opt/homebrew/opt/python@3.12/libexec/bin:$PATH"
```

### 8.3 Building `llama.cpp` with Apple Metal
On macOS, replace the CUDA backend (`-DGGML_CUDA=ON`) with Apple Metal (`-DGGML_METAL=ON`). Metal shaders compile directly to run on Apple Silicon GPU cores.

From the repository root:
```bash
cmake -S adapters/llama_cpp \
  -B adapters/llama_cpp/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON

cmake --build adapters/llama_cpp/build --config Release
```

Verify the native Metal build:
```bash
./adapters/llama_cpp/build/bin/llama-cli --version
```
The output should report `Metal` support enabled.

### 8.4 Python Virtual Environment on macOS
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e ".[dev]"
```

*Note on PyTorch*: On Apple Silicon, PyTorch automatically includes MPS (Metal Performance Shaders) backend acceleration for tensor operations and embeddings (`device="mps"` or `device="cpu"`).

### 8.5 Docker for Sandboxed Code Execution
The `code.workspace` capability executes Python calculations inside an isolated container.
* Install **Docker Desktop** or **OrbStack** for macOS.
* Ensure the Docker daemon is running:
  ```bash
  docker pull python:3.12-slim
  ```
  Docker Desktop on Apple Silicon seamlessly runs `python:3.12-slim` as native `linux/arm64`.

### 8.6 Database Setup
Run the `pgvector` container via Docker Desktop (which provides native ARM64 container support):
```bash
docker run -d \
  --name local-ai-pg \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  pgvector/pgvector:pg16

createdb -h localhost -U postgres local_ai
createdb -h localhost -U postgres local_ai_rag
psql -h localhost -U postgres -d local_ai_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"

.venv/bin/alembic upgrade head
```

### 8.7 Launching & Verifying on macOS
1. **Start the Multi-Model Router**:
   ```bash
   ./scripts/start_llama_server.sh
   ```
   Metal will automatically manage layer offloading into Apple unified memory.
2. **Run Inference Smoke Test**:
   ```bash
   ./scripts/run_smoke_test.sh
   ```
3. **Run Test Suites**:
   ```bash
   .venv/bin/pytest tests/unit/ -v
   ```

