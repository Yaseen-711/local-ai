# Reproducing the Sovereign Industrial AI Workbench

This document records the exact, verified procedure for rebuilding and validating the Sovereign Industrial AI Workbench on a clean, compatible Linux host.

---

## 1. Verified Hardware & Host Environment

* **Host Operating System**: Ubuntu 24.04 / 22.04 LTS (x86_64)
* **GPU**: NVIDIA GeForce RTX 5070 (12 GB VRAM)
* **NVIDIA Driver**: 570.211.01
* **CUDA Toolkit**: 12.8 (`/usr/local/cuda-12.8/bin/nvcc`)
* **Host Toolchain**:
  - Python 3.12 (`python3.12`, `python3.12-venv`)
  - GCC / G++ 13.3.0
  - CMake 3.28.3+
  - Ninja 1.11.1+
  - Docker Engine 24+
  - PostgreSQL client (`psql`, `createdb`)

---

## 2. Step-by-Step Reproduction

### Step 1: Clone Repository & Submodules
Clone the workbench repository and initialize the pinned `llama.cpp` submodule:
```bash
git clone https://github.com/Yaseen-711/local-ai.git local-ai
cd local-ai
git submodule update --init --recursive
```
*Pinned `llama.cpp` commit*: `8887a48f050554f0ee59f56753860c061836b02d`

---

### Step 2: Build Native llama.cpp with CUDA Acceleration
Compile the native `llama-server` binary using CMake and Ninja with CUDA 12.8:
```bash
./scripts/build_llama_cpp.sh
```

This invokes:
```bash
cmake -S adapters/llama_cpp \
  -B adapters/llama_cpp/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON

cmake --build adapters/llama_cpp/build --config Release
```

Verify the server binary exists and is executable:
```bash
test -x adapters/llama_cpp/build/bin/llama-server && echo "llama-server ready"
```

---

### Step 3: Python Environment & Dependencies
Create and activate a Python 3.12 virtual environment, then install the project package in editable mode:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e ".[dev]"
```

---

### Step 4: Model Placement & Air-Gap Configuration
Place the verified GGUF weights into `models/gguf/`:
* `models/gguf/Qwen3.5-9B-Q4_K_M.gguf` (5.3 GB)
* `models/gguf/Qwen3.5-0.8B-Q4_0.gguf` (563 MB)
* `models/gguf/Qwen3.5-9B.mmproj-q8_0.gguf` (624 MB)

Pre-cache Hugging Face embedding and reranking models into your local Hugging Face cache:
* `nomic-ai/nomic-embed-text-v1.5`
* `cross-encoder/ms-marco-MiniLM-L-6-v2`

Enforce air-gapped execution:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAG_OFFLINE_MODE=true
```

---

### Step 5: Database Setup (PostgreSQL with pgvector)
Launch a PostgreSQL container supporting the `pgvector` extension:
```bash
docker run -d \
  --name local-ai-pg \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  pgvector/pgvector:pg16
```

Create the two isolated databases and enable `pgvector` on the RAG database:
```bash
createdb -h localhost -U postgres local_ai
createdb -h localhost -U postgres local_ai_rag
psql -h localhost -U postgres -d local_ai_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Apply database migrations:
```bash
.venv/bin/alembic upgrade head
```

---

### Step 6: Launch Multi-Model Native Inference Router
Start the native `llama-server` router in the background:
```bash
./scripts/start_llama_server.sh
```

Verify that the parent router is healthy and serving both model slots:
```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models | jq .
```

---

### Step 7: Run Smoke Test
Verify basic token generation on `qwen3.5-9b`:
```bash
./scripts/run_smoke_test.sh
```
Expected output:
```text
Local inference is working.
```

---

### Step 8: Ingest Golden Test Pack into Sovereign RAG
Ingest the 3 synthetic corpus documents from `golden_test_pack/`:
```bash
.venv/bin/python -m rag.cli.ingest golden_test_pack/01_equipment_spec.pdf
.venv/bin/python -m rag.cli.ingest golden_test_pack/02_inspection_report.pdf
.venv/bin/python -m rag.cli.ingest golden_test_pack/03_operating_manual.pdf
```
*(Note: Do NOT ingest `04_direct_only_datasheet.pdf`; it is reserved for testing unindexed direct document analysis).*

---

### Step 9: Run Automated Verification Suites

1. **Unit Test Suite (520+ CPU tests)**:
   ```bash
   .venv/bin/pytest tests/unit/ -v
   ```
2. **Full Test Suite (537 tests)**:
   ```bash
   .venv/bin/pytest tests/ -rs
   ```
3. **End-to-End Golden Flow Validation (11 scenarios)**:
   ```bash
   .venv/bin/pytest tests/e2e/test_golden_flow_validation.py -v
   ```
   All 11 scenarios must report `PASSED`.

---

### Step 10: Start Workbench API
Launch the FastAPI REST application:
```bash
.venv/bin/uvicorn apps.api.app:app --host 0.0.0.0 --port 8000
```
API Documentation is available at: `http://localhost:8000/docs`
