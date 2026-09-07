# Operator Troubleshooting & Diagnostic Runbook

This guide provides practical diagnostic procedures and solutions for common operational issues encountered when running the Sovereign Industrial AI Workbench.

---

## 1. Inference Engine & VRAM Diagnostics

### 1.1 CUDA Out-of-Memory (OOM) Errors
* **Symptom**: `llama-server` logs `cudaMalloc failed: out of memory`, or child server processes crash unexpectedly during heavy workloads.
* **Root Cause**: Running multiple large models concurrently or processing large context windows exceeds the 12 GB VRAM capacity of the GPU.
* **Resolution**:
  1. **Switch to Single-Model Hot-Swap Mode**:
     In `scripts/start_llama_server.sh`, change `MODELS_MAX="2"` to `MODELS_MAX="1"`. This ensures only one model is loaded in VRAM at any given moment.
  2. **Verify KV Cache Quantization**:
     Ensure `configs/llama_models.ini` specifies quantized KV caches:
     ```ini
     cache-type-k = q8_0
     cache-type-v = q8_0
     ```
  3. **Force CPU Offloading for Embeddings**:
     If RAG embedding is competing with LLM inference, set the embedding device to CPU in environment variables:
     ```bash
     export RAG_EMBEDDING_DEVICE=cpu
     ```

---

### 1.2 Port 8080 Already in Use / Zombie Processes
* **Symptom**: `scripts/start_llama_server.sh` fails with:
  ```text
  failed to bind to 127.0.0.1:8080: Address already in use
  ```
* **Root Cause**: An earlier instance of `llama-server` or its child router processes did not terminate cleanly.
* **Resolution**:
  Find and kill the active processes:
  ```bash
  # Check listening processes
  lsof -i :8080

  # Terminate all running llama processes
  pkill -9 -f llama-server || true
  pkill -9 -f llama-cli || true

  # Relaunch the server
  ./scripts/start_llama_server.sh
  ```

---

### 1.3 Model Output Truncation
* **Symptom**: Generated text ends abruptly in the middle of a sentence, or `finish_reason` reports `"length"`.
* **Root Cause**: The response exceeded `max_tokens`, or the combined prompt and completion exceeded the 4096 context window.
* **Resolution**:
  - In `GenerationOptions`, increase `max_tokens` (e.g. from `512` to `1024` or `2048`).
  - In RAG workflows, reduce `top_n` from `5` to `3` to allow more token headroom for the final completion.

---

## 2. PostgreSQL & pgvector Diagnostics

### 2.1 Missing `vector` Extension
* **Symptom**: Database queries fail with:
  ```text
  psycopg.errors.UndefinedObject: type "vector" does not exist
  ```
* **Root Cause**: The PostgreSQL database was started without the `pgvector` extension installed, or the extension was not activated on `local_ai_rag`.
* **Resolution**:
  1. Confirm the container uses the official `pgvector` image:
     ```bash
     docker ps | grep pgvector
     ```
  2. Activate the extension manually:
     ```bash
     psql -h localhost -U postgres -d local_ai_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
     ```

---

### 2.2 PostgreSQL Connection Refused
* **Symptom**:
  ```text
  connection to server at "localhost" (127.0.0.1), port 5432 failed: Connection refused
  ```
* **Root Cause**: The database container is stopped or listening on a different interface.
* **Resolution**:
  ```bash
  # Check container status
  docker ps -a | grep local-ai-pg

  # Start container if stopped
  docker start local-ai-pg

  # Verify connectivity
  pg_isready -h localhost -p 5432 -U postgres
  ```

---

## 3. Docker Code Sandbox Diagnostics

### 3.1 Docker Daemon Permission Denied
* **Symptom**:
  ```text
  docker.errors.DockerException: Error while fetching server API version: 
  Permission denied: '/var/run/docker.sock'
  ```
* **Root Cause**: The current host user does not belong to the `docker` group.
* **Resolution**:
  ```bash
  sudo usermod -aG docker "$USER"
  newgrp docker
  # Verify access without sudo:
  docker info
  ```

---

### 3.2 Sandbox Execution Timeout
* **Symptom**: `code.workspace` returns an error with:
  ```text
  TimeoutError: Sandbox execution exceeded 60.0s timeout limit.
  ```
* **Root Cause**: The generated Python script contains an unintentional infinite loop or is computing a heavy numerical operation.
* **Resolution**:
  - The `code.verify_and_repair` capability automatically catches timeouts and feeds the traceback back to the LLM to rewrite the algorithm with lower complexity.
  - If a longer calculation is required, increase `default_timeout_seconds` in `configs/settings.toml` under `[workspace]`.

---

## 4. Decision Engine & Plan Validation Diagnostics

### 4.1 Plan Validation Rejection: `DAG cycle detected`
* **Symptom**: Goal execution fails during planning with:
  ```text
  PlanValidationError: Stage 3 failed: Cyclic dependency detected between tasks: step_2 -> step_3 -> step_2
  ```
* **Root Cause**: The `LLMPlanner` created circular dependencies between tasks.
* **Resolution**:
  - The `DecisionEngine` automatically retries planning up to 2 times with explicit cycle error feedback.
  - If persistent, verify that your goal description specifies a clear sequential outcome (e.g. *"First inspect the P&ID, then query the spec, and finally generate the workbook"*).

---

### 4.2 Capability Not Found: `CapabilityNotFoundError`
* **Symptom**:
  ```text
  CapabilityNotFoundError: Capability 'custom.tool' not found in registry.
  ```
* **Root Cause**: A task in the plan requested a capability identifier that was not registered in `CapabilityRegistry`.
* **Resolution**:
  - Verify that the capability is registered in `AppContext.create_base_capability_registry()` (`apps/context.py`).
  - Run the unit tests to verify registry discovery:
    ```bash
    .venv/bin/pytest tests/unit/test_capability_registry.py -v
    ```

---

## 5. Diagnostic Health Probe

When diagnosing an unhealthy workbench, run the diagnostic probe:

```bash
curl -s http://localhost:8000/telemetry/health | jq .
```

A healthy response should show all subsystems active:
```json
{
  "status": "healthy",
  "inference_server": "online",
  "database": "connected",
  "rag_database": "connected",
  "gpu_vram_used_mb": 7100,
  "gpu_vram_total_mb": 12288,
  "models_available": ["qwen3.5-9b", "qwen3.5-0.8b"]
}
```
