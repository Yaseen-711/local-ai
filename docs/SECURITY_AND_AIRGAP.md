# Sovereignty, Air-Gap & Security Architecture

The Sovereign Industrial AI Workbench is engineered for mission-critical industrial environments (such as refinery operations, process engineering, and pipeline integrity) where data confidentiality, intellectual property protection, and system sovereignty are paramount.

---

## 1. Sovereignty Guarantees

1. **Zero External Network Egress**: The workbench never calls external cloud APIs (e.g. OpenAI, Anthropic, or Hugging Face Hub). All inference, embedding, parsing, and execution occur strictly on localhost.
2. **Zero Telemetry / Phone-Home**: There are no background analytics, tracking beacons, or crash reporters.
3. **Data Residency**: Documents, vector embeddings, and generated engineering artifacts reside exclusively on host-controlled NVMe storage.

---

## 2. Air-Gap Enforcement Mechanisms

### 2.1 Offline Environment Enforcement
The runtime enforces offline execution across all underlying Python libraries:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAG_OFFLINE_MODE=true
```
Attempting to connect to external model repositories triggers an immediate deterministic failure rather than hanging or attempting network fallback.

### 2.2 Localhost-Only Service Bindings
Internal infrastructure services bind strictly to loopback interfaces:
* `llama-server`: Bound to `127.0.0.1:8080`
* `PostgreSQL`: Bound to `127.0.0.1:5432`
* `FastAPI API`: Configurable, defaults to `127.0.0.1:8000`

---

## 3. Sandboxed Code Execution Isolation

The `code.workspace` and `code.verify_and_repair` capabilities execute Python code to perform engineering verification and calculations. To prevent arbitrary code execution vulnerabilities, the environment enforces strict container sandboxing:

```text
Host Operating System
  │
  └── Docker Daemon (`/var/run/docker.sock`)
        │
        └── Ephemeral Container (`python:3.12-slim`)
              ├── Security Profile: --network none (ZERO socket access)
              ├── Resource Cap:     --memory 2g
              ├── Compute Cap:      --cpus 2.0
              ├── Execution Timeout: 60.0 seconds
              └── Workspace Mount:  Isolated ephemeral directory (.workspaces/<uuid>)
```

### Sandbox Security Invariants

| Guardrail | Enforcement Mechanism | Security Impact |
|---|---|---|
| **Network Isolation** | `--network none` | Prevents data exfiltration, socket listeners, and reverse shells. |
| **Memory Limitation** | `--memory 2g` | Prevents denial-of-service via memory exhaustion (OOM). |
| **Compute Quota** | `--cpus 2.0` | Prevents CPU starvation of the host machine and inference server. |
| **Execution Timeout** | `SIGKILL` after 60s | Halts malicious or accidental infinite loops (`while True`). |
| **Ephemeral File System** | Per-task directory | Scratch files are isolated per task and purged after execution. |
| **VRAM Protection** | No GPU passthrough | Containers cannot access GPU memory or interfere with inference models. |

---

## 4. Cryptographic Artifact Integrity

Whenever the workbench generates an engineering artifact (Excel workbook, Word document, PowerPoint presentation, or PDF report):

1. **SHA-256 Hashing**: The binary payload is hashed immediately upon creation:
   $$\text{Hash} = \text{SHA256}(\text{Artifact Bytes})$$
2. **Audit Verification**: The hash is returned in the API response and stored in the database.
3. **Download Header**: When downloaded via `GET /api/v1/artifacts/{id}/download`, the response includes:
   ```http
   X-Artifact-Hash: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   ```
   This guarantees that industrial deliverables have not been altered or tampered with between generation and engineering review.

---

## 5. Thought Sanitization & Leakage Prevention

The internal reasoning models can produce internal chain-of-thought tokens during complex planning and validation.

To protect system prompts and prevent unverified thought leakage:
* **Event Bus Sanitization**: The in-memory SSE event bus (`apps/api/events.py`) enforces strict filtering via `DISALLOWED_EVENT_TYPES` and `DISALLOWED_DATA_KEYS`.
* **Stripped Keys**: Any dictionary keys matching `thought`, `thinking`, or `chain_of_thought` are recursively stripped before events are streamed to clients.
* **Public Boundary**: Clients receive verified operational facts (`task.started`, `task.completed`, `artifact.created`), never unconstrained internal thoughts.

---

## 6. Database Separation & Least Privilege

* **Dual Database Boundary**:
  - `local_ai`: Application persistence, user goals, and task plans.
  - `local_ai_rag`: Raw vector embeddings, document text chunks, and metadata.
* **SQL Injection Prevention**: All queries to PostgreSQL are executed via parameterized SQLAlchemy and psycopg statements; raw string interpolation in SQL is strictly prohibited.
