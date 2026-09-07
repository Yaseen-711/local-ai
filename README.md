# Sovereign Industrial AI Workbench (MRPL Local AI Foundation)

A sovereign, air-gapped, local AI engineering workbench and infrastructure foundation. Built specifically for industrial and refinery environments to provide automated P&ID analysis, technical document understanding, grounded RAG queries, sandboxed code execution, and multi-format engineering artifact generation—running completely on local hardware with zero external cloud egress.

---

## Key Capabilities

| Capability | Identifier | Description | Runtime / Engine |
|---|---|---|---|
| **Sovereign RAG** | `retrieval.rag` | Multi-document hybrid vector search, cross-encoder reranking, source-cited grounded QA, and negative out-of-corpus refusal | PostgreSQL (`local_ai_rag`) + pgvector, Nomic v1.5, MiniLM Cross-Encoder |
| **Vision Inspection** | `vision.inspect` | Visual inspection of Piping & Instrumentation Diagrams (P&IDs), process flow diagrams, and technical drawings | Qwen3.5-9B + Qwen3.5-9B.mmproj-q8_0 (multimodal projector) |
| **Document Understanding** | `document.understand` | Structured extraction of tables, key-value pairs, metadata, and text from complex industrial PDFs | Docling / PyMuPDF fallback parser |
| **Sandboxed Code Execution** | `code.workspace` | Isolated Python calculation and execution sandbox with strict resource limits and no network access | Docker (`python:3.12-slim`), CPU/memory limits, ephemeral volumes |
| **Automated Code Repair** | `code.verify_and_repair` | Multi-turn calculation verification, unit test generation, and automated code repair loop | Foundation Core + Workspace sandbox |
| **Artifact Generation** | `artifact.generate` | Deterministic generation of audit-ready engineering workbooks, technical approval memos, and slide decks | OpenPyXL (XLSX), python-docx (DOCX), python-pptx (PPTX), ReportLab (PDF) |
| **Text Analysis** | `workflow.text_analysis` | Single-pass and two-pass factual extraction, synthesis, and structured JSON output | Foundation Core (`qwen3.5-9b`) |
| **Bounded Agent** | `agent.pydantic_ai` | Autonomous multi-step tool execution with strict authorization policies and budget limits | PydanticAI model adapter + CapabilityToolAdapter |

---

## System Architecture

```text
                                  User / Client Layer
           (HTTP REST API / SSE Event Stream / CLI / Golden E2E Test Suite)
                                          │
                                          ▼
                      FastAPI Application (`apps.api.app`)
           ├── /goals        (Multi-step goal submission, planning, execution)
           ├── /direct       (Direct single-capability execution bypass)
           ├── /rag          (Document ingestion, hybrid search, grounded QA)
           ├── /files        (Document & diagram upload / staging)
           ├── /artifacts    (Download generated XLSX, DOCX, PPTX, PDF)
           └── /telemetry    (System health, GPU VRAM status, execution metrics)
                                          │
                                          ▼
                     Composition Root (`apps.AppContext`)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
      Decision & Planning Layer                     Execution & Orchestration
  ├── StagedEscalationRouter (4-Stage Tier)     ├── GoalOrchestrator (State machine)
  │    Stage 1: Deterministic regex rules       ├── InProcessPlanRunner (DAG scheduler)
  │    Stage 2: Aurelio Semantic Router         ├── PlanValidator (Deterministic 4-stage)
  │    Stage 3: Fast 0.8B Classifier            └── PostgresOrchestrationRepository
  │    Stage 4: Reasoning 9B Classifier              (Persists to `local_ai` DB)
  ├── DecisionEngine (Direct vs Plan)
  └── LLMPlanner (DAG plan generation)
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                         Capability Registry (`orchestration/`)
   [retrieval.rag | vision.inspect | document.understand | code.workspace | ...]
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          Sovereign RAG Subsystem                       Foundation Core
    ├── PostgreSQL `local_ai_rag` (pgvector)     ├── ModelRegistry (TOML configs)
    ├── Nomic-embed-text-v1.5 (768d vectors)     ├── ProviderManager (Thread-safe dispatch)
    └── Cross-Encoder ms-marco-MiniLM-L-6-v2    └── InferenceConnector (Typed protocol)
                                                                │
                                                                ▼
                                                       LlamaCppProvider
                                                                │  HTTP (127.0.0.1:8080)
                                                                ▼
                                                llama-server (Native Router Mode)
                                                ├── Slot 1: Qwen3.5-9B + mmproj (Vision/LLM)
                                                └── Slot 2: Qwen3.5-0.8B (Fast Routing)
                                                                │
                                                                ▼
                                                Hardware: NVIDIA RTX 5070 (12 GB VRAM)
```

---

## Verified Hardware & Runtime Stack

- **Host OS**: Ubuntu Linux (x86_64)
- **GPU**: NVIDIA GeForce RTX 5070 (12 GB VRAM)
- **NVIDIA Driver**: 570.211.01
- **CUDA Toolkit**: 12.8 (`/usr/local/cuda-12.8/bin/nvcc`)
- **Inference Server**: `llama-server` (pinned to commit `8887a48f`, CUDA backend, FlashAttention, CUDA Graphs)
- **Router Configuration**: Multi-model preset router (`configs/llama_models.ini`) serving:
  - `qwen3.5-9b` (Q4_K_M quantization, 4096 context, Q8_0 KV cache, multimodal projector `mmproj-q8_0`)
  - `qwen3.5-0.8b` (Q4_0 quantization, 4096 context, Q8_0 KV cache for fast classification)
- **Databases**:
  - `local_ai`: Application persistence (goals, plans, tasks, execution traces)
  - `local_ai_rag`: Dedicated RAG database with `pgvector` extension

---

## Repository Layout

```text
local-ai/
├── apps/                        # Application composition root and HTTP API
│   ├── api/                     # FastAPI app, routers, schemas, SSE event bus
│   └── context.py               # AppContext process-level composition root
├── configs/                     # System and model configuration files
│   ├── llama_models.ini         # Multi-model presets for llama-server router
│   ├── settings.toml            # Foundation, database, workspace, artifact settings
│   └── models/                  # TOML declarations for registered models
├── connectors/                  # Typed structural capability connectors
├── core/                        # Foundation Core (registry, contracts, providers)
├── docs/                        # Comprehensive technical documentation
│   ├── ARCHITECTURE.md          # Multi-layer architecture and data contracts
│   ├── INFERENCE.md             # Multi-model router and inference configuration
│   ├── MODELS.md                # Model specifications, formats, and offline storage
│   ├── REPRODUCE.md             # Complete reproduction instructions
│   └── SETUP.md                 # Environment setup, dependencies, and verification
├── golden_test_pack/            # Synthetic industrial test pack (PDFs, P&ID PNG)
├── models/                      # Local model weights directory (GGUF files)
├── orchestration/               # Orchestrator, DecisionEngine, Planner, Capabilities
│   ├── capabilities/            # Capability protocol and 8 built-in implementations
│   ├── decision/                # DecisionEngine (intent routing vs planning)
│   ├── execution/               # InProcessPlanRunner DAG execution engine
│   ├── persistence/             # PostgreSQL orchestration repository
│   ├── planning/                # LLMPlanner for DAG plan generation
│   ├── routing/                 # StagedEscalationRouter (4-stage escalation)
│   └── validation/              # Deterministic 4-stage PlanValidator
├── rag/                         # Sovereign RAG subsystem (pgvector, Nomic, Cross-Encoder)
│   ├── chunking/                # Structural text and markdown chunkers
│   ├── embedding/               # Nomic Embed Text v1.5 embedder (768d)
│   ├── indexing/                # Database indexing pipeline
│   ├── ingestion/               # Document ingestion pipeline
│   ├── reranking/               # Joint cross-attention reranker (MiniLM)
│   └── storage/                 # DatabaseManager and pgvector schema
├── scripts/                     # Build, server management, and smoke test scripts
├── tests/                       # Comprehensive test suite (537 tests passing)
│   ├── e2e/                     # End-to-end golden flow validation suite
│   ├── integration/             # Integration tests for server and database
│   └── unit/                    # Fast isolated CPU unit tests
└── workflows/                   # Domain workflows (text analysis, code repair)
```

---

## Golden Test Pack & Verification

The repository includes a dedicated synthetic industrial dataset in `golden_test_pack/`:
- `01_equipment_spec.pdf`: Technical specifications for valve `FV-201A` (Design pressure 45.0 barg, 316L SS).
- `02_inspection_report.pdf`: Routine inspection report (0.12 mm corrosion, routine maintenance).
- `03_operating_manual.pdf`: Operating conditions (Normal operating pressure 38.0 barg).
- `04_direct_only_datasheet.pdf`: Unindexed direct datasheet for heat exchanger `HX-104` (42.5 barg).
- `05_pid_direct_input.png`: High-resolution Piping & Instrumentation Diagram showing `FV-201A`.
- `TEST_GUIDE.md`: Specification of the 11 golden validation scenarios.

All 11 scenarios are automated and verified in `tests/e2e/test_golden_flow_validation.py`:
1. **RAG-only**: Single-document factual query (`FV-201A` design pressure & material).
2. **RAG cross-document**: Joint synthesis across spec, inspection, and operating manual.
3. **Direct PDF ONLY**: Direct single-document QA without corpus ingestion (`HX-104`).
4. **Image + RAG**: Multimodal P&ID tag identification combined with RAG specification lookup.
5. **RAG -> XLSX**: Engineering calculation workbook generation with formulas.
6. **RAG -> PDF/DOCX**: Technical approval note generation from grounded evidence.
7. **RAG -> PPTX**: Executive summary presentation generation.
8. **Code Verification**: Sandboxed calculation executed inside isolated Docker container.
9. **Multi-capability Goal**: Natural language request composed across Vision, RAG, Code, and Artifact generation.
10. **Negative Grounding**: Explicit refusal for non-existent tag `ZX-999` with zero hallucination.
11. **Conflict Handling**: Detection and surfacing of contradictory document values.

---

## Quick Start

### 1. Start Inference Server
```bash
./scripts/start_llama_server.sh
```
Starts `llama-server` in router mode on port `8080`, hosting `qwen3.5-9b` (with vision projector) and `qwen3.5-0.8b`.

### 2. Verify Inference Smoke Test
```bash
./scripts/run_smoke_test.sh
```

### 3. Start PostgreSQL Containers
Ensure PostgreSQL with `pgvector` is running for `local_ai` and `local_ai_rag`:
```bash
docker run -d --name local-ai-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
```

### 4. Run Full Test Suite
```bash
.venv/bin/pytest tests/ -rs
```
Expected result: **537 passed** (or cleanly skipped if specific live services are offline).

### 5. Launch the Workbench API
```bash
.venv/bin/uvicorn apps.api.app:app --host 0.0.0.0 --port 8000
```
Interactive API documentation available at `http://localhost:8000/docs`.

---

## Detailed Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) — Multi-layer design, data contracts, planning engine, capabilities.
- [REST API & SSE Streaming](docs/API_REFERENCE.md) — Endpoint schemas, request/response payloads, and real-time SSE stream protocol.
- [Capabilities Catalog](docs/CAPABILITIES.md) — Parameter schemas, output types, DAG parameter bindings, and extension tutorial.
- [Sovereign RAG Deep Dive](docs/RAG_PIPELINE.md) — Ingestion, chunking, Nomic 768d embeddings, pgvector storage, and grounded QA.
- [Inference Guide](docs/INFERENCE.md) — Native router mode, multi-model presets, vision projector, VRAM management, and hot-swapping.
- [Models Guide](docs/MODELS.md) — GGUF formats, quantization, Hugging Face models, offline air-gap policy, and swapping models.
- [Security & Air-Gap Architecture](docs/SECURITY_AND_AIRGAP.md) — Zero network egress, Docker sandbox isolation, and SHA-256 artifact integrity.
- [Operator Troubleshooting Runbook](docs/TROUBLESHOOTING.md) — Diagnostic runbook for VRAM OOM, database connection, sandbox, and planning issues.
- [Reproduction Guide](docs/REPRODUCE.md) — Step-by-step instructions to build runtime, setup database, and verify suite.
- [Setup Guide](docs/SETUP.md) — Environment variables, Linux & macOS MacBook prerequisites, container configuration.

