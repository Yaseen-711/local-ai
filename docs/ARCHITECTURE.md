# Sovereign Industrial AI Workbench — Architecture Guide

## 1. Architectural Philosophy & Sovereignty

The Local AI Foundation (MRPL Sovereign Industrial AI Workbench) is an air-gapped, fully self-hosted engineering intelligence system. It provides an enterprise-grade layer separating high-level industrial workflows (P&ID diagram analysis, equipment specification queries, engineering verification, and artifact generation) from local model runtimes, vector storage, and hardware acceleration.

### Core Principles

1. **Zero External Egress (True Sovereignty)**: All inference, document processing, vector search, code sandboxing, and artifact rendering execute strictly on local compute with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
2. **Stateless Foundation vs. Stateful Orchestration**: The Foundation Core remains strictly stateless and capability-neutral. Multi-step workflows, task DAGs, plan retries, and persistence reside entirely in the Orchestration and Application layers.
3. **Decoupled Database Isolation**: Orchestration state (goals, plans, task logs) is stored in the `local_ai` PostgreSQL database, while the RAG subsystem operates in an isolated `local_ai_rag` database with `pgvector`.
4. **Deterministic Validation Before Execution**: Generated plans must pass a 4-stage deterministic validator (schema, capability presence, DAG acyclicity, and parameter bindings) before execution begins.
5. **Least-Privilege Isolation**: Python calculation and code repair execute inside an isolated Docker sandbox with zero network access, memory/CPU quotas, and ephemeral volumes.

---

## 2. Multi-Layer Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. API & PRESENTATION LAYER (`apps/api`)                                    │
│    FastAPI Application • SSE EventBus • REST Endpoints:                      │
│    /goals (DAG execution) • /direct (Bypass) • /rag • /files • /artifacts    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ bootstrap via
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ 2. COMPOSITION ROOT (`apps/context.py` - `AppContext`)                       │
│    Wires FoundationCore, InferenceConnector, CapabilityRegistry,             │
│    DecisionEngine, GoalOrchestrator, and PlanRunner into a typed root.      │
└───────────────────┬─────────────────────────────────────┬───────────────────┘
                    │                                     │
┌───────────────────▼───────────────────┐ ┌───────────────▼───────────────────┐
│ 3. DECISION & PLANNING LAYER          │ │ 4. EXECUTION ENGINE               │
│    orchestration/decision/            │ │    orchestration/execution/       │
│    orchestration/routing/             │ │                                   │
│    ├── StagedEscalationRouter         │ │ ├── GoalOrchestrator              │
│    │    Stage 1: Regex Matcher        │ │ ├── InProcessPlanRunner           │
│    │    Stage 2: Semantic Router      │ │ │    (Topological DAG scheduler)  │
│    │    Stage 3: Fast 0.8B Classifier │ │ ├── PlanValidator (4-stage checks)│
│    │    Stage 4: 9B Classifier        │ │ └── PostgresOrchestrationRepo     │
│    ├── DecisionEngine (Direct vs Plan)│ │      (Target DB: `local_ai`)      │
│    └── LLMPlanner (DAG synthesis)     │ │                                   │
└───────────────────┬───────────────────┘ └───────────────┬───────────────────┘
                    │                                     │
┌───────────────────▼─────────────────────────────────────▼───────────────────┐
│ 5. CAPABILITY LAYER (`orchestration/capabilities`)                           │
│    Protocol-driven registered capability descriptors:                       │
│    ├── `retrieval.rag`          (Vector search, cross-encoder, grounded QA) │
│    ├── `vision.inspect`         (Multimodal P&ID & diagram inspection)      │
│    ├── `document.understand`    (Docling / PyMuPDF table/text extraction)   │
│    ├── `code.workspace`         (Docker-isolated Python calculation sandbox)│
│    ├── `code.verify_and_repair` (Multi-turn verification & repair loop)     │
│    ├── `artifact.generate`      (Excel XLSX, Word DOCX, Slides PPTX, PDF)   │
│    ├── `workflow.text_analysis` (Single-pass & two-pass structured analysis)│
│    └── `agent.pydantic_ai`      (Bounded autonomous tool agent)             │
└───────────────────┬─────────────────────────────────────┬───────────────────┘
                    │                                     │
┌───────────────────▼───────────────────┐ ┌───────────────▼───────────────────┐
│ 6. SOVEREIGN RAG SUBSYSTEM (`rag/`)   │ │ 7. FOUNDATION CORE (`core/`)      │
│    Database: `local_ai_rag` (pgvector)│ │    ├── ModelRegistry (TOMLs)      │
│    ├── Ingestion & Chunking           │ │    ├── ProviderManager            │
│    ├── Nomic-embed-text-v1.5 (768d)   │ │    └── FoundationInferenceConnect │
│    ├── Cosine Vector Similarity       │ └───────────────┬───────────────────┘
│    ├── MiniLM Cross-Encoder Reranker  │                 │
│    └── Grounded QA & Refusal Engine   │                 │
└───────────────────────────────────────┘                 │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. PROVIDER LAYER & RUNTIME (`adapters/llama_cpp`)                          │
│    LlamaCppProvider (HTTP Client Mode) ──► 127.0.0.1:8080                   │
│    llama-server (Multi-Model Native Router Mode):                           │
│    ├── Slot 1: `qwen3.5-9b` + `Qwen3.5-9B.mmproj-q8_0` (LLM & Vision)       │
│    └── Slot 2: `qwen3.5-0.8b` (Fast Routing & Intent Classification)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ 9. HARDWARE LAYER                                                           │
│    NVIDIA GeForce RTX 5070 (12 GB VRAM) • Driver 570.211.01 • CUDA 12.8      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer Breakdown & Boundaries

### 3.1 API & Presentation Layer (`apps/api/`)
The external boundary exposing the workbench through asynchronous HTTP endpoints and Server-Sent Events (SSE):
* `POST /goals`: Submits natural language engineering goals. Triggers the `DecisionEngine` to plan and execute a multi-task DAG, streaming step progress over SSE.
* `POST /direct`: Bypasses planning to execute a single capability immediately (e.g. direct text analysis or direct document parsing).
* `POST /rag/query`: Queries the Sovereign RAG knowledge base for grounded facts and citations.
* `POST /rag/ingest`: Ingests technical documentation into `local_ai_rag`.
* `POST /files/upload`: Uploads and stages documents (PDF, DOCX) and diagrams (PNG, JPG) in the ephemeral staging area.
* `GET /artifacts/{id}/download`: Downloads generated engineering artifacts (XLSX, DOCX, PPTX, PDF) with cryptographic SHA-256 verification.
* `GET /telemetry/health`: Probes runtime health, model availability, database connectivity, and GPU VRAM allocation.

### 3.2 Process Composition Root (`apps/context.py` - `AppContext`)
`AppContext` is a frozen dataclass acting as the central dependency injection composition root. It initializes:
* `FoundationCore` and `InferenceConnector`
* All 8 capability implementations
* The `CapabilityRegistry`
* The `StagedEscalationRouter` and `LLMPlanner`
* The `PlanValidator` and `GoalOrchestrator`
* Database connections for orchestration (`local_ai`) and RAG (`local_ai_rag`)

### 3.3 Decision & Planning Layer (`orchestration/decision/`, `routing/`, `planning/`)
When a user submits a natural language request, the `DecisionEngine` determines whether to route directly to a single capability or invoke multi-step DAG planning:

```text
User Request: "Review this P&ID against equipment specs, calculate margin, and generate XLSX"
                                      │
                                      ▼
                           StagedEscalationRouter
   Stage 1: Deterministic regex/prefix match (0ms)
   Stage 2: Semantic Router (Aurelio cosine distance)
   Stage 3: Fast 0.8B LLM Classifier (qwen3.5-0.8b)
   Stage 4: Reasoning 9B LLM Classifier (qwen3.5-9b)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      ExecutionStrategy.DIRECT                  ExecutionStrategy.PLAN_REQUIRED
  (Execute single capability)                              │
                                                           ▼
                                                       LLMPlanner
                                            Synthesizes multi-task DAG plan
                                                           │
                                                           ▼
                                                     PlanValidator
                                            4-Stage Deterministic Verification
                                                           │
                                                           ▼
                                                    GoalOrchestrator
```

### 3.4 Execution & DAG Engine (`orchestration/execution/`)
* **`InProcessPlanRunner`**: Topologically sorts DAG tasks, resolves upstream task outputs to downstream input parameters, executes capabilities concurrently where dependencies allow, and handles task-level retry policies.
* **State Persistence**: Persists goals, plans, and task statuses to PostgreSQL (`local_ai`) using `PostgresOrchestrationRepository`.

### 3.5 The Capability Layer (`orchestration/capabilities/`)
Every capability conforms to the structural `Capability` protocol:
```python
class Capability(Protocol):
    @property
    def capability_id(self) -> str: ...
    def get_descriptor(self) -> CapabilityDescriptor: ...
    def execute(self, params: Dict[str, Any], context: Optional[ExecutionContext] = None) -> TaskResult: ...
```

Registered Capabilities:
1. `retrieval.rag`: Interacts with `local_ai_rag` to perform hybrid search and grounded question answering.
2. `vision.inspect`: Encodes diagram images into base64 and invokes `qwen3.5-9b` with the multimodal projector to detect equipment tags, line numbers, and instrument connections.
3. `document.understand`: Extracts structured hierarchy, markdown text, and tables from PDFs using Docling (with PyMuPDF fallback).
4. `code.workspace`: Spawns a hardened Docker container (`python:3.12-slim`) with no network access (`network_mode="none"`), memory limits (`2g`), and CPU limits (`2.0`) to run calculations.
5. `code.verify_and_repair`: Runs a verification loop that generates test cases, tests code in the sandbox, captures stderr/tracebacks, and repairs code up to 3 turns.
6. `artifact.generate`: Produces binary engineering files (OpenPyXL workbooks with dynamic formulas, python-docx technical memos, python-pptx slide decks).
7. `workflow.text_analysis`: Executes single-pass extraction or two-pass executive synthesis.
8. `agent.pydantic_ai`: Bounded agent that can invoke tools registered in the `CapabilityRegistry` under strict execution policy budgets.

---

## 4. Sovereign RAG Subsystem

The RAG subsystem (`rag/`) is an isolated, air-gapped retrieval engine:

```text
Incoming Document (PDF / DOCX / TXT)
                 │
                 ▼
     Ingestion & Normalization (`rag/ingestion`)
                 │
                 ▼
     Hierarchical Chunking (`rag/chunking`)
                 │
                 ▼
     Nomic Embedding Model (`rag/embedding/nomic.py`)
     • 768 dimensions
     • Prefix: "search_document: "
     • L2 normalization
                 │
                 ▼
     PostgreSQL + pgvector (`local_ai_rag`)
     • Table: document_chunks
     • Cosine similarity search (1 - (embedding <=> query_vec))
                 │
                 ▼
     Candidate Retrieval (Top-K)
                 │
                 ▼
     Cross-Encoder Reranker (`rag/reranking/cross_encoder.py`)
     • Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
     • Pairwise joint cross-attention scoring
     • Used as a ranking signal, NOT a correctness threshold
                 │
                 ▼
     Grounded QA Engine (`rag/domain/grounded_qa.py`)
     ├── Positive Grounding: Cites file, section, and page
     └── Negative Grounding: Explicit refusal if facts are absent
```

### RAG Database Separation Rule
The RAG database is strictly segregated:
* `local_ai`: Application state, user goals, task execution graphs.
* `local_ai_rag`: Chunk embeddings, document text, vector index.
* Contention Rule: Contention between Nomic embedding models, cross-encoder rerankers, and `llama-server` is actively managed. If CUDA VRAM is saturated, reranking seamlessly falls back to optimized CPU execution.

---

## 5. Foundation Core & Inference Runtime

### 5.1 Foundation Core (`core/`)
`FoundationCore` is the lowest-level internal framework. It contains:
* `ModelRegistry`: Declarative model configurations parsed from `configs/models/*.toml`.
* `ProviderManager`: Routes model requests to verified providers.
* `InferenceRequest` & `InferenceResponse`: Strongly typed dataclasses enforcing validation invariants (finite temperatures, bounded tokens, explicit formats).

### 5.2 Provider Layer (`adapters/llama_cpp/`)
The `LlamaCppProvider` interfaces with `llama-server` via HTTP client mode:
* Uses native `llama-server` router mode (`--models-preset configs/llama_models.ini`).
* Communicates with two concurrent model slots on `127.0.0.1:8080`.
* Translates Foundation model aliases (`default`, `qwen3.5`, `vision`) to runtime models.
* Bounded HTTP response streaming with `max_response_bytes` protection.

---

## 6. Deterministic 4-Stage Plan Validation

Before any plan generated by `LLMPlanner` is executed, it must pass through `PlanValidator`:

1. **Stage 1 — Structural Schema Validation**: Confirms the plan contains a non-empty task list, valid IDs, recognized execution modes (`parallel` vs `sequential`), and integer constraints.
2. **Stage 2 — Capability Resolution**: Checks that every task's `capability_id` exists in the active `CapabilityRegistry`.
3. **Stage 3 — DAG Topology Validation**:
   - Builds an adjacency list of task dependencies.
   - Performs Depth-First Search (DFS) cycle detection.
   - Enforces critical path depth limit ($\le 10$) and task count limit ($\le 50$).
4. **Stage 4 — Parameter & Artifact Binding**: Verifies that any upstream reference (e.g. `{{tasks.step_1.output}}`) references a declared predecessor task.

---

## 7. Testing & Verification Architecture

The test suite enforces reliability across three distinct tiers:

1. **Unit Tests (`tests/unit/`, 520+ tests)**:
   - Completely isolated, pure CPU tests.
   - Zero live server, model weight, or database dependencies.
   - Executes in seconds, validating routing rules, plan validation, capability contracts, and error handling.
2. **Integration Tests (`tests/integration/`)**:
   - Tests live communication with `llama-server` and PostgreSQL when services are available.
   - Automatically skips with clear environmental reasons when run offline.
3. **End-to-End Golden Flow Validation (`tests/e2e/test_golden_flow_validation.py`)**:
   - Validates all 11 real-world scenarios from `golden_test_pack/`.
   - Exercises real multi-capability goal execution: Vision inspection $\rightarrow$ RAG lookup $\rightarrow$ Sandboxed Docker calculation $\rightarrow$ Multi-format artifact generation.
