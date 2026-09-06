# Sovereign On-Premise Agentic AI Workbench for Confidential Industrial Work
**Target Organization**: Mangalore Refinery and Petrochemicals Limited (MRPL)  
**Theme**: Smart Automation / Industrial Sovereign AI  
**Deployment Target**: Single On-Premise Workstation / Server (Reference: NVIDIA RTX 5070 12 GB VRAM, Ubuntu Linux)  
**Security Invariant**: 100% Air-Gapped, Zero External Data Egress, Verifiable Sovereignty

---

## 1. Executive Summary & Problem Context

Refineries like MRPL, PSUs, and defense-linked manufacturing facilities handle sensitive, high-consequence operational and engineering documentation daily:
- **Piping & Instrumentation Diagrams (P&IDs)** and unreleased process flow designs.
- **Corrosion & Inspection Reports** for distillation columns, reactors, boilers, and heat exchangers.
- **Capital Expenditure (CAPEX) & Vendor Negotiation Notes**.
- **Board Approval Notes & Plant Management Memos**.
- **Engineering Calculations** (orifice sizing, line hydraulics, relief valve capacity, thickness degradation).
- **Internal Maintenance & Plant Operations Code/Scripts**.

Under strict corporate and national security policies, this data cannot be routed through commercial cloud AI services (e.g., Claude, OpenAI, or Microsoft Copilot). Currently, engineers face an unproductive dilemma: manually drafting notes and calculating formulas, or risking compliance breaches by pasting proprietary data into public tools.

This document establishes the architecture and execution blueprint for the **MRPL Sovereign Agentic AI Workbench**: a completely self-hosted, air-gapped system powered by open-weight models that automates multi-step industrial knowledge work, verifies calculations in an isolated container sandbox, reads multimodal drawings and inspection scans, generates formal corporate deliverables (`.docx`, `.xlsx`, `.pptx`), and provides verifiable runtime telemetry and cryptographic audit manifests demonstrating that **all traffic remains strictly on local loopback interfaces with zero external egress**.

---

## 2. Core Architectural Philosophy & Sovereign Boundary

The workbench builds upon the principle of **architectural ownership**:
1. **Zero Cloud Telemetry or Egress**: All weights, tokenizers, embeddings, parsers, and execution environments reside strictly on local disks and run entirely on local compute.
2. **Multi-Model Heterogeneity**: The backend is never bound to a single model. It supports multiple open-weight models concurrently in native router mode, auto-selecting the right model tier for the task.
3. **Audit-First Multimodal Extensibility**: Rather than prematurely bundling heavy multi-gigabyte vision models, the system first audits the existing open-weight model and local runtime for multimodal/projector capabilities, introducing only the minimal required extension if text-only limitations are empirically proven.
4. **Agent is an Execution Mechanism, Not an Authority**: Autonomous agents plan, iterate, and execute tools, but they operate under strict per-call authorization boundaries (`AgentExecutionPolicy`).
5. **Verifiable Sovereignty**: Sovereignty is proven through continuous real-time socket inspection (`/proc/net/tcp`), loopback-only network isolation, and cryptographic artifact provenance manifests.

---

## 3. Verified Implemented Foundation (Current System Baseline)

The workbench is not theoretical; it builds upon the verified, completed Foundation codebase with **414 automated tests passing** (and 407 passed, 9 cleanly skipped in fully offline/isolated CI):

```
                               MRPL WORKBENCH CORE (BUILT & VERIFIED)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ [Stage 1: Model & Inference Foundation]                                                      │
 │  - Native llama-server router on 127.0.0.1:8080 (RTX 5070 12GB VRAM)                         │
 │  - Concurrent Open-Weight Models: Qwen3.5-9B (Reasoning Tier) + Qwen3.5-0.8B (Lightweight Tier)│
 │  - FoundationCore & FoundationInferenceConnector normalized contracts                       │
 │  - Hard runtime model identity check preventing model spoofing                               │
 ├──────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [Stage 2: Staged Intent Routing & Model Selection]                                           │
 │  - Stage 1: DeterministicRuleMatcher (Exact keywords/regex)                                 │
 │  - Stage 2: AurelioSemanticRouter (Air-gapped embeddings, max-aggregation, threshold >= 0.60) │
 │  - Stage 3: Lightweight LLM Classifier (Qwen3.5-0.8B, fast routing)                         │
 │  - Stage 4: Reasoning LLM Classifier (Qwen3.5-9B, complex disambiguation)                    │
 │  - ModelSelectionPolicy: Dynamic tier-to-model resolution                                    │
 ├──────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [Stage 3: Autonomous Orchestration & Planning]                                               │
 │  - DecisionEngine: Staged escalation and strategy determination                              │
 │  - GoalOrchestrator: Direct capability dispatch & execution lifecycle                        │
 │  - Replanner & PlanValidator: Dynamic DAG evolution with history preservation               │
 │  - Execution Runners: InProcessPlanRunner (sync DAG) + TemporalPlanRunner (durable workflows)│
 ├──────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [Stage 4: PydanticAI Agent Capability ('agent.pydantic_ai')]                                  │
 │  - FoundationPydanticAIModel: Local adapter routing requests through FoundationCore           │
 │  - AgentExecutionPolicy: Per-call dynamic authorization boundary                             │
 │  - CapabilityToolAdapter: Exposes registry capabilities as typed tools with provenance       │
 │  - Bounded Iteration: UsageLimits, CancellationToken, timeout_seconds, and budget limits    │
 │  - Proven Live Multi-Turn Tool Use across both 9B and 0.8B tiers                             │
 ├──────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [Stage 5: Industrial Capabilities]                                                           │
 │  - code.workspace: Docker container-isolated file & bash execution (read, write, edit, run) │
 │  - document.understand: Docling multi-format parser with local OCR fallback                  │
 │  - artifact.generate: Deterministic DOCX, XLSX, and PDF compiler with SHA-256 hashes         │
 │  - workflow.text_analysis: Two-pass structured analytical synthesis                          │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Problem Statement Gap Analysis & Adaptation Matrix

To fully satisfy MRPL's confidential industrial requirements, the system extends the baseline across specific functional vectors:

| MRPL Problem Statement Requirement | Implemented Foundation Baseline | Required Extension for MRPL Workbench |
| :--- | :--- | :--- |
| **Model Auto-Selection across $\ge 2$ Task Types** | `ModelSelectionPolicy` resolves `LIGHTWEIGHT` (0.8B) and `REASONING` (9B). | Route **routine parsing/tagging** to 0.8B; route **engineering math/approval drafting** to 9B; route **P&IDs/drawings** to the verified local multimodal engine. |
| **Scanned Drawings, P&IDs & Visual Inspection** | `document.understand` handles text/tables via Docling OCR. | **Audit-First**: First verify whether existing `Qwen3.5-9B` + `llama.cpp` + `FoundationInferenceConnector` supports image input via projector (`--mmproj`); only introduce a separate VLM (e.g., Qwen2.5-VL) if the audit proves it necessary. |
| **Sandboxed Engineering Code & Step Calculations** | `code.workspace` provides Docker container isolation. | Add specialized refinery calculation libraries (`scipy`, `CoolProp`) and closed-loop test/repair harness validating intermediate steps. |
| **Real Executive Deliverables (Word, PPT, Excel)** | `artifact.generate` compiles standard DOCX, XLSX, PDF. | Add **PPTX slide generation** and **MRPL-specific templates** (Refinery Approval Notes, Technical Inspection Sheets, Calculation Workbooks). |
| **SOP & Manual Grounding (RAG)** | Clean capability boundary reserved (`retrieval.rag`). | Teammate's RAG module registers as a standard capability using local vector storage (LanceDB) and local embeddings; agent calls it via `CapabilityToolAdapter`. |
| **Verifiable Sovereignty & Zero Egress** | Offline test suites verify zero sockets in CI. | Build a **Live Network Audit Monitor** into the API/UI proving 0 bytes external egress and 100% loopback traffic during execution. |
| **Deployable User Interface (The Workbench)** | Python AppContext composition root & CLI tests. | Build a **FastAPI backend** (REST + SSE streaming) paired with an **Interactive Web Workbench UI** (split-screen viewer, DAG visualizer, model badge). |

---

## 5. Target Architecture for the MRPL Workbench

```
                                  MRPL REFINERY USER / ENGINEER
                                                │
                                                ▼
                          ┌─────────────────────────────────────────────┐
                          │   SOVEREIGN INDUSTRIAL WORKBENCH WEB UI     │
                          │  - Split-Screen Scanned PDF/P&ID Viewer     │
                          │  - Interactive Agent Goal Console           │
                          │  - Real-time DAG & Tool Execution Stream    │
                          │  - Dynamic Model Badge (0.8B / 9B / Vision) │
                          │  - Live Network Egress Monitor (0 KB/s Ext) │
                          │  - Deliverables Hub (.docx / .xlsx / .pptx) │
                          └──────────────────────┬──────────────────────┘
                                                 │ HTTP / Server-Sent Events (SSE)
                                                 ▼
                          ┌─────────────────────────────────────────────┐
                          │      FASTAPI APPLICATION DELIVERY LAYER     │
                          │   - /api/v1/goals (Submit industrial task)  │
                          │   - /api/v1/goals/{id}/stream (SSE events)  │
                          │   - /api/v1/artifacts/{id}/download         │
                          │   - /api/v1/telemetry/sovereignty (Sockets) │
                          └──────────────────────┬──────────────────────┘
                                                 │ Holds AppContext
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │ DECISION ENGINE & STAGED INTENT ROUTER                                                      │
  │  - Auto-selects strategy: DIRECT_DETERMINISTIC | DIRECT_CAPABILITY | PLAN_REQUIRED           │
  │  - Model Selection: Task needs -> ModelSelectionPolicy -> Tier Assignment                   │
  └──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
         [Direct Goal Path]                               [Plan Path]
         (Single-step tasks)                     (Multi-step industrial goals)
                  │                                             │
                  └──────────────────────┬──────────────────────┘
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │ AGENT CAPABILITY ('agent.pydantic_ai')                                                      │
  │  - PydanticAI Agent Loop running locally                                                    │
  │  - Per-Call Authorization Seam: AgentExecutionPolicy                                        │
  │  - Advisory Replanning: Emits AgentProposal on unexpected plant findings                    │
  └──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼ (Inference Requests)                        ▼ (Tool Invocations)
  ┌────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐
  │ FoundationPydanticAIModel Adapter          │ │ CapabilityToolAdapter                        │
  │  - Maps PydanticAI to Foundation contracts │ │  - Checks AgentExecutionPolicy on EVERY call │
  │  - Injects native function calling schemas │ │  - Resolves Capability from Registry         │
  │  - Translates ToolReturn and ToolCalls     │ │  - Accumulates Artifact & Data references    │
  └─────────────────────┬──────────────────────┘ └──────────────────────┬───────────────────────┘
                        │                                               │
                        ▼                                               ▼
  ┌────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐
  │ FoundationCore / ProviderManager           │ │ CAPABILITY REGISTRY                          │
  │  - LlamaCppProvider                        │ │  ├─ document.understand (Docling OCR)        │
  │  - Hard Runtime Model Identity Validation  │ │  ├─ vision.inspect (Multimodal Drawing Tool) │
  │  - Multi-Model Router (127.0.0.1:8080):    │ │  ├─ code.workspace (Docker Sandbox)          │
  │    • Qwen3.5-0.8B (Lightweight Tier)       │ │  ├─ artifact.generate (DOCX, XLSX, PPTX, PDF)│
  │    • Qwen3.5-9B (Reasoning Tier)           │ │  ├─ workflow.text_analysis (2-pass synthesis)│
  │    • Multimodal Seam (Audit-First VLM)     │ │  └─ retrieval.rag (Parallel Teammate RAG)    │
  └────────────────────────────────────────────┘ └──────────────────────────────────────────────┘
                        ▲                                               ▲
                        └───────────────────────┬───────────────────────┘
                                                │
                                  100% LOCAL COMPUTE & MEMORY
                        (RTX 5070 GPU / RAM / Local SSD / Docker Daemon)
                                (NO INTERNET / NO EXTERNAL CALLS)
```

---

## 6. Detailed Forward Implementation Roadmap

### Phase 1: Multimodal Capability Audit (Audit-First Step)
**Goal**: Empirically assess whether the existing local model stack (`Qwen3.5-9B` + `llama.cpp` + `FoundationInferenceConnector`) can process image inputs directly before adding any external dependencies.

1. **Audit Scope & Inspection Points**:
   - **Model Architecture Audit**: Inspect the loaded `Qwen3.5-9B` GGUF metadata to confirm whether vision token embeddings or multimodal projection weights are embedded in the model or available as a companion `--mmproj` projector file.
   - **Runtime Server Audit**: Verify `llama-server` command-line flags and endpoint capabilities on `127.0.0.1:8080`. Test if the OpenAI-compatible `/v1/chat/completions` endpoint accepts structured image content payloads (`type: "image_url"` or base64 data URIs) with the current binary.
   - **Connector Seam Audit**: Inspect `core.inference.types.Message` and `FoundationInferenceConnector`. Currently, `Message.content` is defined as `str`. Determine whether multimodal inputs should be represented via a multipart message structure or a structured `image_paths` parameter without breaking downstream consumers.
2. **Decision Gate**:
   - **Case A (Native Projector Supported)**: If a compatible `--mmproj` file enables `Qwen3.5-9B` to process visual tokens directly in `llama-server`, configure the projector preset in `configs/llama_models.ini` with zero architecture changes.
   - **Case B (Text-Only Confirmed)**: If `Qwen3.5-9B` is strictly text-only, proceed to Phase 2 to introduce the smallest required local vision-language model.

---

### Phase 2: Smallest Required Multimodal Extension
**Goal**: If Phase 1 confirms that a dedicated vision model is needed, integrate the minimal open-weight VLM behind existing provider and capability seams.

1. **Target Model Selection (VRAM Budget-Constrained)**:
   - Target an efficient open-weight vision model (e.g., `Qwen2.5-VL-7B-Instruct` or `MiniCPM-V-2.6` in Q4_K_M GGUF format with companion `mmproj`).
   - Fits comfortably in the single RTX 5070 (12 GB VRAM) alongside the lightweight 0.8B model, or dynamically hot-swapped via `llama-server` multi-model slots.
2. **`vision.inspect` Capability Implementation**:
   - Implement `VisionInspectionCapability` in `orchestration/capabilities/builtin/vision/`:
     - Inputs: `image_path` / PDF page render + `query` / inspection instructions.
     - **P&ID Inspection Mode**: Extracts equipment tags (`P-101A/B`, `E-204`), instrument bubbles (`PT-101`, `FIC-202`), line numbers, valve types (gate, globe, check), and flow directions.
     - **Physical Plant Inspection Mode**: Evaluates surface rust, pitting, flange leaks, or analog dial gauge readings from equipment photographs.
   - Registers cleanly into `CapabilityRegistry` as `vision.inspect`.
3. **Agent Seam Exposure**:
   - `CapabilityToolAdapter` exposes `vision.inspect` to `agent.pydantic_ai`, allowing the agent to inspect drawings and diagrams during multi-step analysis.

---

### Phase 3: FastAPI Application Delivery Layer (Server Seam)
**Goal**: Expose the orchestration engine, agent capabilities, and telemetry over standard REST and streaming protocols.

1. **Module Structure**:
   - `apps/api/main.py`: FastAPI application entry point, lifespan management (initializing `AppContext`), and exception handlers.
   - `apps/api/routes/goals.py`:
     - `POST /api/v1/goals`: Accepts goal description, context parameters, input files/references. Dispatches to `DecisionEngine`.
     - `GET /api/v1/goals/{id}`: Returns terminal execution state, final response, generated artifacts, and token metrics.
     - `GET /api/v1/goals/{id}/stream`: Server-Sent Events (SSE) streaming real-time execution events (task started, model output chunk, tool proposed, tool authorized, tool executed, task completed).
   - `apps/api/routes/artifacts.py`:
     - `GET /api/v1/artifacts/{id}/download`: Streams compiled `.docx`, `.xlsx`, `.pdf`, `.pptx` deliverables with accurate MIME types and SHA-256 header validation.
   - `apps/api/routes/telemetry.py`:
     - `GET /api/v1/telemetry/sovereignty`: Returns live network socket inspection data verifying that only loopback connections are active.
     - `GET /api/v1/telemetry/system`: Returns GPU VRAM allocation, CPU usage, and loaded model status.
2. **Streaming Event Protocol**:
   - Standard event envelope: `event: task_update | tool_call | chunk | terminal` with structured JSON payloads containing task IDs, tool arguments, and progress facts.

---

### Phase 4: Industrial Deliverables & Engineering Template Engine
**Goal**: Expand artifact generation from plain documents to formal, styled industrial deliverables required by refinery management.

1. **Presentation Generation (`python-pptx`)**:
   - Extend `ArtifactFormat` in `orchestration/capabilities/builtin/artifact/types.py` to support `PPTX`.
   - Implement `PptxGenerator` in `generators.py`: Compiles structured JSON (slide title, bullet points, table data, callout boxes) into professionally branded slide decks.
2. **Refinery Document Templates**:
   - **MRPL Technical Approval Note (`.docx`)**:
     - Strict corporate formatting: Header (Department, Note Ref Number, Date), Subject Line, 1. Background, 2. Technical Inspection Summary, 3. Risk & Safety Assessment, 4. Financial/Cost Estimate, 5. Recommendation, and Executive Signature Block.
   - **Refinery Engineering Calculation Sheet (`.xlsx`)**:
     - Formatted multi-tab workbook: *Inputs & Design Constants* (e.g., design pressure, temperature, corrosion allowance), *Calculations & Derivations* (Excel formulas preserved, e.g., `=B4*(C4/(2*D4))`), and *Summary & Design Margin Check* with conditional formatting (Pass = Green, Below Margin = Red).
3. **Mathematical Step Preservation**:
   - Standardize structured step representations: Every engineering calculation tool must return intermediate steps ($Formula \rightarrow Substitution \rightarrow Value$) so they are printed verbatim in the final approval note and spreadsheet.

---

### Phase 5: Sovereignty & Air-Gap Telemetry Monitor
**Goal**: Provide visible, objectively verifiable telemetry and cryptographic proofs to MRPL evaluators that all operations remain bounded to the local host and zero external packets leave the network perimeter.

1. **Network Namespace Auditor (`core/telemetry/airgap.py`)**:
   - Continuously monitors active network sockets using Linux `/proc/net/tcp`, `/proc/net/udp`, and process socket descriptors via `psutil`.
   - Classifies every active connection into:
     - `ALLOWED_INTERNAL`: Loopback (`127.0.0.1`, `::1`), local IPC UNIX domain sockets, Docker bridge network (`172.17.0.0/16`).
     - `EXTERNAL_PROHIBITED`: Any connection to non-private WAN IP addresses.
   - Triggers an immediate critical alert and logs an audit failure if any external packet attempt is observed.
2. **Sovereignty Audit Manifest**:
   - With every generated deliverable, emit an accompanying `.audit.json` manifest:
     - Deliverable SHA-256 hash.
     - Input file SHA-256 hashes.
     - Model weights SHA-256 hashes used during the run.
     - Confirmation log verifying zero non-loopback network sockets were open during generation.

---

### Phase 6: Sovereign Industrial Workbench Web UI
**Goal**: A cohesive, modern, on-premise browser interface designed for plant engineers and technical managers.

1. **Frontend Architecture**:
   - Lightweight, responsive dashboard (built using React/Vite/Tailwind or Streamlit for rapid zero-dependency deployment).
   - Served directly from the local FastAPI application (`/` route) so no external CDN or cloud hosting is involved.
2. **Key Dashboard Views**:
   - **View A: Engineering Goal Console**:
     - Pre-configured MRPL workflow cards:
       - *"P&ID Line & Valve Audit"*
       - *"Equipment Inspection & Approval Note Generator"*
       - *"Pump Sizing & Hydraulic Calculation Sheet"*
     - Natural language goal input box with file drop zone (PDF, PNG, JPG, CSV).
   - **View B: Split-Screen Inspector**:
     - *Left Pane*: High-resolution viewer rendering the input scanned document, P&ID drawing, or corrosion photograph.
     - *Right Pane*: Live agent execution feed showing planning steps, model switches, tool inputs/outputs, and final synthesized text.
   - **View C: Real-Time Model & Sovereign Telemetry Bar**:
     - Live badge showing active model: `[Model: Qwen3.5-9B (Reasoning Tier)]` or `[Model: Qwen3.5-0.8B (Fast Tier)]`.
     - Live indicator: `[Sovereignty: 100% Local Loopback | External Traffic: 0 KB/s]`.
     - Local GPU VRAM meter (e.g., `8.2 / 12.0 GB VRAM utilized`).
   - **View D: Deliverables Showcase**:
     - Preview and 1-click download cards for generated `.docx` Approval Notes, `.xlsx` Calculations, and `.pptx` Decks.

---

### Phase 7: End-to-End MRPL Golden Demonstration Scenarios

To prove the expected solution at MRPL, four reproducible end-to-end demonstration scenarios will be prepared with sample open-source industrial datasets:

#### Scenario 1: Scanned Inspection Report -> Executive Approval Note (DOCX)
- **Input**: Scanned multi-page PDF of a refinery heat exchanger inspection (containing ultrasonic thickness measurements, corrosion rates, and inspector handwriting).
- **Execution Flow**:
  1. Goal: *"Review inspection report for Exchanger E-102, identify components exceeding corrosion limits, calculate remaining service life, and draft a formal MRPL Approval Note for tube bundle replacement."*
  2. `DecisionEngine` identifies multi-step workflow -> dispatches to `agent.pydantic_ai`.
  3. Agent invokes `document.understand` to OCR scanned pages and extract thickness tables.
  4. Agent invokes `code.workspace` to run standard remaining-life formula in Python sandbox:
     $$\text{Remaining Life} = \frac{t_{\text{actual}} - t_{\text{minimum}}}{\text{Corrosion Rate}}$$
  5. Agent verifies calculations and invokes `artifact.generate` using the MRPL Approval Note template.
- **Output**: Formatted `MRPL_Approval_Note_E102_Replacement.docx` ready for GM signature.
- **Verification**: Air-gap monitor displays 0 bytes external egress.

#### Scenario 2: Model Auto-Selection Demonstration (Task Differentiation)
- **Execution Flow**:
  1. **Task A (Routine Parsing)**: User submits: *"Extract all equipment tags from this valve maintenance log text."*
     - Decision Engine routes to `ModelTier.LIGHTWEIGHT` (`qwen3.5-0.8b`). Response completes in <1.5 seconds with minimal GPU utilization.
  2. **Task B (Complex Engineering Reasoning)**: User submits: *"Analyze root cause of cavitation failure in pump P-201A based on operating suction pressure and NPSH curves."*
     - Decision Engine auto-escalates to `ModelTier.REASONING` (`qwen3.5-9b`). Model performs multi-step diagnostic reasoning.
- **Verification**: UI visibly highlights model tier switching without restarting the server.

#### Scenario 3: P&ID Drawing Discrepancy Review (Multimodal Vision)
- **Input**: High-resolution P&ID image (e.g., crude distillation unit flow sheet) + text equipment schedule.
- **Execution Flow**:
  1. Goal: *"Verify valve schedule against P&ID Drawing 40-CDU-001 and flag any missing bypass lines or relief valves."*
  2. Agent invokes `vision.inspect` on the drawing image.
  3. Multimodal engine detects instrument bubbles, control valves, and tag labels.
  4. Agent cross-references detected tags against the schedule and generates an Excel discrepancy matrix (`PID_Audit_Findings.xlsx`).
- **Output**: Downloadable Excel sheet highlighting matched vs missing tags.

#### Scenario 4: Sandboxed Engineering Calculations with Intermediate Steps
- **Input**: Pipeline flow parameters (Crude flow rate: $500\text{ m}^3/\text{hr}$, Pipe ID: $300\text{ mm}$, Viscosity: $15\text{ cSt}$).
- **Execution Flow**:
  1. Goal: *"Calculate Reynolds number, flow regime, Darcy friction factor, and pressure drop per 100m in pipe line L-104."*
  2. Agent writes executable Python script using standard Darcy-Weisbach equations.
  3. Script executes inside Docker sandbox (`code.workspace`); stderr and stdout captured.
  4. Agent compiles findings into an interactive Excel workbook (`Line_L104_Hydraulic_Calculations.xlsx`) with intermediate formula steps and dynamic cell references.
- **Output**: Verified calculation report with auditable intermediate formula steps, programmatic sandbox validation, and deterministic Excel formulas.

---

## 7. Acceptance & Evaluation Layer

To ensure industrial reliability and prevent unverified assertions, the workbench establishes concrete, programmatic evaluation criteria across two critical operational pipelines:

### 1. PDF $\rightarrow$ Answer Correctness Evaluation Benchmark
For scanned inspection sheets, equipment manuals, and laboratory test certificates:
- **Evaluation Dataset**: A curated ground-truth test corpus of 10 representative industrial documents (including multi-column tables, scanned degradation logs, and stamps).
- **Extraction Fidelity**:
  - Table cell extraction evaluated against labeled ground-truth tables using Precision, Recall, and F1 score (Target: $\ge 92\%$ F1 on tabular cells).
  - Key-Value entity extraction (e.g., Tag Number, Material Grade, Test Pressure, Inspection Date) evaluated for exact match.
- **Numerical Accuracy & Tolerance**:
  - Critical engineering parameters extracted from PDFs (e.g., wall thickness, operating temperature, corrosion allowance) must match ground-truth values within $\pm 0.001$ relative tolerance.
- **Grounding & Provenance Verification**:
  - Every numerical fact asserted in the generated approval note must link back to an extracted document segment or page coordinate.
  - Automated heuristic check: Scan final synthesized text against extracted document chunks; any ungrounded numerical assertion triggers a benchmark penalty.

### 2. Prompt $\rightarrow$ Code $\rightarrow$ Test $\rightarrow$ Repair $\rightarrow$ Retest Closed Loop
For engineering calculations, hydraulic formulas, and data transformations executed in `code.workspace`:
- **Closed-Loop Execution Contract**:
  1. **Code Generation**: The agent generates Python code along with explicit assertion checks verifying boundary conditions, physical dimensions, and expected units.
  2. **Sandboxed Execution**: The script executes in the isolated Docker container (`code.workspace`). Exit code, stdout, and stderr are captured.
  3. **Automated Error Feedback**: If execution fails (syntax error, runtime exception, or failed assertion), the stderr traceback is passed back to the agent as tool output.
  4. **Self-Correction (Repair)**: The agent analyzes the stack trace, modifies the calculation script, and re-executes.
  5. **Bounded Iteration**: The repair loop is constrained to a maximum of 3 attempts enforced by `AgentExecutionPolicy` and `UsageLimits`.
  6. **Retest & Verification**: The task succeeds only when the code exits with returncode 0 and all self-contained assertions pass.
- **Acceptance Threshold**:
  - 100% of calculation scripts embedded in final deliverables must pass closed-loop verification.
  - Calculations must emit both machine-executable code and symbolic step derivations ($Formula \rightarrow Substitution \rightarrow Result$) in the resulting spreadsheet or document.

---

## 8. RAG Integration Contract (Prepared for Teammate Delivery)

While RAG implementation proceeds in parallel by your teammate, the workbench architecture guarantees zero-friction integration through an explicit capability contract:

```python
# Capability contract ready for teammate's RAG module
class RagRetrievalCapability:
    @property
    def capability_id(self) -> str:
        return "retrieval.rag"

    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any], context: CapabilityContext) -> TaskResult:
        query = inputs.get("query")
        collection = parameters.get("collection", "mrpl_sops")
        top_k = int(parameters.get("top_k", 5))
        # Teammate's LanceDB / local vector store retrieval logic here
        ...
        return TaskResult(
            output={"retrieved_chunks": [...], "sources": [...]},
            references=[DataReference(key="rag_context", ...)],
        )
```

**Integration Guarantee**:
- When the RAG module is ready, it will simply be registered into `CapabilityRegistry` as `registry.register(RagRetrievalCapability(...))`.
- `AgentExecutionPolicy` and `CapabilityToolAdapter` will automatically expose `retrieval.rag` as an approved tool to `agent.pydantic_ai`.
- The agent will immediately begin grounding approval notes and engineering reviews in local MRPL SOPs, refinery manuals, and Indian Standards (IS/OISD standards) without modifying a single line of orchestration or UI code.

---

## 9. Summary of Milestones & Delivery Roadmap

```
  Phase 1: Multimodal Capability Audit [NEXT STEP]
    ├── Inspect Qwen3.5-9B GGUF metadata for vision projection weights
    ├── Test llama-server /v1/chat/completions image payload support on 127.0.0.1:8080
    └── Determine if companion --mmproj is supported or if separate VLM is needed

  Phase 2: Smallest Required Multimodal Extension
    ├── Configure minimal local VLM (e.g., Qwen2.5-VL-7B or MiniCPM-V) only if audit requires it
    ├── Implement vision.inspect capability for P&IDs and equipment photos
    └── Expose vision.inspect to PydanticAI agent via CapabilityToolAdapter

  Phase 3: FastAPI Application Delivery Layer (Server Seam)
    ├── REST Endpoints for Goals, Capabilities, and Artifacts
    ├── Server-Sent Events (SSE) Streaming for Agent Execution
    └── Telemetry & System Status Endpoints

  Phase 4: Industrial Deliverables & MRPL Templates
    ├── PPTX Presentation Generator
    ├── Formatted MRPL Technical Approval Note (.docx)
    └── Engineering Calculation Workbook with Formulas (.xlsx)

  Phase 5: Sovereignty & Air-Gap Telemetry Monitor
    ├── Linux Socket / Traffic Interceptor (/proc/net/tcp)
    ├── Live Prohibited External Connection Detector
    └── Cryptographic Deliverable Provenance Manifest (.audit.json)

  Phase 6: Sovereign Industrial Workbench Web UI
    ├── Split-Screen Document & P&ID Viewer
    ├── Interactive Agent DAG Visualizer
    ├── Live Air-Gap & Model Tier Telemetry Badges
    └── Deliverables Showcase & Download Hub

  Phase 7: End-to-End MRPL Golden Demos & Evaluation Harness
    ├── 4 Repeatable MRPL Refinery Demonstration Scenarios
    ├── PDF -> Answer Correctness Benchmark Harness
    ├── Prompt -> Code -> Test -> Repair -> Retest Validation Suite
    └── Clean Plug-in of Teammate's retrieval.rag Capability
```

---
*This specification acts as the definitive roadmap for adapting the Local AI Foundation into the MRPL Sovereign Industrial AI Workbench.*
