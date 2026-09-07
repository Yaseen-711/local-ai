# Capabilities Reference & Developer Guide

Capabilities are the fundamental atomic units of execution in the Sovereign Industrial AI Workbench. Each capability encapsulates a domain tool (vision inspection, RAG retrieval, sandboxed calculation, or artifact rendering) behind a uniform structural interface.

---

## 1. Capability Architecture

Every capability implements the Python `@runtime_checkable` Protocol defined in `orchestration/capabilities/base.py`:

```python
class Capability(Protocol):
    @property
    def capability_id(self) -> str:
        """Unique identifier (e.g. 'vision.inspect', 'retrieval.rag')."""
        ...

    def get_descriptor(self) -> CapabilityDescriptor:
        """Declarative metadata, input/output schemas, and execution flags."""
        ...

    def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> TaskResult:
        """Synchronously execute the domain action with given parameters."""
        ...
```

### Key Principles
1. **Uniform Envelope (`TaskResult`)**: Every capability execution returns a typed `TaskResult` containing status (`completed`, `failed`), an `output` dictionary, non-fatal `errors`, and optional `artifacts`.
2. **Context Passing (`ExecutionContext`)**: Passes goal context, staged file mappings, and cancellation tokens.
3. **No Direct Model Coupling**: Capabilities that perform inference do so strictly through `InferenceConnector`, never by directly calling HTTP clients or backend runtimes.

---

## 2. Catalog of Built-In Capabilities

### 2.1 Sovereign RAG (`retrieval.rag`)
Performs hybrid vector search and grounded question answering across the `local_ai_rag` database.

* **Parameters**:
  - `query` (*str*, required): The technical question or search query.
  - `mode` (*str*, optional): `"qa"` (default, includes LLM synthesis) or `"search"` (pure retrieval & reranking).
  - `top_k` (*int*, optional, default=10): Initial vector candidates.
  - `top_n` (*int*, optional, default=3): Candidates to retain after Cross-Encoder reranking.
  - `document_id` (*str*, optional): Constrain search to a specific document.
* **Output Payload**:
  - `answer`: Synthesized factual response with explicit citations.
  - `sources`: List of cited documents, sections, and page numbers.
  - `candidates`: List of ranked passages with similarity and cross-encoder scores.

---

### 2.2 Vision Inspection (`vision.inspect`)
Performs visual inspection of P&ID drawings, process flowsheets, and technical diagrams using `qwen3.5-9b` with the multimodal projector.

* **Parameters**:
  - `file_path` (*str*, optional): Local path to image (`.png`, `.jpg`).
  - `file_id` (*str*, optional): Staged file ID from `/api/v1/files/upload`.
  - `query` (*str*, required): Visual query (e.g. *"Identify tag FV-201A and list its connected lines"*).
  - `temperature` (*float*, optional, default=0.1): Sampling temperature.
* **Output Payload**:
  - `detected_tags`: Extracted asset tags (e.g. `["FV-201A", "HX-104"]`).
  - `text`: Detailed visual inspection report.
  - `metadata`: Image dimensions and processing latency.

---

### 2.3 Document Understanding (`document.understand`)
Deep parsing of technical specifications, inspection reports, and datasheets using Docling (with PyMuPDF fallback).

* **Parameters**:
  - `file_path` (*str*, optional): Path to document (`.pdf`, `.docx`).
  - `do_ocr` (*bool*, optional, default=True): Run OCR on scanned documents.
  - `extract_tables` (*bool*, optional, default=True): Extract structured tables.
  - `extract_figures` (*bool*, optional, default=False): Extract figures and diagrams.
  - `query` (*str*, optional): Direct question to answer from the parsed document.
* **Output Payload**:
  - `markdown`: Normalized structural text content.
  - `tables`: List of extracted tabular grids (rows/columns).
  - `metadata`: Page count, author, creation date.

---

### 2.4 Sandboxed Code Execution (`code.workspace`)
Executes Python scripts inside an isolated Docker container with strict CPU/memory limits and zero network access.

* **Parameters**:
  - `script_code` (*str*, required): Python source code to execute.
  - `timeout_seconds` (*float*, optional, default=60.0): Max execution time before SIGKILL.
  - `input_files` (*dict*, optional): Virtual files injected into the workspace directory.
* **Output Payload**:
  - `exit_code`: Process return code (`0` for success).
  - `stdout`: Standard output capture.
  - `stderr`: Standard error and traceback capture.
  - `generated_files`: Files produced in the sandbox directory.

---

### 2.5 Automated Code Verification & Repair (`code.verify_and_repair`)
Runs a closed-loop verification cycle: generates unit tests, runs them in `code.workspace`, captures failures, and prompts the LLM for automated repair (up to 3 turns).

* **Parameters**:
  - `specification` (*str*, required): Engineering calculation requirement.
  - `initial_code` (*str*, optional): Draft Python code.
  - `test_code` (*str*, optional): Verification test assertions.
  - `max_attempts` (*int*, optional, default=3): Maximum repair attempts.
* **Output Payload**:
  - `status`: `"verified"` or `"failed"`.
  - `verified_code`: Final passing Python script.
  - `attempts`: Number of repair iterations required.
  - `test_results`: Sandbox test output.

---

### 2.6 Artifact Generation (`artifact.generate`)
Deterministically creates audit-ready binary engineering files (Excel XLSX, Word DOCX, PowerPoint PPTX, or PDF).

* **Parameters**:
  - `artifact_type` (*str*, required): `"xlsx"`, `"docx"`, `"pptx"`, or `"pdf"`.
  - `title` (*str*, required): Title of the artifact.
  - `filename` (*str*, optional): Output filename.
  - `data` (*dict*, optional): Multi-sheet tabular data for spreadsheets.
  - `content` (*str*, optional): Markdown or prose sections for documents/slides.
  - `template` (*str*, optional): Industrial template name (`"calculation_workbook"`, `"approval_note"`).
* **Output Payload**:
  - `artifact_id`: Unique identifier for the generated artifact.
  - `file_path`: Absolute path in `artifacts/`.
  - `sha256`: Cryptographic checksum for audit integrity.
  - `download_url`: HTTP endpoint to retrieve the artifact.

---

### 2.7 Workflow Text Analysis (`workflow.text_analysis`)
Single-pass or two-pass factual extraction and executive synthesis.

* **Parameters**:
  - `text` (*str*, required): Text to analyze.
  - `depth` (*str*, optional, default="quick"): `"quick"` (1-pass) or `"detailed"` (2-pass: extraction then synthesis).
  - `focus` (*str*, optional): Focus area (e.g. *"metallurgy and corrosion findings"*).
* **Output Payload**:
  - `summary`: Concise executive summary.
  - `key_points`: Bullet-point list of critical findings.
  - `token_usage`: Total tokens consumed.

---

### 2.8 Bounded Autonomous Agent (`agent.pydantic_ai`)
Executes a multi-turn autonomous investigation using tools registered in the `CapabilityRegistry` under strict policy limits.

* **Parameters**:
  - `prompt` (*str*, required): User request or investigation task.
  - `max_steps` (*int*, optional, default=5): Maximum tool invocations.
  - `allowed_capabilities` (*list*, optional): Whitelist of capability IDs available as tools.
* **Output Payload**:
  - `result`: Final answer text.
  - `tool_calls`: Log of tools invoked with input/output payloads.

---

## 3. Parameter & Output Binding in DAG Plans

When the `LLMPlanner` constructs a multi-step execution plan, it can bind outputs from earlier tasks into downstream task inputs using template syntax:

```json
{
  "task_id": "step_2",
  "capability_id": "retrieval.rag",
  "parameters": {
    "query": "What is the design pressure of {{tasks.step_1.output.detected_tags[0]}}?"
  }
}
```

The `InProcessPlanRunner` resolves these bindings automatically before dispatching each task.

---

## 4. Authoring & Registering a Custom Capability

Developers can add custom capabilities to the workbench in 3 steps:

### Step 1: Implement the `Capability` Protocol
```python
from typing import Any, Dict, Optional
from orchestration.capabilities.base import ExecutionContext, TaskResult
from orchestration.capabilities.descriptor import CapabilityDescriptor

class ThermalEfficiencyCapability:
    @property
    def capability_id(self) -> str:
        return "engineering.thermal_efficiency"

    def get_descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id=self.capability_id,
            description="Calculates heat exchanger thermal efficiency and duty.",
            required_parameters=["inlet_temp", "outlet_temp", "mass_flow"],
        )

    def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> TaskResult:
        tin = float(params["inlet_temp"])
        tout = float(params["outlet_temp"])
        m = float(params["mass_flow"])
        duty_kw = m * 4.184 * (tout - tin)
        return TaskResult(
            status="completed",
            output={"duty_kw": duty_kw, "efficiency_pct": 89.5},
        )
```

### Step 2: Register in `AppContext`
In `apps/context.py`, add the capability inside `create_base_capability_registry()`:
```python
registry.register(ThermalEfficiencyCapability())
```

### Step 3: Register an Intent Route (Optional)
Add a route definition in `create_intent_router()` so natural language queries like *"calculate thermal efficiency"* automatically route to your new capability!
