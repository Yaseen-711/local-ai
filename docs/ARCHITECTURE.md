# Local AI Foundation — Core Architecture (Stage 1)

## 1. Architectural Philosophy

The Local AI Foundation is a self-hosted, modular infrastructure layer designed to decouple consumer applications (code review, data analysis, document extraction, RAG, reports, agents, and chat) from underlying model formats, inference runtimes, and deployment topologies.

```text
Workflows / Applications (`workflows/`)
        │
        ▼
Connector Layer (`connectors.InferenceConnector`)
        │
        ▼
Foundation Core (`core.FoundationCore`)
   ├── Model Registry & Config (Declarative TOML definitions, advisory disk availability)
   ├── Provider Manager (Provider dispatch, thread-safe synchronization)
   └── Inference Contracts (Normalized Request ──► Normalized Response)
        │
        ▼
Provider Manager / Provider Adapters (`LlamaCppProvider` via HTTP client mode)
        │
        ▼
Inference Runtime (`llama-server` on 127.0.0.1:8080)
        │
        ▼
Models & Hardware (GGUF / CUDA 12.8 / NVIDIA RTX 5070)
```

---

## 2. Separation of Concerns & State Ownership

1. **Model Registry Ownership**:
   - Owns canonical model IDs, aliases, declared capabilities, and metadata loaded from `configs/models/*.toml`.
   - Owns cached **advisory** filesystem availability (`AvailabilityInfo`).
   - Does **NOT** own runtime process state or loaded model authority.
2. **Provider / Runtime Ownership**:
   - The provider layer (`LlamaCppProvider`) is authoritative for runtime execution and connectivity.
   - Probes runtime health (`RuntimeState.READY`, `UNAVAILABLE`, `ERROR`).
3. **Filesystem vs Configuration**:
   - The filesystem determines path existence and file size.
   - Configuration defines capabilities, roles, aliases, and provider compatibility.
   - Filesystem is never scanned on each inference request; scans occur on startup or explicit `registry.refresh()`.

---

## 3. Data Contracts

### 3.1 Model Schema (`core.models.schema`)
* `ModelDefinition`: Immutable declaration of model metadata, supported providers, aliases, and capabilities.
* `ModelCapabilities`: Declared flags (`chat`, `code`, `reasoning`, `structured_output`, `context_window`).
* `AvailabilityInfo`: Advisory file presence and byte size on disk.

### 3.2 Inference Contracts (`core.inference.types`)
* `Message`: Structured message representation (`MessageRole.SYSTEM`, `USER`, `ASSISTANT`, `TOOL`).
* `GenerationOptions`: Normalized parameters (`temperature`, `top_p`, `max_tokens`, `stop_sequences`, `seed`).
* `InferenceRequest`: Normalized container for model ID, messages list, and generation options.
* `TokenUsage`: Normalized accounting (`prompt_tokens`, `completion_tokens`, `total_tokens`).
* `InferenceResponse`: Normalized output with generated message, finish reason (`FinishReason.STOP`, `LENGTH`), token usage, latency (ms), and optional diagnostic `raw_response`.

---

## 4. Model Configuration Schema

Individual model configuration files reside in `configs/models/<model-id>.toml`:

```toml
[model]
id = "qwen3.5-9b"
display_name = "Qwen 3.5 9B Q4_K_M"
format = "gguf"
path = "models/gguf/Qwen3.5-9B-Q4_K_M.gguf"
aliases = ["qwen3.5", "qwen-9b", "default"]
supported_providers = ["llama_cpp"]
roles = ["general", "coding"]

[capabilities]
chat = true
code = true
reasoning = false
structured_output = true
context_window = 4096

[metadata]
quantization = "Q4_K_M"
parameter_count = "9B"
architecture = "qwen35"
verified_date = "2026-09-01"
```

---

## 5. Reusable Execution Boundary (`FoundationCore.infer`)

The execution boundary is the interface through which higher-level workflows request AI inference:

```text
Higher-Level Workflows (Chat, RAG, Agents, Tool Pipelines, Code Review, Document Gen)
                                  │
                                  ▼
                    FoundationCore.infer(request)
               [or convenience: core.infer_prompt(...)]
                                  │
                                  ▼
               ProviderManager.execute_inference(request)
               ├── 1. Resolve ModelDefinition (via ModelRegistry)
               ├── 2. Select compatible Provider (via ProviderManager)
               └── 3. Translate ID & Dispatch (via BaseProvider.infer)
                                  │
                                  ▼
                   Normalized InferenceResponse
```

### Architectural Principles of the Execution Boundary
1. **Stateless & Reusable**:
   `FoundationCore.infer()` is a reusable callable execution component, **not a fixed pipeline stage**.
   Higher-level workflows may invoke it:
   * Once for simple queries or chat.
   * Multiple times across iterative agent loops or planning steps.
   * After document retrieval in RAG workflows.
   * Before and after tool/sandbox execution.
   * As an internal helper inside code review or PDF/report generation workflows.
2. **Workflow Agnostic**:
   The Foundation execution layer itself does **not** determine whether a task is RAG, an agentic loop, document processing, code analysis, or chat. Those decisions belong strictly to higher-level orchestration layers.
3. **Runtime Model Identifier Translation**:
   The provider owns translation from Foundation model identities/aliases to runtime-specific model identifiers. Foundation-only aliases (such as `"default"` or `"general"`) are resolved to canonical definitions, and the provider sends the verified server identifier to the backend runtime.
4. **Strict Error Boundary (No Silent Fallbacks)**:
   The execution layer reliably reports failure domains (`ModelNotFoundError`, `ProviderNotFoundError`, `ProviderUnavailableError`, `InferenceError`, `ProviderResponseError`). It **does not** silently guess or fall back to another model. Higher-level orchestration layers retain explicit authority over retry, fallback, or error recovery strategies.

---

## 6. The Connector Layer (`connectors.InferenceConnector`)

The Connector Layer provides the capability integration boundary between future orchestration/workflow layers and the underlying Foundation:

```text
Future Orchestrators / Workflows (Chat, RAG, Agents, Code Review, Document Gen)
                                   │
                                   ▼
                InferenceConnector (Structural Protocol)
                                   │
                                   ▼
             FoundationInferenceConnector (In-Process Bridge)
                                   │
                                   ▼
                    FoundationCore.infer(request)
                                   │
                                   ▼
               ProviderManager.execute_inference(request)
                                   │
                                   ▼
                            LlamaCppProvider
                                   │
                                   ▼
                              llama-server
```

### Key Principles of the Connector Layer
1. **Structural Protocol Typing**:
   `InferenceConnector` is defined using Python's `@runtime_checkable Protocol` rather than an ABC. Workflows and test mocks can satisfy the interface structurally (via duck typing) without requiring direct class inheritance.
2. **Strict Boundary Separation**:
   * **Connector**: Provides access to a specific capability (model inference). Owns zero workflow sequencing, loops, branching, or prompt decision-making.
   * **Orchestrator**: Decides *what to do* (state machines, tool coordination, retrieval, multi-step loops). Owns error recovery and fallback policies.
   * **Foundation**: Owns model metadata, provider dispatch, and inference execution.
3. **No Hidden Orchestration**:
   The connector is strictly a capability gateway. It does not inspect prompt contents, does not classify tasks, and does not maintain multi-turn workflow state.
4. **Explicit Error Passthrough**:
   Exceptions (`ModelNotFoundError`, `ProviderUnavailableError`, `InferenceError`) flow straight through the connector to the caller. The connector does not swallow errors or perform silent model fallbacks.
5. **Thread Safety & Execution Semantics**:
   The connector delegates directly to `FoundationCore`. Thread safety for provider access is enforced at the `ProviderManager` and `ModelRegistry` layers via thread synchronization locks (`threading.RLock`). Callers are responsible for coordinating any multi-turn conversation state.

---

## 7. Usage Examples

### 7.1 Using the Connector in Workflows (Dependency Injection)
```python
from connectors import FoundationInferenceConnector, InferenceConnector
from core import FoundationCore, GenerationOptions

# 1. Initialize Foundation
core = FoundationCore.create()

# 2. Instantiate Connector
connector: InferenceConnector = FoundationInferenceConnector(core=core)

# 3. Pass Connector into a higher-level workflow
class CodeReviewWorkflow:
    def __init__(self, inference: InferenceConnector) -> None:
        self._inference = inference

    def review(self, code_snippet: str) -> str:
        response = self._inference.infer_prompt(
            model_id="default",
            prompt=f"Review this code for potential security issues:\n{code_snippet}",
            options=GenerationOptions(temperature=0.1),
        )
        return response.text
```

### 7.2 Structured Execution with `InferenceRequest`
```python
from connectors import FoundationInferenceConnector
from core import FoundationCore, InferenceRequest, GenerationOptions

core = FoundationCore.create()
connector = FoundationInferenceConnector(core)

request = InferenceRequest.from_prompt(
    model_id="default",  # Foundation alias resolved seamlessly
    prompt="Explain binary search in Python.",
    options=GenerationOptions(temperature=0.2, max_tokens=256),
)
response = connector.infer(request)
print("Generated text:", response.text)
print(f"Tokens: {response.usage.total_tokens} (Latency: {response.latency_ms:.1f}ms)")
```

### 7.3 Streamlined Single-Prompt Execution with `infer_prompt`
```python
response = connector.infer_prompt(
    model_id="qwen3.5-9b",
    prompt="Summarize the core findings in 3 bullet points.",
    system_prompt="You are a technical analyst.",
    options=GenerationOptions(temperature=0.1),
)
print("Summary:", response.text)
```

---

## 8. The Workflow Layer (`workflows/`)

The Workflow Layer establishes **conventions and shared contracts** for higher-level systems that consume connectors. It deliberately does **not** impose a base class, Protocol, pipeline engine, or execution framework.

### Design Rationale

Workflows are not polymorphic. A chat workflow, a code review workflow, a RAG workflow, and an agent loop differ fundamentally in input types, output types, control flow, and error handling. Forcing them into a common `Workflow` ABC or Protocol creates a false abstraction. Instead, the layer provides:

1. **Constructor Injection Convention**: Workflows receive connectors via `__init__()`.
2. **Shared Result Envelope**: `WorkflowResult[T]` carries typed output + metadata.
3. **Workflow Error Boundary**: `WorkflowError` distinguishes workflow-level failures from infrastructure errors.

### `WorkflowResult[T]`

A generic dataclass providing a common result envelope:

```python
@dataclass
class WorkflowResult(Generic[T]):
    output: T                              # Primary output, typed per workflow
    model_id: Optional[str] = None         # Model that produced the output (if applicable)
    metadata: Dict[str, Any] = {}          # Workflow-specific diagnostic data
    errors: List[str] = []                 # Non-fatal warnings/issues
```

- `output` is generic (`T`) so each workflow carries its own output type.
- `model_id` is optional because not every workflow step involves inference.
- `errors` captures non-fatal issues; fatal errors raise exceptions.

### `WorkflowError`

A domain exception under `FoundationError`:

```text
FoundationError
├── ConfigurationError
├── ModelRegistryError
│   ├── ModelNotFoundError
│   └── ModelUnavailableError
├── ProviderError
│   ├── ProviderNotFoundError
│   ├── ProviderUnavailableError
│   ├── ProviderResponseError
│   ├── InferenceError
│   └── LifecycleConflictError
└── WorkflowError              ← workflow-level failures
```

Workflows may:
- Let infrastructure errors propagate directly (no catch).
- Catch and wrap infrastructure errors with additional context: `raise WorkflowError(...) from cause`.

Both patterns are valid. The workflow author decides.

### Workflow Convention: Constructor Injection

```python
from connectors import InferenceConnector
from workflows import WorkflowResult

class CodeReviewWorkflow:
    def __init__(self, inference: InferenceConnector) -> None:
        self._inference = inference

    def review(self, code: str) -> WorkflowResult[str]:
        response = self._inference.infer_prompt(
            model_id="default",
            prompt=f"Review this code:\n{code}",
        )
        return WorkflowResult(output=response.text, model_id=response.model_id)
```

### What the Workflow Layer Does NOT Contain

| Non-concern | Reason |
|---|---|
| `Workflow` Protocol or ABC | Workflows are not polymorphic |
| Pipeline / DAG engine | Constrains control flow |
| Workflow registry / discovery | Application-level concern |
| Agent loops or planners | Built when agents are needed |
| RAG / retrieval contracts | Built when RAG is needed |
| Tool execution contracts | Built when tools are needed |
| Session / conversation state | Application-level concern |

### Data / Dependency Flow

```text
Application Code
        │  constructs
        ▼
  SomeWorkflow(inference=connector, ...)
        │  calls domain methods
        ▼
  Workflow calls self._inference.infer_prompt(...)  ← any number of times
        │
        ▼
  InferenceConnector (Protocol)
        │
        ▼
  FoundationInferenceConnector → FoundationCore → ProviderManager → llama-server
```

Testing substitutes the connector:

```text
  Test Code
        │  constructs with fake/mock
        ▼
  SomeWorkflow(inference=fake_connector)
        │
        ▼
  fake_connector returns canned responses
        │
        ▼
  Assert on WorkflowResult
```

---

## 9. Testing Strategy

* **Unit Tests (`tests/unit/`)**: Fast, pure CPU tests with zero GPU, model weight, or live server dependencies. Covers TOML parsing, contracts, registry, provider routing, runtime ID mapping, connector delegation/error propagation, and workflow contract/convention verification. Executes in < 0.2s.
* **Integration Tests (`tests/integration/`)**: Opt-in tests verifying live communication when `llama-server` is started (`scripts/start_llama_server.sh`). Skips cleanly when the server is offline.
