# Local AI Foundation — Core Architecture (Stage 1)

## 1. Architectural Philosophy

The Local AI Foundation is a self-hosted, modular infrastructure layer designed to decouple consumer applications (code review, data analysis, document extraction, RAG, reports, agents, and chat) from underlying model formats, inference runtimes, and deployment topologies.

```text
Applications / Consumers
        │
        ▼
Foundation Core
   ├── Model Registry & Config (Declarative TOML definitions, advisory disk availability)
   ├── Provider Manager (Provider dispatch, thread-safe synchronization)
   └── Inference Contracts (Normalized Request ──► Normalized Response)
        │
        ▼
Provider Adapters (LlamaCppProvider via HTTP client mode)
        │
        ▼
Inference Runtime (llama-server on 127.0.0.1:8080)
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

## 5. Usage Example

```python
from core import FoundationCore, InferenceRequest, GenerationOptions

# 1. Initialize Foundation Core with auto-loaded registry and providers
core = FoundationCore.create()

# 2. Inspect available models
available_models = core.registry.list_available_models()
for model in available_models:
    print(f"Available: {model.id} ({model.display_name})")

# 3. Submit normalized inference request
request = InferenceRequest.from_prompt(
    model_id="qwen3.5-9b",
    prompt="Explain binary search in Python.",
    options=GenerationOptions(temperature=0.2, max_tokens=256),
)

# 4. Receive normalized response
response = core.infer(request)
print("Generated text:", response.text)
print(f"Tokens: {response.usage.total_tokens} (Latency: {response.latency_ms:.1f}ms)")
```

---

## 6. Testing Strategy

* **Unit Tests (`tests/unit/`)**: Fast, pure CPU tests with zero GPU, model weight, or live server dependencies. Uses standard mock fixtures and executes in < 0.2s.
* **Integration Tests (`tests/integration/`)**: Opt-in tests verifying live communication when `llama-server` is started (`scripts/start_llama_server.sh`). Skips cleanly when the server is offline.
