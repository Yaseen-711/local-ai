# Verified Inference Configuration & Runtime Guide

## 1. Inference Engine Overview

The Sovereign Industrial AI Workbench utilizes a native `llama.cpp` inference backend operating in **multi-model router mode**. Rather than spawning isolated single-model CLI commands or maintaining disjoint servers, a single parent `llama-server` process listens on `127.0.0.1:8080` and dynamically manages concurrent model slots defined in `configs/llama_models.ini`.

```text
                               HTTP Request
                                    │
                                    ▼
                 Parent llama-server Router (127.0.0.1:8080)
                 ├── Flag: --models-preset configs/llama_models.ini
                 └── Flag: --models-max 2
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       Slot 1: qwen3.5-9b (4096 ctx)   Slot 2: qwen3.5-0.8b (4096 ctx)
       ├── Model: Qwen3.5-9B-Q4_K_M    └── Model: Qwen3.5-0.8B-Q4_0
       └── Vision: mmproj-q8_0              (Fast intent routing)
            (LLM & Multimodal P&ID)
```

---

## 2. Server Configuration (`configs/llama_models.ini`)

The model router is configured declaratively via INI presets:

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

### Parameter Breakdown

| Setting | Value | Rationale |
|---|---|---|
| `ctx-size` | `4096` | Standard context window sufficient for technical queries, code verification, and chunked RAG synthesis. |
| `batch-size` | `512` | Prompt processing batch size for optimal throughput. |
| `ubatch-size` | `256` | Physical micro-batch size for smooth GPU memory scheduling. |
| `gpu-layers` | `auto` | Offloads all model layers to CUDA VRAM on the RTX 5070 GPU. |
| `cache-type-k` | `q8_0` | 8-bit quantized Key cache for reduced VRAM footprint with zero degradation. |
| `cache-type-v` | `q8_0` | 8-bit quantized Value cache for reduced VRAM footprint. |
| `no-warmup` | `true` | Prevents unnecessary GPU warmup allocations at initial launch. |
| `reasoning` | `off` | Disables verbose internal thinking traces, ensuring deterministic structured responses. |
| `mmproj` | `...mmproj-q8_0.gguf` | Quantized multimodal projector for vision inspection of P&ID diagrams. |

---

## 3. Server Startup & Lifecycle

### Launching the Router
The server is started using the project script:
```bash
./scripts/start_llama_server.sh
```

This executes:
```bash
exec adapters/llama_cpp/build/bin/llama-server \
    --models-preset configs/llama_models.ini \
    --models-max 2 \
    --host 127.0.0.1 \
    --port 8080
```

### Health Check
Verify the server is ready:
```bash
curl -s http://127.0.0.1:8080/health
```
Response:
```json
{"status":"ok"}
```

### Probing Loaded Slots & Models
```bash
curl -s http://127.0.0.1:8080/v1/models
```
Returns definitions for both `qwen3.5-9b` and `qwen3.5-0.8b`.

---

## 4. Multimodal Vision Projector Integration

The `qwen3.5-9b` slot is configured with the `Qwen3.5-9B.mmproj-q8_0.gguf` multimodal projector. This enables direct visual inspection of high-resolution industrial assets (such as P&ID diagrams and process flowsheets) without separate optical character recognition services.

### How Vision Inspection Operates
1. The user or goal passes an image path (e.g. `golden_test_pack/05_pid_direct_input.png`).
2. `VisionInspectionCapability` validates and base64-encodes the image payload.
3. The prompt is injected with multimodal image markers (`[img-...]` / OpenAI-compatible image content blocks).
4. `llama-server` invokes the vision projector to convert image patches into input tokens for the 9B transformer.
5. The model detects tags, equipment boundaries, line ratings, and valve IDs directly from the diagram.

---

## 5. Reasoning Mode & Output Constraints

### Reasoning Configuration
By default, the Qwen 3.5 architecture can produce verbose thinking blocks (`<think>...</think>`). For automated engineering pipelines and tool calling, verbose traces introduce parsing overhead and token latency.

The workbench explicitly disables reasoning traces:
```ini
reasoning = off
```
This causes the model to generate direct, concise, and structured answers.

### Structured Output Constraints
For tasks requiring strict schema compliance (such as `workflow.text_analysis` or `LLMPlanner` DAG output), the Foundation Core applies `OutputConstraint.json()` or custom GBNF grammars. The provider forwards these constraints to `llama-server` to force valid syntax tokens during generation.

---

## 6. VRAM Contention & Resource Management

The workbench is verified on an **NVIDIA GeForce RTX 5070 with 12 GB VRAM**.

### VRAM Budget Allocation

```text
Total Available VRAM: 12,288 MB
├── qwen3.5-9b (Q4_K_M weights + Q8_0 KV cache): ~6,200 MB
├── qwen3.5-0.8b (Q4_0 weights + Q8_0 KV cache):   ~900 MB
├── Qwen3.5-9B.mmproj-q8_0 (Vision Projector):    ~650 MB
├── CUDA Context & System Overhead:                ~800 MB
└── Free Headroom for RAG Embedding / Reranking:  ~3,738 MB
```

### Contention Management Policies
1. **Model Slots**: `--models-max 2` keeps both models hot in GPU memory simultaneously without swapping.
2. **Embedding & Reranker Graceful Fallback**: The Nomic embedding model and MiniLM cross-encoder reranker run on CUDA when headroom is ample. If high context or batch processing elevates GPU memory usage, the reranking pipeline automatically executes on CPU with zero loss in precision.
3. **Sandbox Isolation**: Docker code sandbox containers run with strict memory quotas (`--memory=2g`) and do NOT have access to GPU VRAM, preventing arbitrary Python code from causing CUDA out-of-memory errors.

---

## 7. Single-Model Sequential Hot-Swap Mode (Low VRAM Configuration)

By default, the workbench runs `llama-server` in router mode with `--models-max 2`, keeping both `qwen3.5-9b` and `qwen3.5-0.8b` warm in VRAM concurrently for instantaneous zero-latency routing handoffs.

On hardware with tighter VRAM budgets (such as 8 GB GPUs or 16 GB Apple Silicon MacBooks), you can configure the server to host **only one model at a time with automatic on-demand hot-swapping**.

### 7.1 How to Enable Single-Model Hot-Swapping

In `scripts/start_llama_server.sh`, change `MODELS_MAX` from `2` to `1`:

```bash
# Change in scripts/start_llama_server.sh:
MODELS_MAX="1"
```

Or start `llama-server` directly:
```bash
adapters/llama_cpp/build/bin/llama-server \
    --models-preset configs/llama_models.ini \
    --models-max 1 \
    --host 127.0.0.1 \
    --port 8080
```

### 7.2 How Hot-Swapping Operates Under the Hood

When `--models-max 1` is configured:
1. **LRU Eviction**: `llama-server` allocates a single active slot.
2. **Fast Routing Request**: When a request arrives for `qwen3.5-0.8b` (e.g. Stage 3 intent classification), the router loads the 0.8B weights into memory (~900 MB).
3. **Reasoning / Vision Request**: When the workflow then executes a task requiring `qwen3.5-9b` (e.g. deep engineering reasoning, P&ID visual inspection, or code generation), the parent router automatically unloads `qwen3.5-0.8b`, releases its VRAM and KV cache, and hot-swaps `qwen3.5-9b` into memory.
4. **Subsequent Calls**: If another call to the 9B model arrives, it executes immediately with zero reload overhead because it is already active in the slot.

### 7.3 Memory vs. Latency Trade-Off

| Metric | Concurrent Mode (`--models-max 2`) | Hot-Swap Mode (`--models-max 1`) |
|---|---|---|
| **Peak VRAM Required** | ~7.8 GB (both models + caches) | ~6.2 GB (only the single largest model) |
| **Routing Handoff Latency** | **0 ms** (instant execution) | **~1.5 – 3.0 s** (NVMe weight loading time on swap) |
| **Recommended For** | $\ge$ 12 GB GPUs (RTX 5070 / 4070 / 3080) | 8 GB GPUs (RTX 4060 / 3070) or 16 GB MacBooks |

### 7.4 Dedicated Single-Model Configuration (Bypass Router)

If you prefer to eliminate model swapping entirely and run *exclusively* on the 9B model for all tasks (both routing and reasoning):

1. Edit `configs/llama_models.ini` to define only `[qwen3.5-9b]`:
   ```ini
   [*]
   ctx-size = 4096
   gpu-layers = auto
   cache-type-k = q8_0
   cache-type-v = q8_0
   reasoning = off

   [qwen3.5-9b]
   model = models/gguf/Qwen3.5-9B-Q4_K_M.gguf
   mmproj = models/gguf/Qwen3.5-9B.mmproj-q8_0.gguf
   ```
2. In `configs/settings.toml`, ensure `default_alias = "qwen3.5-9b"`.
3. The `StagedEscalationRouter` will use Stage 1 (regex) and Stage 2 (semantic routing), and if LLM escalation is needed, it will route directly to `qwen3.5-9b`, completely eliminating the 0.8B model dependency.

