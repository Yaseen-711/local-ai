# Verified Models & Offline Storage Guide

## 1. Storage Policy & Air-Gap Compliance

All machine learning model weights are strictly hosted on local storage outside of Git version control. The Sovereign Industrial AI Workbench is architected for air-gapped deployment where external network connections are disallowed.

### Air-Gap Enforcement
The environment strictly enforces offline operation using system environment variables:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAG_OFFLINE_MODE=true
```

Large model files must never be committed to Git. The repository tracks only directory structures via `.gitkeep` files, while ignoring binary weights through `.gitignore`.

```text
models/
├── .gitkeep
└── gguf/
    ├── .gitkeep
    ├── Qwen3.5-9B-Q4_K_M.gguf          (~5.3 GB)
    ├── Qwen3.5-0.8B-Q4_0.gguf          (~563 MB)
    └── Qwen3.5-9B.mmproj-q8_0.gguf     (~624 MB)
```

---

## 2. Model Inventory

The complete workbench operates across 5 verified models covering LLM reasoning, fast routing, multimodal vision, vector embeddings, and cross-encoder reranking:

| Model Role | Model Identifier / Path | Format | Size | Architecture |
|---|---|---|---|---|
| **Primary Reasoning & Coding** | `models/gguf/Qwen3.5-9B-Q4_K_M.gguf` | GGUF (Q4_K_M) | 5.3 GB | Qwen 3.5 9B |
| **Fast Intent Classification** | `models/gguf/Qwen3.5-0.8B-Q4_0.gguf` | GGUF (Q4_0) | 563 MB | Qwen 3.5 0.8B |
| **Multimodal Vision Projector** | `models/gguf/Qwen3.5-9B.mmproj-q8_0.gguf` | GGUF (Q8_0) | 624 MB | Qwen 3.5 Vision |
| **Dense Vector Embeddings** | `nomic-ai/nomic-embed-text-v1.5` | PyTorch / SafeTensors | ~550 MB | Nomic Bert (768d) |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | PyTorch / SafeTensors | ~90 MB | MiniLM-L-6 |

---

## 3. Model Details & Capabilities

### 3.1 Primary Reasoning Model (`qwen3.5-9b`)
* **Path**: `models/gguf/Qwen3.5-9B-Q4_K_M.gguf`
* **Parameters**: 9.2 Billion
* **Quantization**: Q4_K Medium (optimal balance of perplexity and VRAM usage)
* **Context Window**: 4096 tokens (up to 32k supported by architecture)
* **Primary Roles**:
  - Complex industrial problem solving
  - Technical document analysis and table interpretation
  - Code generation, verification, and automated repair
  - Multi-task DAG planning via `LLMPlanner`
  - Grounded question answering with source citations

### 3.2 Fast Classifier Model (`qwen3.5-0.8b`)
* **Path**: `models/gguf/Qwen3.5-0.8B-Q4_0.gguf`
* **Parameters**: 800 Million
* **Quantization**: Q4_0
* **Context Window**: 4096 tokens
* **Primary Roles**:
  - Stage 3 fast intent classification in `StagedEscalationRouter`
  - Rapid entity extraction and query disambiguation
  - High-throughput metadata tagging with sub-50ms latency

### 3.3 Multimodal Vision Projector (`qwen3.5-9b` Vision)
* **Path**: `models/gguf/Qwen3.5-9B.mmproj-q8_0.gguf`
* **Quantization**: Q8_0
* **Role**: Plugs into the `qwen3.5-9b` model slot in `llama-server` to inspect Piping & Instrumentation Diagrams (P&IDs), process engineering flowsheets, and technical schematics.

### 3.4 Vector Embedding Model (`nomic-embed-text-v1.5`)
* **Hugging Face Hub ID**: `nomic-ai/nomic-embed-text-v1.5`
* **Embedding Dimension**: 768
* **Normalization**: L2 unit normalization for cosine similarity search
* **Task Prefixes**:
  - Ingestion: Prepend `search_document: ` to all stored text chunks
  - Retrieval: Prepend `search_query: ` to all incoming search queries
* **Storage**: Ingested vectors are stored in PostgreSQL (`local_ai_rag`) using the `pgvector` extension.

### 3.5 Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`)
* **Hugging Face Hub ID**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Architecture**: Cross-Encoder (joint query-chunk token interaction)
* **Scoring**: Computes a continuous logit score representing query-document relevance.
* **Role in System**: Used strictly as a **ranking signal** to order candidates for final LLM synthesis, never as an arbitrary hard threshold that drops valid evidence.

---

## 4. Declarative Model Configurations

Model metadata and capabilities are declared in TOML files located in `configs/models/`.

### Example: `configs/models/qwen3.5-9b.toml`
```toml
[model]
id = "qwen3.5-9b"
display_name = "Qwen 3.5 9B Q4_K_M"
format = "gguf"
path = "models/gguf/Qwen3.5-9B-Q4_K_M.gguf"
aliases = ["qwen3.5", "qwen-9b", "default"]
supported_providers = ["llama_cpp"]
roles = ["general", "coding", "reasoning"]

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

### Example: `configs/models/qwen3.5-0.8b.toml`
```toml
[model]
id = "qwen3.5-0.8b"
display_name = "Qwen 3.5 0.8B Q4_0"
format = "gguf"
path = "models/gguf/Qwen3.5-0.8B-Q4_0.gguf"
aliases = ["qwen-fast", "router-fast"]
supported_providers = ["llama_cpp"]
roles = ["routing", "classification"]

[capabilities]
chat = true
code = false
reasoning = false
structured_output = true
context_window = 4096

[metadata]
quantization = "Q4_0"
parameter_count = "0.8B"
architecture = "qwen35"
verified_date = "2026-09-01"
```

---

## 5. How to Add or Swap a Different GGUF Model

The Sovereign AI Workbench is strictly model-agnostic. The higher layers (`InferenceConnector`, `DecisionEngine`, `LLMPlanner`, capabilities) interact only with normalized data contracts. You can substitute or add any compatible GGUF model (e.g. Llama-3.1, Mistral-NeMo, DeepSeek-R1-Distill, Gemma 2, Phi-4, or an alternate Qwen variant) in 5 straightforward steps.

### Step 1: Place the New GGUF File

Copy your target GGUF file into the local model directory:
```bash
cp /path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf models/gguf/
```
*(Remember: Large model binaries are ignored by `.gitignore` and must not be committed to Git).*

### Step 2: Create a Declarative Model Configuration

Create a new TOML file in `configs/models/<model-id>.toml` (for example, `configs/models/llama-3.1-8b.toml`):

```toml
[model]
id = "llama-3.1-8b"
display_name = "Llama 3.1 8B Instruct Q4_K_M"
format = "gguf"
path = "models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
aliases = ["llama-3.1", "llama8b", "default"]  # Add "default" if this should be the default model
supported_providers = ["llama_cpp"]
roles = ["general", "coding", "reasoning"]

[capabilities]
chat = true
code = true
reasoning = false
structured_output = true
context_window = 8192

[metadata]
quantization = "Q4_K_M"
parameter_count = "8B"
architecture = "llama"
verified_date = "2026-09-07"
```

### Step 3: Register the Model with `llama-server` Presets

Add the new model block into `configs/llama_models.ini`:

```ini
[llama-3.1-8b]
model = models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
# Optional: if the model supports visual inspection, add:
# mmproj = models/gguf/<projector-name>.gguf
```

*Note on slot limits*: If you are keeping `qwen3.5-0.8b` and replacing `qwen3.5-9b`, ensure `MODELS_MAX="2"` in `scripts/start_llama_server.sh`. If hosting 3 models simultaneously, increase `MODELS_MAX="3"` (ensure your VRAM budget allows this) or switch to single-model hot-swap mode (`MODELS_MAX="1"`).

### Step 4: Set as Default Model (Optional)

If you want the new model to be the primary model for all workbench workflows, update `configs/settings.toml`:

```toml
[providers.llama_cpp]
base_url = "http://127.0.0.1:8080"
timeout_seconds = 60
default_alias = "llama-3.1-8b"
```

### Step 5: Restart the Server & Verify

1. **Restart the inference server**:
   ```bash
   pkill -f llama-server || true
   ./scripts/start_llama_server.sh
   ```

2. **Confirm model registration**:
   ```bash
   curl -s http://127.0.0.1:8080/v1/models | jq .
   ```
   The output should list `llama-3.1-8b` among the registered models.

3. **Verify inference via Python**:
   ```python
   from apps import AppContext

   ctx = AppContext.create()
   response = ctx.inference.infer_prompt(
       model_id="llama-3.1-8b",
       prompt="Verify that you are running and state your model name.",
   )
   print("Response:", response.text)
   ```

### Important Considerations When Swapping Models

1. **Vision Capabilities**: If swapping the primary model with a non-multimodal model, the `vision.inspect` capability will require an active model slot that provides a compatible `mmproj` vision projector file.
2. **Chat Templates**: `llama-server` automatically parses and uses the embedded chat template inside the GGUF file metadata (e.g. ChatML, Llama-3 `<|begin_of_text|>`, Mistral `[INST]`). No manual prompt formatting code needs to be altered in application layers.
3. **Quantization Recommendation**: On an RTX 5070 (12 GB VRAM) or Apple Silicon MacBook (16-32 GB), `Q4_K_M` or `Q5_K_M` quantization provides the best balance between memory efficiency and numerical precision for industrial engineering tasks.

