# Local RAG Developer & Manual Test Harness

A terminal-based inspection and debugging tool for verifying the local RAG pipeline stages on real documents **without involving an LLM**.

```
Document (PDF / Markdown / Text)
       ↓
Docling Document Ingestion (rag/ingestion)
       ↓
Document Normalization (rag/normalization)
       ↓
Structural Chunking (rag/chunking)
       ↓
Metadata & Provenance Enrichment (rag/metadata)
       ↓
Nomic Embedding Generation (rag/embedding)
       ↓
pgvector Persistence & Indexing (rag/indexing)
       ↓
pgvector Cosine Retrieval (rag/retrieval)
       ↓
Cross-Encoder Reranking (rag/reranking)
       ↓
[LLM Generation — NOT IMPLEMENTED]
```

---

## 1. Prerequisites

### PostgreSQL with pgvector
Ensure the PostgreSQL container is running with the `vector` extension enabled:

```bash
# Check if container is running
docker ps --filter "name=local-ai-postgres"

# If not running, start it
docker start local-ai-postgres
# or run via docker-compose:
# docker compose up -d postgres
```

### Python Virtual Environment & Hugging Face Cache
Ensure the local virtual environment is active:
```bash
source .venv/bin/activate
```

The system uses local caches for models:
- Embedding model: `nomic-ai/nomic-embed-text-v1.5`
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

To run in offline mode using the pre-cached weights:
```bash
export HF_HUB_OFFLINE=1
```

---

## 2. Quickstart & Usage

### Interactive Menu Mode
Launch the interactive terminal interface:
```bash
HF_HUB_OFFLINE=1 python -m rag.cli
```

You will be presented with the interactive menu:
```
============================================================
              LOCAL RAG TEST HARNESS CLI
============================================================
[1] Ingest Document (PDF, Markdown, Text)
[2] Ask Question (Retrieve + Rerank)
[3] Inspect Document & Chunks
[4] Clear Test Data
[5] Exit
============================================================
```

### Direct CLI Flags (Non-Interactive / Scriptable)

You can perform one-off operations directly from the command line:

#### Ingest a Document
```bash
HF_HUB_OFFLINE=1 python -m rag.cli --ingest path/to/document.pdf
```

#### Query Document(s)
```bash
HF_HUB_OFFLINE=1 python -m rag.cli --query "What is the primary topic of the document?"
```

#### Custom Top-K and Top-N
Retrieve top 10 candidates from pgvector, rerank and display top 3:
```bash
HF_HUB_OFFLINE=1 python -m rag.cli --query "..." --top-k 10 --top-n 3
```

#### Ingest and Query in One Command
```bash
HF_HUB_OFFLINE=1 python -m rag.cli --ingest sample.pdf --query "Summarize chapter 1"
```

#### Enable Debug Trace
```bash
HF_HUB_OFFLINE=1 python -m rag.cli --debug --query "..."
```

---

## 3. Pipeline Stages & Output Inspection

### Ingestion Inspection
When a document is ingested, the CLI displays step-by-step progress and timing breakdowns:
- `[1/6] Ingesting document via Docling...`
- `[2/6] Normalizing document content...`
- `[3/6] Applying structural chunking...`
- `[4/6] Enriching metadata & provenance...`
- `[5/6] Generating Nomic vector embeddings...`
- `[6/6] Indexing chunks in PostgreSQL + pgvector...`

Summary statistics include:
- Total chunks generated
- Heading distribution (H1, H2, H3, etc.)
- Average and max chunk character lengths
- Total characters and tokens
- Per-stage latency breakdown (seconds)

### Query Inspection: Retrieval vs Reranking

When a question is queried, the CLI displays two distinct ranking blocks side-by-side:

#### 1. Vector Retrieval Results (pgvector)
- **Rank**: 1 to `top_k`
- **Chunk ID**: Deterministic UUID
- **Similarity Score**: Cosine similarity in range `[-1.0, 1.0]` (typically `0.5` - `0.9` for relevant content with unit-normalized Nomic embeddings)
- **Provenance**: Document name, page numbers, heading hierarchy path (`H1 > H2 > H3`)
- **Content Preview**: Truncated snippet of chunk text

#### 2. Cross-Encoder Reranking Results (MiniLM)
- **Rank**: 1 to `top_n`
- **Original Vector Rank**: Prior position from vector retrieval
- **Rank Shift**:
  - `[+N]`: Candidate promoted by reranker
  - `[-N]`: Candidate demoted by reranker
  - `[=]`: Rank remained unchanged
- **Rerank Score**: **Raw logit score** from `cross-encoder/ms-marco-MiniLM-L-6-v2` (`-inf` to `+inf`).
  > **Note on Scores:** The reranker score is an uncalibrated logit score, **not a percentage** and **not a cosine similarity**. High relevance is typically positive (`> 0`, e.g. `+3.2` or `+6.8`), while low relevance or irrelevant content produces negative scores (e.g. `-4.1` or `-8.5`).
- **Full Text**: Complete chunk text showing exactly what would be passed to an LLM.

#### 3. Pipeline Summary
Concludes with:
```
Pipeline Summary:
  Ingested Documents in Store: N
  Vector Retrieval Candidates: K (threshold: 0.0)
  Reranked Top Candidates:     N
  Pipeline Latency:            XX.XX ms (retrieval: XX.XX ms, rerank: XX.XX ms)
  LLM Generation:              NOT IMPLEMENTED (test harness terminates at reranked context)
```

---

## 4. Manual Test Cases Walkthrough

Use the following 7 test scenarios to verify the end-to-end RAG pipeline on real documents:

### Test Case 1: Direct Factual Retrieval
- **Goal:** Verify that a chunk containing an exact definition or number is retrieved and reranked as Rank 1.
- **Action:** Query a specific term, date, or metric present in the ingested document.
- **Expected:**
  - High cosine similarity score (> 0.75).
  - Positive rerank logit score (> 2.0).
  - Rank 1 in both retrieval and reranking.

### Test Case 2: Paraphrased / Semantic Query
- **Goal:** Verify dense semantic retrieval where the query shares few or no exact words with the source text.
- **Action:** Formulate a query using synonyms and conceptual restatements.
- **Expected:**
  - Vector retrieval identifies the semantically relevant chunk despite vocabulary mismatch.
  - Cross-encoder reranker promotes the chunk to top ranks with high relevance logit.

### Test Case 3: Heading Hierarchy & Metadata Query
- **Goal:** Verify that structural breadcrumbs (`Chapter > Section > Subsection`) are preserved and displayed.
- **Action:** Query a topic located deep in a nested section.
- **Expected:**
  - The chunk's `Heading Path` in the CLI output accurately reflects the breadcrumb trail.
  - Page number provenance corresponds to the exact location in the source PDF.

### Test Case 4: Irrelevant / Out-of-Scope Query
- **Goal:** Verify pipeline behavior when the query is completely unrelated to the ingested document.
- **Action:** Query an entirely unrelated topic (e.g., asking about quantum mechanics on a recipe document).
- **Expected:**
  - Vector similarity scores are notably lower (< 0.50).
  - Cross-encoder reranker scores are strongly negative (< -2.0).
  - Clear indication that no chunk is genuinely relevant.

### Test Case 5: Dense / Structured Table Content
- **Goal:** Verify extraction and retrieval of markdown tables or bulleted lists.
- **Action:** Query specific data points located within a table.
- **Expected:**
  - Table structure formatted as clean markdown chunks.
  - Chunk preview renders rows and column alignments legibly.

### Test Case 6: Retrieval vs Reranking Rank Shift
- **Goal:** Verify that the cross-encoder reranker re-evaluates lexical/semantic overlap and corrects vector ordering.
- **Action:** Query a nuanced question where a superficial chunk has high embedding similarity, but a deeper chunk actually answers the question.
- **Expected:**
  - Vector Rank 2 or 3 has a rank shift indicator (`[+1]`, `[+2]`) promoting it to Reranker Rank 1.
  - Rank shift indicators clearly display the movement.

### Test Case 7: Edge Cases & Validation
- **Goal:** Verify robust error handling.
- **Action:** Test empty queries, whitespace queries, nonexistent file paths, unsupported extensions, and empty search indexes.
- **Expected:**
  - Informative error messages without uncaught Python exceptions or crashes.
