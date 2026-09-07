# Sovereign RAG Subsystem — Deep Dive

The Sovereign RAG (Retrieval-Augmented Generation) subsystem provides an air-gapped, verifiable knowledge base engineered specifically for industrial technical documentation, operating manuals, and inspection reports.

---

## 1. Subsystem Architecture

```text
       Ingestion Phase                                    Query Phase
┌──────────────────────────────┐              ┌──────────────────────────────┐
│ Technical Document (PDF/DOCX)│              │ User Query / Engineering Goal│
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│ Ingestion & Normalization    │              │ Query Prefix Injection:      │
│ (Docling / PyMuPDF fallback) │              │ "search_query: <query>"      │
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│ Hierarchical Chunking        │              │ Nomic Embedder (768d, L2)    │
│ (Preserves headings & pages) │              └──────────────┬───────────────┘
└──────────────┬───────────────┘                             │
               │                                             ▼
               ▼                              ┌──────────────────────────────┐
┌──────────────────────────────┐              │ Candidate Retrieval (Top-K)  │
│ Document Prefix Injection:   │              │ PostgreSQL local_ai_rag       │
│ "search_document: <chunk>"   │              │ pgvector Cosine Distance     │
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│ Nomic Embedder (768d, L2)    │              │ Cross-Encoder Reranking      │
└──────────────┬───────────────┘              │ (ms-marco-MiniLM-L-6-v2)     │
               │                              └──────────────┬───────────────┘
               ▼                                             │
┌──────────────────────────────┐                             ▼
│ PostgreSQL Database:         │              ┌──────────────────────────────┐
│ `local_ai_rag` with pgvector │─────────────►│ Grounded QA Synthesis        │
│ Table: document_chunks       │              │ Positive Citing / Refusal    │
└──────────────────────────────┘              └──────────────────────────────┘
```

---

## 2. Ingestion & Structural Chunking

Industrial documents (such as P&ID datasheets or API 510 inspection reports) contain critical structured tables and hierarchical sections. Naive fixed-character chunking breaks table relationships and separates notes from their parent headings.

### Ingestion Pipeline (`rag/ingestion/`)
* Extracts text, tables, and section hierarchies using Docling.
* Converts complex multi-column layouts into normalized Markdown.
* Preserves 1-indexed document page boundaries.

### Chunking Rules (`rag/chunking/`)
* **Token Budget**: 500 to 1000 tokens per chunk with 10% token overlap.
* **Heading Preservation**: Every chunk retains its breadcrumb section trail (`heading_path`, e.g. `["Section 4: Valves", "4.1 Control Valves", "FV-201A"]`).
* **Table Integrity**: Tables are preserved intact as Markdown tables rather than being split across chunks wherever possible.

---

## 3. Embedding Mechanics (`rag/embedding/`)

Vector embeddings are produced using **`nomic-ai/nomic-embed-text-v1.5`**.

* **Vector Dimension**: 768 dimensions.
* **Context Length**: Up to 8192 tokens.
* **Task-Specific Prefixes**: The Nomic architecture requires explicit task prefixes to orient the embedding space:
  - Text Chunks at Ingestion: Prepend `"search_document: "`
  - Search Queries at Retrieval: Prepend `"search_query: "`
* **L2 Normalization**: Vectors are L2-normalized to unit length:
  $$\hat{v} = \frac{v}{\|v\|_2}$$
  This allows dot products and cosine similarity to be computed with maximum efficiency:
  $$\text{sim}(u, v) = \hat{u} \cdot \hat{v}$$

---

## 4. Vector Storage in PostgreSQL (`local_ai_rag`)

The RAG store is housed in a dedicated PostgreSQL database named `local_ai_rag` equipped with the `pgvector` extension.

### Schema: `document_chunks`
```sql
CREATE TABLE document_chunks (
    chunk_id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    heading_path TEXT[],
    page_numbers INT[],
    metadata JSONB,
    embedding vector(768)
);
```

### Retrieval Query
Candidates are queried using pgvector's cosine distance operator (`<=>`):
```sql
SELECT
    chunk_id,
    document_id,
    content,
    heading_path,
    page_numbers,
    1 - (embedding <=> :query_embedding) AS similarity_score
FROM document_chunks
ORDER BY embedding <=> :query_embedding ASC
LIMIT :top_k;
```

---

## 5. Cross-Encoder Reranking (`rag/reranking/`)

Dense vector retrieval (bi-encoders) compresses query and passage into isolated vectors. To achieve high precision on dense engineering data, the candidate pool (top-10) is reranked using **`cross-encoder/ms-marco-MiniLM-L-6-v2`**.

### How Cross-Attention Operates
The Cross-Encoder processes `(query, passage)` jointly through all transformer layers simultaneously, allowing all query tokens to cross-attend to all passage tokens.

### Ranking Signal vs. Hard Cutoff Rule
> [!IMPORTANT]
> The cross-encoder score is treated strictly as a **ranking signal** to order candidates for the LLM synthesis window. It is **not** used as an aggressive binary rejection filter, ensuring that relevant edge-case context is never silently discarded before synthesis.

---

## 6. Grounded Question Answering & Verification

The grounded QA engine (`rag/domain/grounded_qa.py`) enforces strict factual integrity:

### Positive Grounding
Answers must cite their evidence source, including:
- Source document filename (e.g. `01_equipment_spec.pdf`)
- Section heading trail (e.g. `[Specification Sheet > Design Parameters]`)
- Page number (e.g. `Page 1`)

### Negative Grounding (Out-of-Corpus Refusal)
When a user queries an asset or property not present in the corpus (e.g. *"What is the design pressure of ZX-999?"*), the system explicitly refuses:
> *"Asset ZX-999 was not found in the indexed knowledge base. No authoritative specification is available."*

Zero hallucinated values are permitted.

### Conflict Detection
If two documents provide conflicting specifications (e.g., Spec A states `45.0 barg` while Spec B states `50.0 barg`), the system highlights the discrepancy to the engineer rather than picking an arbitrary number.

---

## 7. Command-Line Utilities

The RAG subsystem provides standalone CLI scripts for ingestion and validation:

```bash
# Ingest a document
.venv/bin/python -m rag.cli.ingest golden_test_pack/01_equipment_spec.pdf

# Search candidate chunks
.venv/bin/python -m rag.cli.search "What is the design pressure of FV-201A?"

# Run end-to-end Grounded QA
.venv/bin/python -m rag.cli.qa "What are the design pressure and material of FV-201A?"
```
