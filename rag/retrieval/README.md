# RAG Vector Retrieval / Semantic Similarity Search (`rag/retrieval/`)

This package provides the semantic vector retrieval layer for the Local AI Foundation RAG subsystem, performing cosine similarity search over persisted chunks in PostgreSQL + `pgvector`.

## Architecture Overview

```text
User Query
    ↓
EmbeddingModel.embed_query()
    ↓
Query Vector (768-dim float list)
    ↓
PgVectorRetriever
    ↓
PostgreSQL 17 + pgvector 0.8.6 (<=> cosine distance)
    ↓
Top-K RetrievedChunks
    ↓
[Future Reranker]
    ↓
[Future LLM]
```

## Core Principles & Decoupling

### 1. Query Embedding vs. Retrieval
- The **embedding layer** (`rag/embedding/`) is responsible for converting raw query text into a dense vector embedding (`EmbeddingModel.embed_query()`).
- The **retrieval layer** (`rag/retrieval/`) receives a pre-computed float vector and executes vector similarity search against the database.
- The retriever **does not directly own or depend on any specific embedding model** (such as Nomic), allowing models to be substituted or upgraded without altering retrieval logic.

### 2. Retrieval vs. Reranking & LLM Generation
- Vector retrieval is strictly candidate selection: finding the top-K chunks by latent semantic vector proximity.
- Reranking (e.g. cross-encoders, reciprocal rank fusion, hybrid search) and LLM generation remain separate, independent stages.

## Core Components

### 1. `RetrievedChunk` (`rag.retrieval.models`)
An immutable, typed domain model representing a retrieval match:
- `chunk_id: str`: The unique ID of the matched chunk.
- `document_id: str`: The parent source document identifier.
- `content: str`: Raw, unpolluted text content.
- `metadata: Dict[str, Any]`: Preserved provenance, headings, and custom tags.
- `similarity_score: float`: Cosine similarity score $s \in [-1.0, 1.0]$.
- `rank: int`: 1-indexed position in results (1 = highest similarity).
- `chunk_index: int`: Sequential position in the parent document.

### 2. `VectorRetriever` (`rag.retrieval.interfaces`)
A `@runtime_checkable` Python `Protocol` defining the retrieval contract:
- `@property dimension -> int`: Expected query vector length (e.g. 768).
- `retrieve(query_vector, top_k=5, document_id=None, similarity_threshold=None, filters=None) -> List[RetrievedChunk]`

### 3. `PgVectorRetriever` (`rag.retrieval.retriever`)
Concrete implementation using PostgreSQL 17 and pgvector 0.8.6:
- Metric: **Cosine Similarity** via the `<=>` distance operator.
- Distance translation: pgvector computes distance $d = 1 - \cos(\theta)$. Similarity is converted as $s = 1.0 - d$.
- Deterministic ordering: Results are ordered by `distance ASC` with a secondary tie-breaker on `id ASC`.
- Top-K: Hard `.limit(top_k)` executed in SQL. Chunks are never loaded into Python for in-memory sorting.
- Threshold filtering: If `similarity_threshold` is specified, candidates must satisfy $s \ge \text{threshold}$ ($d \le 1.0 - \text{threshold}$).
- Document scoping: Optional `document_id` filter to narrow search to a specific document.
- Metadata filtering: Optional JSONB containment (`@>`) via `filters` dictionary.
- Empty results: Gracefully returns an empty list `[]` when no chunks match.

## Usage Example

```python
from rag.embedding.nomic import NomicEmbeddingModel
from rag.retrieval import PgVectorRetriever
from rag.storage.database import DatabaseManager

db_manager = DatabaseManager()
retriever = PgVectorRetriever(db_manager)
embedder = NomicEmbeddingModel()

# 1. Embed query text into 768-dim vector
query_vector = embedder.embed_query("How do employees apply for annual leave?")

# 2. Retrieve top 3 candidates with similarity threshold >= 0.5
results = retriever.retrieve(
    query_vector=query_vector,
    top_k=3,
    similarity_threshold=0.5,
)

for res in results:
    print(f"Rank {res.rank}: Chunk {res.chunk_id} (Score: {res.similarity_score:.4f})")
    print(f"Content: {res.content}\n")
```
