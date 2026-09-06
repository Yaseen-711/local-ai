# RAG Vector Indexing & Persistence Layer (`rag/indexing/`)

This package provides the vector persistence and indexing layer for the Local AI Foundation RAG subsystem, persisting domain `Chunk` objects and their computed `EmbeddingResult` representations into PostgreSQL with `pgvector`.

## Architecture Overview

```text
Chunk + EmbeddingResult
           ↓
[PgVectorIndexer Validation]
  - Pairwise identity check: chunk.id == embedding.chunk_id
  - Dimension check: len(vector) == indexer.dimension (768)
  - Numeric integrity: all vector values finite (no NaN, Inf)
  - Metadata serializability: valid JSON
           ↓
[Atomic Database Transaction]
  - Ensure parent DocumentModel exists (satisfies FK)
  - Idempotent upsert via ON CONFLICT (id) DO UPDATE
           ↓
PostgreSQL 17 + pgvector 0.8.6
  - rag_documents
  - rag_chunks (embedding: Vector(768))
```

## Indexing vs. Retrieval Separation

> [!IMPORTANT]
> **Vector Indexing is strictly WRITE / UPDATE / DELETE**:
> - Responsible for data ingestion, transaction safety, validation, upsert, and deletion.
> - **Does NOT perform query operations, similarity ranking, top-k retrieval, or nearest neighbor searches.**
> - Retrieval operations belong strictly to downstream search and retrieval layers.

## Core Components

### 1. `VectorIndexer` (`rag.indexing.interfaces`)
A `@runtime_checkable` Python `Protocol` defining the persistence contract:
- `@property dimension -> int`: Expected vector dimension (e.g. 768).
- `index_chunk(chunk, embedding) -> None`: Persist or update a single chunk.
- `index_chunks(chunks, embeddings) -> int`: Persist or update a batch of chunks atomically.
- `index_document(document, chunks, embeddings) -> int`: Persist a parent document and all its chunks.
- `delete_chunk(chunk_id) -> bool`: Delete a single chunk.
- `delete_document_chunks(document_id) -> int`: Delete all chunks belonging to a document.
- `delete_document(document_id) -> bool`: Delete a document and cascade-delete its chunks.

### 2. `PgVectorIndexer` (`rag.indexing.indexer`)
Concrete implementation using PostgreSQL 17 + pgvector 0.8.6:
- Reuses existing `DatabaseManager` and SQLAlchemy ORM models (`DocumentModel`, `ChunkModel`).
- Vector column: `Vector(768)` matching `nomic-ai/nomic-embed-text-v1.5`.

## Key Operational Behaviors

### 1. Chunk ↔ Vector Relationship
The indexer never trusts input ordering alone. For every element pair `(chunk, embedding)`, it explicitly enforces:
```python
if chunk.id != embedding.chunk_id:
    raise ValueError(...)
```
This guarantees an unambiguous link between chunk text and its mathematical vector representation.

### 2. Metadata Preservation
All chunk metadata (document ID, file path, headings, page numbers, provenance, and custom flags) is preserved intact in PostgreSQL `JSONB`. Additionally, the indexer enriches the stored metadata with embedding provenance (`embedding_model`, `is_normalized`, and optional `token_count`). `chunk.content` is never modified or polluted.

### 3. Idempotency & Upserts
Indexing is safe to execute multiple times on the same chunks:
- Uses PostgreSQL `ON CONFLICT (id) DO UPDATE`.
- Re-indexing an existing chunk updates its content, embedding, and metadata in-place without generating duplicates or primary key errors.

### 4. Transaction Safety & Batch Semantics
Batch operations run within a single atomic database session:
- Pre-validation verifies all pairs before issuing database statements.
- If any database failure occurs during the batch, the entire transaction is rolled back cleanly.

### 5. Deletion & Re-indexing Support
- `delete_chunk(chunk_id)`: Granular single chunk removal.
- `delete_document_chunks(document_id)`: Removes all chunks for a document while keeping the document record.
- `delete_document(document_id)`: Cascading removal of document and all associated chunks.

## Usage Example

```python
from rag.domain.models import Chunk, Document
from rag.embedding.models import EmbeddingResult
from rag.indexing import PgVectorIndexer
from rag.storage.database import DatabaseManager

db_manager = DatabaseManager()
indexer = PgVectorIndexer(db_manager)

chunk = Chunk(
    id="chunk-001",
    document_id="doc-001",
    content="pgvector allows performant vector storage in PostgreSQL.",
    metadata={"heading": "Storage Layer", "page_number": 1},
)

embedding = EmbeddingResult(
    chunk_id="chunk-001",
    vector=[0.05] * 768,
    dimension=768,
    model_name="nomic-ai/nomic-embed-text-v1.5",
    is_normalized=True,
)

# Index single chunk
indexer.index_chunk(chunk, embedding)

# Delete chunk
indexer.delete_chunk("chunk-001")
```
