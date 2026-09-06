# RAG Embedding Generation Layer (`rag/embedding/`)

This package provides the model-agnostic embedding generation layer for the Local AI Foundation RAG subsystem.

## Architecture

The embedding layer bridges structural chunking & metadata enrichment with downstream vector storage and retrieval.

```text
NormalizedDocument
    ↓
StructuralChunker
    ↓
Chunk[]
    ↓
MetadataPipeline (Provenance & Context Enrichment)
    ↓
ChunkEmbeddingService (Batching & Invariant Validation)
    ↓
EmbeddingModel (NomicEmbeddingModel: nomic-embed-text-v1.5)
    ↓
EmbeddingResult[]
    ↓
[Next: Vector Indexing / PgVectorStore]
```

## Core Contracts & Components

### 1. `EmbeddingResult` (`rag.embedding.models`)
An immutable dataclass encapsulating a vector embedding produced for a chunk:
- `chunk_id: str`: The unique ID of the source `Chunk` (provenance link).
- `vector: List[float]`: Dense float embedding vector.
- `dimension: int`: Embedding dimension (e.g. `768`).
- `model_name: str`: Identifier of the embedding model (e.g. `nomic-ai/nomic-embed-text-v1.5`).
- `is_normalized: bool`: Whether the vector is L2 unit-normalized (`True` for cosine similarity).
- `token_count: Optional[int]`: Optional token count.

Validations:
- `chunk_id` and `model_name` must be non-empty strings.
- `vector` must be non-empty and `len(vector) == dimension`.
- `dimension` must be a positive integer.

### 2. `EmbeddingModel` (`rag.embedding.interfaces`)
A `@runtime_checkable` Python `Protocol` defining the standard model interface:
- `@property dimension -> int`: Expected vector length.
- `@property model_name -> str`: Model identifier.
- `@property is_normalized -> bool`: Whether output vectors are normalized.
- `embed_documents(document_texts: Sequence[str]) -> List[List[float]]`: Batch document embedding with document task prefix.
- `embed_query(query_text: str) -> List[float]`: Query embedding with query task prefix.

### 3. `NomicEmbeddingModel` (`rag.embedding.nomic`)
Concrete implementation using `nomic-ai/nomic-embed-text-v1.5`:
- Dimension: `768`
- Document task prefix: `search_document: `
- Query task prefix: `search_query: `
- L2 unit normalization: `True`
- Lazy model loading via `sentence-transformers` with device support (`cuda`, `cpu`) and custom backend callable for offline testing.
- Implements both `EmbeddingModel` and `rag.domain.interfaces.Embedder`.
- Backwards compatible alias: `NomicEmbedder = NomicEmbeddingModel`.

### 4. `ChunkEmbeddingService` (`rag.embedding.service`)
High-level service that coordinates chunk embedding:
- `embed_chunk(chunk: Chunk) -> EmbeddingResult`: Embeds a single chunk.
- `embed_chunks(chunks: Sequence[Chunk], batch_size: int = 32) -> List[EmbeddingResult]`: Batch processes chunks.

Invariants enforced by `ChunkEmbeddingService`:
- Strict 1-to-1 ordering: `results[i].chunk_id == chunks[i].id`.
- Chunks must be valid `Chunk` instances.
- Chunk content must not be empty or whitespace-only.
- Empty sequence `chunks=[]` gracefully returns `[]`.
- Dimension mismatch between model and outputs fails loudly.

## Usage Example

```python
from rag.domain.models import Chunk
from rag.embedding import ChunkEmbeddingService, NomicEmbeddingModel

# Initialize service (defaults to NomicEmbeddingModel)
service = ChunkEmbeddingService()

chunks = [
    Chunk(id="chunk-1", document_id="doc-1", content="PostgreSQL with pgvector for vector search."),
    Chunk(id="chunk-2", document_id="doc-1", content="Nomic Embed Text produces 768-dim embeddings."),
]

# Generate embeddings in batches of 32
results = service.embed_chunks(chunks, batch_size=32)

for res in results:
    print(f"Chunk ID: {res.chunk_id}, Dim: {res.dimension}, Norm: {res.is_normalized}")
```
