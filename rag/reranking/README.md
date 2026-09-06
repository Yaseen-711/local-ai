# RAG Candidate Reranking Layer (`rag/reranking/`)

This package provides the second-stage candidate reranking layer for the Local AI Foundation RAG subsystem, applying cross-encoder deep relevance scoring over retrieved candidates.

## Architecture Overview

```text
Vector Retrieval (PgVectorRetriever)
       ↓
Top-K candidates (RetrievedChunk[])
       ↓
Cross-Encoder Reranker (CrossEncoderReranker)
       ↓
Top-N candidates (RankedChunk[])
       ↓
[Future LLM Generation]
```

## Why Reranking Exists: Bi-Encoder vs. Cross-Encoder

1. **First-Stage Retrieval (Bi-Encoder / Embeddings)**:
   - Encodes query and document chunks independently into 768-dimensional dense vectors.
   - Fast and scalable across millions of chunks via pgvector indexed cosine search.
   - **Trade-off**: Cannot model token-level interactions between query words and chunk words during encoding.

2. **Second-Stage Reranking (Cross-Encoder)**:
   - Takes `(query, chunk_content)` pairs together and feeds them through full multi-head cross-attention.
   - Every token in the query attends to every token in the chunk, yielding significantly higher precision relevance judgments.
   - **Trade-off**: Computationally heavier; best applied to a small pool of candidate chunks (e.g. top 10–50).

## Core Concepts & Design Decisions

### 1. Adaptive RAG Compatibility
In an Adaptive RAG system, an upstream router or judge may determine whether retrieval or reranking is necessary for a given query.
- The `Reranker` is an independent, optional stage.
- `RerankingService.rerank_if_enabled()` allows pipelines to pass through candidates without cross-encoder execution when reranking is toggled off, while emitting the uniform `RankedChunk` schema.

### 2. Score Semantics
The default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Score Type**: Unbounded raw logits (typically ranging from approx. -12.0 to +12.0).
- **Semantics**: Higher values indicate stronger relevance. Negative values indicate irrelevant candidates, while positive values indicate strong relevance.
- Raw scores are preserved exactly as emitted by the model. They are **not** artificially normalized into percentages.

### 3. Top-K vs. Top-N
- **Top-K**: The candidate pool retrieved from the vector database (e.g. `top_k = 20`).
- **Top-N**: The final number of candidates returned after reranking (e.g. `top_n = 5`).
- The reranker accepts an optional `top_n` parameter and returns at most `top_n` candidates sorted by `reranking_score DESC`.

### 4. Preservation of Metrics & Metadata
Every `RankedChunk` retains:
- `original_similarity_score` & `original_retrieval_rank` from first-stage retrieval.
- `reranking_score` & `rerank_rank` from the cross-encoder.
- Complete metadata (document ID, file path, headings, page numbers, citations, custom flags).
- Raw `content` without modification.

### 5. Model Loading & Hardware Device Support
- **Model Caching**: Model weights are loaded once upon first invocation and reused across calls.
- **Hardware**: Supports CPU and CUDA automatically, configurable via `RerankerConfig(device="cpu" | "cuda")`.

## Usage Example

```python
from rag.embedding.nomic import NomicEmbeddingModel
from rag.reranking import CrossEncoderReranker, RerankerConfig
from rag.retrieval import PgVectorRetriever
from rag.storage.database import DatabaseManager

db = DatabaseManager()
retriever = PgVectorRetriever(db)
embedder = NomicEmbeddingModel()
reranker = CrossEncoderReranker(RerankerConfig(batch_size=16))

# 1. First-stage vector retrieval: retrieve top 10 candidates
query = "What is the policy for employee annual leave?"
query_vec = embedder.embed_query(query)
candidates = retriever.retrieve(query_vector=query_vec, top_k=10)

# 2. Second-stage reranking: rerank and select top 3
ranked_results = reranker.rerank(query=query, candidates=candidates, top_n=3)

for chunk in ranked_results:
    print(f"Rank {chunk.rerank_rank} (Rerank: {chunk.reranking_score:.2f} | Vector: {chunk.original_similarity_score:.2f})")
    print(f"Content: {chunk.content}\n")
```
