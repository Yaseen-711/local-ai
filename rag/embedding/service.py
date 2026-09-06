"""Batch chunk embedding service.

Orchestrates converting domain Chunk sequences into EmbeddingResult vectors
with strict 1-to-1 input order preservation, batching, and input validation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from rag.domain.models import Chunk
from rag.embedding.interfaces import EmbeddingModel
from rag.embedding.models import EmbeddingResult


class ChunkEmbeddingService:
    """Service for orchestrating batch vector embedding generation from Chunks."""

    def __init__(self, model: Optional[EmbeddingModel] = None) -> None:
        """Initialize ChunkEmbeddingService.

        Args:
            model: EmbeddingModel implementation. If None, lazily uses default NomicEmbeddingModel.
        """
        if model is None:
            from rag.embedding.nomic import NomicEmbeddingModel

            model = NomicEmbeddingModel()
        self.model = model

    def embed_chunk(self, chunk: Chunk) -> EmbeddingResult:
        """Generate an embedding for a single Chunk.

        Args:
            chunk: Input domain Chunk with content and id.

        Returns:
            EmbeddingResult with vector, dimension, model_name, and chunk_id.
        """
        if not isinstance(chunk, Chunk):
            raise TypeError(f"Expected Chunk instance, got {type(chunk).__name__}")
        if not chunk.id or not str(chunk.id).strip():
            raise ValueError("Chunk id must be a non-empty string")
        if not chunk.content or not chunk.content.strip():
            raise ValueError(f"Chunk '{chunk.id}' content must not be empty or whitespace-only")

        vectors = self.model.embed_documents([chunk.content])
        if not vectors:
            raise RuntimeError(f"Embedding model returned no vector for chunk '{chunk.id}'")

        vector = vectors[0]
        if len(vector) != self.model.dimension:
            raise ValueError(
                f"Embedding vector dimension {len(vector)} does not match model dimension {self.model.dimension}"
            )

        return EmbeddingResult(
            chunk_id=chunk.id,
            vector=vector,
            dimension=self.model.dimension,
            model_name=self.model.model_name,
            is_normalized=self.model.is_normalized,
        )

    def embed_chunks(
        self,
        chunks: Sequence[Chunk],
        batch_size: int = 32,
    ) -> List[EmbeddingResult]:
        """Generate embeddings for a sequence of Chunks in configurable batches.

        Preserves strict 1-to-1 ordering: results[i] corresponds to chunks[i].

        Args:
            chunks: Sequence of domain Chunk objects to embed.
            batch_size: Number of chunks to process per model inference call.

        Returns:
            List of EmbeddingResult objects matching the input sequence order.
        """
        if not isinstance(chunks, Sequence):
            raise TypeError(f"Expected sequence of Chunks, got {type(chunks).__name__}")

        if not chunks:
            return []

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        # Validate all chunks prior to batch processing
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"Item at index {i} is not a Chunk: {type(chunk).__name__}")
            if not chunk.id or not str(chunk.id).strip():
                raise ValueError(f"Chunk at index {i} has empty id")
            if not chunk.content or not chunk.content.strip():
                raise ValueError(f"Chunk '{chunk.id}' at index {i} has empty or whitespace-only content")

        results: List[EmbeddingResult] = []
        total_chunks = len(chunks)

        for offset in range(0, total_chunks, batch_size):
            batch = chunks[offset : offset + batch_size]
            batch_texts = [c.content for c in batch]

            vectors = self.model.embed_documents(batch_texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding model returned {len(vectors)} vectors for a batch of {len(batch)} chunks"
                )

            for chunk, vector in zip(batch, vectors):
                if len(vector) != self.model.dimension:
                    raise ValueError(
                        f"Embedding dimension {len(vector)} for chunk '{chunk.id}' "
                        f"does not match model dimension {self.model.dimension}"
                    )
                results.append(
                    EmbeddingResult(
                        chunk_id=chunk.id,
                        vector=vector,
                        dimension=self.model.dimension,
                        model_name=self.model.model_name,
                        is_normalized=self.model.is_normalized,
                    )
                )

        return results
