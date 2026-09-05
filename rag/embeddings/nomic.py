"""Nomic Embed Text v1.5 Embedder implementation.

Provides NomicEmbedder conforming to the domain Embedder protocol, handling
the required document/query task prefixes and producing 768-dimensional normalized vectors.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence

from rag.domain.interfaces import Embedder

EMBEDDING_DIMENSION = 768
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def l2_normalize(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length (L2 norm)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


class NomicEmbedder(Embedder):
    """Nomic-embed-text-v1.5 implementation of the Embedder protocol.

    Features:
    - 768-dimensional embeddings.
    - Automatic task prefixing:
        - Documents: 'search_document: '
        - Queries: 'search_query: '
    - L2 unit normalization for cosine similarity search.
    - Encapsulated model loading with support for custom backends.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        normalize: bool = True,
        backend: Optional[Callable[[List[str]], List[List[float]]]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._custom_backend = backend
        self.device = device
        self._loaded_model: Optional[Any] = None

    def _get_model(self) -> Any:
        """Lazily load the underlying SentenceTransformer model if no custom backend is set."""
        if self._custom_backend is not None:
            return self._custom_backend
        if self._loaded_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._loaded_model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True,
                )
            except ImportError as exc:
                raise ImportError(
                    f"The 'sentence-transformers' package is required to load {self.model_name}. "
                    "Install it or supply a custom backend callable."
                ) from exc
        return self._loaded_model

    def _encode(self, formatted_texts: List[str]) -> List[List[float]]:
        """Encode formatted texts into 768-dimensional embeddings."""
        if not formatted_texts:
            return []

        if self._custom_backend is not None:
            raw_embeddings = self._custom_backend(formatted_texts)
        else:
            model = self._get_model()
            raw_embeddings = model.encode(
                formatted_texts,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            ).tolist()

        result: List[List[float]] = []
        for idx, emb in enumerate(raw_embeddings):
            if len(emb) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding at index {idx} has invalid dimension {len(emb)}. "
                    f"Expected {EMBEDDING_DIMENSION} dimensions for {self.model_name}."
                )
            vec = [float(x) for x in emb]
            if self.normalize and self._custom_backend is not None:
                vec = l2_normalize(vec)
            result.append(vec)

        return result

    def format_text(self, text: str, is_query: bool = False) -> str:
        """Apply the appropriate Nomic task prefix if not already present."""
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Input text must not be empty or whitespace-only")

        prefix = QUERY_PREFIX if is_query else DOCUMENT_PREFIX
        if text.startswith(prefix):
            return text
        return f"{prefix}{text}"

    # ------------------------------------------------------------------
    # Embedder Protocol Implementation
    # ------------------------------------------------------------------

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Convert a single text string into a 768-dimensional vector embedding.

        Args:
            text: Text content to embed.
            is_query: If True, uses 'search_query: ' prefix; else uses 'search_document: '.

        Returns:
            768-dimensional float list.
        """
        formatted = self.format_text(text, is_query=is_query)
        embeddings = self._encode([formatted])
        return embeddings[0]

    def embed_texts(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        """Convert a sequence of text strings into vector embeddings.

        Args:
            texts: Sequence of strings to embed.
            is_query: If True, uses 'search_query: ' prefix; else uses 'search_document: '.

        Returns:
            List of 768-dimensional float lists, matching the input length.
        """
        if not isinstance(texts, Sequence):
            raise TypeError(f"Expected a sequence of strings, got {type(texts).__name__}")

        if not texts:
            return []

        formatted_batch = [self.format_text(t, is_query=is_query) for t in texts]
        return self._encode(formatted_batch)

    # ------------------------------------------------------------------
    # Convenience Named Methods
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> List[float]:
        """Convenience method to embed a single query with 'search_query: ' prefix."""
        return self.embed_text(query, is_query=True)

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        """Convenience method to embed multiple queries with 'search_query: ' prefix."""
        return self.embed_texts(queries, is_query=True)

    def embed_document(self, document_text: str) -> List[float]:
        """Convenience method to embed a single document with 'search_document: ' prefix."""
        return self.embed_text(document_text, is_query=False)

    def embed_documents(self, document_texts: Sequence[str]) -> List[List[float]]:
        """Convenience method to embed multiple documents with 'search_document: ' prefix."""
        return self.embed_texts(document_texts, is_query=False)
