"""Nomic Embed Text v1.5 embedding model implementation.

Provides NomicEmbeddingModel conforming to the EmbeddingModel protocol and
domain Embedder protocol, handling required task prefixes, producing 768-dimensional
L2-normalized vectors.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence

from rag.domain.interfaces import Embedder
from rag.embedding.interfaces import EmbeddingModel

EMBEDDING_DIMENSION = 768
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def l2_normalize(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length (L2 norm)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


class NomicEmbeddingModel(EmbeddingModel, Embedder):
    """Nomic-embed-text-v1.5 implementation of EmbeddingModel and Embedder.

    Features:
    - 768-dimensional embeddings.
    - Automatic task prefixing:
        - Documents: 'search_document: '
        - Queries: 'search_query: '
    - L2 unit normalization for cosine similarity search.
    - Encapsulated model loading with support for custom backends.
    - Device configuration (cpu, cuda, etc.).
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        normalize: bool = True,
        backend: Optional[Callable[[List[str]], List[List[float]]]] = None,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self.normalize = normalize
        self._custom_backend = backend
        self.device = device
        self._loaded_model: Optional[Any] = None

    @property
    def dimension(self) -> int:
        """The dimensionality of output embeddings."""
        return EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        return self._model_name

    @property
    def is_normalized(self) -> bool:
        """Whether produced embeddings are L2 normalized."""
        return self.normalize

    def _get_model(self) -> Any:
        """Lazily load the underlying SentenceTransformer model if no custom backend is set."""
        if self._custom_backend is not None:
            return self._custom_backend
        if self._loaded_model is None:
            from rag.offline import (
                ensure_offline_environment,
                get_expected_model_path,
                is_model_available_locally,
                is_offline_mode,
                OfflineModelNotFoundError,
            )

            ensure_offline_environment()
            offline_active = is_offline_mode()
            expected_location = get_expected_model_path(self._model_name)

            # Strict fail-closed: verify model exists locally before invoking loader
            if offline_active and not is_model_available_locally(self._model_name):
                raise OfflineModelNotFoundError(
                    model_name=self._model_name,
                    component="NomicEmbeddingModel",
                    expected_location=str(expected_location),
                )

            try:
                from sentence_transformers import SentenceTransformer

                self._loaded_model = SentenceTransformer(
                    self._model_name,
                    device=self.device,
                    trust_remote_code=True,
                    local_files_only=offline_active,
                )
            except ImportError as exc:
                raise ImportError(
                    f"The 'sentence-transformers' package is required to load {self._model_name}. "
                    "Install it or supply a custom backend callable."
                ) from exc
            except Exception as exc:
                if "CUDA out of memory" in str(exc) or "OutOfMemoryError" in type(exc).__name__:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        from sentence_transformers import SentenceTransformer
                        self._loaded_model = SentenceTransformer(
                            self._model_name,
                            device="cpu",
                            trust_remote_code=True,
                            local_files_only=offline_active,
                        )
                        return self._loaded_model
                    except Exception:
                        pass
                if offline_active:
                    raise OfflineModelNotFoundError(
                        model_name=self._model_name,
                        component="NomicEmbeddingModel",
                        expected_location=str(expected_location),
                        details=str(exc),
                    ) from exc
                raise
        return self._loaded_model

    def _encode(self, formatted_texts: List[str]) -> List[List[float]]:
        """Encode formatted texts into 768-dimensional embeddings."""
        if not formatted_texts:
            return []

        if self._custom_backend is not None:
            raw_embeddings = self._custom_backend(formatted_texts)
        else:
            model = self._get_model()
            try:
                raw_embeddings = model.encode(
                    formatted_texts,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True,
                ).tolist()
            except Exception as exc:
                if "CUDA out of memory" in str(exc) or "OutOfMemoryError" in type(exc).__name__:
                    import torch
                    torch.cuda.empty_cache()
                    model.to("cpu")
                    raw_embeddings = model.encode(
                        formatted_texts,
                        normalize_embeddings=self.normalize,
                        convert_to_numpy=True,
                    ).tolist()
                else:
                    raise

        result: List[List[float]] = []
        for idx, emb in enumerate(raw_embeddings):
            if len(emb) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding at index {idx} has invalid dimension {len(emb)}. "
                    f"Expected {EMBEDDING_DIMENSION} dimensions for {self._model_name}."
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
    # EmbeddingModel Protocol Implementation
    # ------------------------------------------------------------------

    def embed_documents(self, document_texts: Sequence[str]) -> List[List[float]]:
        """Convert a sequence of document texts into 768-dimensional vector embeddings."""
        if not isinstance(document_texts, Sequence):
            raise TypeError(f"Expected a sequence of strings, got {type(document_texts).__name__}")

        if not document_texts:
            return []

        formatted_batch = [self.format_text(t, is_query=False) for t in document_texts]
        return self._encode(formatted_batch)

    def embed_query(self, query_text: str) -> List[float]:
        """Convert a single query text into a 768-dimensional vector embedding."""
        formatted = self.format_text(query_text, is_query=True)
        embeddings = self._encode([formatted])
        return embeddings[0]

    # ------------------------------------------------------------------
    # Embedder Protocol Implementation (rag.domain.interfaces.Embedder)
    # ------------------------------------------------------------------

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Convert a single text string into a 768-dimensional vector embedding."""
        formatted = self.format_text(text, is_query=is_query)
        embeddings = self._encode([formatted])
        return embeddings[0]

    def embed_texts(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        """Convert a sequence of text strings into vector embeddings."""
        if not isinstance(texts, Sequence):
            raise TypeError(f"Expected a sequence of strings, got {type(texts).__name__}")

        if not texts:
            return []

        formatted_batch = [self.format_text(t, is_query=is_query) for t in texts]
        return self._encode(formatted_batch)

    # ------------------------------------------------------------------
    # Convenience Named Methods
    # ------------------------------------------------------------------

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        """Convenience method to embed multiple queries with 'search_query: ' prefix."""
        return self.embed_texts(queries, is_query=True)

    def embed_document(self, document_text: str) -> List[float]:
        """Convenience method to embed a single document with 'search_document: ' prefix."""
        return self.embed_text(document_text, is_query=False)


# Backwards compatibility alias
NomicEmbedder = NomicEmbeddingModel
