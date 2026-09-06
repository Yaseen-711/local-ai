"""Local Cross-Encoder implementation of the Reranker protocol.

Provides CrossEncoderReranker using Hugging Face cross-encoder models
(such as cross-encoder/ms-marco-MiniLM-L-6-v2) for deep query-document relevance scoring.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from rag.reranking.interfaces import Reranker
from rag.reranking.models import RankedChunk, RerankerConfig
from rag.retrieval.models import RetrievedChunk


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker evaluating query and candidate pairs together.

    Features:
    - Pairwise joint cross-attention scoring: (query, candidate_content).
    - Batch prediction with configurable batch size and device.
    - Model caching: loads weights once and reuses instance across calls.
    - Custom backend injection for deterministic unit testing.
    - Preserves all candidate metadata and original retrieval scores.
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        backend: Optional[Callable[[List[Tuple[str, str]]], Sequence[float]]] = None,
    ) -> None:
        """Initialize CrossEncoderReranker.

        Args:
            config: RerankerConfig specifying model_name, device, batch_size, max_length.
            backend: Optional callable for mocking predictions without loading model weights.
        """
        self.config = config or RerankerConfig()
        self._custom_backend = backend
        self._model: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Name of the underlying reranking model."""
        return self.config.model_name

    def _get_model(self) -> Any:
        """Lazily load and cache the underlying CrossEncoder model."""
        if self._custom_backend is not None:
            return None

        if self._model is None:
            from rag.offline import (
                ensure_offline_environment,
                get_expected_model_path,
                is_model_available_locally,
                is_offline_mode,
                OfflineModelNotFoundError,
            )

            ensure_offline_environment()
            offline_active = is_offline_mode()
            expected_location = get_expected_model_path(self.config.model_name)

            # Strict fail-closed: verify model exists locally before invoking loader
            if offline_active and not is_model_available_locally(self.config.model_name):
                raise OfflineModelNotFoundError(
                    model_name=self.config.model_name,
                    component="CrossEncoderReranker",
                    expected_location=str(expected_location),
                )

            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.config.model_name,
                    device=self.config.device,
                    max_length=self.config.max_length,
                    local_files_only=offline_active,
                )
            except ImportError as exc:
                raise ImportError(
                    "The 'sentence-transformers' package is required for CrossEncoderReranker. "
                    "Install sentence-transformers or provide a custom backend callable."
                ) from exc
            except Exception as exc:
                if offline_active:
                    raise OfflineModelNotFoundError(
                        model_name=self.config.model_name,
                        component="CrossEncoderReranker",
                        expected_location=str(expected_location),
                        details=str(exc),
                    ) from exc
                raise

        return self._model

    def _validate_inputs(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: Optional[int],
    ) -> None:
        """Validate query text, candidates sequence, and top_n parameter."""
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        if not isinstance(candidates, Sequence):
            raise TypeError(f"candidates must be a Sequence, got {type(candidates).__name__}")

        if top_n is not None:
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError(f"top_n must be a positive integer, got {top_n}")

        for i, c in enumerate(candidates):
            if not isinstance(c, RetrievedChunk):
                raise TypeError(
                    f"Candidate at index {i} is not a RetrievedChunk: {type(c).__name__}"
                )
            if not c.content or not c.content.strip():
                raise ValueError(
                    f"Candidate chunk '{c.chunk_id}' at index {i} has empty content"
                )

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RankedChunk]:
        """Score and re-order candidate chunks relative to a query.

        Args:
            query: Raw user query string.
            candidates: Sequence of RetrievedChunk instances.
            top_n: Optional maximum number of top results to return.

        Returns:
            List of RankedChunk instances ordered by reranking_score descending.
        """
        self._validate_inputs(query, candidates, top_n)

        # Early return for empty candidates without loading model
        if not candidates:
            return []

        # Prepare (query, chunk_content) pairs
        pairs: List[Tuple[str, str]] = [(query, c.content) for c in candidates]

        # Compute cross-encoder relevance scores
        if self._custom_backend is not None:
            raw_scores = self._custom_backend(pairs)
        else:
            model = self._get_model()
            raw_scores = model.predict(
                pairs,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
            )

        if isinstance(raw_scores, (int, float)):
            scores = [float(raw_scores)]
        else:
            scores = [float(s) for s in raw_scores]

        if len(scores) != len(candidates):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )

        # Pair candidates with their computed scores
        scored_pairs = list(zip(candidates, scores))

        # Stable sort: primary descending by score, secondary ascending by chunk_id
        scored_pairs.sort(key=lambda item: (-item[1], item[0].chunk_id))

        # Truncate to top_n if requested
        if top_n is not None:
            scored_pairs = scored_pairs[:top_n]

        # Construct RankedChunk instances with assigned rerank_rank
        ranked_results: List[RankedChunk] = []
        for rank_idx, (candidate, score) in enumerate(scored_pairs, start=1):
            ranked_results.append(
                RankedChunk.from_retrieved_chunk(
                    candidate=candidate,
                    reranking_score=score,
                    rerank_rank=rank_idx,
                )
            )

        return ranked_results
