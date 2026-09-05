"""Semantic Router seam and adapter (Stage 2 intent routing)."""

from typing import Dict, List, Optional

from orchestration.domain.goals import Goal
from orchestration.routing.base import SemanticRouterEncoder
from orchestration.routing.encoders import DeterministicHashEncoder
from orchestration.routing.types import RouteDefinition, RouteResult


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute dot product of two normalized unit vectors."""
    if len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


class AurelioSemanticRouter:
    """Stage 2 semantic vector matcher operating behind a strict air-gapped seam.

    Uses an injected SemanticRouterEncoder (defaulting to DeterministicHashEncoder)
    to calculate cosine similarity between goal descriptions and defined route utterances.
    """

    def __init__(
        self,
        routes: Optional[List[RouteDefinition]] = None,
        encoder: Optional[SemanticRouterEncoder] = None,
        threshold: float = 0.60,
    ) -> None:
        self.encoder = encoder or DeterministicHashEncoder()
        self.threshold = threshold
        self._routes: Dict[str, RouteDefinition] = {}
        self._route_vectors: Dict[str, List[List[float]]] = {}

        if routes:
            for r in routes:
                self.add_route(r)

    def add_route(self, route: RouteDefinition) -> None:
        """Register a route and compute embeddings for its sample utterances."""
        self._routes[route.name] = route
        utterances = route.utterances if route.utterances else [route.description or route.name]
        vectors = self.encoder.encode(utterances)
        self._route_vectors[route.name] = vectors

    def match(self, goal: Goal) -> Optional[RouteResult]:
        """Match goal text against route utterance vectors using cosine similarity."""
        text = goal.description.strip()
        if not text or not self._route_vectors:
            return None

        query_vec = self.encoder.encode([text])[0]

        best_route_name: Optional[str] = None
        best_score: float = -1.0

        for r_name, vectors in self._route_vectors.items():
            for v in vectors:
                sim = _cosine_similarity(query_vec, v)
                if sim > best_score:
                    best_score = sim
                    best_route_name = r_name

        if best_route_name is not None and best_score >= self.threshold:
            route = self._routes[best_route_name]
            return RouteResult(
                route_name=route.name,
                strategy=route.strategy,
                confidence=round(best_score, 4),
                stage_resolved="semantic_router",
                target_capability_id=route.target_capability_id,
                target_model_tier=route.target_model_tier,
                metadata={"similarity_score": best_score},
            )

        return None
