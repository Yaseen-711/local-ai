"""Aurelio Semantic Router seam and adapter (Stage 2 intent routing)."""

import logging
from typing import Dict, List, Optional

try:
    from semantic_router import Route as AurelioRoute, SemanticRouter as AurelioRouter
    from semantic_router.index.local import LocalIndex
    HAS_AURELIO = True
except ImportError:
    HAS_AURELIO = False

from orchestration.domain.goals import Goal
from orchestration.routing.base import SemanticRouterEncoder
from orchestration.routing.encoders import AurelioEncoderAdapter, DeterministicHashEncoder
from orchestration.routing.types import RouteDefinition, RouteResult

logger = logging.getLogger(__name__)


class AurelioSemanticRouter:
    """Stage 2 semantic vector matcher operating behind a strict air-gapped seam.

    Wraps Aurelio SemanticRouter to match goal descriptions against registered route
    utterances in vector space using local CPU embeddings (0 MB VRAM, 100% offline).
    """

    def __init__(
        self,
        routes: Optional[List[RouteDefinition]] = None,
        encoder: Optional[SemanticRouterEncoder] = None,
        threshold: float = 0.60,
    ) -> None:
        if not HAS_AURELIO:
            raise ImportError(
                "Aurelio Semantic Router requires 'semantic-router' to be installed. "
                "Run: pip install semantic-router"
            )

        self.threshold = threshold
        self._routes: Dict[str, RouteDefinition] = {}
        self._aurelio_routes: List[AurelioRoute] = []
        self.encoder = encoder or DeterministicHashEncoder()

        # Wrap with AurelioEncoderAdapter if not already an Aurelio DenseEncoder
        self._adapter = (
            self.encoder
            if hasattr(self.encoder, "type") and hasattr(self.encoder, "score_threshold")
            else AurelioEncoderAdapter(inner=self.encoder, score_threshold=self.threshold)
        )

        self._router: Optional[AurelioRouter] = None

        if routes:
            for r in routes:
                self._routes[r.name] = r
                utterances = r.utterances if r.utterances else [r.description or r.name]
                self._aurelio_routes.append(
                    AurelioRoute(name=r.name, utterances=utterances, score_threshold=self.threshold)
                )
            self._init_router()

    def _init_router(self) -> None:
        """Initialize the underlying Aurelio SemanticRouter with current routes."""
        if not self._aurelio_routes:
            self._router = None
            return

        self._router = AurelioRouter(
            encoder=self._adapter,
            routes=self._aurelio_routes,
            index=LocalIndex(),
            aggregation="max",
            auto_sync="local",
        )

    def add_route(self, route: RouteDefinition) -> None:
        """Register a route with the router and rebuild the local index."""
        self._routes[route.name] = route
        utterances = route.utterances if route.utterances else [route.description or route.name]
        self._aurelio_routes.append(
            AurelioRoute(name=route.name, utterances=utterances, score_threshold=self.threshold)
        )
        self._init_router()

    def match(self, goal: Goal) -> Optional[RouteResult]:
        """Match goal text against route utterance vectors using Aurelio SemanticRouter."""
        text = goal.description.strip()
        if not text or not self._routes or self._router is None:
            return None

        # Execute semantic routing via Aurelio
        choice = self._router(text)

        if choice is not None and choice.name and choice.name in self._routes:
            route = self._routes[choice.name]
            score = float(choice.similarity_score) if choice.similarity_score is not None else 1.0
            return RouteResult(
                route_name=route.name,
                strategy=route.strategy,
                confidence=round(score, 4),
                stage_resolved="semantic_router",
                target_capability_id=route.target_capability_id,
                target_model_tier=route.target_model_tier,
                metadata={"similarity_score": score, "engine": "aurelio_semantic_router"},
            )

        return None
