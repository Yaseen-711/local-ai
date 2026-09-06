"""Staged escalation router composing Deterministic, Semantic, and LLM classifiers."""

import asyncio
from typing import List, Optional

from orchestration.domain.goals import Goal
from orchestration.routing.base import IntentRouter
from orchestration.routing.deterministic import DeterministicRuleMatcher
from orchestration.routing.llm_classifier import LLMIntentClassifier
from orchestration.routing.semantic import AurelioSemanticRouter
from orchestration.routing.types import ExecutionStrategy, ModelTier, RouteDefinition, RouteResult


class StagedEscalationRouter(IntentRouter):
    """Orchestrates staged intent routing from deterministic rules up to LLM reasoning.

    Escalation sequence:
      Stage 1: DeterministicRuleMatcher (in-memory prefix/regex/exact match)
      Stage 2: AurelioSemanticRouter (local embedding vector cosine similarity)
      Stage 3: LLMIntentClassifier with lightweight model
      Stage 4: LLMIntentClassifier with reasoning model
      Fallback: Default PLAN_REQUIRED strategy
    """

    def __init__(
        self,
        routes: List[RouteDefinition],
        deterministic_matcher: Optional[DeterministicRuleMatcher] = None,
        semantic_router: Optional[AurelioSemanticRouter] = None,
        llm_classifier: Optional[LLMIntentClassifier] = None,
        default_route_name: str = "default_plan",
    ) -> None:
        self.routes = routes
        self.deterministic_matcher = deterministic_matcher or DeterministicRuleMatcher(routes=routes)
        self.semantic_router = semantic_router or AurelioSemanticRouter(routes=routes)
        self.llm_classifier = llm_classifier
        self.default_route_name = default_route_name

    def route(self, goal: Goal) -> RouteResult:
        """Route goal through staged escalation tiers."""
        # Stage 1: Deterministic check
        res = self.deterministic_matcher.match(goal)
        if res is not None:
            return res

        # Stage 2: Aurelio Semantic Router check
        if self.semantic_router is not None:
            res = self.semantic_router.match(goal)
            if res is not None:
                return res

        # Stage 3: Lightweight LLM classification
        if self.llm_classifier is not None:
            res = self.llm_classifier.classify(goal, model_tier=ModelTier.LIGHTWEIGHT)
            if res is not None and res.confidence >= 0.70:
                return res

            # Stage 4: Escalated Reasoning model
            res = self.llm_classifier.classify(goal, model_tier=ModelTier.REASONING)
            if res is not None:
                return res

        # Fallback to default PLAN_REQUIRED
        return RouteResult(
            route_name=self.default_route_name,
            strategy=ExecutionStrategy.PLAN_REQUIRED,
            confidence=0.50,
            stage_resolved="fallback",
            metadata={"reason": "No higher-confidence stage resolved"},
        )

    async def route_async(self, goal: Goal) -> RouteResult:
        """Asynchronously route goal."""
        return await asyncio.to_thread(self.route, goal)
