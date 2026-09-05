"""Deterministic rule matcher (Stage 1 intent routing)."""

import re
from typing import Callable, Dict, List, Optional, Pattern

from orchestration.domain.goals import Goal
from orchestration.routing.types import ExecutionStrategy, RouteDefinition, RouteResult


class DeterministicRuleMatcher:
    """Stage 1 matcher using exact strings, prefixes, and regex rules.

    Operates entirely in-memory with zero token cost and zero neural computation.
    """

    def __init__(self, routes: Optional[List[RouteDefinition]] = None) -> None:
        self._exact_matches: Dict[str, RouteDefinition] = {}
        self._prefix_matches: Dict[str, RouteDefinition] = {}
        self._regex_matches: List[tuple[Pattern[str], RouteDefinition]] = []

        if routes:
            for route in routes:
                self.add_route(route)

    def add_route(self, route: RouteDefinition) -> None:
        """Register a route with its deterministic triggers."""
        # Use route name and explicit metadata triggers
        self._exact_matches[route.name.lower()] = route

        prefixes = route.metadata.get("prefixes", [])
        for p in prefixes:
            self._prefix_matches[p.lower()] = route

        patterns = route.metadata.get("patterns", [])
        for pat in patterns:
            self._regex_matches.append((re.compile(pat, re.IGNORECASE), route))

    def match(self, goal: Goal) -> Optional[RouteResult]:
        """Attempt deterministic matching on goal description or context override."""
        # 1. Check explicit route override in goal context
        override = goal.context.get("route")
        if override and str(override).lower() in self._exact_matches:
            route = self._exact_matches[str(override).lower()]
            return RouteResult(
                route_name=route.name,
                strategy=route.strategy,
                confidence=1.0,
                stage_resolved="deterministic",
                target_capability_id=route.target_capability_id,
                target_model_tier=route.target_model_tier,
                metadata={"matched_by": "context_override"},
            )

        text = goal.description.strip()
        text_lower = text.lower()

        # 2. Exact match
        if text_lower in self._exact_matches:
            route = self._exact_matches[text_lower]
            return RouteResult(
                route_name=route.name,
                strategy=route.strategy,
                confidence=1.0,
                stage_resolved="deterministic",
                target_capability_id=route.target_capability_id,
                target_model_tier=route.target_model_tier,
                metadata={"matched_by": "exact_match"},
            )

        # 3. Prefix match
        for prefix, route in self._prefix_matches.items():
            if text_lower.startswith(prefix):
                return RouteResult(
                    route_name=route.name,
                    strategy=route.strategy,
                    confidence=1.0,
                    stage_resolved="deterministic",
                    target_capability_id=route.target_capability_id,
                    target_model_tier=route.target_model_tier,
                    metadata={"matched_by": "prefix", "prefix": prefix},
                )

        # 4. Regex pattern match
        for pattern, route in self._regex_matches:
            match = pattern.search(text)
            if match:
                extracted = match.groupdict() if match.groupdict() else {}
                return RouteResult(
                    route_name=route.name,
                    strategy=route.strategy,
                    confidence=1.0,
                    stage_resolved="deterministic",
                    target_capability_id=route.target_capability_id,
                    target_model_tier=route.target_model_tier,
                    extracted_parameters=extracted,
                    metadata={"matched_by": "regex", "pattern": pattern.pattern},
                )

        return None
