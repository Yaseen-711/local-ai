"""LLM-based intent classifier (Stage 3 and Stage 4 escalation).

Fixes applied in this version:
- Issue 2: ModelSelectionPolicy injected; model_id resolved per model_tier at call time.
- Issue 3: Structured diagnostic exception handling replaces bare except: pass.
- Issue 4: strategy is always authoritative from RouteDefinition, never overridden by LLM output.
"""

import json
import logging
from typing import Dict, List, Optional

from connectors import InferenceConnector
from core.common.errors import SyntaxParsingError
from core.inference.types import GenerationOptions, OutputConstraint
from core.common.parsing import parse_json_payload
from orchestration.domain.goals import Goal
from orchestration.routing.model_selector import ModelSelectionPolicy
from orchestration.routing.types import ExecutionStrategy, ModelTier, RouteDefinition, RouteResult

logger = logging.getLogger(__name__)


class LLMIntentClassifier:
    """Stage 3/4 classifier using a lightweight or capable model via InferenceConnector.

    The caller (StagedEscalationRouter) passes a ``model_tier`` to ``classify()``;
    the injected ``ModelSelectionPolicy`` resolves that tier to a concrete model_id
    at call time.  This ensures Stage 3 uses the LIGHTWEIGHT model and Stage 4 uses
    the REASONING model without hard-coding any model identifiers here.

    RouteDefinition.strategy is always authoritative.  The LLM selects a *route*;
    the route's declared strategy is then used — the LLM JSON ``strategy`` field is
    ignored to prevent the model from overriding execution strategy.
    """

    def __init__(
        self,
        connector: InferenceConnector,
        routes: List[RouteDefinition],
        model_selection_policy: ModelSelectionPolicy,
    ) -> None:
        self._connector = connector
        self._routes = {r.name: r for r in routes}
        self._policy = model_selection_policy

    def classify(self, goal: Goal, model_tier: ModelTier = ModelTier.LIGHTWEIGHT) -> Optional[RouteResult]:
        """Classify a goal using structured LLM inference.

        Returns a RouteResult on success, or None to signal the caller should
        escalate further or fall back.  None is returned (with a log entry) for:
        - Infrastructure failures (connector/provider errors)
        - JSON parsing failures
        - Responses where the LLM-selected route is not in the known route set

        The caller (StagedEscalationRouter) treats None as a safe signal to escalate.
        """
        # Resolve concrete model for this tier — never falls back silently on ValueError.
        try:
            model_id = self._policy.resolve_model_id(model_tier)
        except ValueError as exc:
            logger.error(
                "LLMIntentClassifier: cannot resolve model for tier '%s': %s — skipping LLM stage",
                model_tier.value, exc,
            )
            return None

        route_options = [
            {
                "name": r.name,
                "description": r.description or r.name,
            }
            for r in self._routes.values()
        ]

        system_prompt = (
            "You are an intent classification system. Given a user goal and available routes, "
            "select the most appropriate route by name.\n"
            "Respond strictly in JSON with keys: route_name (string), confidence (float between 0 and 1).\n"
            "Do NOT include a 'strategy' key — the strategy is determined by the system, not by you."
        )

        user_prompt = (
            f"Goal: {goal.description}\n"
            f"Available Routes:\n{json.dumps(route_options, indent=2)}\n\n"
            "Return JSON matching the schema."
        )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # --- Infrastructure / provider call ---
        try:
            response = self._connector.infer_prompt(
                model_id=model_id,
                prompt=full_prompt,
                options=GenerationOptions(
                    temperature=0.0,
                    max_tokens=128,
                    constraint=OutputConstraint.json(),
                ),
            )
        except Exception as exc:
            # Any connector / provider / network failure is an infrastructure error.
            # Log as warning (the staged router will escalate / fall back).
            logger.warning(
                "LLMIntentClassifier: inference failure (model=%s, tier=%s): %s: %s",
                model_id, model_tier.value, type(exc).__name__, exc,
            )
            return None

        # --- JSON parsing ---
        try:
            data = parse_json_payload(response.text)
        except (SyntaxParsingError, json.JSONDecodeError, ValueError, SyntaxError, TypeError) as exc:
            logger.warning(
                "LLMIntentClassifier: JSON parsing failure (model=%s, tier=%s): %s: %s — raw: %r",
                model_id, model_tier.value, type(exc).__name__, exc, response.text[:200],
            )
            return None

        # --- Route selection ---
        r_name = data.get("route_name")
        if not r_name:
            logger.info(
                "LLMIntentClassifier: LLM response missing 'route_name' (model=%s, tier=%s) — no-match",
                model_id, model_tier.value,
            )
            return None

        if r_name not in self._routes:
            logger.info(
                "LLMIntentClassifier: LLM selected unrecognized route '%s' (model=%s, tier=%s) — no-match",
                r_name, model_id, model_tier.value,
            )
            return None

        route = self._routes[r_name]

        # Issue 4: strategy is ALWAYS from RouteDefinition — the LLM cannot override it.
        strategy = route.strategy

        conf = float(data.get("confidence", 0.85))
        stage_name = "escalated" if model_tier == ModelTier.REASONING else "llm_classifier"

        return RouteResult(
            route_name=route.name,
            strategy=strategy,
            confidence=conf,
            stage_resolved=stage_name,
            target_capability_id=route.target_capability_id,
            target_model_tier=route.target_model_tier,
            metadata={"raw_llm_output": data, "resolved_model_id": model_id},
        )
