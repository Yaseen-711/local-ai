"""Data contracts for intent routing and strategy classification."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionStrategy(str, Enum):
    """Execution strategy selected for a Goal."""
    DIRECT_DETERMINISTIC = "direct_deterministic"
    DIRECT_CAPABILITY = "direct_capability"
    PLAN_REQUIRED = "plan_required"
    REJECT = "reject"


class ModelTier(str, Enum):
    """Abstract model tier decoupled from concrete model IDs and hardware."""
    LIGHTWEIGHT = "lightweight"
    CAPABLE = "capable"
    REASONING = "reasoning"


@dataclass(frozen=True)
class RouteDefinition:
    """Declarative specification of a recognized intent route.

    Attributes:
        name: Unique route name identifier (e.g. 'direct_echo', 'text_analysis').
        strategy: Execution strategy for this route.
        target_capability_id: Optional capability to invoke for DIRECT_CAPABILITY.
        target_model_tier: Optional model tier recommended for this route.
        utterances: Sample phrases / templates used for semantic vector matching.
        description: Human-readable explanation of the route intent.
        metadata: Arbitrary route-specific metadata.
    """
    name: str
    strategy: ExecutionStrategy
    target_capability_id: Optional[str] = None
    target_model_tier: Optional[ModelTier] = None
    utterances: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteResult:
    """Outcome of intent routing and strategy classification.

    Attributes:
        route_name: Name of the selected route.
        strategy: Selected execution strategy.
        confidence: Normalized score in [0.0, 1.0].
        stage_resolved: The escalation stage that resolved this route
            ('deterministic', 'semantic_router', 'llm_classifier', 'escalated').
        target_capability_id: Target capability if DIRECT_CAPABILITY.
        target_model_tier: Target model tier.
        extracted_parameters: Any parameters extracted from goal during routing.
        metadata: Diagnostic or routing metadata.
    """
    route_name: str
    strategy: ExecutionStrategy
    confidence: float
    stage_resolved: str
    target_capability_id: Optional[str] = None
    target_model_tier: Optional[ModelTier] = None
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
