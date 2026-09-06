"""Decision types and result contracts for the Decision & Planning layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from orchestration.orchestrator import DirectGoalResult
from orchestration.planning.types import CandidatePlan
from orchestration.routing.types import ExecutionStrategy, RouteResult
from orchestration.validation.types import ValidationResult


@dataclass(frozen=True)
class DecisionPolicy:
    """Configuration policy governing the DecisionEngine."""
    max_validation_retries: int = 1
    fallback_strategy: ExecutionStrategy = ExecutionStrategy.PLAN_REQUIRED
    allow_direct_execution: bool = True


@dataclass
class DecisionResult:
    """Outcome of processing a goal through the DecisionEngine.
    
    Exposes direct execution results or planning/validation outcomes
    without storing execution results inside Goal.context.
    """
    decision_type: ExecutionStrategy
    goal_id: str
    plan_id: Optional[str] = None
    plan: Optional[Any] = None
    route_result: Optional[RouteResult] = None
    direct_result: Optional[DirectGoalResult] = None
    candidate_plan: Optional[CandidatePlan] = None
    validation_result: Optional[ValidationResult] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
