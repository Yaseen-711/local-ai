"""Domain data contracts and value objects for AgentCapability.

Provides parameter definitions, iteration/tool budgeting records, tool execution
telemetry, advisory replan proposal structures, and typed execution outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from orchestration.routing.types import ModelTier


@dataclass
class AgentBudget:
    """Tracks and enforces turn and tool invocation limits for an agent task attempt."""
    max_iterations: int = 10
    max_tool_calls: int = 20
    current_iterations: int = 0
    current_tool_calls: int = 0

    def record_iteration(self) -> None:
        """Increment model iteration turn counter."""
        self.current_iterations += 1

    def record_tool_call(self) -> None:
        """Increment tool execution counter."""
        self.current_tool_calls += 1

    @property
    def is_iteration_exhausted(self) -> bool:
        """Check if maximum allowed model turns have been reached."""
        return self.current_iterations >= self.max_iterations

    @property
    def is_tool_call_exhausted(self) -> bool:
        """Check if maximum allowed tool invocations have been reached."""
        return self.current_tool_calls >= self.max_tool_calls

    @property
    def is_exhausted(self) -> bool:
        """Check if either turn or tool limit has been exceeded."""
        return self.is_iteration_exhausted or self.is_tool_call_exhausted


@dataclass(frozen=True)
class AgentToolCallRecord:
    """Diagnostic trace record of an individual tool invocation by the agent."""
    tool_name: str
    capability_id: str
    arguments: Dict[str, Any]
    output_summary: str
    execution_id: str
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace record for metadata and reporting."""
        return asdict(self)


@dataclass(frozen=True)
class AgentProposal:
    """Advisory orchestration extension point proposed by the agent.

    The agent has NO authority to mutate Goals, Plans, or Tasks directly.
    If an agent determines that additional workflow tasks or replanning are
    advisable, it outputs an AgentProposal that the calling GoalOrchestrator
    or Replanner may evaluate and act upon.
    """
    type: str
    reason: str
    suggested_tasks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize proposal for TaskResult output."""
        return asdict(self)


@dataclass(frozen=True)
class AgentExecutionOutput:
    """Typed outcome payload produced by AgentCapability."""
    response: str
    iterations: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    proposal: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution output to serializable dictionary for TaskResult."""
        return {
            "response": self.response,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "proposal": self.proposal,
        }


@dataclass(frozen=True)
class AgentParameters:
    """Normalized configuration parameters for an agent execution."""
    allowed_capabilities: List[str] = field(default_factory=list)
    model_tier: ModelTier = ModelTier.REASONING
    max_iterations: int = 10
    max_tool_calls: int = 20
    timeout_seconds: float = 120.0
    system_prompt: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentParameters":
        """Construct AgentParameters from raw parameters dictionary."""
        tier_raw = data.get("model_tier", ModelTier.REASONING)
        if isinstance(tier_raw, str):
            try:
                tier = ModelTier(tier_raw.lower())
            except ValueError:
                tier = ModelTier.REASONING
        elif isinstance(tier_raw, ModelTier):
            tier = tier_raw
        else:
            tier = ModelTier.REASONING

        allowed = data.get("allowed_capabilities", [])
        if isinstance(allowed, (list, set, tuple)):
            allowed_list = [str(c) for c in allowed]
        else:
            allowed_list = []

        return cls(
            allowed_capabilities=allowed_list,
            model_tier=tier,
            max_iterations=int(data.get("max_iterations", 10)),
            max_tool_calls=int(data.get("max_tool_calls", 20)),
            timeout_seconds=float(data.get("timeout_seconds", 120.0)),
            system_prompt=str(data["system_prompt"]) if "system_prompt" in data and data["system_prompt"] is not None else None,
        )
