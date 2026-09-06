"""Built-in Agent Capability using PydanticAI.

Provides:
  - AgentCapability: Capability protocol implementation (agent.pydantic_ai).
  - FoundationPydanticAIModel: Local PydanticAI Model adapter.
  - CapabilityToolAdapter: Bridges CapabilityRegistry to PydanticAI tools.
  - AgentExecutionPolicy: Per-call capability authorization boundary.
"""

from orchestration.capabilities.builtin.agent.capability import AgentCapability
from orchestration.capabilities.builtin.agent.model_adapter import (
    FoundationPydanticAIModel,
)
from orchestration.capabilities.builtin.agent.policy import (
    AgentExecutionPolicy,
    UnauthorizedCapabilityError,
)
from orchestration.capabilities.builtin.agent.tool_adapter import (
    CapabilityToolAdapter,
)
from orchestration.capabilities.builtin.agent.types import (
    AgentBudget,
    AgentExecutionOutput,
    AgentParameters,
    AgentProposal,
    AgentToolCallRecord,
)

__all__ = [
    "AgentCapability",
    "FoundationPydanticAIModel",
    "CapabilityToolAdapter",
    "AgentExecutionPolicy",
    "UnauthorizedCapabilityError",
    "AgentParameters",
    "AgentBudget",
    "AgentProposal",
    "AgentToolCallRecord",
    "AgentExecutionOutput",
]
