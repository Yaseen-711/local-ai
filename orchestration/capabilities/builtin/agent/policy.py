"""Simple, deterministic authority and execution policy for agent capabilities.

Enforces per-tool-call authorization:
Agent proposed tool -> AgentExecutionPolicy -> CapabilityToolAdapter -> CapabilityRegistry -> Capability.execute()

The agent is never the permission authority.
"""

from __future__ import annotations

import logging
from typing import Set

logger = logging.getLogger(__name__)


class UnauthorizedCapabilityError(PermissionError):
    """Raised when an agent attempts to invoke a capability that is not authorized."""
    pass


class AgentExecutionPolicy:
    """Deterministic policy enforcing capability authorization boundaries per tool call.

    Validates every individual tool invocation requested by the agent against the
    approved capability whitelist established for the task/attempt.
    """

    def is_authorized(self, capability_id: str, allowed_capabilities: Set[str]) -> bool:
        """Evaluate if capability_id is authorized by the policy."""
        if not capability_id:
            return False
        return capability_id in allowed_capabilities

    def authorize_tool_call(self, capability_id: str, allowed_capabilities: Set[str]) -> None:
        """Enforce authorization for a tool call.

        Args:
            capability_id: Canonical capability identifier being invoked.
            allowed_capabilities: Whitelisted set of capability IDs authorized for this task.

        Raises:
            UnauthorizedCapabilityError: If capability_id is not in allowed_capabilities.
        """
        if not self.is_authorized(capability_id, allowed_capabilities):
            msg = (
                f"Unauthorized capability invocation: '{capability_id}'. "
                f"Allowed capabilities for this task: {sorted(allowed_capabilities)}"
            )
            logger.warning(msg)
            raise UnauthorizedCapabilityError(msg)
