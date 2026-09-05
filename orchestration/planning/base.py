"""Base planner protocol and interface."""

import asyncio
from typing import Protocol, runtime_checkable

from orchestration.planning.types import CandidatePlan, PlanningContext


@runtime_checkable
class Planner(Protocol):
    """Protocol for components that generate CandidatePlans from PlanningContext.
    
    Planners propose candidate structures but have no authority over validation,
    activation, persistence, or execution.
    """

    def plan(self, context: PlanningContext) -> CandidatePlan:
        """Synchronously propose a CandidatePlan for the given goal and context."""
        ...

    async def plan_async(self, context: PlanningContext) -> CandidatePlan:
        """Asynchronously propose a CandidatePlan for the given goal and context."""
        ...
