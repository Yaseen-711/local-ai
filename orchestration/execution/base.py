"""Execution boundary protocol for orchestration plan runners.

Defines the pluggable execution boundary between GoalOrchestrator and
concrete execution backends (InProcessPlanRunner for synchronous execution,
or future distributed backends such as Temporal).
"""

from typing import Protocol, runtime_checkable

from orchestration.domain.plans import Plan


@runtime_checkable
class PlanRunner(Protocol):
    """Execution boundary protocol for driving an orchestration Plan to completion.

    GoalOrchestrator depends strictly on this protocol rather than a concrete
    runner, ensuring the execution engine remains swappable without altering
    orchestration coordination logic.
    """

    def run(self, plan: Plan) -> Plan:
        """Drive an orchestration Plan to completion.

        Args:
            plan: The Plan to execute.

        Returns:
            The executed Plan in its terminal state (COMPLETED, FAILED, or CANCELLED).
        """
        ...
