"""Capability structural protocol and execution context.

A capability represents a semantic system function (e.g. model inference,
workflow execution, data transformation) that can be invoked with parameters
and inputs.

A capability is completely decoupled from the orchestration domain:
it does NOT know about Task, Plan, Attempt, or Dependency entities.
The execution runner is responsible for translating orchestration tasks
into capability invocations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from orchestration.domain.results import TaskResult


@dataclass(frozen=True)
class CapabilityContext:
    """Narrow, read-only invocation context for capability execution.

    Contains strictly execution-scoped diagnostic and tracking metadata.
    Must NOT contain service locators, database handles, or orchestration
    domain entities.

    Note:
        frozen=True protects against attribute reassignment on the context
        instance itself (e.g. context.execution_id = ...); it does not prevent
        in-place mutation of nested dictionary keys within metadata.

    Attributes:
        execution_id: Unique correlation identifier for this execution attempt.
        metadata: Arbitrary advisory metadata (e.g. caller tags, timeouts).
    """
    execution_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Capability(Protocol):
    """Semantic system function contract.

    A capability executes a specific system function independent of how work
    units are orchestrated or scheduled.
    """

    @property
    def capability_id(self) -> str:
        """Canonical declarative identifier for this capability (e.g. 'inference.prompt')."""
        ...

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        """Execute the capability function.

        Args:
            parameters: Declarative configuration options (e.g. model_id, temperature).
            inputs: Resolved data payloads (e.g. prompt, text, upstream outputs).
            context: Narrow invocation telemetry context.

        Returns:
            TaskResult value object containing the output and metadata.

        Raises:
            Exception: On execution or validation failure. The runner captures
                any exception and maps it to a TaskError.
        """
        ...
