"""Temporal activities for executing orchestration tasks via CapabilityRegistry.

Activities run within Temporal workers and invoke semantic capabilities
without any awareness of upstream/downstream DAG structure or Goal meaning.
"""

from typing import Any, Dict

from temporalio import activity


from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.execution.temporal.types import (
    TaskActivityInput,
    TaskActivityOutput,
)


class TaskExecutionActivity:
    """Activity adapter executing task capabilities via CapabilityRegistry."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        """Initialize activity with a populated CapabilityRegistry.

        Args:
            registry: CapabilityRegistry used to resolve capability_id.
        """
        self._registry = registry

    @activity.defn(name="execute_task")
    async def execute_task(self, input: TaskActivityInput) -> TaskActivityOutput:
        """Execute a single task capability attempt.

        Args:
            input: TaskActivityInput specifying capability_id, attempt_id,
                parameters, and resolved inputs.

        Returns:
            TaskActivityOutput with execution facts (status, output, or error).
        """
        if not self._registry.has(input.capability_id):
            return TaskActivityOutput(
                task_id=input.task_id,
                attempt_id=input.attempt_id,
                status="FAILED",
                error_message=f"Capability '{input.capability_id}' not found in registry.",
                error_category="CAPABILITY",
                error_code="CAPABILITY_NOT_FOUND",
            )

        capability = self._registry.get(input.capability_id)
        context = CapabilityContext(execution_id=input.attempt_id)

        try:
            result = capability.execute(
                parameters=input.parameters,
                inputs=input.inputs,
                context=context,
            )

            serialized_references = [
                {
                    "key": ref.key,
                    "source_task_id": ref.source_task_id,
                    "uri": ref.uri,
                    "mime_type": ref.mime_type,
                    "metadata": ref.metadata,
                }
                for ref in result.references
            ]

            serialized_artifacts = [
                {
                    "artifact_id": art.artifact_id,
                    "name": art.name,
                    "uri": art.uri,
                    "mime_type": art.mime_type,
                    "size_bytes": art.size_bytes,
                    "metadata": art.metadata,
                }
                for art in result.artifacts
            ]

            return TaskActivityOutput(
                task_id=input.task_id,
                attempt_id=input.attempt_id,
                status="COMPLETED",
                output=result.output,
                references=serialized_references,
                artifacts=serialized_artifacts,
            )
        except Exception as exc:
            return TaskActivityOutput(
                task_id=input.task_id,
                attempt_id=input.attempt_id,
                status="FAILED",
                error_message=str(exc),
                error_category="EXECUTION",
                error_code=type(exc).__name__,
            )
