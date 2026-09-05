"""In-process synchronous execution runner for orchestration plans.

The runner is the bridge between the orchestration domain (Plan, Task, Attempt)
and semantic system functions (Capability, CapabilityRegistry).

It owns:
1. Plan and Task lifecycle state transitions.
2. Topological DAG execution and dependency readiness evaluation.
3. Translating Task parameters and input_references into capability arguments.
4. Invoking capabilities with a narrow CapabilityContext.
5. Capturing results or mapping exceptions to TaskError.

It does NOT:
- Expose Task to Capability.
- Manage Goal lifecycle (owned at higher application/workflow levels).
- Depend on Celery, PostgreSQL, Docker, or async runtimes.
"""

from typing import Any, Dict

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.plans import Plan
from orchestration.domain.tasks import Task
from orchestration.domain.types import (
    PlanStatus,
    TaskErrorCategory,
    TaskStatus,
)
from orchestration.domain.results import TaskError


class InProcessPlanRunner:
    """Synchronous in-process execution engine for orchestration Plans."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        """Initialize runner with a populated CapabilityRegistry.

        Args:
            registry: CapabilityRegistry used to resolve capability_id to
                executable Capability implementations.
        """
        self._registry = registry

    def run(self, plan: Plan) -> Plan:
        """Drive an active or draft Plan to completion in-process.

        Args:
            plan: The Plan to execute. If in DRAFT status, it is activated.

        Returns:
            The executed Plan in its terminal state (COMPLETED or FAILED).

        Raises:
            ValueError: If the plan is already in a terminal state (COMPLETED,
                FAILED, CANCELLED) or has no tasks.
        """
        if plan.status == PlanStatus.DRAFT:
            plan.activate()

        if plan.status != PlanStatus.ACTIVE:
            raise ValueError(
                f"Cannot run plan '{plan.plan_id}': status is '{plan.status.value}', "
                f"expected 'active' (or 'draft')."
            )

        if not plan.tasks:
            raise ValueError(
                f"Cannot run plan '{plan.plan_id}': plan has no tasks."
            )

        while True:
            # 1. Update readiness across all uncompleted tasks
            current_statuses = {t_id: t.status for t_id, t in plan.tasks.items()}
            for task in plan.tasks.values():
                task.update_readiness(current_statuses)

            # 2. Check for ready tasks
            ready_tasks = [
                t for t in plan.tasks.values() if t.status == TaskStatus.READY
            ]

            # 3. If no tasks are ready, evaluate termination
            if not ready_tasks:
                # All tasks completed successfully
                if all(t.status == TaskStatus.COMPLETED for t in plan.tasks.values()):
                    plan.mark_completed()
                    return plan

                # Any task failed or remaining tasks are blocked
                if any(t.status == TaskStatus.FAILED for t in plan.tasks.values()):
                    plan.mark_failed()
                    return plan

                if any(t.status in (TaskStatus.BLOCKED, TaskStatus.CANCELLED) for t in plan.tasks.values()):
                    plan.mark_failed()
                    return plan

                # Tasks remain pending but none are ready (unresolvable dependency)
                if any(t.status == TaskStatus.PENDING for t in plan.tasks.values()):
                    plan.mark_failed()
                    return plan

                # Catch-all termination
                plan.mark_failed()
                return plan

            # 4. Execute the first ready task
            task = ready_tasks[0]
            self._execute_task(task, plan)

    def _execute_task(self, task: Task, plan: Plan) -> None:
        """Execute a single ready task and record the outcome on its lifecycle.

        Args:
            task: Task in READY status to execute.
            plan: Plan owning this task (for input reference resolution).
        """
        attempt_id = f"att-{task.task_id}-{len(task.attempts) + 1}"
        attempt = task.start_attempt(attempt_id)

        # 1. Capability resolution
        if not self._registry.has(task.capability_id):
            error = TaskError(
                message=f"Capability '{task.capability_id}' not found in registry.",
                category=TaskErrorCategory.CAPABILITY,
                error_code="CAPABILITY_NOT_FOUND",
            )
            task.fail_attempt(attempt.attempt_id, error)
            return

        capability = self._registry.get(task.capability_id)

        # 2. Input translation & execution within safety boundary
        parameters = dict(task.parameters)
        context = CapabilityContext(execution_id=attempt.attempt_id)

        try:
            inputs = self._resolve_inputs(task, plan)
            result = capability.execute(
                parameters=parameters,
                inputs=inputs,
                context=context,
            )
            task.complete_attempt(attempt.attempt_id, result)
        except Exception as exc:
            error = TaskError.from_exception(
                exc,
                category=TaskErrorCategory.EXECUTION,
            )
            task.fail_attempt(attempt.attempt_id, error)

    def _resolve_inputs(self, task: Task, plan: Plan) -> Dict[str, Any]:
        """Resolve a task's logical input_references from upstream plan outputs.

        Runner-local resolution for in-process execution:
        - If ref.source_task_id is set, upstream task must exist in plan and have
          completed with a TaskResult; raises ValueError otherwise.
        - If ref.uri is set, preserves the URI hint.
        - If neither is provided, raises ValueError.

        Args:
            task: Task whose input references need resolution.
            plan: Plan containing upstream tasks.

        Returns:
            Dictionary of resolved inputs keyed by logical input name.

        Raises:
            ValueError: If an upstream task does not exist, has no result,
                or if a reference is unresolvable.
        """
        resolved: Dict[str, Any] = {}
        for logical_name, ref in task.input_references.items():
            if ref.source_task_id is not None:
                source_task = plan.tasks.get(ref.source_task_id)
                if source_task is None:
                    raise ValueError(
                        f"Cannot resolve input reference '{logical_name}' for task '{task.task_id}': "
                        f"upstream task '{ref.source_task_id}' does not exist in plan '{plan.plan_id}'."
                    )
                if source_task.result is None:
                    raise ValueError(
                        f"Cannot resolve input reference '{logical_name}' for task '{task.task_id}': "
                        f"upstream task '{ref.source_task_id}' has no result (status: '{source_task.status.value}')."
                    )
                source_output = source_task.result.output
                # If output is a dict and contains the referenced key, extract it
                if isinstance(source_output, dict) and ref.key in source_output:
                    resolved[logical_name] = source_output[ref.key]
                elif hasattr(source_output, ref.key):
                    resolved[logical_name] = getattr(source_output, ref.key)
                else:
                    # Otherwise pass the output directly
                    resolved[logical_name] = source_output
            elif ref.uri is not None:
                resolved[logical_name] = ref.uri
            else:
                raise ValueError(
                    f"Cannot resolve input reference '{logical_name}' for task '{task.task_id}': "
                    f"reference has neither 'source_task_id' nor 'uri'."
                )
        return resolved
