"""Temporal plan runner implementing the PlanRunner execution boundary.

Connects to a Temporal cluster/client, manages workflow lifecycle
(start, wait, get_status, cancel, run), and returns factual execution outputs.
"""

import asyncio
from typing import Any, Coroutine, Dict, Optional, TypeVar

from temporalio.client import Client

from orchestration.domain.plans import Plan
from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.types import (
    PlanStatus,
    TaskErrorCategory,
    TaskStatus,
)
from orchestration.execution.base import (
    ExecutionHandle,
    PlanExecutionResult,
    PlanExecutionSnapshot,
    PlanRunner,
)
from orchestration.execution.temporal.types import (
    InputReferenceDTO,
    PlanWorkflowInput,
    PlanWorkflowOutput,
    TaskDefinitionDTO,
)
from orchestration.execution.temporal.workflows import PlanExecutionWorkflow

T = TypeVar("T")


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously from sync callers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    raise RuntimeError(
        "Synchronous PlanRunner methods (start, wait, run, get_status, cancel) "
        "cannot be called from within a running asyncio event loop because doing so "
        "would block the event loop. Use the async methods (start_async, wait_async, "
        "run_async, get_status_async, cancel_async) instead."
    )



class TemporalPlanRunner(PlanRunner):
    """Durable distributed PlanRunner backed by Temporal workflows and activities."""

    def __init__(self, client: Client, task_queue: str) -> None:
        """Initialize TemporalPlanRunner.

        Args:
            client: Temporal client connected to a Temporal service/cluster.
            task_queue: Task queue name for workflow and activity scheduling.
        """
        self._client = client
        self._task_queue = task_queue

    # ---------------------------------------------------------------------------
    # Synchronous PlanRunner Protocol Implementation
    # ---------------------------------------------------------------------------

    def start(self, plan: Plan) -> ExecutionHandle:
        """Initiate plan execution on Temporal.

        Args:
            plan: The Plan to execute.

        Returns:
            ExecutionHandle identifying the running execution.
        """
        return _run_sync(self.start_async(plan))

    def wait(
        self,
        handle: ExecutionHandle,
        timeout: Optional[float] = None,
    ) -> PlanExecutionResult:
        """Wait for the Temporal workflow execution to reach a terminal state.

        Args:
            handle: The ExecutionHandle returned by start().
            timeout: Optional timeout in seconds.

        Returns:
            PlanExecutionResult containing terminal execution facts.
        """
        return _run_sync(self.wait_async(handle, timeout=timeout))

    def get_status(self, handle: ExecutionHandle) -> PlanExecutionSnapshot:
        """Query the non-blocking status of the execution.

        Args:
            handle: The ExecutionHandle to query.

        Returns:
            PlanExecutionSnapshot reflecting the execution state.
        """
        return _run_sync(self.get_status_async(handle))

    def cancel(self, handle: ExecutionHandle) -> None:
        """Request cancellation of the running execution.

        Args:
            handle: The ExecutionHandle to cancel.
        """
        _run_sync(self.cancel_async(handle))

    def run(self, plan: Plan) -> PlanExecutionResult:
        """Convenience method: start execution and wait for completion.

        Args:
            plan: The Plan to execute.

        Returns:
            PlanExecutionResult containing terminal execution facts.
        """
        return _run_sync(self.run_async(plan))

    # ---------------------------------------------------------------------------
    # Asynchronous Native Methods
    # ---------------------------------------------------------------------------

    async def start_async(self, plan: Plan) -> ExecutionHandle:
        """Initiate plan execution asynchronously.

        Args:
            plan: The Plan to execute.

        Returns:
            ExecutionHandle identifying the running execution.

        Raises:
            ValueError: If plan is already terminal or has no tasks.
        """
        if plan.status == PlanStatus.DRAFT:
            plan.activate()

        if plan.status != PlanStatus.ACTIVE:
            raise ValueError(
                f"Cannot run plan '{plan.plan_id}': status is '{plan.status.value}', "
                f"expected 'active' (or 'draft')."
            )

        if not plan.tasks:
            raise ValueError(f"Cannot run plan '{plan.plan_id}': plan has no tasks.")

        # Serialize Plan into PlanWorkflowInput DTO
        task_dtos = []
        for task in plan.tasks.values():
            deps = [d.upstream_task_id for d in task.dependencies]
            input_refs = {
                k: InputReferenceDTO(
                    key=ref.key,
                    source_task_id=ref.source_task_id,
                    uri=ref.uri,
                )
                for k, ref in task.input_references.items()
            }
            attempt_id = f"att-{task.task_id}-{len(task.attempts) + 1}"
            task_dtos.append(
                TaskDefinitionDTO(
                    task_id=task.task_id,
                    capability_id=task.capability_id,
                    attempt_id=attempt_id,
                    parameters=dict(task.parameters),
                    dependencies=deps,
                    input_references=input_refs,
                )
            )

        wf_input = PlanWorkflowInput(
            plan_id=plan.plan_id,
            goal_id=plan.goal_id,
            tasks=task_dtos,
        )

        wf_handle = await self._client.start_workflow(
            PlanExecutionWorkflow.run,
            wf_input,
            id=f"plan-{plan.plan_id}",
            task_queue=self._task_queue,
        )

        return ExecutionHandle(
            execution_id=wf_handle.id,
            plan_id=plan.plan_id,
            backend_info={
                "engine": "temporal",
                "run_id": wf_handle.result_run_id,
            },
        )

    async def wait_async(
        self,
        handle: ExecutionHandle,
        timeout: Optional[float] = None,
    ) -> PlanExecutionResult:
        """Wait for the Temporal workflow execution to reach a terminal state.

        Args:
            handle: The ExecutionHandle returned by start().
            timeout: Optional timeout in seconds.

        Returns:
            PlanExecutionResult containing terminal execution facts.
        """
        wf_handle = self._client.get_workflow_handle(handle.execution_id)
        if timeout is not None:
            output = await asyncio.wait_for(
                wf_handle.result(),
                timeout=timeout,
            )
        else:
            output = await wf_handle.result()

        if isinstance(output, dict):
            raw_status = output.get("status", "FAILED")
            raw_results = output.get("task_results", {})
            raw_errors = output.get("task_errors", {})
            raw_attempts = output.get("task_attempts", {})
            raw_refs = output.get("task_references", {})
            raw_arts = output.get("task_artifacts", {})
        else:
            raw_status = output.status
            raw_results = output.task_results
            raw_errors = output.task_errors
            raw_attempts = getattr(output, "task_attempts", {})
            raw_refs = getattr(output, "task_references", {})
            raw_arts = getattr(output, "task_artifacts", {})

        task_results: Dict[str, TaskResult] = {}
        for t_id, out in raw_results.items():
            refs = [
                DataReference(
                    key=r["key"],
                    source_task_id=r.get("source_task_id"),
                    uri=r.get("uri"),
                    mime_type=r.get("mime_type", "application/json"),
                    metadata=r.get("metadata", {}),
                )
                for r in raw_refs.get(t_id, [])
            ]
            arts = [
                ArtifactReference(
                    artifact_id=a["artifact_id"],
                    name=a["name"],
                    uri=a["uri"],
                    mime_type=a["mime_type"],
                    size_bytes=a.get("size_bytes"),
                    metadata=a.get("metadata", {}),
                )
                for a in raw_arts.get(t_id, [])
            ]
            task_results[t_id] = TaskResult(
                output=out,
                references=refs,
                artifacts=arts,
            )

        task_errors: Dict[str, TaskError] = {}
        for t_id, err_dict in raw_errors.items():
            cat_str = err_dict.get("category", "EXECUTION")
            try:
                category = TaskErrorCategory[cat_str]
            except KeyError:
                category = TaskErrorCategory.EXECUTION

            task_errors[t_id] = TaskError(
                message=err_dict.get("message", "Task execution failed."),
                category=category,
                error_code=err_dict.get("error_code"),
                details=err_dict.get("details", {}),
            )

        try:
            status = PlanStatus(raw_status.lower())
        except ValueError:
            status = PlanStatus.FAILED

        return PlanExecutionResult(
            execution_id=handle.execution_id,
            plan_id=handle.plan_id,
            status=status,
            task_results=task_results,
            task_errors=task_errors,
            task_attempts=raw_attempts,
        )

    async def get_status_async(self, handle: ExecutionHandle) -> PlanExecutionSnapshot:
        """Query the non-blocking status of the execution.

        Args:
            handle: The ExecutionHandle to query.

        Returns:
            PlanExecutionSnapshot reflecting the execution state.
        """
        wf_handle = self._client.get_workflow_handle(handle.execution_id)
        snapshot_data = await wf_handle.query(PlanExecutionWorkflow.get_snapshot)

        raw_task_statuses = snapshot_data.get("task_statuses", {})
        task_statuses: Dict[str, TaskStatus] = {}
        for t_id, s_str in raw_task_statuses.items():
            try:
                task_statuses[t_id] = TaskStatus(s_str.lower())
            except ValueError:
                task_statuses[t_id] = TaskStatus.PENDING

        is_cancelled = snapshot_data.get("is_cancelled", False)
        if is_cancelled:
            status = PlanStatus.CANCELLED
            is_terminal = True
        elif all(s == TaskStatus.COMPLETED for s in task_statuses.values()) and task_statuses:
            status = PlanStatus.COMPLETED
            is_terminal = True
        elif any(s in (TaskStatus.FAILED, TaskStatus.BLOCKED) for s in task_statuses.values()):
            status = PlanStatus.FAILED
            is_terminal = True
        else:
            status = PlanStatus.ACTIVE
            is_terminal = False

        return PlanExecutionSnapshot(
            execution_id=handle.execution_id,
            plan_id=handle.plan_id,
            status=status,
            task_statuses=task_statuses,
            is_terminal=is_terminal,
        )

    async def cancel_async(self, handle: ExecutionHandle) -> None:
        """Request cancellation of the running execution asynchronously.

        Args:
            handle: The ExecutionHandle to cancel.
        """
        wf_handle = self._client.get_workflow_handle(handle.execution_id)
        await wf_handle.cancel()

    async def run_async(self, plan: Plan) -> PlanExecutionResult:
        """Asynchronously start execution and wait for completion.

        Args:
            plan: The Plan to execute.

        Returns:
            PlanExecutionResult containing terminal execution facts.
        """
        handle = await self.start_async(plan)
        return await self.wait_async(handle)
