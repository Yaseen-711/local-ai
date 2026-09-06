"""Temporal workflow for executing orchestration plans.

A deterministic, mechanical DAG executor:
1. Evaluates task dependency readiness.
2. Schedules eligible tasks concurrently as activities.
3. Resolves inputs from upstream task outputs.
4. Returns factual execution outputs without semantic interpretation.
"""

import asyncio
from datetime import timedelta
from typing import Any, Dict, List

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from orchestration.execution.temporal.types import (
        PlanWorkflowInput,
        PlanWorkflowOutput,
        TaskActivityInput,
        TaskActivityOutput,
        TaskDefinitionDTO,
    )


@workflow.defn
class PlanExecutionWorkflow:
    """Mechanical DAG execution workflow for an approved Plan."""

    def __init__(self) -> None:
        self.task_statuses: Dict[str, str] = {}
        self.task_outputs: Dict[str, Any] = {}
        self.task_references: Dict[str, List[Dict[str, Any]]] = {}
        self.task_artifacts: Dict[str, List[Dict[str, Any]]] = {}
        self.task_errors: Dict[str, Dict[str, Any]] = {}
        self.task_attempts: Dict[str, str] = {}
        self._cancelled: bool = False

    @workflow.run
    async def run(self, input: PlanWorkflowInput) -> PlanWorkflowOutput:
        """Drive DAG execution to completion.

        Args:
            input: PlanWorkflowInput with plan structure and task DAG definitions.

        Returns:
            PlanWorkflowOutput containing terminal execution facts.
        """
        if not input.tasks:
            return PlanWorkflowOutput(
                plan_id=input.plan_id,
                status="FAILED",
                task_results={},
                task_errors={"plan": {"message": "Plan has no tasks."}},
                task_statuses={},
                task_attempts={},
                task_references={},
                task_artifacts={},
            )

        tasks_by_id = {t.task_id: t for t in input.tasks}
        for t_id in tasks_by_id:
            self.task_statuses[t_id] = "PENDING"

        running_tasks: Dict[str, asyncio.Task] = {}

        try:
            while True:
                if self._cancelled:
                    return self._abort_and_cancel(input.plan_id, running_tasks)

                # 1. Evaluate readiness of all PENDING tasks
                for t_id, task_def in tasks_by_id.items():
                    if self.task_statuses[t_id] == "PENDING":
                        deps = task_def.dependencies
                        if all(self.task_statuses.get(dep) == "COMPLETED" for dep in deps):
                            self.task_statuses[t_id] = "READY"
                        elif any(
                            self.task_statuses.get(dep) in ("FAILED", "BLOCKED", "CANCELLED")
                            for dep in deps
                        ):
                            self.task_statuses[t_id] = "BLOCKED"

                # 2. Dispatch all READY tasks concurrently
                ready_tasks = [
                    t_def
                    for t_id, t_def in tasks_by_id.items()
                    if self.task_statuses[t_id] == "READY"
                ]
                for task_def in ready_tasks:
                    t_id = task_def.task_id
                    try:
                        resolved_inputs = self._resolve_inputs(task_def)
                    except ValueError as exc:
                        self.task_statuses[t_id] = "FAILED"
                        self.task_errors[t_id] = {
                            "message": str(exc),
                            "category": "EXECUTION",
                            "error_code": "INPUT_RESOLUTION_ERROR",
                            "details": {},
                        }
                        if task_def.attempt_id:
                            self.task_attempts[t_id] = task_def.attempt_id
                        continue

                    self.task_statuses[t_id] = "RUNNING"
                    attempt_id = task_def.attempt_id or f"att-{t_id}-1"
                    self.task_attempts[t_id] = attempt_id
                    activity_input = TaskActivityInput(
                        task_id=t_id,
                        capability_id=task_def.capability_id,
                        attempt_id=attempt_id,
                        parameters=task_def.parameters,
                        inputs=resolved_inputs,
                    )
                    fut = asyncio.create_task(
                        workflow.execute_activity(
                            "execute_task",
                            activity_input,
                            start_to_close_timeout=timedelta(minutes=5),
                            result_type=TaskActivityOutput,
                        )
                    )
                    running_tasks[t_id] = fut

                # 3. Wait for at least one running activity to complete
                if running_tasks:
                    done, _ = await workflow.wait(
                        running_tasks.values(),
                        return_when="FIRST_COMPLETED",
                    )
                    for t_id, fut in list(running_tasks.items()):
                        if fut in done:
                            del running_tasks[t_id]
                            try:
                                output = fut.result()
                                if isinstance(output, dict):
                                    status = output.get("status")
                                    out_val = output.get("output")
                                    refs_val = output.get("references", [])
                                    arts_val = output.get("artifacts", [])
                                    err_msg = output.get("error_message")
                                    err_cat = output.get("error_category")
                                    err_code = output.get("error_code")
                                    err_det = output.get("error_details", {})
                                else:
                                    status = output.status
                                    out_val = output.output
                                    refs_val = output.references
                                    arts_val = output.artifacts
                                    err_msg = output.error_message
                                    err_cat = output.error_category
                                    err_code = output.error_code
                                    err_det = output.error_details

                                if status == "COMPLETED":
                                    self.task_statuses[t_id] = "COMPLETED"
                                    self.task_outputs[t_id] = out_val
                                    self.task_references[t_id] = refs_val
                                    self.task_artifacts[t_id] = arts_val
                                else:
                                    self.task_statuses[t_id] = "FAILED"
                                    self.task_errors[t_id] = {
                                        "message": err_msg or "Task failed",
                                        "category": err_cat or "EXECUTION",
                                        "error_code": err_code,
                                        "details": err_det,
                                    }
                            except asyncio.CancelledError:
                                self.task_statuses[t_id] = "CANCELLED"
                                raise
                            except Exception as exc:
                                self.task_statuses[t_id] = "FAILED"
                                self.task_errors[t_id] = {
                                    "message": str(exc),
                                    "category": "EXECUTION",
                                    "error_code": type(exc).__name__,
                                    "details": {},
                                }
                    continue

                # 4. If no running tasks and no ready tasks, evaluate termination
                for t_id in tasks_by_id:
                    if self.task_statuses[t_id] == "PENDING":
                        self.task_statuses[t_id] = "BLOCKED"

                if all(s == "COMPLETED" for s in self.task_statuses.values()):
                    return PlanWorkflowOutput(
                        plan_id=input.plan_id,
                        status="COMPLETED",
                        task_results=self.task_outputs,
                        task_errors=self.task_errors,
                        task_statuses=self.task_statuses,
                        task_attempts=self.task_attempts,
                        task_references=self.task_references,
                        task_artifacts=self.task_artifacts,
                    )

                # Termination on failures or blocked tasks
                return PlanWorkflowOutput(
                    plan_id=input.plan_id,
                    status="FAILED",
                    task_results=self.task_outputs,
                    task_errors=self.task_errors,
                    task_statuses=self.task_statuses,
                    task_attempts=self.task_attempts,
                    task_references=self.task_references,
                    task_artifacts=self.task_artifacts,
                )
        except asyncio.CancelledError:
            return self._abort_and_cancel(input.plan_id, running_tasks)

    def _resolve_inputs(self, task_def: TaskDefinitionDTO) -> Dict[str, Any]:
        """Resolve inputs from upstream task outputs or URI references.

        Raises:
            ValueError: If an upstream task does not exist or has no output,
                or if a reference is invalid.
        """
        resolved: Dict[str, Any] = {}
        for name, ref in task_def.input_references.items():
            if ref.source_task_id is not None:
                if ref.source_task_id not in self.task_outputs:
                    raise ValueError(
                        f"Cannot resolve input reference '{name}' for task '{task_def.task_id}': "
                        f"upstream task '{ref.source_task_id}' has no completed output."
                    )
                src_output = self.task_outputs[ref.source_task_id]
                if isinstance(src_output, dict) and ref.key in src_output:
                    resolved[name] = src_output[ref.key]
                elif hasattr(src_output, ref.key):
                    resolved[name] = getattr(src_output, ref.key)
                elif ref.key == "output":
                    resolved[name] = src_output
                else:
                    raise ValueError(
                        f"Cannot resolve input reference '{name}' for task '{task_def.task_id}': "
                        f"key '{ref.key}' not found in output of upstream task '{ref.source_task_id}'."
                    )
            elif ref.uri is not None:
                resolved[name] = ref.uri
            else:
                raise ValueError(
                    f"Cannot resolve input reference '{name}' for task '{task_def.task_id}': "
                    f"reference has neither 'source_task_id' nor 'uri'."
                )
        return resolved

    def _abort_and_cancel(
        self,
        plan_id: str,
        running_tasks: Dict[str, asyncio.Task],
    ) -> PlanWorkflowOutput:
        """Cancel all in-flight activities and return CANCELLED output."""
        for fut in running_tasks.values():
            fut.cancel()
        for t_id in self.task_statuses:
            if self.task_statuses[t_id] in ("PENDING", "READY", "RUNNING"):
                self.task_statuses[t_id] = "CANCELLED"
        return PlanWorkflowOutput(
            plan_id=plan_id,
            status="CANCELLED",
            task_results=self.task_outputs,
            task_errors=self.task_errors,
            task_statuses=self.task_statuses,
            task_attempts=self.task_attempts,
            task_references=self.task_references,
            task_artifacts=self.task_artifacts,
        )

    @workflow.signal
    def cancel_plan(self) -> None:
        """Signal to gracefully request plan cancellation."""
        self._cancelled = True

    @workflow.query
    def get_snapshot(self) -> Dict[str, Any]:
        """Query the current state of tasks in the workflow."""
        return {
            "is_cancelled": self._cancelled,
            "task_statuses": dict(self.task_statuses),
        }
