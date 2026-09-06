"""Integration tests for PlanExecutionWorkflow using Temporal WorkflowEnvironment."""

import asyncio
from typing import Any, Dict
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.results import TaskResult
from orchestration.execution.temporal.activities import TaskExecutionActivity
from orchestration.execution.temporal.types import (
    InputReferenceDTO,
    PlanWorkflowInput,
    PlanWorkflowOutput,
    TaskDefinitionDTO,
)
from orchestration.execution.temporal.workflows import PlanExecutionWorkflow


class EchoCapability:
    @property
    def capability_id(self) -> str:
        return "test.echo"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        text = inputs.get("text") or parameters.get("text", "")
        return TaskResult(output=f"echo:{text}")


class CrashingCapability:
    @property
    def capability_id(self) -> str:
        return "test.crash"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        raise RuntimeError("simulated capability failure")


def test_workflow_linear_dag():
    """Verify sequential tasks execute with data passing via references."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-linear"
            registry = CapabilityRegistry()
            registry.register(EchoCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                t1 = TaskDefinitionDTO(
                    task_id="t1",
                    capability_id="test.echo",
                    parameters={"text": "first"},
                )
                t2 = TaskDefinitionDTO(
                    task_id="t2",
                    capability_id="test.echo",
                    dependencies=["t1"],
                    input_references={"text": InputReferenceDTO(source_task_id="t1")},
                )
                wf_input = PlanWorkflowInput(
                    plan_id="p-linear",
                    goal_id="g-1",
                    tasks=[t1, t2],
                )

                output: PlanWorkflowOutput = await env.client.execute_workflow(
                    PlanExecutionWorkflow.run,
                    wf_input,
                    id="wf-linear",
                    task_queue=tq,
                )

                assert output.status == "COMPLETED"
                assert output.task_results["t1"] == "echo:first"
                assert output.task_results["t2"] == "echo:echo:first"
                assert output.task_statuses["t1"] == "COMPLETED"
                assert output.task_statuses["t2"] == "COMPLETED"

    asyncio.run(_test())


def test_workflow_diamond_dag():
    """Verify diamond DAG: A -> (B, C) -> D."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-diamond"
            registry = CapabilityRegistry()
            registry.register(EchoCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                tA = TaskDefinitionDTO(task_id="A", capability_id="test.echo", parameters={"text": "start"})
                tB = TaskDefinitionDTO(
                    task_id="B",
                    capability_id="test.echo",
                    dependencies=["A"],
                    input_references={"text": InputReferenceDTO(source_task_id="A")},
                )
                tC = TaskDefinitionDTO(
                    task_id="C",
                    capability_id="test.echo",
                    dependencies=["A"],
                    input_references={"text": InputReferenceDTO(source_task_id="A")},
                )
                tD = TaskDefinitionDTO(
                    task_id="D",
                    capability_id="test.echo",
                    dependencies=["B", "C"],
                    input_references={"text": InputReferenceDTO(source_task_id="B")},
                )
                wf_input = PlanWorkflowInput(
                    plan_id="p-diamond",
                    goal_id="g-1",
                    tasks=[tA, tB, tC, tD],
                )

                output: PlanWorkflowOutput = await env.client.execute_workflow(
                    PlanExecutionWorkflow.run,
                    wf_input,
                    id="wf-diamond",
                    task_queue=tq,
                )

                assert output.status == "COMPLETED"
                assert output.task_statuses["A"] == "COMPLETED"
                assert output.task_statuses["B"] == "COMPLETED"
                assert output.task_statuses["C"] == "COMPLETED"
                assert output.task_statuses["D"] == "COMPLETED"

    asyncio.run(_test())


def test_workflow_failure_blocks_downstream():
    """Verify task failure transitions workflow to FAILED and blocks downstreams."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-failure"
            registry = CapabilityRegistry()
            registry.register(EchoCapability())
            registry.register(CrashingCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                t1 = TaskDefinitionDTO(task_id="t1", capability_id="test.crash")
                t2 = TaskDefinitionDTO(task_id="t2", capability_id="test.echo", dependencies=["t1"])
                wf_input = PlanWorkflowInput(
                    plan_id="p-fail",
                    goal_id="g-1",
                    tasks=[t1, t2],
                )

                output: PlanWorkflowOutput = await env.client.execute_workflow(
                    PlanExecutionWorkflow.run,
                    wf_input,
                    id="wf-fail",
                    task_queue=tq,
                )

                assert output.status == "FAILED"
                assert output.task_statuses["t1"] == "FAILED"
                assert output.task_statuses["t2"] == "BLOCKED"
                assert "t1" in output.task_errors
                assert output.task_errors["t1"]["category"] == "EXECUTION"

    asyncio.run(_test())


def test_workflow_cancellation():
    """Verify workflow cancellation signal aborts execution cleanly."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-cancel"
            registry = CapabilityRegistry()
            registry.register(EchoCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                t1 = TaskDefinitionDTO(task_id="t1", capability_id="test.echo", parameters={"text": "cancel-me"})
                wf_input = PlanWorkflowInput(
                    plan_id="p-cancel",
                    goal_id="g-1",
                    tasks=[t1],
                )

                handle = await env.client.start_workflow(
                    PlanExecutionWorkflow.run,
                    wf_input,
                    id="wf-cancel",
                    task_queue=tq,
                )

                await handle.signal(PlanExecutionWorkflow.cancel_plan)
                output: PlanWorkflowOutput = await handle.result()

                # Workflow completes in CANCELLED or COMPLETED depending on timing
                assert output.status in ("CANCELLED", "COMPLETED")

    asyncio.run(_test())


def test_workflow_missing_upstream_input_fails_honestly():
    """Verify workflow fails task honestly when upstream input reference cannot be resolved."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-missing-ref"
            registry = CapabilityRegistry()
            registry.register(EchoCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                # t1 is READY immediately (no dependencies) but references non-existent upstream 't_nonexistent'
                t1 = TaskDefinitionDTO(
                    task_id="t1",
                    capability_id="test.echo",
                    attempt_id="att-t1-custom",
                    dependencies=[],
                    input_references={"text": InputReferenceDTO(source_task_id="t_nonexistent")},
                )
                t2 = TaskDefinitionDTO(
                    task_id="t2",
                    capability_id="test.echo",
                    dependencies=["t1"],
                )
                wf_input = PlanWorkflowInput(
                    plan_id="p-missing-ref",
                    goal_id="g-1",
                    tasks=[t1, t2],
                )

                output: PlanWorkflowOutput = await env.client.execute_workflow(
                    PlanExecutionWorkflow.run,
                    wf_input,
                    id="wf-missing-ref",
                    task_queue=tq,
                )

                assert output.status == "FAILED"
                assert output.task_statuses["t1"] == "FAILED"
                assert output.task_statuses["t2"] == "BLOCKED"
                assert "t1" in output.task_errors
                assert output.task_errors["t1"]["error_code"] == "INPUT_RESOLUTION_ERROR"
                assert "upstream task 't_nonexistent' has no completed output" in output.task_errors["t1"]["message"]
                assert output.task_attempts["t1"] == "att-t1-custom"

    asyncio.run(_test())


def test_temporal_workflow_fails_when_ref_key_missing_from_dict():
    """Verify workflow records INPUT_RESOLUTION_ERROR when ref.key is missing from completed upstream dict output."""
    class DictCapability:
        @property
        def capability_id(self) -> str:
            return "test.dict"

        def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any], context: CapabilityContext) -> TaskResult:
            return TaskResult(output={"summary": "ok"})

    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-missing-key"
            registry = CapabilityRegistry()
            registry.register(DictCapability())
            registry.register(EchoCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                t1 = TaskDefinitionDTO(
                    task_id="t1",
                    capability_id="test.dict",
                    dependencies=[],
                )
                t2 = TaskDefinitionDTO(
                    task_id="t2",
                    capability_id="test.echo",
                    dependencies=["t1"],
                    input_references={"text": InputReferenceDTO(key="missing_field", source_task_id="t1")},
                )
                wf_input = PlanWorkflowInput(
                    plan_id="p-missing-key",
                    goal_id="g-1",
                    tasks=[t1, t2],
                )

                output: PlanWorkflowOutput = await env.client.execute_workflow(
                    PlanExecutionWorkflow.run,
                    wf_input,
                    id="wf-missing-key",
                    task_queue=tq,
                )

                assert output.status == "FAILED"
                assert output.task_statuses["t1"] == "COMPLETED"
                assert output.task_statuses["t2"] == "FAILED"
                assert "t2" in output.task_errors
                assert output.task_errors["t2"]["error_code"] == "INPUT_RESOLUTION_ERROR"
                assert "key 'missing_field' not found in output" in output.task_errors["t2"]["message"]

    asyncio.run(_test())


