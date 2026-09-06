"""Integration tests for TemporalPlanRunner."""

import asyncio
from typing import Any, Dict
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.dependencies import Dependency
from orchestration.domain.plans import Plan
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.goals import Goal
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.execution.temporal import (
    PlanExecutionWorkflow,
    TaskExecutionActivity,
    TemporalPlanRunner,
)
from orchestration.orchestrator import GoalOrchestrator


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


def test_temporal_runner_run_async_returns_facts():
    """Verify TemporalPlanRunner.run_async returns factual PlanExecutionResult without mutating plan."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-runner-single"
            registry = CapabilityRegistry()
            registry.register(EchoCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                runner = TemporalPlanRunner(client=env.client, task_queue=tq)
                plan = Plan(plan_id="p-single", goal_id="g-1", title="Single Task Plan")
                plan.add_task(
                    Task(
                        task_id="t1",
                        plan_id="p-single",
                        title="T1",
                        capability_id="test.echo",
                        parameters={"text": "hello runner"},
                    )
                )

                result = await runner.run_async(plan)

                # Runner returns execution facts
                assert result.status == PlanStatus.COMPLETED
                assert "t1" in result.task_results
                assert result.task_results["t1"].output == "echo:hello runner"
                assert result.task_attempts["t1"] == "att-t1-1"

                # Domain plan remains unmutated by the runner (GoalOrchestrator owns mutation)
                assert plan.status == PlanStatus.ACTIVE
                assert plan.tasks["t1"].status == TaskStatus.PENDING

    asyncio.run(_test())


def test_temporal_runner_decoupled_lifecycle():
    """Verify start_async, get_status_async, and wait_async."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-runner-lifecycle"
            registry = CapabilityRegistry()
            registry.register(EchoCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                runner = TemporalPlanRunner(client=env.client, task_queue=tq)
                plan = Plan(plan_id="p-decoupled", goal_id="g-1", title="Decoupled Plan")
                plan.add_task(
                    Task(
                        task_id="t1",
                        plan_id="p-decoupled",
                        title="T1",
                        capability_id="test.echo",
                        parameters={"text": "lifecycle data"},
                    )
                )

                handle = await runner.start_async(plan)
                assert handle.plan_id == "p-decoupled"
                assert handle.execution_id == "plan-p-decoupled"

                # Query status
                snapshot = await runner.get_status_async(handle)
                assert snapshot.plan_id == "p-decoupled"

                # Wait for completion
                result = await runner.wait_async(handle)
                assert result.status == PlanStatus.COMPLETED
                assert "t1" in result.task_results
                assert result.task_results["t1"].output == "echo:lifecycle data"

    asyncio.run(_test())


def test_temporal_runner_orchestrator_integration():
    """Verify GoalOrchestrator coordinates TemporalPlanRunner asynchronously."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-runner-chain"
            registry = CapabilityRegistry()
            registry.register(EchoCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                runner = TemporalPlanRunner(client=env.client, task_queue=tq)
                orchestrator = GoalOrchestrator(runner=runner)

                goal = Goal(goal_id="g-chain", description="Chain Goal")
                plan = Plan(plan_id="p-chain", goal_id="g-chain", title="Chain Plan")
                t1 = Task(
                    task_id="t1",
                    plan_id="p-chain",
                    title="Step 1",
                    capability_id="test.echo",
                    parameters={"text": "chain-val"},
                )
                t2 = Task(
                    task_id="t2",
                    plan_id="p-chain",
                    title="Step 2",
                    capability_id="test.echo",
                    dependencies=[Dependency(upstream_task_id="t1", downstream_task_id="t2")],
                    input_references={"text": DataReference(key="output", source_task_id="t1")},
                )
                plan.add_task(t1)
                plan.add_task(t2)

                result_goal = await orchestrator.execute_goal_async(goal, plan)

                assert result_goal.status == GoalStatus.COMPLETED
                assert plan.status == PlanStatus.COMPLETED
                assert plan.tasks["t1"].status == TaskStatus.COMPLETED
                assert plan.tasks["t2"].status == TaskStatus.COMPLETED
                assert plan.tasks["t2"].result is not None
                assert plan.tasks["t2"].result.output == "echo:echo:chain-val"

    asyncio.run(_test())


class ProducerWithRefsAndArtifactsCapability:
    @property
    def capability_id(self) -> str:
        return "test.producer_refs_artifacts"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        from orchestration.domain.references import ArtifactReference, DataReference

        return TaskResult(
            output="generated document",
            references=[
                DataReference(key="doc_ref", uri="ref://doc-1", mime_type="text/plain", metadata={"version": 1}),
            ],
            artifacts=[
                ArtifactReference(
                    artifact_id="art-1",
                    name="report.pdf",
                    uri="artifact://files/report.pdf",
                    mime_type="application/pdf",
                    size_bytes=1024,
                    metadata={"pages": 2},
                )
            ],
        )


def test_temporal_runner_preserves_references_and_artifacts():
    """Verify TemporalPlanRunner preserves TaskResult.references and TaskResult.artifacts on task.result."""
    async def _test():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            tq = "tq-runner-refs"
            registry = CapabilityRegistry()
            registry.register(ProducerWithRefsAndArtifactsCapability())
            act = TaskExecutionActivity(registry)

            async with Worker(
                env.client,
                task_queue=tq,
                workflows=[PlanExecutionWorkflow],
                activities=[act.execute_task],
            ):
                runner = TemporalPlanRunner(client=env.client, task_queue=tq)
                orchestrator = GoalOrchestrator(runner=runner)

                goal = Goal(goal_id="g-refs", description="Refs Goal")
                plan = Plan(plan_id="p-refs", goal_id="g-refs", title="Refs Plan")
                t1 = Task(
                    task_id="t1",
                    plan_id="p-refs",
                    title="Generate Doc",
                    capability_id="test.producer_refs_artifacts",
                )
                plan.add_task(t1)

                result_goal = await orchestrator.execute_goal_async(goal, plan)

                assert result_goal.status == GoalStatus.COMPLETED
                assert plan.status == PlanStatus.COMPLETED
                assert plan.tasks["t1"].status == TaskStatus.COMPLETED
                task_res = plan.tasks["t1"].result
                assert task_res is not None
                assert task_res.output == "generated document"
                assert len(task_res.references) == 1
                assert task_res.references[0].key == "doc_ref"
                assert task_res.references[0].uri == "ref://doc-1"
                assert task_res.references[0].mime_type == "text/plain"
                assert task_res.references[0].metadata == {"version": 1}

                assert len(task_res.artifacts) == 1
                assert task_res.artifacts[0].artifact_id == "art-1"
                assert task_res.artifacts[0].name == "report.pdf"
                assert task_res.artifacts[0].uri == "artifact://files/report.pdf"
                assert task_res.artifacts[0].mime_type == "application/pdf"
                assert task_res.artifacts[0].size_bytes == 1024
                assert task_res.artifacts[0].metadata == {"pages": 2}

    asyncio.run(_test())

