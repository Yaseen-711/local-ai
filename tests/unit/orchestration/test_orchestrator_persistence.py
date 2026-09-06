"""Integration tests for GoalOrchestrator with OrchestrationRepository persistence."""

import asyncio
import pytest
from typing import Any, Optional

from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.tasks import Task
from orchestration.domain.results import TaskResult
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.execution.base import (
    ExecutionHandle,
    PlanExecutionResult,
    PlanExecutionSnapshot,
    PlanRunner,
)
from orchestration.execution.runner import InProcessPlanRunner
from orchestration.orchestrator import GoalOrchestrator
from orchestration.persistence.engine import (
    create_db_engine,
    create_session_factory,
)
from orchestration.persistence.models import Base
from orchestration.persistence.repository import (
    PostgresOrchestrationRepository,
)


class EchoCapability(Capability):
    @property
    def capability_id(self) -> str:
        return "test.echo"

    def execute(
        self,
        parameters: dict,
        inputs: dict,
        context: CapabilityContext,
    ) -> TaskResult:
        return TaskResult(output=f"Echo: {parameters.get('text', '')}")


@pytest.fixture
def repo():
    """In-memory SQLite repository fixture."""
    engine = create_db_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = create_session_factory(engine)
    repository = PostgresOrchestrationRepository(session_or_factory=session_factory)
    try:
        yield repository
    finally:
        repository.close()


@pytest.fixture
def registry():
    reg = CapabilityRegistry()
    reg.register(EchoCapability())
    return reg


def test_orchestrator_persists_milestones_on_success(
    repo: PostgresOrchestrationRepository, registry: CapabilityRegistry
):
    runner = InProcessPlanRunner(registry=registry)
    orchestrator = GoalOrchestrator(runner=runner, repository=repo)

    goal = Goal(goal_id="g-pers-1", description="Success Goal")
    plan = Plan(plan_id="p-pers-1", goal_id="g-pers-1", title="Success Plan")
    plan.add_task(
        Task(
            task_id="t1",
            plan_id="p-pers-1",
            title="Echo Task",
            capability_id="test.echo",
            parameters={"text": "hello persistence"},
        )
    )

    # Initial state: not in repository yet
    assert repo.goals.get("g-pers-1") is None
    assert repo.plans.get("p-pers-1") is None

    # Execute
    res_goal = orchestrator.execute_goal(goal, plan)
    assert res_goal.status == GoalStatus.COMPLETED
    assert plan.status == PlanStatus.COMPLETED

    # Verify both Goal and Plan were persisted to the database in terminal state
    persisted_goal = repo.goals.get("g-pers-1")
    assert persisted_goal is not None
    assert persisted_goal.status == GoalStatus.COMPLETED
    assert persisted_goal.active_plan_id == "p-pers-1"
    assert persisted_goal.completed_at is not None

    persisted_plan = repo.plans.get("p-pers-1")
    assert persisted_plan is not None
    assert persisted_plan.status == PlanStatus.COMPLETED
    assert persisted_plan.tasks["t1"].status == TaskStatus.COMPLETED
    assert persisted_plan.tasks["t1"].result is not None
    assert persisted_plan.tasks["t1"].result.output == "Echo: hello persistence"
    assert len(persisted_plan.tasks["t1"].attempts) == 1


def test_orchestrator_persists_milestones_on_failure(
    repo: PostgresOrchestrationRepository, registry: CapabilityRegistry
):
    runner = InProcessPlanRunner(registry=registry)
    orchestrator = GoalOrchestrator(runner=runner, repository=repo)

    goal = Goal(goal_id="g-pers-fail", description="Failure Goal")
    plan = Plan(plan_id="p-pers-fail", goal_id="g-pers-fail", title="Failure Plan")
    plan.add_task(
        Task(
            task_id="t-missing",
            plan_id="p-pers-fail",
            title="Missing Cap",
            capability_id="test.non_existent",
        )
    )

    res_goal = orchestrator.execute_goal(goal, plan)
    assert res_goal.status == GoalStatus.FAILED
    assert plan.status == PlanStatus.FAILED

    # Verify persisted failure states
    persisted_goal = repo.goals.get("g-pers-fail")
    assert persisted_goal is not None
    assert persisted_goal.status == GoalStatus.FAILED

    persisted_plan = repo.plans.get("p-pers-fail")
    assert persisted_plan is not None
    assert persisted_plan.status == PlanStatus.FAILED
    assert persisted_plan.tasks["t-missing"].status == TaskStatus.FAILED
    assert persisted_plan.tasks["t-missing"].error is not None


def test_orchestrator_persists_milestones_on_cancellation(
    repo: PostgresOrchestrationRepository,
):
    class CancellableRunner(PlanRunner):
        def start(self, plan: Plan) -> ExecutionHandle:
            return ExecutionHandle(execution_id="h-cancel", plan_id=plan.plan_id)

        def wait(
            self, handle: ExecutionHandle, timeout: Optional[float] = None
        ) -> PlanExecutionResult:
            return PlanExecutionResult(
                execution_id=handle.execution_id,
                plan_id=handle.plan_id,
                status=PlanStatus.CANCELLED,
            )

        def get_status(self, handle: ExecutionHandle) -> PlanExecutionSnapshot:
            return PlanExecutionSnapshot(
                execution_id=handle.execution_id,
                plan_id=handle.plan_id,
                status=PlanStatus.CANCELLED,
                task_statuses={},
                is_terminal=True,
            )

        def cancel(self, handle: ExecutionHandle) -> None:
            pass

        def run(self, plan: Plan) -> Any:
            handle = self.start(plan)
            return self.wait(handle)

    runner = CancellableRunner()
    orchestrator = GoalOrchestrator(runner=runner, repository=repo)

    goal = Goal(goal_id="g-cancel", description="Cancel Goal")
    plan = Plan(plan_id="p-cancel", goal_id="g-cancel", title="Cancel Plan")
    plan.add_task(
        Task(
            task_id="t-c",
            plan_id="p-cancel",
            title="Cancel Task",
            capability_id="test.echo",
        )
    )

    goal.status = GoalStatus.ACTIVE
    orchestrator.cancel_goal(goal, plan)

    persisted_goal = repo.goals.get("g-cancel")
    assert persisted_goal is not None
    assert persisted_goal.status == GoalStatus.CANCELLED

    persisted_plan = repo.plans.get("p-cancel")
    assert persisted_plan is not None
    assert persisted_plan.status == PlanStatus.CANCELLED
    assert persisted_plan.tasks["t-c"].status == TaskStatus.CANCELLED


def test_orchestrator_persists_milestones_async(
    repo: PostgresOrchestrationRepository,
):
    async def _test():
        class AsyncRunner:
            async def start_async(self, plan: Plan) -> ExecutionHandle:
                return ExecutionHandle(execution_id="h-async", plan_id=plan.plan_id)

            async def wait_async(
                self, handle: ExecutionHandle
            ) -> PlanExecutionResult:
                return PlanExecutionResult(
                    execution_id=handle.execution_id,
                    plan_id=handle.plan_id,
                    status=PlanStatus.COMPLETED,
                )

            async def cancel_async(self, handle: ExecutionHandle) -> None:
                pass

            def start(self, plan: Plan) -> ExecutionHandle:
                raise NotImplementedError

            def wait(
                self, handle: ExecutionHandle, timeout: Optional[float] = None
            ) -> PlanExecutionResult:
                raise NotImplementedError

            def get_status(self, handle: ExecutionHandle) -> PlanExecutionSnapshot:
                raise NotImplementedError

            def cancel(self, handle: ExecutionHandle) -> None:
                raise NotImplementedError

            def run(self, plan: Plan) -> Any:
                raise NotImplementedError

        orchestrator = GoalOrchestrator(runner=AsyncRunner(), repository=repo)  # type: ignore[arg-type]

        goal = Goal(goal_id="g-async-pers", description="Async Persist Goal")
        plan = Plan(
            plan_id="p-async-pers", goal_id="g-async-pers", title="Async Plan"
        )
        plan.add_task(
            Task(
                task_id="t-a",
                plan_id="p-async-pers",
                title="Async Task",
                capability_id="test.echo",
            )
        )

        res_goal = await orchestrator.execute_goal_async(goal, plan)
        assert res_goal.status == GoalStatus.COMPLETED

        persisted_goal = repo.goals.get("g-async-pers")
        assert persisted_goal is not None
        assert persisted_goal.status == GoalStatus.COMPLETED

        persisted_plan = repo.plans.get("p-async-pers")
        assert persisted_plan is not None
        assert persisted_plan.status == PlanStatus.COMPLETED

    asyncio.run(_test())
