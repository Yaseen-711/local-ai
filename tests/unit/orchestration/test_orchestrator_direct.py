"""Unit tests for GoalOrchestrator direct-goal execution."""

import pytest

from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.goals import Goal
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.types import GoalStatus, TaskErrorCategory
from orchestration.execution.runner import InProcessPlanRunner
from orchestration.orchestrator import DirectGoalResult, GoalOrchestrator
from orchestration.persistence.engine import create_db_engine, create_session_factory
from orchestration.persistence.models import Base
from orchestration.persistence.repository import PostgresOrchestrationRepository


class EchoCapability:
    @property
    def capability_id(self) -> str:
        return "test.echo"

    def execute(self, parameters, inputs, context):
        msg = parameters.get("message") or inputs.get("message", "echo")
        return TaskResult(output={"echo": msg})


class FailingCapability:
    @property
    def capability_id(self) -> str:
        return "test.failing"

    def execute(self, parameters, inputs, context):
        raise RuntimeError("Direct capability failed")


@pytest.fixture
def orchestrator_with_registry():
    registry = CapabilityRegistry()
    registry.register(EchoCapability())
    registry.register(FailingCapability())
    runner = InProcessPlanRunner(registry=registry)
    return GoalOrchestrator(runner=runner, registry=registry)


def test_execute_direct_goal_success(orchestrator_with_registry):
    goal = Goal(
        goal_id="g-dir-1",
        description="Direct echo goal",
        context={"user_id": "u123"},
    )

    result = orchestrator_with_registry.execute_direct_goal(
        goal=goal,
        capability_id="test.echo",
        parameters={"message": "hello direct"},
    )

    assert isinstance(result, DirectGoalResult)
    assert result.goal.status == GoalStatus.COMPLETED
    assert result.goal.completed_at is not None
    assert result.result is not None
    assert result.result.output == {"echo": "hello direct"}
    assert result.error is None
    # Crucial: Goal.context remains pure and untouched!
    assert result.goal.context == {"user_id": "u123"}
    assert "direct_result" not in result.goal.context


def test_execute_direct_goal_missing_capability(orchestrator_with_registry):
    goal = Goal(goal_id="g-dir-2", description="Missing cap goal")

    result = orchestrator_with_registry.execute_direct_goal(
        goal=goal,
        capability_id="nonexistent.cap",
    )

    assert result.goal.status == GoalStatus.FAILED
    assert result.error is not None
    assert result.error.category == TaskErrorCategory.CAPABILITY
    assert "not found" in result.error.message
    assert result.result is None


def test_execute_direct_goal_failure(orchestrator_with_registry):
    goal = Goal(goal_id="g-dir-3", description="Failing goal")

    result = orchestrator_with_registry.execute_direct_goal(
        goal=goal,
        capability_id="test.failing",
    )

    assert result.goal.status == GoalStatus.FAILED
    assert result.error is not None
    assert "Direct capability failed" in result.error.message


def test_execute_deterministic_goal(orchestrator_with_registry):
    goal = Goal(goal_id="g-det-1", description="Deterministic goal")

    def handler():
        return TaskResult(output={"system": "ok", "version": "1.0"})

    result = orchestrator_with_registry.execute_deterministic_goal(
        goal=goal,
        handler=handler,
    )

    assert result.goal.status == GoalStatus.COMPLETED
    assert result.result is not None
    assert result.result.output == {"system": "ok", "version": "1.0"}


def test_execute_direct_goal_async(orchestrator_with_registry):
    async def _test():
        goal = Goal(goal_id="g-dir-async", description="Async direct goal")
        result = await orchestrator_with_registry.execute_direct_goal_async(
            goal=goal,
            capability_id="test.echo",
            parameters={"message": "async echo"},
        )
        assert result.goal.status == GoalStatus.COMPLETED
        assert result.result.output == {"echo": "async echo"}

    import asyncio
    asyncio.run(_test())


def test_execute_direct_goal_persistence():
    engine = create_db_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = create_session_factory(engine)
    repo = PostgresOrchestrationRepository(session_or_factory=session_factory)

    registry = CapabilityRegistry()
    registry.register(EchoCapability())
    runner = InProcessPlanRunner(registry=registry)
    orchestrator = GoalOrchestrator(runner=runner, repository=repo, registry=registry)

    goal = Goal(goal_id="g-persisted-direct", description="Persisted direct goal")
    result = orchestrator.execute_direct_goal(
        goal=goal,
        capability_id="test.echo",
        parameters={"message": "persisted"},
    )

    assert result.goal.status == GoalStatus.COMPLETED
    loaded_goal = repo.goals.get("g-persisted-direct")
    assert loaded_goal is not None
    assert loaded_goal.status == GoalStatus.COMPLETED
    assert loaded_goal.active_plan_id == "direct:test.echo"
    repo.close()
