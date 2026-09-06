"""Unit tests for DecisionEngine routing, direct execution, planning, and validation."""

import asyncio
from unittest.mock import MagicMock
import pytest

from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.decision import DecisionEngine, DecisionPolicy, DecisionResult
from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.results import TaskResult
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.execution import InProcessPlanRunner
from orchestration.orchestrator import GoalOrchestrator
from orchestration.planning import CandidatePlan, CandidateTask, TemplatePlanner
from orchestration.routing.deterministic import DeterministicRuleMatcher
from orchestration.routing.staged import StagedEscalationRouter
from orchestration.routing.types import ExecutionStrategy, RouteDefinition
from orchestration.validation.validator import PlanValidator


class EchoCapability:
    def __init__(self, cid: str = "echo"):
        self._cid = cid

    @property
    def capability_id(self) -> str:
        return self._cid

    def execute(self, parameters, inputs, context):
        return TaskResult(output={"echo": parameters.get("message", "ok")})


@pytest.fixture
def test_setup():
    registry = CapabilityRegistry()
    cap = EchoCapability("echo")
    registry.register(
        cap,
        descriptor=CapabilityDescriptor(
            capability_id="echo",
            description="Echoes message",
            parameter_schema={"type": "object", "required": ["message"]},
        ),
    )

    runner = InProcessPlanRunner(registry=registry)
    orchestrator = GoalOrchestrator(runner=runner, registry=registry)

    routes = [
        RouteDefinition(
            name="ping",
            strategy=ExecutionStrategy.DIRECT_DETERMINISTIC,
            metadata={"prefixes": ["ping"]},
        ),
        RouteDefinition(
            name="direct_echo",
            strategy=ExecutionStrategy.DIRECT_CAPABILITY,
            target_capability_id="echo",
            metadata={"prefixes": ["echo"]},
        ),
        RouteDefinition(
            name="pipeline",
            strategy=ExecutionStrategy.PLAN_REQUIRED,
            metadata={"prefixes": ["run pipeline"]},
        ),
        RouteDefinition(
            name="malicious",
            strategy=ExecutionStrategy.REJECT,
            metadata={"prefixes": ["rm -rf", "drop table"]},
        ),
    ]
    router = StagedEscalationRouter(routes=routes)

    # Template planner for "pipeline"
    def pipeline_template(ctx):
        return CandidatePlan(
            plan_id="p_pipe",
            goal_id=ctx.goal.goal_id,
            title="Echo Pipeline",
            tasks=[
                CandidateTask(
                    task_id="t1",
                    title="Echo Task",
                    capability_id="echo",
                    parameters={"message": "hello pipeline"},
                )
            ],
        )

    planner = TemplatePlanner(templates={"default": pipeline_template})
    validator = PlanValidator(capability_registry=registry)

    engine = DecisionEngine(
        router=router,
        orchestrator=orchestrator,
        planner=planner,
        validator=validator,
        deterministic_handlers={"ping": lambda g: {"pong": True}},
    )

    return engine, orchestrator, registry


def test_direct_deterministic_execution(test_setup):
    engine, orchestrator, _ = test_setup
    goal = Goal(goal_id="g_ping", description="ping server")

    result = engine.process_goal(goal)
    assert result.decision_type == ExecutionStrategy.DIRECT_DETERMINISTIC
    assert result.direct_result is not None
    assert result.direct_result.result is not None
    assert result.direct_result.result.output == {"pong": True}
    assert goal.status == GoalStatus.COMPLETED
    # Verify Goal.context was not polluted with output
    assert "pong" not in goal.context


def test_direct_capability_execution(test_setup):
    engine, orchestrator, _ = test_setup
    goal = Goal(
        goal_id="g_echo",
        description="echo fast",
        context={"parameters": {"message": "hello direct"}},
    )

    result = engine.process_goal(goal)
    assert result.decision_type == ExecutionStrategy.DIRECT_CAPABILITY
    assert result.direct_result is not None
    assert result.direct_result.error is None
    assert result.direct_result.result is not None
    assert result.direct_result.result.output == {"echo": "hello direct"}
    assert goal.status == GoalStatus.COMPLETED
    # Verify Goal.context was not polluted
    assert "echo" not in goal.context


def test_plan_required_execution_success(test_setup):
    engine, orchestrator, _ = test_setup
    goal = Goal(goal_id="g_pipe", description="run pipeline test")

    result = engine.process_goal(goal)
    assert result.decision_type == ExecutionStrategy.PLAN_REQUIRED
    assert result.validation_result.is_valid is True
    assert result.plan_id == "p_pipe"
    assert goal.status == GoalStatus.COMPLETED


def test_plan_required_validation_failure_prevents_execution(test_setup):
    engine, orchestrator, registry = test_setup

    # Planner that produces an invalid plan (unknown capability)
    bad_planner = TemplatePlanner(
        templates={
            "default": lambda ctx: CandidatePlan(
                plan_id="p_bad",
                goal_id=ctx.goal.goal_id,
                title="Bad Plan",
                tasks=[
                    CandidateTask(
                        task_id="t1",
                        title="Invalid Cap Task",
                        capability_id="nonexistent_cap",
                    )
                ],
            )
        }
    )

    bad_engine = DecisionEngine(
        router=engine._router,
        orchestrator=orchestrator,
        planner=bad_planner,
        validator=engine._validator,
    )

    goal = Goal(goal_id="g_fail", description="run pipeline bad")
    result = bad_engine.process_goal(goal)

    assert result.decision_type == ExecutionStrategy.PLAN_REQUIRED
    assert result.validation_result.is_valid is False
    assert any(e.code == "UNKNOWN_CAPABILITY" for e in result.validation_result.errors)
    # Goal should remain PENDING (not activated)
    assert goal.status == GoalStatus.PENDING


def test_reject_strategy(test_setup):
    engine, orchestrator, _ = test_setup
    goal = Goal(goal_id="g_bad", description="drop table users")

    result = engine.process_goal(goal)
    assert result.decision_type == ExecutionStrategy.REJECT
    assert goal.status == GoalStatus.FAILED


def test_async_processing(test_setup):
    engine, orchestrator, _ = test_setup

    async def _run():
        g1 = Goal(goal_id="g_async_ping", description="ping server async")
        res1 = await engine.process_goal_async(g1)
        assert res1.decision_type == ExecutionStrategy.DIRECT_DETERMINISTIC
        assert g1.status == GoalStatus.COMPLETED

        g2 = Goal(goal_id="g_async_pipe", description="run pipeline async")
        res2 = await engine.process_goal_async(g2)
        assert res2.decision_type == ExecutionStrategy.PLAN_REQUIRED
        assert g2.status == GoalStatus.COMPLETED

    asyncio.run(_run())
