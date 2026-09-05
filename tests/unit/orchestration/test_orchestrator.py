"""Unit and integration tests for GoalOrchestrator."""

from typing import Any, Dict, Optional
import pytest

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.execution.base import PlanRunner
from orchestration.execution.runner import InProcessPlanRunner
from orchestration.orchestrator import GoalOrchestrator


class EchoCapability:
    """Test capability echoing input."""

    @property
    def capability_id(self) -> str:
        return "test.echo"

    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any], context: CapabilityContext) -> TaskResult:
        msg = inputs.get("msg") or parameters.get("msg", "")
        return TaskResult(output=f"echo:{msg}")


class FailingCapability:
    """Test capability that raises an error."""

    @property
    def capability_id(self) -> str:
        return "test.fail"

    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any], context: CapabilityContext) -> TaskResult:
        raise RuntimeError("simulated capability failure")


class MockPlanRunner:
    """Custom runner satisfying PlanRunner protocol for testing boundary decoupling."""

    def __init__(self, target_status: PlanStatus = PlanStatus.COMPLETED) -> None:
        self.target_status = target_status
        self.last_plan: Optional[Plan] = None

    def run(self, plan: Plan) -> Plan:
        self.last_plan = plan
        if self.target_status == PlanStatus.COMPLETED:
            if plan.status == PlanStatus.DRAFT:
                plan.activate()
            plan.mark_completed()
        elif self.target_status == PlanStatus.FAILED:
            if plan.status == PlanStatus.DRAFT:
                plan.activate()
            plan.mark_failed()
        elif self.target_status == PlanStatus.CANCELLED:
            plan.cancel()
        return plan


def _create_runner() -> InProcessPlanRunner:
    """Helper creating InProcessPlanRunner with test capabilities."""
    registry = CapabilityRegistry()
    registry.register(EchoCapability())
    registry.register(FailingCapability())
    return InProcessPlanRunner(registry=registry)


# ---------------------------------------------------------------------------
# Protocol and Decoupling Tests
# ---------------------------------------------------------------------------

def test_in_process_plan_runner_satisfies_plan_runner_protocol():
    """Verify InProcessPlanRunner satisfies the PlanRunner execution boundary protocol."""
    runner = _create_runner()
    assert isinstance(runner, PlanRunner)


def test_orchestrator_accepts_custom_plan_runner():
    """Verify GoalOrchestrator works with any implementation satisfying PlanRunner."""
    mock_runner = MockPlanRunner(target_status=PlanStatus.COMPLETED)
    assert isinstance(mock_runner, PlanRunner)

    orchestrator = GoalOrchestrator(runner=mock_runner)
    assert orchestrator.runner is mock_runner

    goal = Goal(goal_id="g-mock", description="Mock goal")
    plan = Plan(plan_id="p-mock", goal_id="g-mock", title="Mock plan")
    plan.add_task(Task(task_id="t1", plan_id="p-mock", title="T1", capability_id="test.echo"))

    result_goal = orchestrator.execute_goal(goal, plan)
    assert result_goal.status == GoalStatus.COMPLETED
    assert result_goal.active_plan_id == "p-mock"
    assert plan.status == PlanStatus.COMPLETED


# ---------------------------------------------------------------------------
# Lifecycle Coordination Tests
# ---------------------------------------------------------------------------

def test_execute_goal_success():
    """Verify GoalOrchestrator binds Goal, runs Plan, and synchronizes COMPLETED."""
    runner = _create_runner()
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(goal_id="g-1", description="Analyze market trend")
    plan = Plan(plan_id="p-1", goal_id="g-1", title="Market Plan")
    plan.add_task(
        Task(
            task_id="t-1",
            plan_id="p-1",
            title="Echo Task",
            capability_id="test.echo",
            parameters={"msg": "trend data"},
        )
    )

    result_goal = orchestrator.execute_goal(goal, plan)

    assert result_goal.status == GoalStatus.COMPLETED
    assert result_goal.completed_at is not None
    assert result_goal.active_plan_id == "p-1"
    assert plan.status == PlanStatus.COMPLETED
    assert plan.tasks["t-1"].status == TaskStatus.COMPLETED
    assert plan.tasks["t-1"].result.output == "echo:trend data"


def test_execute_goal_with_multi_step_data_flow():
    """Verify multi-step plan execution through GoalOrchestrator."""
    runner = _create_runner()
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(goal_id="g-pipe", description="Data pipeline goal")
    plan = Plan(plan_id="p-pipe", goal_id="g-pipe", title="Data Pipeline")
    t1 = Task(
        task_id="t1",
        plan_id="p-pipe",
        title="Step 1",
        capability_id="test.echo",
        parameters={"msg": "initial"},
    )
    t2 = Task(
        task_id="t2",
        plan_id="p-pipe",
        title="Step 2",
        capability_id="test.echo",
        dependencies=[Dependency(upstream_task_id="t1", downstream_task_id="t2")],
        input_references={"msg": DataReference(key="out", source_task_id="t1")},
    )
    plan.add_task(t1)
    plan.add_task(t2)

    result_goal = orchestrator.execute_goal(goal, plan)

    assert result_goal.status == GoalStatus.COMPLETED
    assert t1.status == TaskStatus.COMPLETED
    assert t2.status == TaskStatus.COMPLETED
    assert t2.result.output == "echo:echo:initial"


def test_execute_goal_failure_synchronization():
    """Verify task failure marks Plan FAILED and synchronizes Goal FAILED."""
    runner = _create_runner()
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(goal_id="g-fail", description="Failing goal")
    plan = Plan(plan_id="p-fail", goal_id="g-fail", title="Failing Plan")
    plan.add_task(
        Task(
            task_id="t-err",
            plan_id="p-fail",
            title="Error Task",
            capability_id="test.fail",
        )
    )

    result_goal = orchestrator.execute_goal(goal, plan)

    assert result_goal.status == GoalStatus.FAILED
    assert result_goal.completed_at is not None
    assert plan.status == PlanStatus.FAILED


# ---------------------------------------------------------------------------
# Validation & Guard Tests
# ---------------------------------------------------------------------------

def test_execute_goal_non_pending_raises():
    """Verify attempting to execute a non-PENDING goal raises ValueError."""
    runner = _create_runner()
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(goal_id="g-1", description="Test")
    goal.status = GoalStatus.ACTIVE  # Already active
    plan = Plan(plan_id="p-1", goal_id="g-1", title="Plan")
    plan.add_task(Task(task_id="t1", plan_id="p-1", title="T1", capability_id="test.echo"))

    with pytest.raises(ValueError, match="expected 'pending'"):
        orchestrator.execute_goal(goal, plan)


def test_execute_goal_mismatched_goal_id_raises():
    """Verify plan belonging to a different goal raises ValueError."""
    runner = _create_runner()
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(goal_id="goal-actual", description="Actual")
    plan = Plan(plan_id="p-1", goal_id="goal-different", title="Plan")
    plan.add_task(Task(task_id="t1", plan_id="p-1", title="T1", capability_id="test.echo"))

    with pytest.raises(ValueError, match="belongs to goal 'goal-different', expected 'goal-actual'"):
        orchestrator.execute_goal(goal, plan)


def test_execute_goal_terminal_plan_status_raises():
    """Verify executing an already completed plan raises ValueError."""
    runner = _create_runner()
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(goal_id="g-1", description="Test")
    plan = Plan(plan_id="p-1", goal_id="g-1", title="Plan")
    plan.add_task(Task(task_id="t1", plan_id="p-1", title="T1", capability_id="test.echo"))
    plan.activate()
    plan.mark_completed()

    with pytest.raises(ValueError, match="expected 'draft' or 'active'"):
        orchestrator.execute_goal(goal, plan)


def test_execute_goal_runner_exception_marks_goal_failed():
    """Verify unhandled runner exception transitions Goal to FAILED and re-raises."""

    class CrashingRunner:
        def run(self, plan: Plan) -> Plan:
            raise RuntimeError("unexpected engine failure")

    orchestrator = GoalOrchestrator(runner=CrashingRunner())  # type: ignore[arg-type]
    goal = Goal(goal_id="g-crash", description="Crash test")
    plan = Plan(plan_id="p-crash", goal_id="g-crash", title="Crash plan")
    plan.add_task(Task(task_id="t1", plan_id="p-crash", title="T1", capability_id="test.echo"))

    with pytest.raises(RuntimeError, match="unexpected engine failure"):
        orchestrator.execute_goal(goal, plan)

    assert goal.status == GoalStatus.FAILED
    assert goal.completed_at is not None


# ---------------------------------------------------------------------------
# Cancellation Tests
# ---------------------------------------------------------------------------

def test_cancel_goal_with_plan():
    """Verify cancel_goal cancels both goal and plan."""
    runner = _create_runner()
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(goal_id="g-cancel", description="To cancel")
    plan = Plan(plan_id="p-cancel", goal_id="g-cancel", title="To cancel")
    t1 = Task(task_id="t1", plan_id="p-cancel", title="T1", capability_id="test.echo")
    plan.add_task(t1)

    orchestrator.cancel_goal(goal, plan)

    assert goal.status == GoalStatus.CANCELLED
    assert plan.status == PlanStatus.CANCELLED
    assert t1.status == TaskStatus.CANCELLED


def test_cancel_goal_mismatched_plan_raises():
    """Verify cancel_goal with mismatched plan raises ValueError."""
    runner = _create_runner()
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(goal_id="g-own", description="Owner")
    plan = Plan(plan_id="p-foreign", goal_id="g-other", title="Other")

    with pytest.raises(ValueError, match="expected 'g-own'"):
        orchestrator.cancel_goal(goal, plan)


def test_cancel_goal_already_completed_raises():
    """Verify cancelling an already completed goal raises ValueError."""
    mock_runner = MockPlanRunner(target_status=PlanStatus.COMPLETED)
    orchestrator = GoalOrchestrator(runner=mock_runner)

    goal = Goal(goal_id="g-done", description="Done")
    plan = Plan(plan_id="p-done", goal_id="g-done", title="Done")
    plan.add_task(Task(task_id="t1", plan_id="p-done", title="T1", capability_id="test.echo"))

    orchestrator.execute_goal(goal, plan)
    assert goal.status == GoalStatus.COMPLETED

    with pytest.raises(ValueError, match="already completed"):
        orchestrator.cancel_goal(goal, plan)


# ---------------------------------------------------------------------------
# AppContext Integration Test
# ---------------------------------------------------------------------------

def test_app_context_create_goal_orchestrator(tmp_path):
    """Verify AppContext factory creates a ready GoalOrchestrator."""
    from apps.context import AppContext

    configs_dir = tmp_path / "configs" / "models"
    models_dir = tmp_path / "models" / "gguf"
    configs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    (models_dir / "test-model.gguf").write_bytes(b"mock-weight-data")

    (configs_dir / "test-model.toml").write_bytes(b"""\
[model]
id = "test-model"
format = "gguf"
path = "models/gguf/test-model.gguf"
supported_providers = ["llama_cpp"]
""")

    settings_file = tmp_path / "settings.toml"
    settings_file.write_bytes(b"""\
[foundation]
environment = "unit-test"
models_dir = "models"
configs_dir = "configs/models"

[providers.llama_cpp]
base_url = "http://127.0.0.1:8080"
""")

    ctx = AppContext.create(
        repo_root=tmp_path,
        configs_dir=configs_dir,
        settings_path=settings_file,
    )

    orchestrator = ctx.create_goal_orchestrator()
    assert isinstance(orchestrator, GoalOrchestrator)
    assert isinstance(orchestrator.runner, PlanRunner)
