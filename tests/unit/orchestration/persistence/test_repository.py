"""Unit tests for SQLAlchemy Orchestration Repository."""

import pytest
from datetime import datetime, timezone

from orchestration.domain.attempts import Attempt
from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import (
    AttemptStatus,
    GoalStatus,
    PlanStatus,
    TaskErrorCategory,
    TaskStatus,
)
from orchestration.persistence.engine import (
    create_db_engine,
    create_session_factory,
)
from orchestration.persistence.models import Base
from orchestration.persistence.repository import (
    PostgresOrchestrationRepository,
)


@pytest.fixture
def repo():
    """Create an in-memory SQLite database and return a clean PostgresOrchestrationRepository."""
    engine = create_db_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = create_session_factory(engine)
    repository = PostgresOrchestrationRepository(session_or_factory=session_factory)
    try:
        yield repository
    finally:
        repository.close()


def test_save_and_get_goal(repo: PostgresOrchestrationRepository):
    now = datetime.now(timezone.utc)
    goal = Goal(
        goal_id="g-test-1",
        description="Write documentation",
        status=GoalStatus.PENDING,
        context={"author": "alice"},
        created_at=now,
    )

    with repo.transaction():
        repo.goals.save(goal)

    loaded = repo.goals.get("g-test-1")
    assert loaded is not None
    assert loaded.goal_id == "g-test-1"
    assert loaded.description == "Write documentation"
    assert loaded.status == GoalStatus.PENDING
    assert loaded.context == {"author": "alice"}
    assert loaded.active_plan_id is None

    # Update goal status
    goal.activate("p-test-1")
    with repo.transaction():
        repo.goals.save(goal)

    updated = repo.goals.get("g-test-1")
    assert updated is not None
    assert updated.status == GoalStatus.ACTIVE
    assert updated.active_plan_id == "p-test-1"

    # List goals
    all_goals = repo.goals.list_goals()
    assert len(all_goals) == 1
    assert all_goals[0].goal_id == "g-test-1"


def test_save_and_get_plan_with_dag(repo: PostgresOrchestrationRepository):
    # Goal must exist due to foreign key
    goal = Goal(goal_id="g-dag", description="DAG Goal")
    with repo.transaction():
        repo.goals.save(goal)

    plan = Plan(
        plan_id="p-dag",
        goal_id="g-dag",
        title="Pipeline Plan",
        status=PlanStatus.DRAFT,
    )
    t1 = Task(
        task_id="t1",
        plan_id="p-dag",
        title="Fetch Data",
        capability_id="data.fetch",
        parameters={"source": "api"},
    )
    t2 = Task(
        task_id="t2",
        plan_id="p-dag",
        title="Transform Data",
        capability_id="data.transform",
        input_references={
            "raw_data": DataReference(key="output", source_task_id="t1")
        },
        dependencies=[Dependency(upstream_task_id="t1", downstream_task_id="t2")],
    )
    plan.add_task(t1)
    plan.add_task(t2)
    plan.record_revision("rev-1", "Initial pipeline setup")

    with repo.transaction():
        repo.plans.save(plan)

    loaded = repo.plans.get("p-dag")
    assert loaded is not None
    assert loaded.plan_id == "p-dag"
    assert loaded.goal_id == "g-dag"
    assert loaded.status == PlanStatus.DRAFT
    assert len(loaded.tasks) == 2
    assert "t1" in loaded.tasks
    assert "t2" in loaded.tasks
    assert loaded.tasks["t2"].input_references["raw_data"].source_task_id == "t1"
    assert len(loaded.tasks["t2"].dependencies) == 1
    assert loaded.tasks["t2"].dependencies[0].upstream_task_id == "t1"
    assert len(loaded.revisions) == 1
    assert loaded.revisions[0].revision_id == "rev-1"


def test_plan_update_preserves_attempts(repo: PostgresOrchestrationRepository):
    goal = Goal(goal_id="g-att", description="Attempt History Goal")
    with repo.transaction():
        repo.goals.save(goal)

    plan = Plan(
        plan_id="p-att",
        goal_id="g-att",
        title="Attempt History Plan",
        status=PlanStatus.ACTIVE,
    )
    task = Task(
        task_id="t-flaky",
        plan_id="p-att",
        title="Flaky Task",
        capability_id="test.flaky",
        status=TaskStatus.READY,
    )
    plan.add_task(task)

    # First attempt starts
    task.start_attempt("att-1")
    with repo.transaction():
        repo.plans.save(plan)

    loaded_1 = repo.plans.get("p-att")
    assert loaded_1 is not None
    assert loaded_1.tasks["t-flaky"].status == TaskStatus.RUNNING
    assert len(loaded_1.tasks["t-flaky"].attempts) == 1
    assert loaded_1.tasks["t-flaky"].attempts[0].status == AttemptStatus.RUNNING

    # First attempt fails
    task.fail_attempt(
        "att-1",
        TaskError(message="Temporary glitch", category=TaskErrorCategory.INFRASTRUCTURE),
    )
    assert task.status == TaskStatus.FAILED

    # Reset task to READY to start second attempt (simulating retry logic)
    task.status = TaskStatus.READY
    task.start_attempt("att-2")
    task.complete_attempt("att-2", TaskResult(output="Success on retry"))
    assert task.status == TaskStatus.COMPLETED

    with repo.transaction():
        repo.plans.save(plan)

    loaded_2 = repo.plans.get("p-att")
    assert loaded_2 is not None
    t_flaky = loaded_2.tasks["t-flaky"]
    assert t_flaky.status == TaskStatus.COMPLETED
    assert t_flaky.result is not None
    assert t_flaky.result.output == "Success on retry"

    # CRITICAL: Both attempt 1 and attempt 2 are preserved in order!
    assert len(t_flaky.attempts) == 2
    assert t_flaky.attempts[0].attempt_id == "att-1"
    assert t_flaky.attempts[0].status == AttemptStatus.FAILURE
    assert t_flaky.attempts[0].error is not None
    assert t_flaky.attempts[0].error.message == "Temporary glitch"

    assert t_flaky.attempts[1].attempt_id == "att-2"
    assert t_flaky.attempts[1].status == AttemptStatus.SUCCESS
    assert t_flaky.attempts[1].result is not None
    assert t_flaky.attempts[1].result.output == "Success on retry"


def test_task_identity_across_revisions(repo: PostgresOrchestrationRepository):
    """Verify that unchanged tasks retain identity while changed tasks get new IDs across revisions."""
    goal = Goal(goal_id="g-rev", description="Revisions Goal")
    with repo.transaction():
        repo.goals.save(goal)

    plan = Plan(
        plan_id="p-rev",
        goal_id="g-rev",
        title="Revision Test Plan",
        status=PlanStatus.DRAFT,
    )
    t1 = Task(
        task_id="t1",
        plan_id="p-rev",
        title="Task 1",
        capability_id="test.echo",
        parameters={"text": "initial"},
    )
    t2 = Task(
        task_id="t2",
        plan_id="p-rev",
        title="Task 2 (Original)",
        capability_id="test.echo",
        parameters={"version": 1},
    )
    plan.add_task(t1)
    plan.add_task(t2)
    plan.record_revision("rev-1", "Initial plan specification")

    with repo.transaction():
        repo.plans.save(plan)

    # Replanning:
    # t1 succeeded
    t1.status = TaskStatus.READY
    t1.start_attempt("att-t1-1")
    t1.complete_attempt("att-t1-1", TaskResult(output="done"))

    # t2's specification is changed: under our invariant, it requires a NEW task_id: t2_revised
    t2.status = TaskStatus.SKIPPED  # Old task bypassed
    t2_revised = Task(
        task_id="t2_revised",
        plan_id="p-rev",
        title="Task 2 (Revised)",
        capability_id="test.echo",
        parameters={"version": 2, "extra": True},
    )
    t3 = Task(
        task_id="t3",
        plan_id="p-rev",
        title="Task 3 (New)",
        capability_id="test.echo",
    )
    plan.add_task(t2_revised)
    plan.add_task(t3)

    plan.record_revision(
        "rev-2", "Revised task 2 specification and added task 3"
    )

    with repo.transaction():
        repo.plans.save(plan)

    # Both revisions are preserved in database
    loaded = repo.plans.get("p-rev")
    assert loaded is not None
    assert len(loaded.revisions) == 2
    assert len(loaded.tasks) == 4  # t1, t2, t2_revised, t3 all coexist cleanly

    # Historical reconstruction of Revision 1:
    hist_1 = repo.plans.get_historical_plan_revision("p-rev", 1)
    assert hist_1 is not None
    assert len(hist_1.tasks) == 2
    assert "t1" in hist_1.tasks
    assert "t2" in hist_1.tasks
    assert "t2_revised" not in hist_1.tasks
    assert hist_1.tasks["t2"].parameters == {"version": 1}

    # Historical reconstruction of Revision 2:
    hist_2 = repo.plans.get_historical_plan_revision("p-rev", 2)
    assert hist_2 is not None
    assert len(hist_2.tasks) == 4
    assert "t1" in hist_2.tasks
    assert "t2" in hist_2.tasks
    assert "t2_revised" in hist_2.tasks
    assert "t3" in hist_2.tasks
    assert hist_2.tasks["t2"].parameters == {"version": 1}
    assert hist_2.tasks["t2_revised"].parameters == {"version": 2, "extra": True}


def test_transaction_rollback(repo: PostgresOrchestrationRepository):
    goal = Goal(goal_id="g-rollback", description="Rollback Goal")

    with pytest.raises(RuntimeError):
        with repo.transaction():
            repo.goals.save(goal)
            raise RuntimeError("Database error during transaction")

    # Assert goal was rolled back and does not exist
    assert repo.goals.get("g-rollback") is None
