"""Unit tests for domain <-> ORM mappers in orchestration persistence."""

from datetime import datetime, timezone

from orchestration.domain.attempts import Attempt
from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan, PlanRevision
from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskError, TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import (
    AttemptStatus,
    GoalStatus,
    PlanStatus,
    TaskErrorCategory,
    TaskStatus,
)
from orchestration.persistence.mappers import (
    attempt_to_model,
    build_revision_snapshot_payload,
    goal_to_model,
    model_to_attempt,
    model_to_goal,
    model_to_plan,
    model_to_task,
    plan_to_model,
    reconstruct_historical_plan,
    task_error_to_dict,
    task_result_to_dict,
    task_to_model,
)


def test_goal_mapper_roundtrip():
    now = datetime.now(timezone.utc)
    goal = Goal(
        goal_id="g-123",
        description="Test Goal Description",
        status=GoalStatus.ACTIVE,
        context={"user_id": "u-456", "priority": "high"},
        active_plan_id="p-789",
        created_at=now,
        completed_at=None,
    )

    model = goal_to_model(goal)
    assert model.goal_id == "g-123"
    assert model.description == "Test Goal Description"
    assert model.status == "active"
    assert model.context == {"user_id": "u-456", "priority": "high"}
    assert model.active_plan_id == "p-789"

    roundtrip = model_to_goal(model)
    assert roundtrip.goal_id == goal.goal_id
    assert roundtrip.description == goal.description
    assert roundtrip.status == GoalStatus.ACTIVE
    assert roundtrip.context == goal.context
    assert roundtrip.active_plan_id == goal.active_plan_id
    assert roundtrip.created_at == goal.created_at
    assert roundtrip.completed_at is None


def test_value_object_serialization():
    data_ref = DataReference(
        key="extracted_data",
        source_task_id="t-1",
        uri="/tmp/data.json",
        mime_type="application/json",
        metadata={"rows": 100},
    )
    art_ref = ArtifactReference(
        artifact_id="art-1",
        name="report.pdf",
        uri="/tmp/report.pdf",
        mime_type="application/pdf",
        size_bytes=4096,
        metadata={"pages": 3},
    )
    result = TaskResult(
        output={"summary": "Success"},
        references=[data_ref],
        artifacts=[art_ref],
        metadata={"latency_ms": 120},
    )
    res_dict = task_result_to_dict(result)
    assert res_dict is not None
    assert res_dict["output"] == {"summary": "Success"}
    assert len(res_dict["references"]) == 1
    assert res_dict["references"][0]["key"] == "extracted_data"
    assert len(res_dict["artifacts"]) == 1
    assert res_dict["artifacts"][0]["size_bytes"] == 4096

    error = TaskError(
        message="Model timeout exceeded",
        category=TaskErrorCategory.TIMEOUT,
        error_code="TIMEOUT_EXCEEDED",
        details={"deadline_s": 30},
        cause_exception_type="asyncio.TimeoutError",
    )
    err_dict = task_error_to_dict(error)
    assert err_dict is not None
    assert err_dict["category"] == "timeout"
    assert err_dict["error_code"] == "TIMEOUT_EXCEEDED"
    assert err_dict["cause_exception_type"] == "asyncio.TimeoutError"


def test_task_and_attempt_mapper_roundtrip():
    now = datetime.now(timezone.utc)
    task = Task(
        task_id="t-task1",
        plan_id="p-plan1",
        title="Execute Inference",
        capability_id="inference.chat",
        description="Process chat query",
        parameters={"temperature": 0.7},
        input_references={
            "input_doc": DataReference(key="doc", source_task_id="t-prev")
        },
        status=TaskStatus.COMPLETED,
        created_at=now,
        started_at=now,
        completed_at=now,
        result=TaskResult(output="Generated response"),
    )
    attempt = Attempt(
        attempt_id="att-task1-1",
        task_id="t-task1",
        attempt_number=1,
        started_at=now,
        completed_at=now,
        status=AttemptStatus.SUCCESS,
        result=TaskResult(output="Generated response"),
        metadata={"worker_id": "worker-1"},
    )
    task.attempts.append(attempt)

    t_model = task_to_model(task)
    assert t_model.task_id == "t-task1"
    assert t_model.capability_id == "inference.chat"
    assert t_model.status == "completed"
    assert len(t_model.attempts) == 1
    assert t_model.attempts[0].attempt_id == "att-task1-1"
    assert t_model.attempts[0].status == "success"

    reconstructed_task = model_to_task(t_model)
    assert reconstructed_task.task_id == task.task_id
    assert reconstructed_task.status == TaskStatus.COMPLETED
    assert reconstructed_task.result is not None
    assert reconstructed_task.result.output == "Generated response"
    assert len(reconstructed_task.attempts) == 1
    assert reconstructed_task.attempts[0].attempt_id == "att-task1-1"
    assert reconstructed_task.attempts[0].status == AttemptStatus.SUCCESS
    assert reconstructed_task.input_references["input_doc"].key == "doc"


def test_plan_mapper_roundtrip_with_dag_and_revisions():
    now = datetime.now(timezone.utc)
    plan = Plan(
        plan_id="p-plan100",
        goal_id="g-goal100",
        title="DAG Plan",
        status=PlanStatus.ACTIVE,
        created_at=now,
    )
    t1 = Task(
        task_id="t1",
        plan_id="p-plan100",
        title="Step 1",
        capability_id="test.echo",
        parameters={"text": "hello"},
    )
    t2 = Task(
        task_id="t2",
        plan_id="p-plan100",
        title="Step 2",
        capability_id="test.echo",
        parameters={"text": "world"},
        dependencies=[Dependency(upstream_task_id="t1", downstream_task_id="t2")],
    )
    plan.add_task(t1)
    plan.add_task(t2)

    # Record revision
    rev = plan.record_revision("rev-1", "Initial plan creation")
    assert rev.task_ids == ["t1", "t2"]

    plan_model = plan_to_model(plan)
    assert plan_model.plan_id == "p-plan100"
    assert len(plan_model.tasks) == 2
    assert len(plan_model.revisions) == 1

    # Verify snapshot_payload in revision model
    rev_model = plan_model.revisions[0]
    assert rev_model.revision_id == "rev-1"
    assert rev_model.revision_number == 1
    assert rev_model.task_ids == ["t1", "t2"]
    assert "t1" in rev_model.snapshot_payload["tasks"]
    assert "t2" in rev_model.snapshot_payload["tasks"]
    assert len(rev_model.snapshot_payload["dependencies"]) == 1

    # Reconstruct domain Plan
    domain_plan = model_to_plan(plan_model)
    assert domain_plan.plan_id == "p-plan100"
    assert len(domain_plan.tasks) == 2
    assert len(domain_plan.tasks["t2"].dependencies) == 1
    assert domain_plan.tasks["t2"].dependencies[0].upstream_task_id == "t1"
    assert len(domain_plan.revisions) == 1
    assert domain_plan.revisions[0].task_ids == ["t1", "t2"]
    # Domain PlanRevision does not have snapshot_payload
    assert not hasattr(domain_plan.revisions[0], "snapshot_payload")


def test_reconstruct_historical_plan_from_snapshot():
    now = datetime.now(timezone.utc)
    plan = Plan(
        plan_id="p-hist",
        goal_id="g-hist",
        title="Historical Test",
        status=PlanStatus.DRAFT,
        created_at=now,
    )
    t1 = Task(
        task_id="t1",
        plan_id="p-hist",
        title="Original T1",
        capability_id="cap.v1",
        parameters={"param": "initial"},
    )
    plan.add_task(t1)
    plan.record_revision("rev-1", "Revision 1 specification")

    # Mutate plan for Revision 2
    plan.status = PlanStatus.ACTIVE
    t1.status = TaskStatus.COMPLETED
    t1.result = TaskResult(output="done")

    t2 = Task(
        task_id="t2",
        plan_id="p-hist",
        title="New Task T2",
        capability_id="cap.v2",
        parameters={"param": "second"},
    )
    plan.add_task(t2)
    plan.record_revision("rev-2", "Added T2 after T1 finished")

    # Map to ORM
    plan_model = plan_to_model(plan)

    # Reconstruct Revision 1 from snapshot
    hist_rev1 = reconstruct_historical_plan(plan_model, revision_number=1)
    assert hist_rev1 is not None
    assert hist_rev1.plan_id == "p-hist"
    assert len(hist_rev1.tasks) == 1
    assert "t1" in hist_rev1.tasks
    assert "t2" not in hist_rev1.tasks
    # Planned historical state: Task is PENDING with original parameters
    assert hist_rev1.tasks["t1"].title == "Original T1"
    assert hist_rev1.tasks["t1"].parameters == {"param": "initial"}
    assert hist_rev1.tasks["t1"].status == TaskStatus.PENDING
    assert hist_rev1.tasks["t1"].result is None

    # Reconstruct Revision 2 from snapshot
    hist_rev2 = reconstruct_historical_plan(plan_model, revision_number=2)
    assert hist_rev2 is not None
    assert len(hist_rev2.tasks) == 2
    assert "t1" in hist_rev2.tasks
    assert "t2" in hist_rev2.tasks
