"""Unit tests for Replanner work reuse and PlanRevision integration."""

from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.results import TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import PlanStatus, TaskStatus
from orchestration.planning import (
    CandidatePlan,
    CandidateTask,
    PlanningContext,
    Replanner,
    TemplatePlanner,
)


def test_replanner_preserves_completed_work():
    # Setup prior plan with:
    # t1 (COMPLETED with result)
    # t2 (FAILED)
    prior_plan = Plan(plan_id="plan_1", goal_id="g1", title="Original Plan", status=PlanStatus.ACTIVE)

    t1 = Task(
        task_id="t1",
        plan_id="plan_1",
        title="Extraction",
        capability_id="text_extract",
        status=TaskStatus.COMPLETED,
        result=TaskResult(output={"extracted": "abc"}),
    )
    t2 = Task(
        task_id="t2",
        plan_id="plan_1",
        title="Synthesis",
        capability_id="summarize",
        status=TaskStatus.FAILED,
        dependencies=[Dependency("t1", "t2")],
    )
    prior_plan.add_task(t1)
    prior_plan.add_task(t2)
    prior_plan.record_revision(revision_id="rev_1", reason="Initial plan")

    # Inner planner proposes replacement task t3
    def replan_template(ctx: PlanningContext) -> CandidatePlan:
        assert "t1" in ctx.available_task_ids
        assert "t1" in ctx.completed_tasks
        assert ctx.completed_tasks["t1"].output == {"extracted": "abc"}

        return CandidatePlan(
            plan_id=ctx.prior_plan.plan_id,
            goal_id=ctx.goal.goal_id,
            title="Revised Plan",
            tasks=[
                CandidateTask(
                    task_id="t3",  # New task with new ID
                    title="Alternative Synthesis",
                    capability_id="summarize",
                    dependencies=[Dependency("t1", "t3")],
                )
            ],
            dependencies=[Dependency("t1", "t3")],
        )

    template_planner = TemplatePlanner(templates={"default": replan_template})
    replanner = Replanner(planner=template_planner)

    goal = Goal(goal_id="g1", description="Process data")
    revised_candidate = replanner.replan(
        goal=goal,
        prior_plan=prior_plan,
        reason="Task t2 failed; replacing with t3",
        preserve_completed=True,
    )

    # Verify candidate contains both reused t1 and new t3
    task_ids = [t.task_id for t in revised_candidate.tasks]
    assert "t1" in task_ids
    assert "t3" in task_ids
    assert revised_candidate.metadata["reused_task_count"] == 1

    # Convert to domain Plan and verify state preservation
    new_plan = revised_candidate.to_plan()
    assert "t1" in new_plan.tasks
    assert "t3" in new_plan.tasks
    assert new_plan.tasks["t3"].dependencies[0].upstream_task_id == "t1"

    # Invariant 1: Reused completed task retains status, result, and identity
    assert new_plan.tasks["t1"].status == TaskStatus.COMPLETED
    assert new_plan.tasks["t1"].result is not None
    assert new_plan.tasks["t1"].result.output == {"extracted": "abc"}
    assert new_plan.tasks["t1"].task_id == "t1"

    # New task has fresh PENDING status and no result
    assert new_plan.tasks["t3"].status == TaskStatus.PENDING
    assert new_plan.tasks["t3"].result is None

    # Invariant 3: Prior PlanRevisions are preserved on new_plan
    assert len(new_plan.revisions) == 1
    assert new_plan.revisions[0].revision_id == "rev_1"
    assert new_plan.revisions[0].revision_number == 1

    # Appending a new revision creates rev_2 with revision_number == 2
    rev2 = new_plan.record_revision(revision_id="rev_2", reason="Task t2 failed; replacing with t3")
    assert len(new_plan.revisions) == 2
    assert rev2.revision_number == 2
    assert "t1" in rev2.task_ids
    assert "t3" in rev2.task_ids

    # Verify persistence mapping works on the replanned aggregate
    from orchestration.persistence.mappers import plan_to_model, model_to_plan
    plan_model = plan_to_model(new_plan)
    roundtripped_plan = model_to_plan(plan_model)
    assert roundtripped_plan.tasks["t1"].status == TaskStatus.COMPLETED
    assert roundtripped_plan.tasks["t1"].result.output == {"extracted": "abc"}
    assert len(roundtripped_plan.revisions) == 2
    assert roundtripped_plan.revisions[1].revision_number == 2
