"""Domain <-> ORM Mappers for Orchestration Persistence.

Pure transformation functions between pure domain dataclasses/value objects
and SQLAlchemy 2.0 ORM models.
Includes generation of persistence-level snapshot_payload for PlanRevisionModel
without modifying domain PlanRevision.
"""

from typing import Any, Dict, List, Optional

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
from orchestration.persistence.models import (
    AttemptModel,
    DependencyModel,
    GoalModel,
    PlanModel,
    PlanRevisionModel,
    TaskModel,
)


# ---------------------------------------------------------------------------
# Value Object Serialization & Deserialization
# ---------------------------------------------------------------------------


def data_reference_to_dict(ref: DataReference) -> Dict[str, Any]:
    """Serialize a DataReference value object to a dictionary."""
    return {
        "key": ref.key,
        "source_task_id": ref.source_task_id,
        "uri": ref.uri,
        "mime_type": ref.mime_type,
        "metadata": dict(ref.metadata),
    }


def dict_to_data_reference(d: Dict[str, Any]) -> DataReference:
    """Deserialize a dictionary to a DataReference value object."""
    return DataReference(
        key=d["key"],
        source_task_id=d.get("source_task_id"),
        uri=d.get("uri"),
        mime_type=d.get("mime_type", "application/json"),
        metadata=dict(d.get("metadata", {})),
    )


def artifact_reference_to_dict(ref: ArtifactReference) -> Dict[str, Any]:
    """Serialize an ArtifactReference value object to a dictionary."""
    return {
        "artifact_id": ref.artifact_id,
        "name": ref.name,
        "uri": ref.uri,
        "mime_type": ref.mime_type,
        "size_bytes": ref.size_bytes,
        "metadata": dict(ref.metadata),
    }


def dict_to_artifact_reference(d: Dict[str, Any]) -> ArtifactReference:
    """Deserialize a dictionary to an ArtifactReference value object."""
    return ArtifactReference(
        artifact_id=d["artifact_id"],
        name=d["name"],
        uri=d["uri"],
        mime_type=d.get("mime_type", "application/octet-stream"),
        size_bytes=d.get("size_bytes"),
        metadata=dict(d.get("metadata", {})),
    )


def task_result_to_dict(res: Optional[TaskResult]) -> Optional[Dict[str, Any]]:
    """Serialize a TaskResult value object to a dictionary."""
    if res is None:
        return None
    return {
        "output": res.output,
        "references": [data_reference_to_dict(r) for r in res.references],
        "artifacts": [artifact_reference_to_dict(a) for a in res.artifacts],
        "metadata": dict(res.metadata),
    }


def dict_to_task_result(d: Optional[Dict[str, Any]]) -> Optional[TaskResult]:
    """Deserialize a dictionary to a TaskResult value object."""
    if d is None:
        return None
    refs = [dict_to_data_reference(r) for r in d.get("references", [])]
    arts = [dict_to_artifact_reference(a) for a in d.get("artifacts", [])]
    return TaskResult(
        output=d.get("output"),
        references=refs,
        artifacts=arts,
        metadata=dict(d.get("metadata", {})),
    )


def task_error_to_dict(err: Optional[TaskError]) -> Optional[Dict[str, Any]]:
    """Serialize a TaskError value object to a dictionary."""
    if err is None:
        return None
    return {
        "message": err.message,
        "category": err.category.value,
        "error_code": err.error_code,
        "details": dict(err.details),
        "cause_exception_type": err.cause_exception_type,
    }


def dict_to_task_error(d: Optional[Dict[str, Any]]) -> Optional[TaskError]:
    """Deserialize a dictionary to a TaskError value object."""
    if d is None:
        return None
    category = TaskErrorCategory(d.get("category", TaskErrorCategory.EXECUTION.value))
    return TaskError(
        message=d.get("message", ""),
        category=category,
        error_code=d.get("error_code", "TASK_EXECUTION_ERROR"),
        details=dict(d.get("details", {})),
        cause_exception_type=d.get("cause_exception_type"),
    )


# ---------------------------------------------------------------------------
# Goal Aggregate Mapping
# ---------------------------------------------------------------------------


def goal_to_model(goal: Goal) -> GoalModel:
    """Map a Goal domain entity to a GoalModel ORM instance."""
    return GoalModel(
        goal_id=goal.goal_id,
        description=goal.description,
        status=goal.status.value,
        context=dict(goal.context),
        active_plan_id=goal.active_plan_id,
        created_at=goal.created_at,
        completed_at=goal.completed_at,
    )


def model_to_goal(model: GoalModel) -> Goal:
    """Map a GoalModel ORM instance to a Goal domain entity."""
    return Goal(
        goal_id=model.goal_id,
        description=model.description,
        status=GoalStatus(model.status),
        context=dict(model.context),
        active_plan_id=model.active_plan_id,
        created_at=model.created_at,
        completed_at=model.completed_at,
    )


# ---------------------------------------------------------------------------
# Attempt & Task Mapping
# ---------------------------------------------------------------------------


def attempt_to_model(attempt: Attempt) -> AttemptModel:
    """Map an Attempt domain entity to an AttemptModel ORM instance."""
    return AttemptModel(
        attempt_id=attempt.attempt_id,
        task_id=attempt.task_id,
        attempt_number=attempt.attempt_number,
        status=attempt.status.value,
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        result=task_result_to_dict(attempt.result),
        error=task_error_to_dict(attempt.error),
        metadata_json=dict(attempt.metadata),
    )


def model_to_attempt(model: AttemptModel) -> Attempt:
    """Map an AttemptModel ORM instance to an Attempt domain entity."""
    return Attempt(
        attempt_id=model.attempt_id,
        task_id=model.task_id,
        attempt_number=model.attempt_number,
        started_at=model.started_at,
        completed_at=model.completed_at,
        status=AttemptStatus(model.status),
        result=dict_to_task_result(model.result),
        error=dict_to_task_error(model.error),
        metadata=dict(model.metadata_json),
    )


def task_to_model(task: Task) -> TaskModel:
    """Map a Task domain entity to a TaskModel ORM instance."""
    serialized_inputs = {
        logical_name: data_reference_to_dict(ref)
        for logical_name, ref in task.input_references.items()
    }
    model = TaskModel(
        task_id=task.task_id,
        plan_id=task.plan_id,
        title=task.title,
        capability_id=task.capability_id,
        description=task.description,
        parameters=dict(task.parameters),
        input_references=serialized_inputs,
        status=task.status.value,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        result=task_result_to_dict(task.result),
        error=task_error_to_dict(task.error),
    )
    # Populate attempts
    model.attempts = [attempt_to_model(a) for a in task.attempts]
    return model


def model_to_task(
    model: TaskModel, dependencies: Optional[List[Dependency]] = None
) -> Task:
    """Map a TaskModel ORM instance to a Task domain entity."""
    deserialized_inputs = {
        name: dict_to_data_reference(raw_ref)
        for name, raw_ref in model.input_references.items()
    }
    if dependencies is None:
        dependencies = [
            Dependency(
                upstream_task_id=dep_model.upstream_task_id,
                downstream_task_id=dep_model.downstream_task_id,
            )
            for dep_model in getattr(model, "dependencies", [])
        ]
    task = Task(
        task_id=model.task_id,
        plan_id=model.plan_id,
        title=model.title,
        capability_id=model.capability_id,
        description=model.description,
        parameters=dict(model.parameters),
        input_references=deserialized_inputs,
        dependencies=dependencies,
        status=TaskStatus(model.status),
        created_at=model.created_at,
        started_at=model.started_at,
        completed_at=model.completed_at,
        result=dict_to_task_result(model.result),
        error=dict_to_task_error(model.error),
    )
    # Reconstruct attempts in order
    sorted_attempts = sorted(model.attempts, key=lambda a: a.attempt_number)
    task.attempts = [model_to_attempt(a) for a in sorted_attempts]
    return task


# ---------------------------------------------------------------------------
# Plan Aggregate Mapping (Including Snapshot Payload Generation)
# ---------------------------------------------------------------------------


def build_revision_snapshot_payload(
    plan: Plan, task_ids: List[str]
) -> Dict[str, Any]:
    """Generate an immutable task specification snapshot for a PlanRevision.

    Captures the declarative specification of tasks (title, capability, parameters,
    input_references, dependencies) as they exist at the time of revision recording.
    """
    tasks_snapshot: Dict[str, Any] = {}
    dependencies_snapshot: List[Dict[str, str]] = []

    for tid in task_ids:
        task = plan.tasks.get(tid)
        if task is None:
            continue
        tasks_snapshot[tid] = {
            "task_id": task.task_id,
            "title": task.title,
            "capability_id": task.capability_id,
            "description": task.description,
            "parameters": dict(task.parameters),
            "input_references": {
                name: data_reference_to_dict(ref)
                for name, ref in task.input_references.items()
            },
        }
        for dep in task.dependencies:
            dependencies_snapshot.append({
                "upstream_task_id": dep.upstream_task_id,
                "downstream_task_id": dep.downstream_task_id,
            })

    return {
        "tasks": tasks_snapshot,
        "dependencies": dependencies_snapshot,
    }


def plan_to_model(plan: Plan) -> PlanModel:
    """Map a Plan domain aggregate to a PlanModel ORM instance.

    Synchronizes the plan, tasks, dependencies, attempts, and revisions.
    Generates snapshot_payload for any revisions.
    """
    plan_model = PlanModel(
        plan_id=plan.plan_id,
        goal_id=plan.goal_id,
        title=plan.title,
        status=plan.status.value,
        created_at=plan.created_at,
        completed_at=plan.completed_at,
    )

    # 1. Map Tasks and Attempts
    task_models_by_id: Dict[str, TaskModel] = {}
    for task in plan.tasks.values():
        t_model = task_to_model(task)
        plan_model.tasks.append(t_model)
        task_models_by_id[task.task_id] = t_model

    # 2. Map Dependencies
    for task in plan.tasks.values():
        t_model = task_models_by_id[task.task_id]
        for dep in task.dependencies:
            dep_model = DependencyModel(
                upstream_task_id=dep.upstream_task_id,
                downstream_task_id=dep.downstream_task_id,
            )
            t_model.dependencies.append(dep_model)

    # 3. Map PlanRevisions with snapshot_payload
    for rev in plan.revisions:
        snapshot_payload = build_revision_snapshot_payload(plan, rev.task_ids)
        rev_model = PlanRevisionModel(
            revision_id=rev.revision_id,
            plan_id=rev.plan_id,
            revision_number=rev.revision_number,
            reason=rev.reason,
            task_ids=list(rev.task_ids),
            snapshot_payload=snapshot_payload,
            created_at=rev.created_at,
        )
        plan_model.revisions.append(rev_model)

    return plan_model


def model_to_plan(model: PlanModel) -> Plan:
    """Map a PlanModel ORM instance to a complete Plan domain aggregate."""
    plan = Plan(
        plan_id=model.plan_id,
        goal_id=model.goal_id,
        title=model.title,
        status=PlanStatus(model.status),
        created_at=model.created_at,
        completed_at=model.completed_at,
    )

    # 1. Reconstruct Tasks, Dependencies, and Attempts
    for t_model in model.tasks:
        task = model_to_task(t_model)
        plan.tasks[task.task_id] = task

    # 2. Reconstruct PlanRevisions as pure domain entities (without snapshot_payload)
    sorted_revisions = sorted(model.revisions, key=lambda r: r.revision_number)
    plan.revisions = [
        PlanRevision(
            revision_id=rev_model.revision_id,
            plan_id=rev_model.plan_id,
            revision_number=rev_model.revision_number,
            reason=rev_model.reason,
            task_ids=list(rev_model.task_ids),
            created_at=rev_model.created_at,
        )
        for rev_model in sorted_revisions
    ]

    return plan


def reconstruct_historical_plan(
    model: PlanModel, revision_number: int
) -> Optional[Plan]:
    """Reconstruct a historical Plan specification from a PlanRevision's snapshot_payload.

    Reproduces the exact planned task specifications and dependencies at the time
    that revision was recorded, independent of subsequent execution state mutations.
    """
    rev_model = next(
        (r for r in model.revisions if r.revision_number == revision_number), None
    )
    if rev_model is None:
        return None

    snapshot = rev_model.snapshot_payload or {}
    raw_tasks = snapshot.get("tasks", {})
    raw_deps = snapshot.get("dependencies", [])

    historical_plan = Plan(
        plan_id=model.plan_id,
        goal_id=model.goal_id,
        title=model.title,
        status=PlanStatus.DRAFT,
        created_at=rev_model.created_at,
    )

    # Pre-index dependencies by downstream task ID
    deps_by_downstream: Dict[str, List[Dependency]] = {}
    for raw_dep in raw_deps:
        d = Dependency(
            upstream_task_id=raw_dep["upstream_task_id"],
            downstream_task_id=raw_dep["downstream_task_id"],
        )
        deps_by_downstream.setdefault(raw_dep["downstream_task_id"], []).append(d)

    # Reconstitute planned tasks from snapshot
    for tid, raw_task in raw_tasks.items():
        inputs = {
            name: dict_to_data_reference(raw_ref)
            for name, raw_ref in raw_task.get("input_references", {}).items()
        }
        task = Task(
            task_id=raw_task["task_id"],
            plan_id=model.plan_id,
            title=raw_task.get("title", ""),
            capability_id=raw_task.get("capability_id", ""),
            description=raw_task.get("description", ""),
            parameters=dict(raw_task.get("parameters", {})),
            input_references=inputs,
            dependencies=deps_by_downstream.get(tid, []),
            status=TaskStatus.PENDING,
            created_at=rev_model.created_at,
        )
        historical_plan.tasks[tid] = task

    return historical_plan
