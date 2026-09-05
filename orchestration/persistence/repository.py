"""SQLAlchemy Implementation of Orchestration Repositories.

Provides atomic aggregate root persistence for Goal and Plan aggregates,
preserving immutable PlanRevision history, Task identity across revisions,
and historical Attempt records.
"""

from contextlib import contextmanager
from typing import Generator, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.persistence.base import (
    GoalRepository,
    OrchestrationRepository,
    PlanRepository,
)
from orchestration.persistence.mappers import (
    attempt_to_model,
    build_revision_snapshot_payload,
    data_reference_to_dict,
    goal_to_model,
    model_to_goal,
    model_to_plan,
    plan_to_model,
    reconstruct_historical_plan,
    task_error_to_dict,
    task_result_to_dict,
    task_to_model,
)
from orchestration.persistence.models import (
    DependencyModel,
    GoalModel,
    PlanModel,
    PlanRevisionModel,
    TaskModel,
)


class PostgresGoalRepository(GoalRepository):
    """SQLAlchemy implementation of GoalRepository."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, goal: Goal) -> None:
        """Atomically persist or update a Goal entity."""
        existing = self._session.get(GoalModel, goal.goal_id)
        if existing is None:
            model = goal_to_model(goal)
            self._session.add(model)
        else:
            existing.description = goal.description
            existing.status = goal.status.value
            existing.context = dict(goal.context)
            existing.active_plan_id = goal.active_plan_id
            existing.completed_at = goal.completed_at

    def get(self, goal_id: str) -> Optional[Goal]:
        """Fetch a Goal by its unique identifier."""
        model = self._session.get(GoalModel, goal_id)
        if model is None:
            return None
        return model_to_goal(model)

    def list_goals(self, limit: int = 100, offset: int = 0) -> List[Goal]:
        """List goals ordered by creation time descending."""
        stmt = (
            select(GoalModel)
            .order_by(GoalModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        models = self._session.scalars(stmt).all()
        return [model_to_goal(m) for m in models]


class PostgresPlanRepository(PlanRepository):
    """SQLAlchemy implementation of PlanRepository.

    Treats Plan as an Aggregate Root, synchronizing tasks, dependencies,
    attempts, and revisions atomically.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, plan: Plan) -> None:
        """Atomically persist or update an entire Plan aggregate.

        Preserves existing task identities, prior attempts, and records
        new immutable plan revisions.
        """
        stmt = (
            select(PlanModel)
            .where(PlanModel.plan_id == plan.plan_id)
            .options(
                selectinload(PlanModel.tasks).selectinload(TaskModel.attempts),
                selectinload(PlanModel.tasks).selectinload(TaskModel.dependencies),
                selectinload(PlanModel.revisions),
            )
        )
        existing_plan = self._session.scalars(stmt).first()

        if existing_plan is None:
            plan_model = plan_to_model(plan)
            self._session.add(plan_model)
            return

        # 1. Update scalar plan attributes
        existing_plan.title = plan.title
        existing_plan.status = plan.status.value
        existing_plan.completed_at = plan.completed_at

        # 2. Synchronize Tasks and Attempts
        existing_tasks_by_id = {t.task_id: t for t in existing_plan.tasks}

        for task in plan.tasks.values():
            if task.task_id in existing_tasks_by_id:
                t_model = existing_tasks_by_id[task.task_id]
                t_model.status = task.status.value
                t_model.started_at = task.started_at
                t_model.completed_at = task.completed_at
                t_model.result = task_result_to_dict(task.result)
                t_model.error = task_error_to_dict(task.error)
                t_model.parameters = dict(task.parameters)
                t_model.description = task.description
                t_model.input_references = {
                    name: data_reference_to_dict(ref)
                    for name, ref in task.input_references.items()
                }

                # Synchronize attempts without deleting prior attempts
                existing_att_by_id = {a.attempt_id: a for a in t_model.attempts}
                for attempt in task.attempts:
                    if attempt.attempt_id in existing_att_by_id:
                        att_model = existing_att_by_id[attempt.attempt_id]
                        att_model.status = attempt.status.value
                        att_model.completed_at = attempt.completed_at
                        att_model.result = task_result_to_dict(attempt.result)
                        att_model.error = task_error_to_dict(attempt.error)
                        att_model.metadata_json = dict(attempt.metadata)
                    else:
                        new_att = attempt_to_model(attempt)
                        t_model.attempts.append(new_att)
            else:
                new_t_model = task_to_model(task)
                existing_plan.tasks.append(new_t_model)
                existing_tasks_by_id[task.task_id] = new_t_model

        # 3. Synchronize Dependencies
        for task in plan.tasks.values():
            t_model = existing_tasks_by_id[task.task_id]
            # Replace dependencies on task
            current_dep_keys = {
                (d.upstream_task_id, d.downstream_task_id) for d in task.dependencies
            }
            existing_dep_keys = {
                (d.upstream_task_id, d.downstream_task_id) for d in t_model.dependencies
            }

            if current_dep_keys != existing_dep_keys:
                t_model.dependencies.clear()
                for dep in task.dependencies:
                    t_model.dependencies.append(
                        DependencyModel(
                            upstream_task_id=dep.upstream_task_id,
                            downstream_task_id=dep.downstream_task_id,
                        )
                    )

        # 4. Synchronize PlanRevisions (Append-only snapshots)
        existing_rev_nums = {r.revision_number for r in existing_plan.revisions}
        for rev in plan.revisions:
            if rev.revision_number not in existing_rev_nums:
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
                existing_plan.revisions.append(rev_model)
                existing_rev_nums.add(rev.revision_number)

    def get(self, plan_id: str) -> Optional[Plan]:
        """Load a complete Plan aggregate including tasks, DAG edges, attempts, and revisions."""
        stmt = (
            select(PlanModel)
            .where(PlanModel.plan_id == plan_id)
            .options(
                selectinload(PlanModel.tasks).selectinload(TaskModel.attempts),
                selectinload(PlanModel.tasks).selectinload(TaskModel.dependencies),
                selectinload(PlanModel.revisions),
            )
        )
        model = self._session.scalars(stmt).first()
        if model is None:
            return None
        return model_to_plan(model)

    def list_for_goal(self, goal_id: str) -> List[Plan]:
        """List all plans serving a goal."""
        stmt = (
            select(PlanModel)
            .where(PlanModel.goal_id == goal_id)
            .order_by(PlanModel.created_at.asc())
            .options(
                selectinload(PlanModel.tasks).selectinload(TaskModel.attempts),
                selectinload(PlanModel.tasks).selectinload(TaskModel.dependencies),
                selectinload(PlanModel.revisions),
            )
        )
        models = self._session.scalars(stmt).all()
        return [model_to_plan(m) for m in models]

    def get_historical_plan_revision(
        self, plan_id: str, revision_number: int
    ) -> Optional[Plan]:
        """Reconstitute a historical Plan revision from its immutable snapshot payload."""
        stmt = (
            select(PlanModel)
            .where(PlanModel.plan_id == plan_id)
            .options(selectinload(PlanModel.revisions))
        )
        model = self._session.scalars(stmt).first()
        if model is None:
            return None
        return reconstruct_historical_plan(model, revision_number)


class PostgresOrchestrationRepository(OrchestrationRepository):
    """Unified Orchestration Repository managing transactional sessions."""

    def __init__(
        self, session_or_factory: sessionmaker[Session] | Session
    ) -> None:
        if isinstance(session_or_factory, sessionmaker):
            self._session_factory: Optional[sessionmaker[Session]] = session_or_factory
            self._session: Session = session_or_factory()
            self._owns_session = True
        else:
            self._session_factory = None
            self._session = session_or_factory
            self._owns_session = False

        self._goals = PostgresGoalRepository(self._session)
        self._plans = PostgresPlanRepository(self._session)

    @property
    def session(self) -> Session:
        """Direct access to underlying SQLAlchemy Session."""
        return self._session

    @property
    def goals(self) -> GoalRepository:
        """Goal repository."""
        return self._goals

    @property
    def plans(self) -> PlanRepository:
        """Plan repository."""
        return self._plans

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Provide an atomic transaction boundary with automatic commit/rollback."""
        if self._session.in_transaction():
            # Nested savepoint or existing transaction
            with self._session.begin_nested():
                yield
        else:
            with self._session.begin():
                yield

    def close(self) -> None:
        """Close underlying session if owned."""
        if self._owns_session:
            self._session.close()
