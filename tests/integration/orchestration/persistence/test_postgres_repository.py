"""Live PostgreSQL integration test for orchestration persistence.

Skipped automatically if a live PostgreSQL instance is not reachable at the
configured database URL.
"""

import os
import socket
import pytest
from datetime import datetime, timezone

from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.tasks import Task
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.persistence.engine import (
    create_db_engine,
    create_session_factory,
)
from orchestration.persistence.models import Base, GoalModel, PlanModel
from orchestration.persistence.repository import (
    PostgresOrchestrationRepository,
)


def is_postgres_available(host: str = "127.0.0.1", port: int = 5432) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


POSTGRES_AVAILABLE = is_postgres_available()
POSTGRES_URL = os.getenv(
    "LOCAL_AI_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/local_ai",
)


@pytest.mark.integration
@pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="PostgreSQL instance not reachable on 127.0.0.1:5432",
)
def test_live_postgres_persistence_roundtrip():
    engine = create_db_engine(POSTGRES_URL)
    Base.metadata.create_all(bind=engine)
    session_factory = create_session_factory(engine)
    repo = PostgresOrchestrationRepository(session_or_factory=session_factory)

    try:
        now = datetime.now(timezone.utc)
        goal = Goal(
            goal_id="g-live-pg",
            description="Live PostgreSQL Goal",
            status=GoalStatus.ACTIVE,
            context={"env": "postgres"},
            created_at=now,
        )
        plan = Plan(
            plan_id="p-live-pg",
            goal_id="g-live-pg",
            title="Live PostgreSQL Plan",
            status=PlanStatus.ACTIVE,
            created_at=now,
        )
        task = Task(
            task_id="t-live-1",
            plan_id="p-live-pg",
            title="Postgres Task",
            capability_id="test.echo",
            status=TaskStatus.READY,
        )
        plan.add_task(task)

        with repo.transaction():
            repo.goals.save(goal)
            repo.plans.save(plan)

        loaded_goal = repo.goals.get("g-live-pg")
        assert loaded_goal is not None
        assert loaded_goal.goal_id == "g-live-pg"
        assert loaded_goal.context == {"env": "postgres"}

        loaded_plan = repo.plans.get("p-live-pg")
        assert loaded_plan is not None
        assert loaded_plan.plan_id == "p-live-pg"
        assert "t-live-1" in loaded_plan.tasks

    finally:
        # Cleanup
        repo.close()
        with session_factory() as cleanup_session:
            p = cleanup_session.get(PlanModel, "p-live-pg")
            g = cleanup_session.get(GoalModel, "g-live-pg")
            if p is not None:
                cleanup_session.delete(p)
            if g is not None:
                cleanup_session.delete(g)
            cleanup_session.commit()
