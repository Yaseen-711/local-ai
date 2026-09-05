"""Initial orchestration persistence schema.

Revision ID: 001_initial_orchestration_schema
Revises: None
Create Date: 2026-09-05 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_orchestration_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    # 1. orchestration_goals
    op.create_table(
        "orchestration_goals",
        sa.Column("goal_id", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("context", json_type, nullable=False, server_default="{}"),
        sa.Column("active_plan_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("goal_id"),
    )
    op.create_index(
        "ix_orchestration_goals_status",
        "orchestration_goals",
        ["status"],
        unique=False,
    )

    # 2. orchestration_plans
    op.create_table(
        "orchestration_plans",
        sa.Column("plan_id", sa.String(length=64), nullable=False),
        sa.Column("goal_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["goal_id"],
            ["orchestration_goals.goal_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("plan_id"),
    )
    op.create_index(
        "ix_orchestration_plans_goal_id",
        "orchestration_plans",
        ["goal_id"],
        unique=False,
    )
    op.create_index(
        "ix_orchestration_plans_status",
        "orchestration_plans",
        ["status"],
        unique=False,
    )

    # 3. orchestration_plan_revisions
    op.create_table(
        "orchestration_plan_revisions",
        sa.Column("revision_id", sa.String(length=64), nullable=False),
        sa.Column("plan_id", sa.String(length=64), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("task_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("snapshot_payload", json_type, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["orchestration_plans.plan_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("revision_id"),
        sa.UniqueConstraint(
            "plan_id", "revision_number", name="uq_plan_revision_number"
        ),
    )
    op.create_index(
        "ix_orchestration_plan_revisions_plan_id",
        "orchestration_plan_revisions",
        ["plan_id"],
        unique=False,
    )

    # 4. orchestration_tasks
    op.create_table(
        "orchestration_tasks",
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("plan_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("capability_id", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("parameters", json_type, nullable=False, server_default="{}"),
        sa.Column("input_references", json_type, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", json_type, nullable=True),
        sa.Column("error", json_type, nullable=True),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["orchestration_plans.plan_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("task_id"),
    )
    op.create_index(
        "ix_orchestration_tasks_plan_id",
        "orchestration_tasks",
        ["plan_id"],
        unique=False,
    )
    op.create_index(
        "ix_orchestration_tasks_capability_id",
        "orchestration_tasks",
        ["capability_id"],
        unique=False,
    )
    op.create_index(
        "ix_orchestration_tasks_status",
        "orchestration_tasks",
        ["status"],
        unique=False,
    )

    # 5. orchestration_dependencies
    op.create_table(
        "orchestration_dependencies",
        sa.Column("upstream_task_id", sa.String(length=64), nullable=False),
        sa.Column("downstream_task_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["upstream_task_id"],
            ["orchestration_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["downstream_task_id"],
            ["orchestration_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("upstream_task_id", "downstream_task_id"),
    )
    op.create_index(
        "ix_orchestration_dependencies_downstream",
        "orchestration_dependencies",
        ["downstream_task_id"],
        unique=False,
    )

    # 6. orchestration_attempts
    op.create_table(
        "orchestration_attempts",
        sa.Column("attempt_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", json_type, nullable=True),
        sa.Column("error", json_type, nullable=True),
        sa.Column("metadata_json", json_type, nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["orchestration_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("attempt_id"),
        sa.UniqueConstraint(
            "task_id", "attempt_number", name="uq_task_attempt_number"
        ),
    )
    op.create_index(
        "ix_orchestration_attempts_task_id",
        "orchestration_attempts",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        "ix_orchestration_attempts_status",
        "orchestration_attempts",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("orchestration_attempts")
    op.drop_table("orchestration_dependencies")
    op.drop_table("orchestration_tasks")
    op.drop_table("orchestration_plan_revisions")
    op.drop_table("orchestration_plans")
    op.drop_table("orchestration_goals")
