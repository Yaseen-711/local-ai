"""SQLAlchemy 2.0 Declarative ORM models for orchestration persistence.

Tables:
- orchestration_goals: Goal entity records
- orchestration_plans: Plan entity records
- orchestration_plan_revisions: PlanRevision immutable snapshot records with snapshot_payload
- orchestration_tasks: Task entity records
- orchestration_dependencies: Task DAG dependency directed edges
- orchestration_attempts: Individual task execution run attempts
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

# Cross-dialect JSON column: uses JSONB in PostgreSQL, standard JSON in SQLite
JSON_VARIANT = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    """Base class for orchestration SQLAlchemy models."""
    pass


class GoalModel(Base):
    """Relational model for the Goal aggregate root."""

    __tablename__ = "orchestration_goals"

    goal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    context: Mapped[Dict[str, Any]] = mapped_column(
        JSON_VARIANT, nullable=False, default=dict
    )
    active_plan_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    plans: Mapped[List["PlanModel"]] = relationship(
        "PlanModel", back_populates="goal", cascade="all, delete-orphan"
    )


class PlanModel(Base):
    """Relational model for the Plan aggregate root."""

    __tablename__ = "orchestration_plans"

    plan_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    goal_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("orchestration_goals.goal_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    goal: Mapped["GoalModel"] = relationship("GoalModel", back_populates="plans")
    tasks: Mapped[List["TaskModel"]] = relationship(
        "TaskModel",
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="TaskModel.created_at",
    )
    revisions: Mapped[List["PlanRevisionModel"]] = relationship(
        "PlanRevisionModel",
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="PlanRevisionModel.revision_number",
    )


class PlanRevisionModel(Base):
    """Relational model for immutable PlanRevision history with snapshot payload."""

    __tablename__ = "orchestration_plan_revisions"

    revision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("orchestration_plans.plan_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    task_ids: Mapped[List[str]] = mapped_column(
        JSON_VARIANT, nullable=False, default=list
    )
    snapshot_payload: Mapped[Dict[str, Any]] = mapped_column(
        JSON_VARIANT, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "plan_id", "revision_number", name="uq_plan_revision_number"
        ),
    )

    plan: Mapped["PlanModel"] = relationship("PlanModel", back_populates="revisions")


class TaskModel(Base):
    """Relational model for discrete units of work within a plan."""

    __tablename__ = "orchestration_tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("orchestration_plans.plan_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    capability_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    parameters: Mapped[Dict[str, Any]] = mapped_column(
        JSON_VARIANT, nullable=False, default=dict
    )
    input_references: Mapped[Dict[str, Any]] = mapped_column(
        JSON_VARIANT, nullable=False, default=dict
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON_VARIANT, nullable=True
    )
    error: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON_VARIANT, nullable=True
    )

    plan: Mapped["PlanModel"] = relationship("PlanModel", back_populates="tasks")
    attempts: Mapped[List["AttemptModel"]] = relationship(
        "AttemptModel",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="AttemptModel.attempt_number",
    )
    dependencies: Mapped[List["DependencyModel"]] = relationship(
        "DependencyModel",
        foreign_keys="[DependencyModel.downstream_task_id]",
        back_populates="downstream_task",
        cascade="all, delete-orphan",
    )


class DependencyModel(Base):
    """Relational model for directed prerequisite edges between tasks in a plan."""

    __tablename__ = "orchestration_dependencies"

    upstream_task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("orchestration_tasks.task_id", ondelete="CASCADE"),
        primary_key=True,
    )
    downstream_task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("orchestration_tasks.task_id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    downstream_task: Mapped["TaskModel"] = relationship(
        "TaskModel",
        foreign_keys=[downstream_task_id],
        back_populates="dependencies",
    )


class AttemptModel(Base):
    """Relational model for individual task execution run records."""

    __tablename__ = "orchestration_attempts"

    attempt_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("orchestration_tasks.task_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON_VARIANT, nullable=True
    )
    error: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON_VARIANT, nullable=True
    )
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON_VARIANT, nullable=False, default=dict
    )

    __table_args__ = (
        UniqueConstraint("task_id", "attempt_number", name="uq_task_attempt_number"),
    )

    task: Mapped["TaskModel"] = relationship("TaskModel", back_populates="attempts")
