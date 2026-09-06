"""Schemas for goal lifecycle and DAG orchestration endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from apps.api.schemas.common import ArtifactReferenceSchema, DataReferenceSchema


class CreateGoalRequest(BaseModel):
    """Input payload to create a new industrial Goal."""
    description: Optional[str] = Field(default=None, description="Goal objective statement or description")
    title: Optional[str] = Field(default=None, description="Optional concise title")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input files or data payloads (e.g. file_id)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters or policy flags")

    @model_validator(mode="after")
    def check_at_least_one(self) -> "CreateGoalRequest":
        if not self.description and not self.title:
            raise ValueError("Either 'description' or 'title' must be provided for the goal.")
        return self


class TaskSchema(BaseModel):
    """Representation of an individual task in a plan."""
    task_id: str
    capability_id: str
    title: str
    status: str
    dependencies: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Any] = None
    error: Optional[str] = None


class GoalResponse(BaseModel):
    """Public representation of a Goal entity."""
    goal_id: str
    description: str
    title: Optional[str] = None
    status: str
    active_plan_id: Optional[str] = None
    created_at: str
    context: Dict[str, Any] = Field(default_factory=dict)
    links: Dict[str, str] = Field(default_factory=dict)


class CandidatePlanResponse(BaseModel):
    """Decision and candidate DAG plan response."""
    goal_id: str
    plan_id: Optional[str] = None
    strategy: str
    route: Optional[str] = None
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
    tasks: List[TaskSchema] = Field(default_factory=list)


class GoalExecutionResponse(BaseModel):
    """Response returned when a goal execution is accepted and started."""
    goal_id: str
    plan_id: Optional[str] = None
    status: str
    events_url: str


class GoalDetailResponse(BaseModel):
    """Detailed view of a Goal, its active plan, and executed task outputs."""
    goal: GoalResponse
    plan: Optional[CandidatePlanResponse] = None
    tasks: List[TaskSchema] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[ArtifactReferenceSchema] = Field(default_factory=list)


class CancelGoalResponse(BaseModel):
    """Response returned when a goal cancellation is requested."""
    goal_id: str
    status: str
    message: str = "Goal cancellation requested."
