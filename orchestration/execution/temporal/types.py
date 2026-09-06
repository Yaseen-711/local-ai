"""Data Transfer Objects (DTOs) for Temporal execution.

These serializable value objects are used strictly across Temporal workflow
and activity boundaries, keeping domain entities completely decoupled from
wire-serialization concerns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskActivityInput:
    """Input payload dispatched to a TaskExecutionActivity.

    Attributes:
        task_id: Identifier of the Task being executed.
        capability_id: Semantic capability to resolve in the worker registry.
        attempt_id: Unique identifier for this execution attempt.
        parameters: Static task parameters.
        inputs: Dynamically resolved inputs from upstream task outputs.
    """

    task_id: str
    capability_id: str
    attempt_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskActivityOutput:
    """Output payload returned from a TaskExecutionActivity.

    Attributes:
        task_id: Identifier of the executed Task.
        attempt_id: Attempt identifier for correlation.
        status: Execution status ("COMPLETED" or "FAILED").
        output: Raw output payload on success.
        references: Semantic data references returned on success.
        artifacts: Artifact reference metadata dictionaries.
        error_message: Error description on failure.
        error_category: String representation of TaskErrorCategory.
        error_code: Optional machine-readable error code.
        error_details: Additional debugging context on failure.
    """

    task_id: str
    attempt_id: str
    status: str
    output: Any = None
    references: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputReferenceDTO:
    """Serializable representation of a DataReference or ArtifactReference.

    Attributes:
        key: Key to extract from upstream output.
        source_task_id: Optional ID of upstream task supplying data.
        uri: Optional external URI for artifact references.
    """

    key: str = "output"
    source_task_id: Optional[str] = None
    uri: Optional[str] = None


@dataclass
class TaskDefinitionDTO:
    """Serializable task specification for workflow execution.

    Attributes:
        task_id: Unique task identifier.
        capability_id: Capability identifier string.
        attempt_id: Explicit domain attempt identifier for this execution.
        parameters: Task execution parameters.
        dependencies: List of upstream task IDs required before execution.
        input_references: Mapping of input parameter names to reference specs.
    """

    task_id: str
    capability_id: str
    attempt_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    input_references: Dict[str, InputReferenceDTO] = field(default_factory=dict)


@dataclass
class PlanWorkflowInput:
    """Input payload to start a PlanExecutionWorkflow.

    Attributes:
        plan_id: Unique plan identifier.
        goal_id: Identifier of the goal this plan serves.
        tasks: List of task definitions composing the plan's DAG.
    """

    plan_id: str
    goal_id: str
    tasks: List[TaskDefinitionDTO] = field(default_factory=list)


@dataclass
class PlanWorkflowOutput:
    """Terminal execution facts returned by PlanExecutionWorkflow.

    Attributes:
        plan_id: Identifier of the executed plan.
        status: Terminal status string ("COMPLETED", "FAILED", "CANCELLED").
        task_results: Map of task_id to successful output data.
        task_errors: Map of task_id to error details dictionary.
        task_statuses: Map of task_id to final status string.
        task_attempts: Map of task_id to explicit attempt identifier string.
    """

    plan_id: str
    status: str
    task_results: Dict[str, Any] = field(default_factory=dict)
    task_errors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    task_statuses: Dict[str, str] = field(default_factory=dict)
    task_attempts: Dict[str, str] = field(default_factory=dict)
    task_references: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    task_artifacts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
