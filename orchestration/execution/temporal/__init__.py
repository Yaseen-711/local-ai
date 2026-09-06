"""Temporal execution package for Local AI Foundation orchestration.

Provides distributed durable execution for Plans via Temporal:
- PlanExecutionWorkflow: Mechanical DAG scheduling workflow.
- TaskExecutionActivity: Activity worker adapter for CapabilityRegistry.
- TemporalPlanRunner: Concrete PlanRunner driving plans via Temporal.
"""

from orchestration.execution.temporal.activities import TaskExecutionActivity
from orchestration.execution.temporal.runner import TemporalPlanRunner
from orchestration.execution.temporal.workflows import PlanExecutionWorkflow

__all__ = [
    "PlanExecutionWorkflow",
    "TaskExecutionActivity",
    "TemporalPlanRunner",
]
