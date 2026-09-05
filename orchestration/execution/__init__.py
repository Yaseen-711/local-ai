"""Execution layer for Local AI Foundation orchestration.

Provides in-process execution runners that drive Plans to completion.
"""

from orchestration.execution.base import (
    ExecutionHandle,
    PlanExecutionResult,
    PlanExecutionSnapshot,
    PlanRunner,
)
from orchestration.execution.runner import InProcessPlanRunner

__all__ = [
    "ExecutionHandle",
    "PlanExecutionResult",
    "PlanExecutionSnapshot",
    "PlanRunner",
    "InProcessPlanRunner",
]
