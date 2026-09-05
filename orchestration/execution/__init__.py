"""Execution layer for Local AI Foundation orchestration.

Provides in-process execution runners that drive Plans to completion.
"""

from orchestration.execution.runner import InProcessPlanRunner

__all__ = [
    "InProcessPlanRunner",
]
