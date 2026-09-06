"""Data contracts and schemas for Code -> Test -> Repair -> Retest workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CodeTaskCategory(str, Enum):
    """Category of coding task governing verification strategy."""
    GENERAL_CODE = "general_code"
    ENGINEERING_CALCULATION = "engineering_calculation"


@dataclass(frozen=True)
class EngineeringTolerance:
    """Tolerances and unit constraints for engineering verification."""
    abs_tol: Optional[float] = None
    rel_tol: Optional[float] = 1e-4
    unit: Optional[str] = None


@dataclass(frozen=True)
class EngineeringAssertion:
    """Machine-checkable specification for an engineering calculation target."""
    name: str
    expected_value: float
    tolerance: EngineeringTolerance = field(default_factory=EngineeringTolerance)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: Optional[str] = None


@dataclass
class ExecutionAttemptRecord:
    """Comprehensive execution record of a single code verification attempt."""
    attempt_number: int
    code: str
    test_code: str
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    timed_out: bool = False
    error_summary: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class CodeTestRepairResult:
    """Outcome of a Code -> Test -> Repair -> Retest workflow execution."""
    status: str  # "success" or "failed"
    category: CodeTaskCategory
    final_code: str
    test_code: str
    attempts: List[ExecutionAttemptRecord] = field(default_factory=list)
    total_attempts: int = 0
    verification_output: str = ""
    terminal_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
