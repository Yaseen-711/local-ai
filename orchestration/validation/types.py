"""Types and data structures for 4-stage deterministic plan validation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationStage(str, Enum):
    """The four deterministic plan validation stages."""
    STRUCTURAL = "structural"
    CAPABILITY = "capability"
    HARD_CONSTRAINTS = "hard_constraints"
    FEASIBILITY = "feasibility"


@dataclass(frozen=True)
class ValidationError:
    """Error encountered during plan validation."""
    stage: ValidationStage
    code: str
    message: str
    task_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating a candidate or domain plan."""
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(
        self,
        stage: ValidationStage,
        code: str,
        message: str,
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an error and invalidate the result."""
        self.is_valid = False
        self.errors.append(
            ValidationError(
                stage=stage,
                code=code,
                message=message,
                task_id=task_id,
                details=details or {},
            )
        )

    def add_warning(self, warning: str) -> None:
        """Record a non-fatal warning."""
        self.warnings.append(warning)
