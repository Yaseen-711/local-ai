"""Shared workflow data contracts for Local AI Foundation.

Provides a common result envelope for workflow execution. Workflows may return
WorkflowResult directly, or return domain-specific types — this is a convention,
not a requirement.
"""

from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, TypeVar, Any

T = TypeVar("T")


@dataclass
class WorkflowResult(Generic[T]):
    """Common result envelope for workflow execution.

    Type Parameters:
        T: The type of the primary output (text, dict, list, structured data, etc.).

    Workflows may return this directly, or return domain-specific types.
    This is a convention, not a requirement.

    Attributes:
        output: Primary workflow output. Type is determined by the workflow.
        model_id: Model that produced the output, if applicable. Optional because
            not every workflow step involves inference (e.g. formatting, validation).
        metadata: Arbitrary workflow-specific metadata for diagnostics or downstream use.
        errors: Non-fatal warnings or issues encountered during execution. Fatal
            errors should raise exceptions (WorkflowError or infrastructure errors)
            rather than appearing in this list.
    """
    output: T
    model_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
