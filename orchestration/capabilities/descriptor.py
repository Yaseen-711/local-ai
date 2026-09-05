"""CapabilityDescriptor — declarative metadata, schemas, and availability for capabilities.

Decoupled from runtime execution and concrete implementation classes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Declarative specification and catalog metadata for a system capability.

    Attributes:
        capability_id: Unique identifier matching the Capability.capability_id.
        description: Human-readable description of what the capability does.
        parameter_schema: Expected configuration parameters and constraints.
        input_schema: Expected runtime input payload names and types.
        output_schema: Expected output keys and types in TaskResult.output.
        is_available: Advisory flag indicating whether the capability is available
            in the current runtime environment.
    """
    capability_id: str
    description: str
    parameter_schema: Dict[str, Any] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    is_available: bool = True
    is_deprecated: bool = False
    deprecation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
