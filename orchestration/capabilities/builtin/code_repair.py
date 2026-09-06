"""Built-in Code Verification and Repair capability.

Exposes the Code -> Test -> Repair -> Retest workflow as an orchestratable capability
with declarative semantics, bounded execution, and full provenance tracking.
"""

from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any, Dict, List, Optional

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.domain.results import TaskResult
from workflows.code_repair.types import (
    CodeTaskCategory,
    CodeTestRepairResult,
    EngineeringAssertion,
    EngineeringTolerance,
)
from workflows.code_repair.workflow import CodeTestRepairWorkflow

logger = logging.getLogger(__name__)


class CodeVerificationRepairCapability:
    """Capability orchestrating Code -> Test -> Repair -> Retest cycles inside code.workspace."""

    def __init__(self, workflow: Optional[CodeTestRepairWorkflow] = None) -> None:
        self._workflow = workflow or CodeTestRepairWorkflow()

    @property
    def capability_id(self) -> str:
        return "code.verify_and_repair"

    def get_descriptor(self) -> CapabilityDescriptor:
        """Declarative catalog descriptor for this capability."""
        return CapabilityDescriptor(
            capability_id=self.capability_id,
            description="Generate, sandbox-execute, test, and repair code with bounded retries and objective verification.",
            parameter_schema={
                "category": {"type": "string", "enum": ["general_code", "engineering_calculation"], "default": "general_code"},
                "max_repair_attempts": {"type": "integer", "default": 3},
                "timeout_seconds": {"type": "number", "default": 30.0},
                "test_command": {"type": "string"},
                "entrypoint": {"type": "string"},
                "call_kwargs": {"type": "object"},
            },
            input_schema={
                "prompt": {"type": "string", "required": True},
                "assertions": {"type": "array"},
            },
            output_schema={
                "status": {"type": "string"},
                "final_code": {"type": "string"},
                "test_code": {"type": "string"},
                "total_attempts": {"type": "integer"},
                "attempts": {"type": "array"},
                "verification_output": {"type": "string"},
            },
        )

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        """Execute Code -> Test -> Repair -> Retest cycle adhering to Capability protocol."""
        prompt = str(
            inputs.get("prompt")
            or inputs.get("specification")
            or inputs.get("text")
            or inputs.get("description")
            or parameters.get("prompt")
            or parameters.get("specification")
            or parameters.get("description")
            or (next(iter(inputs.values())) if inputs else "")
            or ""
        ).strip()

        if not prompt:
            raise ValueError(f"Capability '{self.capability_id}' requires a non-empty 'prompt' in inputs or parameters.")

        cat_str = str(parameters.get("category") or inputs.get("category") or "general_code").lower()
        try:
            category = CodeTaskCategory(cat_str)
        except ValueError:
            category = CodeTaskCategory.GENERAL_CODE

        # Parse assertions if provided
        raw_assertions = inputs.get("assertions") or parameters.get("assertions") or []
        assertions: List[EngineeringAssertion] = []
        for a in raw_assertions:
            if isinstance(a, EngineeringAssertion):
                assertions.append(a)
            elif isinstance(a, dict):
                tol_dict = a.get("tolerance", {})
                tol = EngineeringTolerance(
                    abs_tol=tol_dict.get("abs_tol"),
                    rel_tol=tol_dict.get("rel_tol", 1e-4),
                    unit=tol_dict.get("unit"),
                )
                assertions.append(
                    EngineeringAssertion(
                        name=str(a.get("name", "val")),
                        expected_value=float(a.get("expected_value", 0.0)),
                        tolerance=tol,
                        min_value=float(a["min_value"]) if "min_value" in a and a["min_value"] is not None else None,
                        max_value=float(a["max_value"]) if "max_value" in a and a["max_value"] is not None else None,
                        description=a.get("description"),
                    )
                )

        max_repairs = int(parameters.get("max_repair_attempts") or inputs.get("max_repair_attempts") or 3)
        timeout_s = float(parameters.get("timeout_seconds") or inputs.get("timeout_seconds") or 30.0)
        test_cmd = parameters.get("test_command") or inputs.get("test_command")
        entrypoint = parameters.get("entrypoint") or inputs.get("entrypoint")
        call_kwargs = parameters.get("call_kwargs") or inputs.get("call_kwargs")

        result: CodeTestRepairResult = self._workflow.execute(
            prompt=prompt,
            category=category,
            assertions=assertions,
            max_repair_attempts=max_repairs,
            timeout_seconds=timeout_s,
            test_command=test_cmd,
            entrypoint=entrypoint,
            call_kwargs=call_kwargs,
        )

        output_dict = asdict(result)
        # Convert enum for JSON serializability
        output_dict["category"] = result.category.value

        return TaskResult(
            output=output_dict,
            metadata={
                "capability_id": self.capability_id,
                "execution_id": context.execution_id,
                "status": result.status,
                "total_attempts": result.total_attempts,
            },
        )
