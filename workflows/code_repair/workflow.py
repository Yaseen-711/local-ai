"""Code -> Test -> Repair -> Retest Workflow.

Implements the bounded code generation, isolated execution, testing, and repair
lifecycle exclusively using the existing code.workspace capability.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from connectors.inference import InferenceConnector
from core.common.parsing import parse_json_payload
from core.inference.types import GenerationOptions
from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.code.capability import WorkspaceCodingCapability
from orchestration.capabilities.builtin.code.types import (
    WorkspaceAction,
    WorkspaceCommandResponse,
)
from workflows.code_repair.types import (
    CodeTaskCategory,
    CodeTestRepairResult,
    EngineeringAssertion,
    EngineeringTolerance,
    ExecutionAttemptRecord,
)

logger = logging.getLogger(__name__)

HARD_MAX_REPAIR_ATTEMPTS = 3
DEFAULT_EXECUTION_TIMEOUT = 30.0


def extract_compact_diagnostic(cmd_response: WorkspaceCommandResponse, max_chars: int = 800) -> str:
    """Extract a compact, highly relevant error summary from command response."""
    if cmd_response.timed_out:
        return f"Execution timed out. Process was killed after exceeding timeout limit."

    combined = (cmd_response.stderr.strip() or cmd_response.stdout.strip())
    if not combined:
        return f"Command failed with exit code {cmd_response.exit_code} and empty output."

    lines = combined.splitlines()
    failure_lines = []
    capture = False
    for line in lines:
        if any(marker in line for marker in (
            "FAIL:", "ERROR:", "Traceback (most recent call last):",
            "AssertionError", "SyntaxError", "TypeError", "NameError",
            "ZeroDivisionError", "ValueError", "ImportError"
        )):
            capture = True
        if capture:
            failure_lines.append(line)

    if failure_lines:
        summary = "\n".join(failure_lines)
    else:
        summary = "\n".join(lines[-12:])

    if len(summary) > max_chars:
        summary = summary[-max_chars:]

    return f"Process exited with code {cmd_response.exit_code}:\n{summary}"


def build_engineering_test_harness(
    assertions: List[EngineeringAssertion],
    module_name: str = "solution",
    entrypoint: Optional[str] = None,
    call_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an objective, machine-checkable test harness for engineering calculations.

    Guarantees that the model cannot self-certify its answer: reference values,
    tolerances, dimensional constraints, and physical boundaries are asserted objectively.
    """
    tests = []
    kwargs_repr = repr(call_kwargs or {})

    for idx, ass in enumerate(assertions):
        rel_tol = ass.tolerance.rel_tol if ass.tolerance.rel_tol is not None else 1e-4
        abs_tol = ass.tolerance.abs_tol if ass.tolerance.abs_tol is not None else 0.0
        expected = ass.expected_value
        name = ass.name
        unit = ass.tolerance.unit

        test_body = [
            f"    def test_assertion_{idx}_{name}(self):",
            f"        # Objective check for {name}: expected {expected} (rel_tol={rel_tol}, abs_tol={abs_tol})",
            f"        res = getattr(self.result, '{name}', None)",
            f"        if res is None and isinstance(self.result, dict):",
            f"            res = self.result.get('{name}')",
            f"        self.assertIsNotNone(res, 'Target calculation value \"{name}\" not found in output')",
            f"        actual_val = float(res['value'] if isinstance(res, dict) and 'value' in res else res)",
            f"        self.assertTrue(",
            f"            math.isclose(actual_val, {expected}, rel_tol={rel_tol}, abs_tol={abs_tol}),",
            f"            f'Calculated {name}={{actual_val}} violates tolerance from reference {expected} (rel={rel_tol}, abs={abs_tol})'",
            f"        )",
        ]

        if ass.min_value is not None:
            test_body.append(
                f"        self.assertGreaterEqual(actual_val, {ass.min_value}, 'Boundary violation: {name} below minimum {ass.min_value}')"
            )
        if ass.max_value is not None:
            test_body.append(
                f"        self.assertLessEqual(actual_val, {ass.max_value}, 'Boundary violation: {name} exceeds maximum {ass.max_value}')"
            )
        if unit:
            test_body.extend([
                f"        actual_unit = res.get('unit') if isinstance(res, dict) else getattr(self.result, '{name}_unit', None)",
                f"        if actual_unit is not None:",
                f"            self.assertEqual(str(actual_unit).lower(), '{unit.lower()}', 'Unit mismatch for {name}')",
            ])

        tests.append("\n".join(test_body))

    all_tests = "\n\n".join(tests)

    code = f'''"""Auto-generated objective engineering verification harness."""
import math
import sys
import unittest

import {module_name}

class TestEngineeringVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inputs = {kwargs_repr}
        if hasattr({module_name}, "{entrypoint or 'calculate'}"):
            fn = getattr({module_name}, "{entrypoint or 'calculate'}")
            cls.result = fn(**inputs)
        elif hasattr({module_name}, "run"):
            cls.result = {module_name}.run(**inputs)
        else:
            cls.result = {module_name}

{all_tests}

if __name__ == "__main__":
    unittest.main(verbosity=2)
'''
    return code


class CodeTestRepairWorkflow:
    """Orchestrates Code -> Test -> Repair -> Retest cycles strictly within code.workspace."""

    def __init__(
        self,
        connector: Optional[InferenceConnector] = None,
        workspace_capability: Optional[WorkspaceCodingCapability] = None,
        model_id: str = "default",
        generator_fn: Optional[Callable[[str, Optional[str]], Tuple[str, str]]] = None,
    ) -> None:
        """Initialize workflow.

        Args:
            connector: InferenceConnector for model code generation and repair.
            workspace_capability: WorkspaceCodingCapability providing the sandbox execution boundary.
            model_id: Target inference model.
            generator_fn: Optional deterministic generator callback for testing/custom agent integration.
        """
        self._connector = connector
        self._workspace_capability = workspace_capability or WorkspaceCodingCapability()
        self._model_id = model_id
        self._generator_fn = generator_fn

    def _generate_code(
        self,
        prompt: str,
        category: CodeTaskCategory,
        assertions: Optional[List[EngineeringAssertion]] = None,
        previous_code: Optional[str] = None,
        failure_diagnostic: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generate or repair code and test harness."""
        # 1. Use custom generator if provided
        if self._generator_fn is not None:
            return self._generator_fn(prompt, failure_diagnostic)

        if self._connector is None:
            raise RuntimeError("CodeTestRepairWorkflow requires either an InferenceConnector or a generator_fn.")

        # 2. Build model prompt
        is_repair = failure_diagnostic is not None
        if not is_repair:
            system = (
                "You are an expert systems software and engineering code generator. "
                "Generate complete, robust Python solution code and unit tests. "
                "Output a JSON object with 'solution_code' and 'test_code' keys. "
                "Do not include prose outside the JSON."
            )
            user_msg = (
                f"Task Category: {category.value}\n"
                f"Specification:\n{prompt}\n\n"
                "Provide functional Python code in 'solution.py' and test harness in 'test_solution.py'. "
                "Return JSON: {\"solution_code\": \"...\", \"test_code\": \"...\"}"
            )
        else:
            system = (
                "You are an expert software repair agent. Analyze the compact test failure diagnostic "
                "and repair the code. Output a JSON object with 'solution_code' and 'test_code' keys."
            )
            user_msg = (
                f"Original Task:\n{prompt}\n\n"
                f"Current Solution Code:\n{previous_code}\n\n"
                f"Verification Failure Diagnostic:\n{failure_diagnostic}\n\n"
                "Fix the bug causing this test failure. Return JSON: {\"solution_code\": \"...\", \"test_code\": \"...\"}"
            )

        resp = self._connector.infer_prompt(
            model_id=self._model_id,
            prompt=user_msg,
            system_prompt=system,
            options=GenerationOptions(temperature=0.1, max_tokens=2048),
        )

        content = resp.message.content or ""
        solution_code = ""
        test_code = ""

        try:
            parsed = parse_json_payload(content)
            if isinstance(parsed, dict):
                solution_code = str(parsed.get("solution_code", ""))
                test_code = str(parsed.get("test_code", ""))
        except Exception:
            # Fallback regex extraction for markdown code blocks
            blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)
            if blocks:
                solution_code = blocks[0].strip()
                if len(blocks) > 1:
                    test_code = blocks[1].strip()

        # If engineering assertions are provided, override or supplement with objective test harness
        if category == CodeTaskCategory.ENGINEERING_CALCULATION and assertions:
            test_code = build_engineering_test_harness(assertions=assertions)

        if not solution_code:
            solution_code = content.strip()

        return solution_code, test_code

    def execute(
        self,
        prompt: str,
        category: CodeTaskCategory = CodeTaskCategory.GENERAL_CODE,
        assertions: Optional[List[EngineeringAssertion]] = None,
        max_repair_attempts: int = HARD_MAX_REPAIR_ATTEMPTS,
        timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT,
        test_command: Optional[str] = None,
        entrypoint: Optional[str] = None,
        call_kwargs: Optional[Dict[str, Any]] = None,
    ) -> CodeTestRepairResult:
        """Execute the Code -> Test -> Repair -> Retest cycle with bounded retries.

        Args:
            prompt: Problem statement or coding requirement.
            category: GENERAL_CODE or ENGINEERING_CALCULATION.
            assertions: Optional list of objective engineering checks.
            max_repair_attempts: Maximum attempts allowed (strictly capped at 3).
            timeout_seconds: Timeout per sandbox command execution.
            test_command: Custom verification command (default: 'python3 -m unittest test_solution.py -v').
            entrypoint: Target function name for engineering calculation harness.
            call_kwargs: Inputs for engineering calculation harness.

        Returns:
            CodeTestRepairResult containing outcome, attempt history, and verified code.
        """
        # Strictly enforce hard maximum ceiling of 3 attempts
        effective_max = min(max(1, max_repair_attempts), HARD_MAX_REPAIR_ATTEMPTS)

        attempts_history: List[ExecutionAttemptRecord] = []
        current_code = ""
        current_test_code = ""
        diagnostic: Optional[str] = None
        cmd_run = test_command or "python3 -B -m unittest test_solution.py -v"

        ctx = CapabilityContext(execution_id=f"code-repair-{int(time.time()*1000)}")

        for attempt_idx in range(1, effective_max + 1):
            t0 = time.perf_counter()
            logger.info("Executing code verification attempt %d/%d", attempt_idx, effective_max)

            # 1. Generate or repair code
            if attempt_idx == 1:
                current_code, current_test_code = self._generate_code(
                    prompt=prompt,
                    category=category,
                    assertions=assertions,
                )
            else:
                current_code, repaired_test = self._generate_code(
                    prompt=prompt,
                    category=category,
                    assertions=assertions,
                    previous_code=current_code,
                    failure_diagnostic=diagnostic,
                )
                if repaired_test and category != CodeTaskCategory.ENGINEERING_CALCULATION:
                    current_test_code = repaired_test

            # For engineering calculations with assertions, ensure objective test harness is strictly used
            if category == CodeTaskCategory.ENGINEERING_CALCULATION and assertions:
                current_test_code = build_engineering_test_harness(
                    assertions=assertions,
                    entrypoint=entrypoint,
                    call_kwargs=call_kwargs,
                )

            # 2. Write code and tests strictly into existing sandbox workspace
            self._workspace_capability.execute(
                parameters={"action": WorkspaceAction.WRITE_FILE.value, "path": "solution.py", "content": current_code},
                inputs={},
                context=ctx,
            )
            self._workspace_capability.execute(
                parameters={"action": WorkspaceAction.WRITE_FILE.value, "path": "test_solution.py", "content": current_test_code},
                inputs={},
                context=ctx,
            )

            # 3. Execute tests strictly via code.workspace
            cmd_task_res = self._workspace_capability.execute(
                parameters={
                    "action": WorkspaceAction.RUN_COMMAND.value,
                    "command": cmd_run,
                    "timeout_seconds": timeout_seconds,
                },
                inputs={},
                context=ctx,
            )

            cmd_out = cmd_task_res.output
            if isinstance(cmd_out, dict):
                cmd_resp = WorkspaceCommandResponse(
                    command=cmd_out.get("command", cmd_run),
                    stdout=cmd_out.get("stdout", ""),
                    stderr=cmd_out.get("stderr", ""),
                    exit_code=int(cmd_out.get("exit_code", -1)),
                    execution_time_ms=float(cmd_out.get("execution_time_ms", 0.0)),
                    success=bool(cmd_out.get("success", False)),
                    timed_out=bool(cmd_out.get("timed_out", False)),
                    error=cmd_out.get("error"),
                )
            else:
                cmd_resp = cmd_out

            dt_ms = (time.perf_counter() - t0) * 1000

            # 4. Check verification pass/fail
            passed = cmd_resp.success and cmd_resp.exit_code == 0 and not cmd_resp.timed_out
            error_summary = None if passed else extract_compact_diagnostic(cmd_resp)

            attempt_record = ExecutionAttemptRecord(
                attempt_number=attempt_idx,
                code=current_code,
                test_code=current_test_code,
                exit_code=cmd_resp.exit_code,
                stdout=cmd_resp.stdout,
                stderr=cmd_resp.stderr,
                success=passed,
                timed_out=cmd_resp.timed_out,
                error_summary=error_summary,
                duration_ms=dt_ms,
            )
            attempts_history.append(attempt_record)

            if passed:
                logger.info("Verification passed on attempt %d", attempt_idx)
                return CodeTestRepairResult(
                    status="success",
                    category=category,
                    final_code=current_code,
                    test_code=current_test_code,
                    attempts=attempts_history,
                    total_attempts=len(attempts_history),
                    verification_output=cmd_resp.stdout or cmd_resp.stderr,
                    metadata={"attempts_count": len(attempts_history), "command": cmd_run},
                )

            # Verification failed, prepare compact diagnostic for next repair iteration
            diagnostic = error_summary
            logger.warning(
                "Attempt %d failed (exit %d). Diagnostic: %s",
                attempt_idx, cmd_resp.exit_code, (error_summary or "")[:150]
            )

        # 5. All attempts exhausted: Terminal Failure
        last_error = attempts_history[-1].error_summary if attempts_history else "Unknown error"
        terminal_msg = f"Repair budget exhausted after {len(attempts_history)} attempts. Last error:\n{last_error}"

        return CodeTestRepairResult(
            status="failed",
            category=category,
            final_code=current_code,
            test_code=current_test_code,
            attempts=attempts_history,
            total_attempts=len(attempts_history),
            verification_output=attempts_history[-1].stdout or attempts_history[-1].stderr if attempts_history else "",
            terminal_error=terminal_msg,
            metadata={"attempts_count": len(attempts_history), "command": cmd_run},
        )
