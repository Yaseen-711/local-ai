"""Unit tests for Code -> Test -> Repair -> Retest workflow."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.code.capability import WorkspaceCodingCapability
from orchestration.capabilities.builtin.code_repair import CodeVerificationRepairCapability
from orchestration.domain.results import TaskResult
from workflows.code_repair.types import (
    CodeTaskCategory,
    CodeTestRepairResult,
    EngineeringAssertion,
    EngineeringTolerance,
)
from workflows.code_repair.workflow import (
    HARD_MAX_REPAIR_ATTEMPTS,
    CodeTestRepairWorkflow,
    build_engineering_test_harness,
    extract_compact_diagnostic,
)


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    ws = tmp_path / "sandbox_ws"
    ws.mkdir(parents=True)
    return ws


@pytest.fixture
def workspace_cap(workspace_dir: Path) -> WorkspaceCodingCapability:
    return WorkspaceCodingCapability(
        workspace_dir=workspace_dir,
        default_executor_type="local_subprocess",
    )


def test_first_attempt_success(workspace_cap: WorkspaceCodingCapability):
    """1. Initial code execution succeeds on attempt 1 without repair."""
    def generator(prompt, diagnostic):
        solution = "def add(a, b):\n    return a + b\n"
        test = (
            "import unittest\n"
            "import solution\n\n"
            "class TestAdd(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(solution.add(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        return solution, test

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Write an add function with tests")

    assert res.status == "success"
    assert res.total_attempts == 1
    assert len(res.attempts) == 1
    assert res.attempts[0].success is True
    assert res.attempts[0].exit_code == 0
    assert "OK" in (res.verification_output or res.attempts[0].stderr)


def test_syntax_runtime_failure_captured(workspace_cap: WorkspaceCodingCapability):
    """2. Syntax or runtime crash in generated code is captured as attempt failure."""
    def generator(prompt, diagnostic):
        # Invalid Python syntax
        solution = "def broken(\n"
        test = "import unittest\nimport solution\nclass T(unittest.TestCase):\n    def test_syntax(self): pass\nif __name__ == '__main__': unittest.main()\n"
        return solution, test

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Write code", max_repair_attempts=1)

    assert res.status == "failed"
    assert res.total_attempts == 1
    assert res.attempts[0].success is False
    assert res.attempts[0].exit_code != 0
    assert "SyntaxError" in (res.attempts[0].error_summary or "")


def test_assertion_test_failure_captured(workspace_cap: WorkspaceCodingCapability):
    """3. Failed assertion in test harness is captured with non-zero exit code."""
    def generator(prompt, diagnostic):
        solution = "def multiply(a, b):\n    return a + b\n"  # bug: + instead of *
        test = (
            "import unittest\n"
            "import solution\n\n"
            "class TestMul(unittest.TestCase):\n"
            "    def test_mul(self):\n"
            "        self.assertEqual(solution.multiply(3, 4), 12)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        return solution, test

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Multiply two numbers", max_repair_attempts=1)

    assert res.status == "failed"
    assert res.total_attempts == 1
    assert res.attempts[0].success is False
    assert res.attempts[0].exit_code != 0
    assert "AssertionError" in (res.attempts[0].error_summary or "")


def test_repair_followed_by_successful_retest(workspace_cap: WorkspaceCodingCapability):
    """4. Attempt 1 fails, compact diagnostic triggers repair, Attempt 2 passes."""
    def generator(prompt, diagnostic):
        if diagnostic is None:
            # Buggy attempt 1
            solution = "def calculate_area(r):\n    return 3.14 * r\n"  # bug: missing r**2
            test = (
                "import unittest\n"
                "import solution\n\n"
                "class TestArea(unittest.TestCase):\n"
                "    def test_area(self):\n"
                "        self.assertAlmostEqual(solution.calculate_area(2), 12.56, places=2)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )
            return solution, test
        else:
            # Repaired attempt 2
            assert "AssertionError" in diagnostic or "FAIL" in diagnostic
            solution = "def calculate_area(r):\n    return 3.14 * (r ** 2)\n"
            test = (
                "import unittest\n"
                "import solution\n\n"
                "class TestArea(unittest.TestCase):\n"
                "    def test_area(self):\n"
                "        self.assertAlmostEqual(solution.calculate_area(2), 12.56, places=2)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )
            return solution, test

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Calculate circle area")

    assert res.status == "success"
    assert res.total_attempts == 2
    assert len(res.attempts) == 2
    assert res.attempts[0].success is False
    assert res.attempts[1].success is True
    assert res.attempts[1].exit_code == 0
    assert "calculate_area(r):" in res.final_code
    assert "(r ** 2)" in res.final_code


def test_repair_exhaustion_terminal_failure(workspace_cap: WorkspaceCodingCapability):
    """5. Exactly 3 failed repair attempts result in terminal failure."""
    calls = [0]

    def generator(prompt, diagnostic):
        calls[0] += 1
        # Always return failing code
        solution = f"def compute():\n    return {calls[0]}\n"
        test = (
            "import unittest\n"
            "import solution\n\n"
            "class TestComp(unittest.TestCase):\n"
            "    def test_val(self):\n"
            "        self.assertEqual(solution.compute(), 999)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        return solution, test

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Compute 999", max_repair_attempts=3)

    assert res.status == "failed"
    assert res.total_attempts == 3
    assert len(res.attempts) == 3
    assert all(not a.success for a in res.attempts)
    assert "Repair budget exhausted" in (res.terminal_error or "")


def test_hard_three_attempt_cap_enforcement(workspace_cap: WorkspaceCodingCapability):
    """6. Requesting > 3 attempts is strictly clamped to hard limit of 3."""
    calls = [0]

    def generator(prompt, diagnostic):
        calls[0] += 1
        return "def bad(): return 1\n", "import unittest\nif __name__ == '__main__': raise RuntimeError('fail')\n"

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Infinite fail", max_repair_attempts=10)

    # Must be capped at HARD_MAX_REPAIR_ATTEMPTS = 3
    assert res.total_attempts == 3
    assert len(res.attempts) == 3
    assert calls[0] == 3


def test_attempt_provenance_preservation(workspace_cap: WorkspaceCodingCapability):
    """7. Each attempt record preserves code snapshots, exit codes, and durations."""
    attempt_counter = [0]

    def generator(prompt, diagnostic):
        attempt_counter[0] += 1
        sol = f"# Version {attempt_counter[0]}\ndef val(): return {attempt_counter[0]}\n"
        test = (
            "import unittest\nimport solution\n"
            "class T(unittest.TestCase):\n"
            "    def test_it(self): self.assertEqual(solution.val(), 2)\n"
            "if __name__ == '__main__': unittest.main()\n"
        )
        return sol, test

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Produce 2")

    assert res.status == "success"
    assert res.total_attempts == 2

    # Attempt 1 provenance
    att1 = res.attempts[0]
    assert att1.attempt_number == 1
    assert "Version 1" in att1.code
    assert att1.exit_code != 0
    assert att1.success is False
    assert att1.duration_ms > 0

    # Attempt 2 provenance
    att2 = res.attempts[1]
    assert att2.attempt_number == 2
    assert "Version 2" in att2.code
    assert att2.exit_code == 0
    assert att2.success is True
    assert att2.duration_ms > 0


def test_timeout_resource_limit_propagation(workspace_cap: WorkspaceCodingCapability):
    """8. Command timeout inside workspace propagates as timed_out attempt failure."""
    def generator(prompt, diagnostic):
        solution = "import time\ndef hang():\n    time.sleep(5)\n"
        test = (
            "import unittest, solution\n"
            "class T(unittest.TestCase):\n"
            "    def test_h(self): solution.hang()\n"
            "if __name__ == '__main__': unittest.main()\n"
        )
        return solution, test

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Hang test", timeout_seconds=0.2, max_repair_attempts=1)

    assert res.status == "failed"
    assert res.attempts[0].timed_out is True
    assert res.attempts[0].success is False
    assert "timed out" in (res.attempts[0].error_summary or "").lower()


def test_engineering_reference_tolerance_verification(workspace_cap: WorkspaceCodingCapability):
    """9. Engineering calculations verify reference values, relative tolerances, and boundary checks."""
    assertions = [
        EngineeringAssertion(
            name="pump_head_m",
            expected_value=45.2,
            tolerance=EngineeringTolerance(rel_tol=0.02, unit="m"),  # 2% tolerance
            min_value=0.0,
            max_value=100.0,
        ),
        EngineeringAssertion(
            name="efficiency",
            expected_value=0.78,
            tolerance=EngineeringTolerance(abs_tol=0.05),
            min_value=0.0,
            max_value=1.0,
        ),
    ]

    # Test 1: Solution within tolerances passes
    def good_generator(prompt, diagnostic):
        solution = (
            "def calculate():\n"
            "    return {\n"
            "        'pump_head_m': {'value': 45.5, 'unit': 'm'},\n"
            "        'efficiency': 0.79,\n"
            "    }\n"
        )
        return solution, ""

    wf_good = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=good_generator,
    )
    res_good = wf_good.execute(
        prompt="Pump calculation",
        category=CodeTaskCategory.ENGINEERING_CALCULATION,
        assertions=assertions,
    )
    assert res_good.status == "success"
    assert res_good.total_attempts == 1

    # Test 2: Solution outside tolerance fails
    def bad_generator(prompt, diagnostic):
        solution = (
            "def calculate():\n"
            "    return {\n"
            "        'pump_head_m': {'value': 55.0, 'unit': 'm'},\n"  # Violates 2% of 45.2
            "        'efficiency': 0.79,\n"
            "    }\n"
        )
        return solution, ""

    wf_bad = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=bad_generator,
    )
    res_bad = wf_bad.execute(
        prompt="Pump calculation",
        category=CodeTaskCategory.ENGINEERING_CALCULATION,
        assertions=assertions,
        max_repair_attempts=1,
    )
    assert res_bad.status == "failed"
    assert "violates tolerance" in (res_bad.attempts[0].error_summary or "")


def test_proof_execution_always_goes_through_code_workspace(workspace_cap: WorkspaceCodingCapability):
    """10. Verify that all file writes and executions are routed strictly through WorkspaceCodingCapability."""
    dispatched_actions = []
    original_execute = workspace_cap.execute

    def spy_execute(parameters, inputs, context):
        dispatched_actions.append(parameters.get("action"))
        return original_execute(parameters, inputs, context)

    workspace_cap.execute = spy_execute  # type: ignore

    def generator(prompt, diagnostic):
        return "def f(): return 42\n", "import unittest, solution\nclass T(unittest.TestCase):\n def test_f(self): self.assertEqual(solution.f(), 42)\nif __name__ == '__main__': unittest.main()\n"

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    res = wf.execute(prompt="Spy execution test")

    assert res.status == "success"
    # Verify exact actions dispatched through capability: write_file solution, write_file test, run_command
    assert dispatched_actions == ["write_file", "write_file", "run_command"]


def test_capability_wrapper_contract(workspace_cap: WorkspaceCodingCapability):
    """11. Verify CodeVerificationRepairCapability satisfies the Capability protocol."""
    def generator(prompt, diagnostic):
        return "def f(): return 10\n", "import unittest, solution\nclass T(unittest.TestCase):\n def test_f(self): self.assertEqual(solution.f(), 10)\nif __name__ == '__main__': unittest.main()\n"

    wf = CodeTestRepairWorkflow(
        workspace_capability=workspace_cap,
        generator_fn=generator,
    )
    cap = CodeVerificationRepairCapability(workflow=wf)
    assert cap.capability_id == "code.verify_and_repair"

    ctx = CapabilityContext(execution_id="test-exec-1")
    task_res = cap.execute(
        parameters={"category": "general_code", "max_repair_attempts": 2},
        inputs={"prompt": "Return 10"},
        context=ctx,
    )

    assert isinstance(task_res, TaskResult)
    assert task_res.output["status"] == "success"
    assert task_res.output["total_attempts"] == 1
    assert task_res.metadata["capability_id"] == "code.verify_and_repair"
    assert task_res.metadata["status"] == "success"
