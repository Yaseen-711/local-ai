"""Live end-to-end integration test demonstrating Code -> Test -> Repair -> Retest in Docker sandbox.

Validates the full milestone lifecycle against the real Docker container environment:
1. Generates engineering calculation code with an intentional calculation flaw.
2. Executes test harness inside live Docker sandbox (python:3.12-slim, network_mode='none').
3. Captures real Docker exit code (1), stderr, and test assertion failure.
4. Supplies compact diagnostic to agent repair logic.
5. Writes repaired code to Docker workspace and retests in container.
6. Retest passes with exit code 0 inside the isolated container.
7. Validates that no code was executed directly on host and attempt history is preserved.
"""

from pathlib import Path
import pytest

from orchestration.capabilities.builtin.code import (
    DockerWorkspaceExecutor,
    WorkspaceCodingCapability,
)
from workflows.code_repair.types import (
    CodeTaskCategory,
    CodeTestRepairResult,
    EngineeringAssertion,
    EngineeringTolerance,
)
from workflows.code_repair.workflow import CodeTestRepairWorkflow


@pytest.fixture
def docker_workspace_dir(tmp_path: Path) -> Path:
    ws = tmp_path / "live_docker_ws"
    ws.mkdir(parents=True)
    return ws


def test_live_docker_code_test_repair_retest_e2e(docker_workspace_dir: Path):
    """Demonstrate real Code -> Test -> Repair -> Retest cycle inside isolated Docker sandbox."""
    import docker

    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f"Docker daemon is not available: {exc}")

    # 1. Initialize real Docker sandbox executor (hard network isolation, CPU/RAM caps)
    executor = DockerWorkspaceExecutor(
        workspace_dir=docker_workspace_dir,
        image="python:3.12-slim",
        network_mode="none",
        memory_limit="1g",
        nano_cpus=2_000_000_000,
    )

    try:
        workspace_cap = WorkspaceCodingCapability(executor=executor)

        # 2. Engineering specification: Heat exchanger duty calculation
        # Q = m_dot * Cp * (T_out - T_in)
        # m_dot = 15.0 kg/s, Cp = 4.184 kJ/(kg*K), T_in = 25 C, T_out = 65 C -> Delta T = 40 K
        # Expected Q = 15.0 * 4.184 * 40.0 = 2510.4 kW
        assertions = [
            EngineeringAssertion(
                name="duty_kw",
                expected_value=2510.4,
                tolerance=EngineeringTolerance(rel_tol=0.001, unit="kW"),  # 0.1% tolerance
                min_value=0.0,
                max_value=10000.0,
                description="Heat transfer duty in kW",
            ),
        ]

        attempt_counter = [0]

        def agent_code_generator(prompt: str, diagnostic: str | None) -> tuple[str, str]:
            attempt_counter[0] += 1
            if diagnostic is None:
                # Attempt 1: Real initial code with flawed formula (wrong specific heat constant 2.5 instead of 4.184)
                code = (
                    '"""Heat exchanger thermal duty calculation."""\n\n'
                    'def calculate(mass_flow=15.0, temp_in=25.0, temp_out=65.0):\n'
                    '    # Intentional bug: incorrect Cp constant used\n'
                    '    cp = 2.5  # Incorrect Cp for water (should be 4.184)\n'
                    '    delta_t = temp_out - temp_in\n'
                    '    q = mass_flow * cp * delta_t\n'
                    '    return {"duty_kw": {"value": q, "unit": "kW"}}\n'
                )
                return code, ""
            else:
                # Attempt 2: Agent repairs based on real Docker failure diagnostic
                assert "Process exited with code 1" in diagnostic or "violates tolerance" in diagnostic
                repaired_code = (
                    '"""Heat exchanger thermal duty calculation - REPAIRED."""\n\n'
                    'def calculate(mass_flow=15.0, temp_in=25.0, temp_out=65.0):\n'
                    '    # Repaired: verified Cp of water is 4.184 kJ/(kg*K)\n'
                    '    cp = 4.184\n'
                    '    delta_t = temp_out - temp_in\n'
                    '    q = mass_flow * cp * delta_t\n'
                    '    return {"duty_kw": {"value": q, "unit": "kW"}}\n'
                )
                return repaired_code, ""

        # 3. Instantiate workflow bound to Docker sandbox
        workflow = CodeTestRepairWorkflow(
            workspace_capability=workspace_cap,
            generator_fn=agent_code_generator,
        )

        # 4. Execute workflow inside live Docker container
        result: CodeTestRepairResult = workflow.execute(
            prompt="Calculate crude preheat heat exchanger thermal duty for 15 kg/s water heated from 25C to 65C.",
            category=CodeTaskCategory.ENGINEERING_CALCULATION,
            assertions=assertions,
            max_repair_attempts=3,
            timeout_seconds=30.0,
        )

        # 5. Verify outcomes
        assert result.status == "success", f"Workflow failed: {result.terminal_error}"
        assert result.total_attempts == 2, f"Expected exactly 2 attempts, got {result.total_attempts}"
        assert len(result.attempts) == 2

        # Verify Attempt 1: Real failure inside Docker
        att1 = result.attempts[0]
        assert att1.attempt_number == 1
        assert att1.exit_code == 1
        assert att1.success is False
        assert "violates tolerance from reference 2510.4" in att1.error_summary
        assert "1500.0" in att1.error_summary or "1500.0" in att1.stderr  # 15 * 2.5 * 40 = 1500.0

        # Verify Attempt 2: Real pass inside Docker
        att2 = result.attempts[1]
        assert att2.attempt_number == 2
        assert att2.exit_code == 0
        assert att2.success is True
        assert "OK" in att2.stderr or "OK" in att2.stdout
        assert "cp = 4.184" in result.final_code

    finally:
        executor.cleanup()
