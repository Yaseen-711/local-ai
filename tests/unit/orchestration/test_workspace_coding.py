"""Unit tests for Workspace Coding capability, subprocess executor, and Docker sandboxing."""

from pathlib import Path
import pytest

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.code import (
    DockerUnavailableError,
    DockerWorkspaceExecutor,
    LocalSubprocessWorkspaceExecutor,
    WorkspaceAction,
    WorkspaceCodingCapability,
    WorkspaceCommandRequest,
    WorkspaceExecutor,
    WorkspaceFileRead,
)


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    ws = tmp_path / "sandbox_workspace"
    ws.mkdir(parents=True)
    return ws


def test_subprocess_executor_file_operations(workspace_dir: Path):
    """Verify LocalSubprocessWorkspaceExecutor read, write, and edit operations."""
    executor = LocalSubprocessWorkspaceExecutor(workspace_dir=workspace_dir)
    assert isinstance(executor, WorkspaceExecutor)

    # 1. Write file
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    executor.write_file("src/app.py", content)
    assert (workspace_dir / "src" / "app.py").exists()

    # 2. Windowed read
    read_res = executor.read_file("src/app.py", start_line=2, end_line=4)
    assert isinstance(read_res, WorkspaceFileRead)
    assert read_res.path == "src/app.py"
    assert read_res.start_line == 2
    assert read_res.end_line == 4
    assert read_res.total_lines == 5
    assert read_res.content == "line 2\nline 3\nline 4\n"

    # 3. Edit file
    executor.edit_file("src/app.py", target="line 3", replacement="line 3 (modified)")
    full_read = executor.read_file("src/app.py")
    assert "line 3 (modified)" in full_read.content

    # 4. List directory
    entries = executor.list_dir("src")
    assert len(entries) >= 1
    assert entries[0]["name"] == "app.py"
    assert entries[0]["is_dir"] is False


def test_subprocess_executor_path_traversal_rejection(workspace_dir: Path):
    """Verify strict path traversal enforcement prevents accessing paths outside root."""
    executor = LocalSubprocessWorkspaceExecutor(workspace_dir=workspace_dir)

    with pytest.raises(PermissionError, match="traverses outside workspace"):
        executor.write_file("../../outside.txt", "malicious content")

    with pytest.raises(PermissionError, match="traverses outside workspace"):
        executor.read_file("../other.txt")

    with pytest.raises(PermissionError, match="traverses outside workspace"):
        executor.edit_file("/etc/passwd", "root", "toor")


def test_subprocess_executor_command_execution(workspace_dir: Path):
    """Verify LocalSubprocessWorkspaceExecutor executes shell commands."""
    executor = LocalSubprocessWorkspaceExecutor(workspace_dir=workspace_dir)

    req = WorkspaceCommandRequest(
        command="python3 -c 'print(40 + 2)'",
        timeout_seconds=10.0,
    )
    res = executor.run_command(req)

    assert res.success is True
    assert res.exit_code == 0
    assert res.stdout.strip() == "42"
    assert res.timed_out is False
    assert res.execution_time_ms > 0


def test_subprocess_executor_command_timeout(workspace_dir: Path):
    """Verify command execution properly terminates and flags timeout."""
    executor = LocalSubprocessWorkspaceExecutor(workspace_dir=workspace_dir)

    req = WorkspaceCommandRequest(
        command="sleep 5",
        timeout_seconds=0.2,
    )
    res = executor.run_command(req)

    assert res.success is False
    assert res.timed_out is True
    assert "timed out" in (res.error or "").lower()


def test_docker_executor_unavailable_error(workspace_dir: Path, monkeypatch):
    """Verify DockerWorkspaceExecutor raises DockerUnavailableError when daemon is down."""
    import docker
    from docker.errors import DockerException

    def fake_from_env():
        raise DockerException("Cannot connect to the Docker daemon at unix:///var/run/docker.sock")

    monkeypatch.setattr(docker, "from_env", fake_from_env)

    # Must raise DockerUnavailableError and NEVER silently fall back to host execution
    with pytest.raises(DockerUnavailableError, match="Docker daemon is unreachable"):
        DockerWorkspaceExecutor(workspace_dir=workspace_dir)



def test_workspace_capability_dispatch_all_actions(workspace_dir: Path):
    """Verify WorkspaceCodingCapability dispatches read, write, edit, list, and run_command."""
    cap = WorkspaceCodingCapability(
        workspace_dir=workspace_dir,
        default_executor_type="local_subprocess",
    )
    assert cap.capability_id == "code.workspace"

    ctx = CapabilityContext(execution_id="exec-ws-1")

    # 1. Write file
    w_res = cap.execute(
        parameters={"action": "write_file", "path": "calc.py", "content": "print('calc loaded')\n"},
        inputs={},
        context=ctx,
    )
    assert w_res.output["status"] == "written"
    assert (workspace_dir / "calc.py").exists()

    # 2. Read file
    r_res = cap.execute(
        parameters={"action": "read_file", "path": "calc.py"},
        inputs={},
        context=ctx,
    )
    assert "calc loaded" in r_res.output["content"]

    # 3. Edit file
    e_res = cap.execute(
        parameters={"action": "edit_file", "path": "calc.py", "target": "calc loaded", "replacement": "calc v2"},
        inputs={},
        context=ctx,
    )
    assert e_res.output["status"] == "edited"

    # 4. List dir
    l_res = cap.execute(
        parameters={"action": "list_dir", "path": "."},
        inputs={},
        context=ctx,
    )
    assert l_res.output["count"] >= 1

    # 5. Run command
    c_res = cap.execute(
        parameters={"action": "run_command", "command": "python3 calc.py"},
        inputs={},
        context=ctx,
    )
    assert c_res.output["success"] is True
    assert c_res.output["stdout"].strip() == "calc v2"


def test_docker_executor_live_execution(workspace_dir: Path):
    """Verify live Docker sandbox execution if Docker daemon is reachable."""
    import docker
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f"Docker daemon not running or accessible: {exc}")

    executor = DockerWorkspaceExecutor(
        workspace_dir=workspace_dir,
        image="python:3.12-slim",
        network_mode="none",
    )
    try:
        executor.write_file("docker_test.py", "print(99 + 1)\n")
        req = WorkspaceCommandRequest(command="python3 docker_test.py", timeout_seconds=30.0)
        res = executor.run_command(req)

        assert res.success is True
        assert res.exit_code == 0
        assert res.stdout.strip() == "100"
    finally:
        executor.cleanup()
