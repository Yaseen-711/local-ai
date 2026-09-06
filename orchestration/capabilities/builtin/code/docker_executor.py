"""Docker container workspace executor.

Default and mandatory isolation boundary for software development and command execution.
Enforces container sandboxing with hard CPU/RAM limits, path traversal protection,
and network isolation (network_mode='none').

Strict Invariant:
  Fails explicitly with DockerUnavailableError if Docker is unreachable.
  NEVER silently falls back to host execution.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

from orchestration.capabilities.builtin.code.base import (
    DockerUnavailableError,
    WorkspaceExecutor,
)
from orchestration.capabilities.builtin.code.subprocess_executor import _safe_resolve
from orchestration.capabilities.builtin.code.types import (
    WorkspaceCommandRequest,
    WorkspaceCommandResponse,
    WorkspaceFileRead,
)

logger = logging.getLogger(__name__)


class DockerWorkspaceExecutor(WorkspaceExecutor):
    """Containerized Docker workspace executor for secure isolated code execution."""

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        image: Optional[str] = None,
        docker_image: Optional[str] = None,
        memory_limit: Optional[str] = None,
        mem_limit: Optional[str] = None,
        nano_cpus: int = 2_000_000_000,
        network_mode: str = "none",
    ) -> None:
        import docker
        from docker.errors import DockerException

        self._image = docker_image or image or "python:3.12-slim"
        self._memory_limit = mem_limit or memory_limit or "2g"
        self._nano_cpus = nano_cpus
        self._network_mode = network_mode


        if workspace_dir is not None:
            self._root = Path(workspace_dir).resolve()
            self._root.mkdir(parents=True, exist_ok=True)
            self._temp_dir = None
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="docker_workspace_")
            self._root = Path(self._temp_dir).resolve()

        self._container = None

        # Connect to Docker daemon — must fail explicitly if unavailable
        try:
            self._client = docker.from_env()
            self._client.ping()
        except DockerException as exc:
            self.cleanup()
            raise DockerUnavailableError(
                f"Docker daemon is unreachable. Docker is required as the default execution boundary "
                f"to prevent unsandboxed host execution: {exc}"
            ) from exc

        self._start_container()


    @property
    def root(self) -> Path:
        return self._root

    def _start_container(self) -> None:
        """Start the sandboxed container running idle with workspace bind-mounted."""
        try:
            self._container = self._client.containers.run(
                image=self._image,
                command=["tail", "-f", "/dev/null"],
                detach=True,
                remove=True,
                network_mode=self._network_mode,
                mem_limit=self._memory_limit,
                nano_cpus=self._nano_cpus,
                volumes={
                    str(self._root): {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
                working_dir="/workspace",
            )
        except Exception as exc:
            self.cleanup()
            raise RuntimeError(f"Failed to start Docker sandbox container: {exc}") from exc

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> WorkspaceFileRead:
        target = _safe_resolve(self._root, path)
        if not target.exists():
            raise FileNotFoundError(f"File not found in workspace: {path}")
        if not target.is_file():
            raise ValueError(f"Target is not a file: {path}")

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total_lines = len(lines)

        s_line = max(1, start_line) if start_line is not None else 1
        e_line = min(total_lines, end_line) if end_line is not None else total_lines

        if s_line > total_lines:
            selected_content = ""
        else:
            selected_content = "".join(lines[s_line - 1 : e_line])

        return WorkspaceFileRead(
            path=path,
            content=selected_content,
            start_line=s_line,
            end_line=e_line,
            total_lines=total_lines,
        )

    def write_file(self, path: str, content: str, overwrite: bool = True) -> None:
        target = _safe_resolve(self._root, path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"File already exists in workspace: {path}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def edit_file(self, path: str, target: str, replacement: str) -> None:
        file_path = _safe_resolve(self._root, path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found in workspace: {path}")

        current_content = file_path.read_text(encoding="utf-8")
        if target not in current_content:
            raise ValueError(f"Target string not found in file '{path}' for editing.")

        occurrences = current_content.count(target)
        if occurrences > 1:
            raise ValueError(f"Target string appears {occurrences} times in '{path}'. Must be unique.")

        new_content = current_content.replace(target, replacement, 1)
        file_path.write_text(new_content, encoding="utf-8")

    def list_dir(
        self,
        path: str = ".",
        recursive: bool = False,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        target_dir = _safe_resolve(self._root, path)
        if not target_dir.exists():
            raise FileNotFoundError(f"Directory not found in workspace: {path}")

        items = []
        if recursive:
            for root, dirs, files in os.walk(target_dir):
                rel_root = Path(root).relative_to(self._root)
                depth = len(rel_root.parts)
                if depth > max_depth:
                    dirs.clear()
                    continue
                for d in sorted(dirs):
                    p = Path(root) / d
                    items.append({
                        "name": d,
                        "path": str(p.relative_to(self._root)),
                        "is_dir": True,
                        "size_bytes": 0,
                    })
                for f in sorted(files):
                    p = Path(root) / f
                    items.append({
                        "name": f,
                        "path": str(p.relative_to(self._root)),
                        "is_dir": False,
                        "size_bytes": p.stat().st_size,
                    })
        else:
            for entry in sorted(target_dir.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
                items.append({
                    "name": entry.name,
                    "path": str(entry.relative_to(self._root)),
                    "is_dir": entry.is_dir(),
                    "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                })

        return items

    def run_command(self, request: WorkspaceCommandRequest) -> WorkspaceCommandResponse:
        if self._container is None:
            raise RuntimeError("Sandbox container is not running.")

        workdir = "/workspace"
        if request.working_dir:
            _safe_resolve(self._root, request.working_dir)
            clean_sub = request.working_dir.strip("/\\")
            workdir = f"/workspace/{clean_sub}"

        cmd = ["/bin/sh", "-c", request.command]
        start_time = time.perf_counter()
        timed_out = False
        exit_code = -1
        stdout = ""
        stderr = ""
        error = None

        def _exec():
            return self._container.exec_run(
                cmd=cmd,
                workdir=workdir,
                environment=request.environment,
                demux=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_exec)
            try:
                exec_result = future.result(timeout=request.timeout_seconds)
                exit_code = exec_result.exit_code
                out_bytes, err_bytes = exec_result.output
                stdout = out_bytes.decode("utf-8", errors="replace") if out_bytes else ""
                stderr = err_bytes.decode("utf-8", errors="replace") if err_bytes else ""
            except concurrent.futures.TimeoutError:
                timed_out = True
                exit_code = 124
                error = f"Command timed out after {request.timeout_seconds} seconds."
                # Kill container and restart clean instance
                try:
                    self._container.kill()
                except Exception:
                    pass
                self._start_container()
            except Exception as exc:
                error = f"Container execution failure: {exc}"
                stderr = str(exc)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return WorkspaceCommandResponse(
            command=request.command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_ms=elapsed_ms,
            success=exit_code == 0 and not timed_out,
            timed_out=timed_out,
            error=error,
        )

    def cleanup(self) -> None:
        container = getattr(self, "_container", None)
        if container is not None:
            try:
                container.stop(timeout=1)
            except Exception:
                pass
            self._container = None

        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

