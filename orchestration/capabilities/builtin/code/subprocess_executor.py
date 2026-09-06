"""Local subprocess workspace executor (explicit developer/test option ONLY).

Security note:
  This executor runs on the host system without container isolation.
  It must only be used when explicitly configured for local testing or development.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from orchestration.capabilities.builtin.code.base import WorkspaceExecutor
from orchestration.capabilities.builtin.code.types import (
    WorkspaceCommandRequest,
    WorkspaceCommandResponse,
    WorkspaceFileRead,
)


def _safe_resolve(base_dir: Path, relative_path: str) -> Path:
    """Resolve path and verify it stays strictly inside base_dir (path traversal protection)."""
    target = Path(relative_path)
    if target.is_absolute():
        resolved = target.resolve()
    else:
        resolved = (base_dir / target).resolve()

    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError:
        raise PermissionError(f"Access denied: path '{relative_path}' traverses outside workspace.")
    return resolved



class LocalSubprocessWorkspaceExecutor(WorkspaceExecutor):
    """Local non-containerized workspace executor for explicit developer/test usage."""

    def __init__(self, workspace_dir: Optional[Path] = None) -> None:
        if workspace_dir is not None:
            self._root = Path(workspace_dir).resolve()
            self._root.mkdir(parents=True, exist_ok=True)
            self._temp_dir = None
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="local_workspace_")
            self._root = Path(self._temp_dir).resolve()

    @property
    def root(self) -> Path:
        return self._root

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
        workdir = _safe_resolve(self._root, request.working_dir) if request.working_dir else self._root
        if not workdir.exists():
            raise FileNotFoundError(f"Working directory does not exist: {request.working_dir}")

        env = os.environ.copy()
        env.update(request.environment)

        start_time = time.perf_counter()
        timed_out = False
        exit_code = -1
        stdout = ""
        stderr = ""
        error = None

        try:
            proc = subprocess.run(
                request.command,
                shell=True,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            error = f"Command timed out after {request.timeout_seconds} seconds."
            exit_code = 124  # Standard timeout exit code
        except Exception as exc:
            error = f"Execution failed: {exc}"
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
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
