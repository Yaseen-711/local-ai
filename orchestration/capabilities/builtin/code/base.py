"""Workspace executor protocol and domain exceptions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from orchestration.capabilities.builtin.code.types import (
    WorkspaceCommandRequest,
    WorkspaceCommandResponse,
    WorkspaceFileRead,
)


class DockerUnavailableError(RuntimeError):
    """Raised when Docker execution is required but the Docker daemon is unreachable."""
    pass


@runtime_checkable
class WorkspaceExecutor(Protocol):
    """Protocol for isolated workspace manipulation and execution engines."""

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> WorkspaceFileRead:
        """Read file contents with optional line range windowing."""
        ...

    def write_file(self, path: str, content: str, overwrite: bool = True) -> None:
        """Write content to a file within the workspace."""
        ...

    def edit_file(self, path: str, target: str, replacement: str) -> None:
        """Atomically replace target string with replacement string in a file."""
        ...

    def list_dir(
        self,
        path: str = ".",
        recursive: bool = False,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """List directory tree contents with metadata."""
        ...

    def run_command(self, request: WorkspaceCommandRequest) -> WorkspaceCommandResponse:
        """Execute a shell command within the workspace."""
        ...

    def cleanup(self) -> None:
        """Release container or temporary directory resources."""
        ...
