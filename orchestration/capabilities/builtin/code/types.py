"""Data contracts for workspace coding and command execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class WorkspaceAction(str, Enum):
    """Supported discrete workspace actions."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"
    LIST_DIR = "list_dir"
    RUN_COMMAND = "run_command"


@dataclass(frozen=True)
class WorkspaceFileRead:
    """Result of reading a workspace file with windowing support.

    Attributes:
        path: Relative path within the workspace.
        content: Extracted text content.
        start_line: 1-indexed start line.
        end_line: 1-indexed end line.
        total_lines: Total line count of the file.
    """
    path: str
    content: str
    start_line: int
    end_line: int
    total_lines: int


@dataclass(frozen=True)
class WorkspaceCommandRequest:
    """Request to execute a command within the workspace container/sandbox.

    Attributes:
        command: Shell or executable command string.
        working_dir: Subdirectory relative to workspace root (defaults to workspace root).
        timeout_seconds: Maximum wall-clock execution time before SIGKILL.
        environment: Execution-specific environment variable overrides.
    """
    command: str
    working_dir: Optional[str] = None
    timeout_seconds: float = 60.0
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceCommandResponse:
    """Outcome of command execution.

    Attributes:
        command: The executed command string.
        stdout: Standard output text.
        stderr: Standard error text.
        exit_code: Process return code (0 indicates success).
        execution_time_ms: Elapsed wall-clock time in milliseconds.
        success: Whether exit_code == 0 and no timeout occurred.
        timed_out: Whether execution was terminated due to exceeding timeout.
        error: Optional failure diagnosis message.
    """
    command: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    success: bool
    timed_out: bool = False
    error: Optional[str] = None
