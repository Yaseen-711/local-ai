"""Workspace coding capability package.

Provides isolated workspace file manipulation and containerized command execution.
"""

from orchestration.capabilities.builtin.code.base import (
    DockerUnavailableError,
    WorkspaceExecutor,
)
from orchestration.capabilities.builtin.code.capability import (
    WorkspaceCodingCapability,
)
from orchestration.capabilities.builtin.code.docker_executor import (
    DockerWorkspaceExecutor,
)
from orchestration.capabilities.builtin.code.subprocess_executor import (
    LocalSubprocessWorkspaceExecutor,
)
from orchestration.capabilities.builtin.code.types import (
    WorkspaceAction,
    WorkspaceCommandRequest,
    WorkspaceCommandResponse,
    WorkspaceFileRead,
)

__all__ = [
    "DockerUnavailableError",
    "WorkspaceExecutor",
    "DockerWorkspaceExecutor",
    "LocalSubprocessWorkspaceExecutor",
    "WorkspaceCodingCapability",
    "WorkspaceAction",
    "WorkspaceFileRead",
    "WorkspaceCommandRequest",
    "WorkspaceCommandResponse",
]
