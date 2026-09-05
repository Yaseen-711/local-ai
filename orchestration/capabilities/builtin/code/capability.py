"""Built-in Workspace Coding capability.

Provides a bounded workspace execution environment for coding agents/models to inspect,
read, write, edit files, and execute build/test commands within an isolated Docker sandbox.
"""

from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.code.base import (
    DockerUnavailableError,
    WorkspaceExecutor,
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
)
from orchestration.domain.results import TaskResult

logger = logging.getLogger(__name__)


class WorkspaceCodingCapability:
    """Capability providing bounded workspace tools inside an isolated execution container.

    Semantic contract:
        Parameters / Inputs:
            - 'action' (str, required): One of 'read_file', 'write_file', 'edit_file',
              'list_dir', 'run_command'.
            - 'executor_type' (str, optional): 'docker' (default) or 'local_subprocess' (explicit only).
            - For 'read_file':
                'path' (str), 'start_line' (int, optional), 'end_line' (int, optional).
            - For 'write_file':
                'path' (str), 'content' (str), 'overwrite' (bool, default: True).
            - For 'edit_file':
                'path' (str), 'target' (str), 'replacement' (str).
            - For 'list_dir':
                'path' (str, default: '.'), 'recursive' (bool, default: False), 'max_depth' (int, default: 3).
            - For 'run_command':
                'command' (str), 'working_dir' (str, optional), 'timeout_seconds' (float, default: 60.0).
    """

    def __init__(
        self,
        executor: Optional[WorkspaceExecutor] = None,
        workspace_dir: Optional[Path] = None,
        default_executor_type: str = "docker",
    ) -> None:
        self._executor = executor
        self._workspace_dir = workspace_dir
        self._default_executor_type = default_executor_type

    @property
    def capability_id(self) -> str:
        return "code.workspace"

    def _resolve_executor(self, executor_type: str) -> WorkspaceExecutor:
        if self._executor is not None:
            return self._executor

        if executor_type == "docker":
            return DockerWorkspaceExecutor(workspace_dir=self._workspace_dir)
        elif executor_type == "local_subprocess":
            return LocalSubprocessWorkspaceExecutor(workspace_dir=self._workspace_dir)
        else:
            raise ValueError(
                f"Unknown executor_type '{executor_type}'. Expected 'docker' or 'local_subprocess'."
            )

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        # 1. Resolve action
        action_str = str(
            parameters.get("action")
            or inputs.get("action")
            or ""
        ).lower()

        if not action_str:
            raise ValueError(
                f"Capability '{self.capability_id}' requires an 'action' in parameters or inputs. "
                f"Supported actions: {[a.value for a in WorkspaceAction]}"
            )

        try:
            action = WorkspaceAction(action_str)
        except ValueError:
            valid = [a.value for a in WorkspaceAction]
            raise ValueError(f"Invalid workspace action '{action_str}'. Expected one of: {valid}")

        # 2. Resolve executor (default: docker; fails if Docker is unreachable)
        exec_type = str(
            parameters.get("executor_type")
            or inputs.get("executor_type")
            or self._default_executor_type
        ).lower()

        executor = self._resolve_executor(exec_type)

        # 3. Dispatch action
        if action == WorkspaceAction.READ_FILE:
            path = str(inputs.get("path") or parameters.get("path") or "")
            if not path:
                raise ValueError("Action 'read_file' requires parameter 'path'.")
            start_line = parameters.get("start_line") or inputs.get("start_line")
            end_line = parameters.get("end_line") or inputs.get("end_line")
            s_int = int(start_line) if start_line is not None else None
            e_int = int(end_line) if end_line is not None else None

            read_res = executor.read_file(path, start_line=s_int, end_line=e_int)
            return TaskResult(
                output=asdict(read_res),
                metadata={"action": action.value, "executor": type(executor).__name__},
            )

        elif action == WorkspaceAction.WRITE_FILE:
            path = str(inputs.get("path") or parameters.get("path") or "")
            if not path:
                raise ValueError("Action 'write_file' requires parameter 'path'.")
            content = str(inputs.get("content") if "content" in inputs else parameters.get("content", ""))
            overwrite = bool(parameters.get("overwrite", inputs.get("overwrite", True)))

            executor.write_file(path, content, overwrite=overwrite)
            return TaskResult(
                output={"status": "written", "path": path, "size_bytes": len(content.encode("utf-8"))},
                metadata={"action": action.value, "executor": type(executor).__name__},
            )

        elif action == WorkspaceAction.EDIT_FILE:
            path = str(inputs.get("path") or parameters.get("path") or "")
            target = str(inputs.get("target") if "target" in inputs else parameters.get("target", ""))
            replacement = str(inputs.get("replacement") if "replacement" in inputs else parameters.get("replacement", ""))
            if not path or not target:
                raise ValueError("Action 'edit_file' requires parameters 'path' and 'target'.")

            executor.edit_file(path, target=target, replacement=replacement)
            return TaskResult(
                output={"status": "edited", "path": path},
                metadata={"action": action.value, "executor": type(executor).__name__},
            )

        elif action == WorkspaceAction.LIST_DIR:
            path = str(inputs.get("path") or parameters.get("path") or ".")
            recursive = bool(parameters.get("recursive", inputs.get("recursive", False)))
            max_depth = int(parameters.get("max_depth", inputs.get("max_depth", 3)))

            entries = executor.list_dir(path=path, recursive=recursive, max_depth=max_depth)
            return TaskResult(
                output={"entries": entries, "count": len(entries)},
                metadata={"action": action.value, "executor": type(executor).__name__},
            )

        elif action == WorkspaceAction.RUN_COMMAND:
            cmd = str(inputs.get("command") or parameters.get("command") or "")
            if not cmd:
                raise ValueError("Action 'run_command' requires parameter 'command'.")
            working_dir = parameters.get("working_dir") or inputs.get("working_dir")
            timeout_s = float(parameters.get("timeout_seconds") or inputs.get("timeout_seconds") or 60.0)
            env = dict(parameters.get("environment") or inputs.get("environment") or {})

            req = WorkspaceCommandRequest(
                command=cmd,
                working_dir=str(working_dir) if working_dir else None,
                timeout_seconds=timeout_s,
                environment=env,
            )
            cmd_res = executor.run_command(req)

            fail_on_error = bool(parameters.get("fail_on_error", inputs.get("fail_on_error", False)))
            if fail_on_error and not cmd_res.success:
                raise RuntimeError(
                    f"Command '{cmd}' failed (exit code {cmd_res.exit_code}): {cmd_res.stderr or cmd_res.error}"
                )

            return TaskResult(
                output=asdict(cmd_res),
                metadata={
                    "action": action.value,
                    "executor": type(executor).__name__,
                    "exit_code": cmd_res.exit_code,
                    "success": cmd_res.success,
                },
            )

        else:
            raise ValueError(f"Unhandled action: {action}")
