"""Capability-to-tool adapter bridging CapabilityRegistry to PydanticAI tools.

Exposes explicitly authorized capabilities as PydanticAI tools with strongly-typed
signatures, per-call policy authorization, execution context isolation, and provenance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from pydantic_ai.tools import Tool

from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.builtin.agent.policy import (
    AgentExecutionPolicy,
    UnauthorizedCapabilityError,
)
from orchestration.capabilities.builtin.agent.types import AgentToolCallRecord
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskResult

logger = logging.getLogger(__name__)


class CapabilityToolAdapter:
    """Bridges Foundation capabilities to PydanticAI tools with per-call authorization.

    Ensures that existing capabilities never import PydanticAI. Evaluates AgentExecutionPolicy
    dynamically on every tool invocation before dispatching to Capability.execute().
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        policy: Optional[AgentExecutionPolicy] = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or AgentExecutionPolicy()
        self._collected_references: List[DataReference] = []
        self._collected_artifacts: List[ArtifactReference] = []
        self._tool_traces: List[AgentToolCallRecord] = []

    @property
    def collected_references(self) -> List[DataReference]:
        """All DataReferences emitted by child capabilities during this run."""
        return list(self._collected_references)

    @property
    def collected_artifacts(self) -> List[ArtifactReference]:
        """All ArtifactReferences emitted by child capabilities during this run."""
        return list(self._collected_artifacts)

    @property
    def tool_traces(self) -> List[AgentToolCallRecord]:
        """Diagnostic trace records of all executed tool calls."""
        return list(self._tool_traces)

    def reset_state(self) -> None:
        """Reset execution accumulation state for a new agent task."""
        self._collected_references = []
        self._collected_artifacts = []
        self._tool_traces = []

    def _execute_capability_with_tracking(
        self,
        capability_id: str,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        allowed_capabilities: Set[str],
        parent_context: Optional[CapabilityContext] = None,
    ) -> Any:
        """Evaluate policy, construct isolated context, execute capability, and track telemetry."""
        # 1. Per-tool-call authorization check
        self._policy.authorize_tool_call(capability_id, allowed_capabilities)

        # 2. Lookup capability from registry
        capability: Capability = self._registry.get(capability_id)

        # 3. Construct isolated CapabilityContext
        parent_id = parent_context.execution_id if parent_context else "direct"
        child_execution_id = f"agent-tool-{uuid.uuid4().hex[:8]}"
        context = CapabilityContext(
            execution_id=child_execution_id,
            metadata={"parent_execution_id": parent_id, "capability_id": capability_id},
        )

        # 4. Execute capability
        try:
            result: TaskResult = capability.execute(
                parameters=parameters,
                inputs=inputs,
                context=context,
            )
            # 5. Collect outputs, references, and artifacts for provenance
            if result.references:
                self._collected_references.extend(result.references)
            if result.artifacts:
                self._collected_artifacts.extend(result.artifacts)

            output_summary = str(result.output)[:200] if result.output is not None else "None"
            self._tool_traces.append(
                AgentToolCallRecord(
                    tool_name=capability_id,
                    capability_id=capability_id,
                    arguments={"parameters": parameters, "inputs": inputs},
                    output_summary=output_summary,
                    execution_id=child_execution_id,
                    success=True,
                )
            )
            return result.output
        except Exception as exc:
            self._tool_traces.append(
                AgentToolCallRecord(
                    tool_name=capability_id,
                    capability_id=capability_id,
                    arguments={"parameters": parameters, "inputs": inputs},
                    output_summary="Failed",
                    execution_id=child_execution_id,
                    success=False,
                    error=str(exc),
                )
            )
            raise

    def build_tools(
        self,
        allowed_capabilities: Set[str],
        parent_context: Optional[CapabilityContext] = None,
    ) -> List[Tool]:
        """Build PydanticAI tools for all authorized capabilities present in the registry."""
        tools: List[Tool] = []

        # 1. code.workspace
        if "code.workspace" in allowed_capabilities and self._registry.has("code.workspace"):
            def code_workspace(
                action: str,
                path: str = "",
                content: str = "",
                target: str = "",
                replacement: str = "",
                command: str = "",
                recursive: bool = False,
                max_depth: int = 3,
                start_line: Optional[int] = None,
                end_line: Optional[int] = None,
                timeout_seconds: float = 60.0,
            ) -> Any:
                """Execute isolated workspace coding action inside Docker sandbox.

                Actions:
                  - 'read_file': read content from path (optional: start_line, end_line)
                  - 'write_file': write content to path
                  - 'edit_file': replace target string with replacement in path
                  - 'list_dir': list entries in path (optional: recursive, max_depth)
                  - 'run_command': execute command in sandbox workspace (optional: timeout_seconds)
                """
                params = {
                    "action": action,
                    "recursive": recursive,
                    "max_depth": max_depth,
                    "start_line": start_line,
                    "end_line": end_line,
                    "timeout_seconds": timeout_seconds,
                }
                inputs = {
                    "path": path,
                    "content": content,
                    "target": target,
                    "replacement": replacement,
                    "command": command,
                }
                return self._execute_capability_with_tracking(
                    capability_id="code.workspace",
                    parameters=params,
                    inputs=inputs,
                    allowed_capabilities=allowed_capabilities,
                    parent_context=parent_context,
                )

            tools.append(
                Tool(
                    code_workspace,
                    name="code_workspace",
                    description="Execute isolated workspace actions (read_file, write_file, edit_file, list_dir, run_command) inside Docker.",
                )
            )

        # 2. document.understand
        if "document.understand" in allowed_capabilities and self._registry.has("document.understand"):
            def document_understand(
                file_path: str,
                do_ocr: bool = True,
                extract_tables: bool = True,
                extract_figures: bool = False,
                max_pages: Optional[int] = None,
            ) -> Any:
                """Parse document file (PDF, DOCX, etc.) with layout analysis and OCR."""
                params = {
                    "do_ocr": do_ocr,
                    "extract_tables": extract_tables,
                    "extract_figures": extract_figures,
                    "max_pages": max_pages,
                }
                inputs = {"file_path": file_path}
                return self._execute_capability_with_tracking(
                    capability_id="document.understand",
                    parameters=params,
                    inputs=inputs,
                    allowed_capabilities=allowed_capabilities,
                    parent_context=parent_context,
                )

            tools.append(
                Tool(
                    document_understand,
                    name="document_understand",
                    description="Parse documents with layout analysis, table extraction, and OCR.",
                )
            )

        # 3. artifact.generate
        if "artifact.generate" in allowed_capabilities and self._registry.has("artifact.generate"):
            def artifact_generate(
                artifact_type: str,
                filename: Optional[str] = None,
                title: Optional[str] = None,
                data: Optional[Any] = None,
                content: Optional[str] = None,
                template: Optional[str] = None,
                template_data: Optional[Any] = None,
            ) -> Any:
                """Generate deterministic XLSX, DOCX, PPTX, or PDF binary artifact with SHA-256 provenance."""
                params: Dict[str, Any] = {
                    "artifact_type": artifact_type,
                    "filename": filename,
                    "title": title,
                }
                if template:
                    params["template"] = template
                inputs: Dict[str, Any] = {
                    "data": data,
                    "content": content,
                }
                if template_data is not None:
                    inputs["template_data"] = template_data
                return self._execute_capability_with_tracking(
                    capability_id="artifact.generate",
                    parameters=params,
                    inputs=inputs,
                    allowed_capabilities=allowed_capabilities,
                    parent_context=parent_context,
                )

            tools.append(
                Tool(
                    artifact_generate,
                    name="artifact_generate",
                    description="Generate deterministic XLSX, DOCX, PPTX, or PDF files from structured data, markdown, or industrial templates.",
                )
            )

        # 4. workflow.text_analysis
        if "workflow.text_analysis" in allowed_capabilities and self._registry.has("workflow.text_analysis"):
            def text_analysis(text: str) -> Any:
                """Perform structured multi-step text analysis (summary, key points, sentiments)."""
                params: Dict[str, Any] = {}
                inputs = {"text": text}
                return self._execute_capability_with_tracking(
                    capability_id="workflow.text_analysis",
                    parameters=params,
                    inputs=inputs,
                    allowed_capabilities=allowed_capabilities,
                    parent_context=parent_context,
                )

            tools.append(
                Tool(
                    text_analysis,
                    name="text_analysis",
                    description="Run structured multi-step text analysis on a text payload.",
                )
            )

        # 5. code.verify_and_repair
        if "code.verify_and_repair" in allowed_capabilities and self._registry.has("code.verify_and_repair"):
            def code_verify_and_repair(
                prompt: str,
                category: str = "general_code",
                assertions: Optional[List[Dict[str, Any]]] = None,
                max_repair_attempts: int = 3,
                timeout_seconds: float = 30.0,
            ) -> Any:
                """Execute bounded code generation, isolated execution, testing, and repair cycle."""
                params = {
                    "category": category,
                    "max_repair_attempts": max_repair_attempts,
                    "timeout_seconds": timeout_seconds,
                }
                inputs = {
                    "prompt": prompt,
                    "assertions": assertions or [],
                }
                return self._execute_capability_with_tracking(
                    capability_id="code.verify_and_repair",
                    parameters=params,
                    inputs=inputs,
                    allowed_capabilities=allowed_capabilities,
                    parent_context=parent_context,
                )

            tools.append(
                Tool(
                    code_verify_and_repair,
                    name="code_verify_and_repair",
                    description="Generate, sandbox-execute, test, and repair code with bounded retries and objective verification.",
                )
            )

        return tools
