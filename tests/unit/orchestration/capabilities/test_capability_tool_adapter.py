"""Unit tests for CapabilityToolAdapter and per-call AgentExecutionPolicy."""

from typing import Any, Dict
from unittest.mock import MagicMock
import pytest

from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.builtin.agent.policy import (
    AgentExecutionPolicy,
    UnauthorizedCapabilityError,
)
from orchestration.capabilities.builtin.agent.tool_adapter import (
    CapabilityToolAdapter,
)
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskResult


class DummyCapability:
    def __init__(self, cid: str, return_refs: bool = False, return_arts: bool = False):
        self._cid = cid
        self.return_refs = return_refs
        self.return_arts = return_arts
        self.last_params: Dict[str, Any] = {}
        self.last_inputs: Dict[str, Any] = {}
        self.last_context: Any = None

    @property
    def capability_id(self) -> str:
        return self._cid

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        self.last_params = parameters
        self.last_inputs = inputs
        self.last_context = context

        refs = []
        if self.return_refs:
            refs.append(DataReference(key="doc_key", uri="file:///tmp/doc.json"))

        arts = []
        if self.return_arts:
            arts.append(
                ArtifactReference(
                    artifact_id="art-1",
                    name="report.pdf",
                    uri="file:///tmp/report.pdf",
                    mime_type="application/pdf",
                )
            )

        return TaskResult(
            output={"status": "success", "echo": inputs.get("text") or inputs.get("path")},
            references=refs,
            artifacts=arts,
        )


def test_tool_adapter_builds_approved_tools():
    """Verify only capabilities present in allowed_capabilities and registry are built."""
    registry = CapabilityRegistry()
    registry.register(DummyCapability("workflow.text_analysis"))
    registry.register(DummyCapability("document.understand"))
    registry.register(DummyCapability("code.workspace"))

    adapter = CapabilityToolAdapter(registry=registry)

    # Allowed only text_analysis
    tools = adapter.build_tools(allowed_capabilities={"workflow.text_analysis"})
    assert len(tools) == 1
    assert tools[0].name == "text_analysis"

    # Allowed text_analysis and document.understand
    tools2 = adapter.build_tools(
        allowed_capabilities={"workflow.text_analysis", "document.understand"}
    )
    assert len(tools2) == 2
    tool_names = {t.name for t in tools2}
    assert tool_names == {"text_analysis", "document_understand"}


def test_tool_adapter_rejects_unauthorized_capability_at_invocation():
    """Verify requesting an unauthorized capability at tool execution time raises UnauthorizedCapabilityError."""
    registry = CapabilityRegistry()
    cap = DummyCapability("workflow.text_analysis")
    registry.register(cap)

    policy = AgentExecutionPolicy()
    adapter = CapabilityToolAdapter(registry=registry, policy=policy)

    tools = adapter.build_tools(allowed_capabilities={"workflow.text_analysis"})
    tool_fn = tools[0].function

    # Calling with empty allowed_capabilities directly via internal executor
    with pytest.raises(UnauthorizedCapabilityError, match="Unauthorized capability invocation"):
        adapter._execute_capability_with_tracking(
            capability_id="workflow.text_analysis",
            parameters={},
            inputs={"text": "hi"},
            allowed_capabilities=set(),  # unauthorized
        )


def test_tool_adapter_argument_unpacking_and_context_isolation():
    """Verify tool arguments unpack properly into parameters/inputs and pass isolated context."""
    registry = CapabilityRegistry()
    cap = DummyCapability("workflow.text_analysis")
    registry.register(cap)

    adapter = CapabilityToolAdapter(registry=registry)
    parent_ctx = CapabilityContext(execution_id="parent-exec-123")

    tools = adapter.build_tools(
        allowed_capabilities={"workflow.text_analysis"},
        parent_context=parent_ctx,
    )
    assert len(tools) == 1
    tool_fn = tools[0].function

    res = tool_fn(text="sample text to analyze")
    assert res == {"status": "success", "echo": "sample text to analyze"}
    assert cap.last_inputs == {"text": "sample text to analyze"}
    assert cap.last_context.execution_id.startswith("agent-tool-")
    assert cap.last_context.metadata["parent_execution_id"] == "parent-exec-123"
    assert cap.last_context.metadata["capability_id"] == "workflow.text_analysis"


def test_tool_adapter_collects_artifacts_and_references():
    """Verify child artifacts and data references are tracked by adapter."""
    registry = CapabilityRegistry()
    cap1 = DummyCapability("document.understand", return_refs=True)
    cap2 = DummyCapability("artifact.generate", return_arts=True)
    registry.register(cap1)
    registry.register(cap2)

    adapter = CapabilityToolAdapter(registry=registry)
    tools = adapter.build_tools(
        allowed_capabilities={"document.understand", "artifact.generate"}
    )
    tools_by_name = {t.name: t.function for t in tools}

    # Execute document_understand
    tools_by_name["document_understand"](file_path="doc.pdf")
    assert len(adapter.collected_references) == 1
    assert adapter.collected_references[0].key == "doc_key"

    # Execute artifact_generate
    tools_by_name["artifact_generate"](artifact_type="pdf", filename="report.pdf")
    assert len(adapter.collected_artifacts) == 1
    assert adapter.collected_artifacts[0].name == "report.pdf"

    # Verify tool traces
    traces = adapter.tool_traces
    assert len(traces) == 2
    assert traces[0].tool_name == "document.understand"
    assert traces[0].success is True
    assert traces[1].tool_name == "artifact.generate"
    assert traces[1].success is True

    # Test reset_state
    adapter.reset_state()
    assert len(adapter.collected_references) == 0
    assert len(adapter.collected_artifacts) == 0
    assert len(adapter.tool_traces) == 0
