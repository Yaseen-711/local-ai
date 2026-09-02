"""Unit tests for workflow layer contracts and conventions."""

from unittest.mock import MagicMock

import pytest

from connectors import InferenceConnector
from core.common.errors import (
    FoundationError,
    InferenceError,
    ModelRegistryError,
    ProviderError,
    ProviderUnavailableError,
    WorkflowError,
)
from core.common.types import FinishReason, MessageRole
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
    Message,
    TokenUsage,
)
from workflows import WorkflowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(
    model_id: str = "qwen3.5-9b",
    text: str = "Mock output",
) -> InferenceResponse:
    return InferenceResponse(
        request_id="req-wf-1",
        model_id=model_id,
        message=Message(role=MessageRole.ASSISTANT, content=text),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        latency_ms=20.0,
    )


class _ExampleWorkflow:
    """Minimal example workflow for testing conventions.

    Receives an InferenceConnector via constructor injection.
    """

    def __init__(self, inference: InferenceConnector) -> None:
        self._inference = inference

    def analyze(self, code: str) -> WorkflowResult[str]:
        response = self._inference.infer_prompt(
            model_id="default",
            prompt=f"Analyze this code:\n{code}",
            options=GenerationOptions(temperature=0.1),
        )
        return WorkflowResult(
            output=response.text,
            model_id=response.model_id,
            metadata={"latency_ms": response.latency_ms},
        )

    def analyze_multi_step(self, code: str) -> WorkflowResult[str]:
        """Multi-step workflow: first analyze, then summarize."""
        step1 = self._inference.infer_prompt(
            model_id="default",
            prompt=f"List issues in this code:\n{code}",
        )
        step2 = self._inference.infer_prompt(
            model_id="default",
            prompt=f"Summarize these findings:\n{step1.text}",
        )
        return WorkflowResult(
            output=step2.text,
            model_id=step2.model_id,
            metadata={"steps_executed": 2},
        )

    def analyze_with_recovery(self, code: str) -> WorkflowResult[str]:
        """Workflow that catches infrastructure errors and wraps them."""
        try:
            response = self._inference.infer_prompt(
                model_id="default",
                prompt=f"Analyze:\n{code}",
            )
            return WorkflowResult(output=response.text, model_id=response.model_id)
        except ProviderUnavailableError as exc:
            raise WorkflowError(
                f"Code analysis failed: inference backend unavailable"
            ) from exc


# ---------------------------------------------------------------------------
# Tests: WorkflowResult construction
# ---------------------------------------------------------------------------

def test_workflow_result_construction():
    """Verify WorkflowResult can be constructed with all fields."""
    result: WorkflowResult[str] = WorkflowResult(
        output="Review complete",
        model_id="qwen3.5-9b",
        metadata={"confidence": 0.95, "language": "python"},
        errors=["Minor: unused import detected"],
    )
    assert result.output == "Review complete"
    assert result.model_id == "qwen3.5-9b"
    assert result.metadata == {"confidence": 0.95, "language": "python"}
    assert result.errors == ["Minor: unused import detected"]


def test_workflow_result_with_defaults():
    """Verify WorkflowResult defaults: metadata={}, errors=[], model_id=None."""
    result: WorkflowResult[int] = WorkflowResult(output=42)
    assert result.output == 42
    assert result.model_id is None
    assert result.metadata == {}
    assert result.errors == []


# ---------------------------------------------------------------------------
# Tests: WorkflowError hierarchy
# ---------------------------------------------------------------------------

def test_workflow_error_hierarchy():
    """Verify WorkflowError is a FoundationError but NOT a ProviderError or ModelRegistryError."""
    err = WorkflowError("workflow failed")
    assert isinstance(err, FoundationError)
    assert isinstance(err, WorkflowError)
    assert not isinstance(err, ProviderError)
    assert not isinstance(err, ModelRegistryError)


def test_workflow_error_wraps_cause():
    """Verify WorkflowError can wrap an infrastructure error via exception chaining."""
    infra_error = InferenceError("GPU out of memory")
    try:
        raise WorkflowError("Analysis step failed") from infra_error
    except WorkflowError as wf_err:
        assert str(wf_err) == "Analysis step failed"
        assert wf_err.__cause__ is infra_error
        assert isinstance(wf_err.__cause__, InferenceError)


# ---------------------------------------------------------------------------
# Tests: Constructor injection and connector usage
# ---------------------------------------------------------------------------

def test_workflow_receives_connector_via_constructor():
    """Verify a workflow stores and can call a connector received via __init__."""
    mock_core = MagicMock()
    mock_core.infer_prompt.return_value = _make_mock_response(text="Analysis output")

    from connectors import FoundationInferenceConnector
    connector = FoundationInferenceConnector(core=mock_core)

    workflow = _ExampleWorkflow(inference=connector)
    result = workflow.analyze("def foo(): pass")

    assert result.output == "Analysis output"
    assert result.model_id == "qwen3.5-9b"
    assert "latency_ms" in result.metadata
    mock_core.infer_prompt.assert_called_once()


def test_workflow_with_fake_connector():
    """Verify a workflow works with a duck-typed fake connector (no inheritance)."""

    class FakeConnector:
        def __init__(self):
            self.calls = []

        def infer(self, request: InferenceRequest) -> InferenceResponse:
            self.calls.append(("infer", request))
            return _make_mock_response(text="Fake infer output")

        def infer_prompt(self, model_id, prompt, system_prompt=None,
                         options=None, request_id=None) -> InferenceResponse:
            self.calls.append(("infer_prompt", model_id, prompt))
            return _make_mock_response(text="Fake prompt output")

    fake = FakeConnector()
    # Verify structural typing
    assert isinstance(fake, InferenceConnector)

    workflow = _ExampleWorkflow(inference=fake)
    result = workflow.analyze("x = 1")

    assert result.output == "Fake prompt output"
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "infer_prompt"


def test_workflow_repeated_connector_calls():
    """Verify a workflow can call infer_prompt() multiple times (multi-step)."""
    mock_core = MagicMock()
    mock_core.infer_prompt.side_effect = [
        _make_mock_response(text="Step 1: found 3 issues"),
        _make_mock_response(text="Summary: 3 issues found"),
    ]

    from connectors import FoundationInferenceConnector
    connector = FoundationInferenceConnector(core=mock_core)

    workflow = _ExampleWorkflow(inference=connector)
    result = workflow.analyze_multi_step("buggy code")

    assert result.output == "Summary: 3 issues found"
    assert result.metadata["steps_executed"] == 2
    assert mock_core.infer_prompt.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Error propagation and wrapping
# ---------------------------------------------------------------------------

def test_workflow_infrastructure_error_propagates():
    """Verify infrastructure errors propagate through a workflow that doesn't catch them."""
    mock_core = MagicMock()
    mock_core.infer_prompt.side_effect = ProviderUnavailableError("server down")

    from connectors import FoundationInferenceConnector
    connector = FoundationInferenceConnector(core=mock_core)

    workflow = _ExampleWorkflow(inference=connector)

    # analyze() does NOT catch ProviderUnavailableError, so it must propagate
    with pytest.raises(ProviderUnavailableError, match="server down"):
        workflow.analyze("some code")


def test_workflow_catches_and_wraps_infrastructure_error():
    """Verify a workflow CAN catch infrastructure errors and wrap in WorkflowError."""
    mock_core = MagicMock()
    mock_core.infer_prompt.side_effect = ProviderUnavailableError("backend offline")

    from connectors import FoundationInferenceConnector
    connector = FoundationInferenceConnector(core=mock_core)

    workflow = _ExampleWorkflow(inference=connector)

    with pytest.raises(WorkflowError, match="inference backend unavailable") as exc_info:
        workflow.analyze_with_recovery("some code")

    # Verify the original cause is preserved
    assert isinstance(exc_info.value.__cause__, ProviderUnavailableError)
    assert "backend offline" in str(exc_info.value.__cause__)
