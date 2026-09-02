"""Unit tests for the Connector Layer."""

from unittest.mock import MagicMock
import pytest

from connectors import FoundationInferenceConnector, InferenceConnector
from core.common.errors import (
    InferenceError,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from core.common.types import FinishReason, MessageRole
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
    Message,
    TokenUsage,
)


def _make_mock_response(model_id: str = "qwen3.5-9b", text: str = "Test response") -> InferenceResponse:
    return InferenceResponse(
        request_id="req-123",
        model_id=model_id,
        message=Message(role=MessageRole.ASSISTANT, content=text),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        latency_ms=25.0,
    )


def test_inference_connector_protocol_compliance():
    """Verify that FoundationInferenceConnector satisfies the InferenceConnector Protocol."""
    mock_core = MagicMock()
    connector = FoundationInferenceConnector(core=mock_core)

    assert isinstance(connector, InferenceConnector)
    assert connector.core is mock_core


def test_custom_duck_typed_class_satisfies_protocol():
    """Verify that any duck-typed class with infer and infer_prompt satisfies InferenceConnector."""
    class CustomMockConnector:
        def infer(self, request: InferenceRequest) -> InferenceResponse:
            return _make_mock_response()

        def infer_prompt(
            self,
            model_id: str,
            prompt: str,
            system_prompt=None,
            options=None,
            request_id=None,
        ) -> InferenceResponse:
            return _make_mock_response()

    custom = CustomMockConnector()
    assert isinstance(custom, InferenceConnector)


def test_connector_infer_delegation():
    """Verify FoundationInferenceConnector.infer delegates faithfully to FoundationCore.infer."""
    mock_core = MagicMock()
    expected_resp = _make_mock_response(text="Delegated infer output")
    mock_core.infer.return_value = expected_resp

    connector = FoundationInferenceConnector(core=mock_core)
    request = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    result = connector.infer(request)

    assert result is expected_resp
    assert result.text == "Delegated infer output"
    mock_core.infer.assert_called_once_with(request)


def test_connector_infer_prompt_delegation():
    """Verify FoundationInferenceConnector.infer_prompt delegates faithfully to FoundationCore.infer_prompt."""
    mock_core = MagicMock()
    expected_resp = _make_mock_response(text="Delegated prompt output")
    mock_core.infer_prompt.return_value = expected_resp

    connector = FoundationInferenceConnector(core=mock_core)
    opts = GenerationOptions(temperature=0.3)

    result = connector.infer_prompt(
        model_id="default",
        prompt="Explain recursion",
        system_prompt="You are a tutor.",
        options=opts,
        request_id="req-custom-1",
    )

    assert result is expected_resp
    assert result.text == "Delegated prompt output"
    mock_core.infer_prompt.assert_called_once_with(
        model_id="default",
        prompt="Explain recursion",
        system_prompt="You are a tutor.",
        options=opts,
        request_id="req-custom-1",
    )


@pytest.mark.parametrize(
    "error_cls, error_msg",
    [
        (ModelNotFoundError, "Model 'unknown-model' is not configured."),
        (ProviderUnavailableError, "llama-server is offline."),
        (InferenceError, "Runtime generation error: context limit."),
    ],
)
def test_connector_transparent_error_propagation(error_cls, error_msg):
    """Verify that exceptions from FoundationCore propagate cleanly through the connector without being swallowed."""
    mock_core = MagicMock()
    mock_core.infer.side_effect = error_cls(error_msg)
    mock_core.infer_prompt.side_effect = error_cls(error_msg)

    connector = FoundationInferenceConnector(core=mock_core)
    request = InferenceRequest.from_prompt(model_id="any-model", prompt="Hello")

    # infer() propagates error
    with pytest.raises(error_cls, match=error_msg):
        connector.infer(request)

    # infer_prompt() propagates error
    with pytest.raises(error_cls, match=error_msg):
        connector.infer_prompt(model_id="any-model", prompt="Hello")
