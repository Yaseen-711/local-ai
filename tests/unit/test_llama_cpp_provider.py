"""Unit tests for LlamaCppProvider with mocked HTTP transport."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from core.common.errors import (
    InferenceError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from core.common.types import FinishReason, MessageRole, ModelFormat, RuntimeState
from core.inference.providers.llama_cpp import LlamaCppProvider
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    Message,
    OutputConstraint,
)
from core.models.schema import ModelCapabilities, ModelDefinition



@pytest.fixture
def mock_model_def() -> ModelDefinition:
    return ModelDefinition(
        id="qwen3.5-9b",
        display_name="Qwen 3.5 9B",
        format=ModelFormat.GGUF,
        relative_path=Path("models/gguf/test.gguf"),
        supported_providers=["llama_cpp"],
        aliases=["qwen3.5", "default"],
        capabilities=ModelCapabilities(chat=True, code=True),
    )


def test_provider_properties():
    provider = LlamaCppProvider(base_url="http://127.0.0.1:8080", timeout_seconds=30.0)
    assert provider.provider_name == "llama_cpp"
    assert provider.base_url == "http://127.0.0.1:8080"
    assert provider.timeout_seconds == 30.0


def test_build_payload(mock_model_def):
    provider = LlamaCppProvider()
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[
            Message.system("System instruction"),
            Message.user("User prompt"),
        ],
        options=GenerationOptions(
            temperature=0.5,
            top_p=0.9,
            max_tokens=256,
            stop_sequences=["\n\n"],
            seed=42,
            extra_options={"top_k": 40},
        ),
    )

    payload = provider._build_payload(req, mock_model_def)
    assert payload["model"] == "qwen3.5-9b"
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 256
    assert payload["stop"] == ["\n\n"]
    assert payload["seed"] == 42
    assert payload["top_k"] == 40
    assert payload["stream"] is False
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": "System instruction"}
    assert payload["messages"][1] == {"role": "user", "content": "User prompt"}


def test_build_payload_resolves_alias_to_runtime_id(mock_model_def):
    """Verify that a request using a Foundation alias resolves to the runtime-compatible model ID."""
    provider = LlamaCppProvider()
    # Caller requests via Foundation alias "default"
    req = InferenceRequest.from_prompt(model_id="default", prompt="Hello")

    payload = provider._build_payload(req, mock_model_def)
    # Must send canonical model_def.id ("qwen3.5-9b") to runtime, NOT Foundation alias "default"
    assert payload["model"] == "qwen3.5-9b"


def test_build_payload_uses_metadata_alias():
    """Verify that metadata['llama_cpp_alias'] is used if declared."""
    provider = LlamaCppProvider()
    custom_model_def = ModelDefinition(
        id="foundation-id-1",
        display_name="Custom Model",
        format=ModelFormat.GGUF,
        relative_path=Path("models/gguf/test.gguf"),
        supported_providers=["llama_cpp"],
        metadata={"llama_cpp_alias": "custom-server-alias"},
    )
    req = InferenceRequest.from_prompt(model_id="foundation-id-1", prompt="Hello")

    payload = provider._build_payload(req, custom_model_def)
    assert payload["model"] == "custom-server-alias"


@patch("urllib.request.urlopen")
def test_infer_success(mock_urlopen, mock_model_def):
    provider = LlamaCppProvider()
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    mock_response_data = {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    response = provider.infer(req, mock_model_def)

    assert response.model_id == "qwen3.5-9b"
    assert response.text == "Hello! How can I assist you?"
    assert response.finish_reason == FinishReason.STOP
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 8
    assert response.usage.total_tokens == 18
    assert response.latency_ms >= 0.0
    assert response.raw_response == mock_response_data


@patch("urllib.request.urlopen")
def test_infer_connection_refused(mock_urlopen, mock_model_def):
    provider = LlamaCppProvider()
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    with pytest.raises(ProviderUnavailableError, match="Failed to connect to llama-server"):
        provider.infer(req, mock_model_def)


@patch("urllib.request.urlopen")
def test_infer_http_error(mock_urlopen, mock_model_def):
    provider = LlamaCppProvider()
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    err_fp = io.BytesIO(b"Internal Server Error: context limit exceeded")
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="http://127.0.0.1:8080/v1/chat/completions",
        code=500,
        msg="Internal Error",
        hdrs={},
        fp=err_fp,
    )

    with pytest.raises(InferenceError, match="llama-server returned HTTP error 500"):
        provider.infer(req, mock_model_def)


@patch("urllib.request.urlopen")
def test_infer_malformed_json(mock_urlopen, mock_model_def):
    provider = LlamaCppProvider()
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"NOT VALID JSON"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    with pytest.raises(ProviderResponseError, match="Failed to parse JSON response"):
        provider.infer(req, mock_model_def)


@patch("urllib.request.urlopen")
def test_infer_missing_choices(mock_urlopen, mock_model_def):
    provider = LlamaCppProvider()
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"choices": []}'
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    with pytest.raises(ProviderResponseError, match="missing or empty 'choices'"):
        provider.infer(req, mock_model_def)


@patch("urllib.request.urlopen")
def test_check_health_ready(mock_urlopen):
    provider = LlamaCppProvider()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    state = provider.check_health()
    assert state == RuntimeState.READY


@patch("urllib.request.urlopen")
def test_check_health_unavailable(mock_urlopen):
    provider = LlamaCppProvider()
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    state = provider.check_health()
    assert state == RuntimeState.UNAVAILABLE


@patch("urllib.request.urlopen")
def test_is_model_loaded(mock_urlopen, mock_model_def):
    provider = LlamaCppProvider()

    # Server reporting "qwen3.5-9b"
    mock_payload = {"data": [{"id": "qwen3.5-9b"}]}
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    assert provider.is_model_loaded(mock_model_def) is True

    # Server reporting a completely different model
    mock_payload_other = {"data": [{"id": "other-model-7b"}]}
    mock_resp.read.return_value = json.dumps(mock_payload_other).encode("utf-8")
    assert provider.is_model_loaded(mock_model_def) is False


def test_build_payload_with_json_constraint(mock_model_def):
    """Verify that OutputConstraint.json() adds response_format json_object to payload."""
    provider = LlamaCppProvider()
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[Message.user("Generate JSON")],
        options=GenerationOptions(constraint=OutputConstraint.json()),
    )
    payload = provider._build_payload(req, mock_model_def)
    assert payload.get("response_format") == {"type": "json_object"}
    assert "grammar" not in payload


def test_build_payload_with_grammar_constraint(mock_model_def):
    """Verify that OutputConstraint.from_grammar() adds grammar string to payload."""
    provider = LlamaCppProvider()
    grammar_str = 'root ::= "yes" | "no"'
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[Message.user("Binary choice")],
        options=GenerationOptions(constraint=OutputConstraint.from_grammar(grammar_str)),
    )
    payload = provider._build_payload(req, mock_model_def)
    assert payload.get("grammar") == grammar_str
    assert "response_format" not in payload



def test_build_payload_without_constraint_has_no_response_format(mock_model_def):
    """Verify that default unconstrained generation does not inject response_format or grammar."""
    provider = LlamaCppProvider()
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[Message.user("Explain recursion")],
        options=GenerationOptions(constraint=None),
    )
    payload = provider._build_payload(req, mock_model_def)
    assert "response_format" not in payload
    assert "grammar" not in payload
