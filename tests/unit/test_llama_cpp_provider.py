"""Unit tests for LlamaCppProvider with mocked HTTP transport."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from core.common.errors import (
    ConfigurationError,
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


def test_base_url_scheme_validation():
    """Verify base_url scheme validation in LlamaCppProvider."""
    # Valid http and https
    p1 = LlamaCppProvider(base_url="http://127.0.0.1:8080")
    assert p1.base_url == "http://127.0.0.1:8080"

    p2 = LlamaCppProvider(base_url="https://remote-llm.internal:8443")
    assert p2.base_url == "https://remote-llm.internal:8443"

    # Invalid schemes and formats
    with pytest.raises(ConfigurationError, match="requires an 'http://' or 'https://' URL"):
        LlamaCppProvider(base_url="file:///etc/passwd")

    with pytest.raises(ConfigurationError, match="requires an 'http://' or 'https://' URL"):
        LlamaCppProvider(base_url="ftp://example.com")

    with pytest.raises(ConfigurationError, match="requires an 'http://' or 'https://' URL"):
        LlamaCppProvider(base_url="")


def test_max_response_bytes_validation():
    """Verify max_response_bytes validation at provider boundary."""
    p = LlamaCppProvider(max_response_bytes=1024)
    assert p.max_response_bytes == 1024

    with pytest.raises(ConfigurationError, match="max_response_bytes must be a positive integer"):
        LlamaCppProvider(max_response_bytes=0)

    with pytest.raises(ConfigurationError, match="max_response_bytes must be a positive integer"):
        LlamaCppProvider(max_response_bytes=-100)

    with pytest.raises(ConfigurationError, match="max_response_bytes must be a positive integer"):
        LlamaCppProvider(max_response_bytes=True)  # type: ignore[arg-type]


def test_extra_options_reserved_keys_rejected(mock_model_def):
    """Verify that passing reserved normalized keys in extra_options raises InferenceError."""
    provider = LlamaCppProvider()

    reserved_samples = ["model", "messages", "stream", "temperature", "response_format", "grammar"]
    for key in reserved_samples:
        req = InferenceRequest(
            model_id="qwen3.5-9b",
            messages=[Message.user("Hello")],
            options=GenerationOptions(extra_options={key: "rogue_value"}),
        )
        with pytest.raises(InferenceError, match="extra_options cannot override normalized parameter"):
            provider._build_payload(req, mock_model_def)


def test_extra_options_unreserved_keys_allowed(mock_model_def):
    """Verify that unreserved backend-specific keys in extra_options are merged cleanly."""
    provider = LlamaCppProvider()
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[Message.user("Hello")],
        options=GenerationOptions(extra_options={"mirostat": 2, "mirostat_tau": 5.0}),
    )
    payload = provider._build_payload(req, mock_model_def)
    assert payload.get("mirostat") == 2
    assert payload.get("mirostat_tau") == 5.0


def test_unsupported_constraint_format_rejected(mock_model_def):
    """Verify that an unsupported OutputConstraint format raises InferenceError."""
    provider = LlamaCppProvider()
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[Message.user("Hello")],
        options=GenerationOptions(constraint=OutputConstraint(format="yaml")),
    )
    with pytest.raises(InferenceError, match="does not support OutputConstraint format 'yaml'"):
        provider._build_payload(req, mock_model_def)


def test_empty_grammar_constraint_rejected(mock_model_def):
    """Verify that grammar constraint with empty text raises InferenceError."""
    provider = LlamaCppProvider()
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[Message.user("Hello")],
        options=GenerationOptions(constraint=OutputConstraint(format="grammar", grammar="")),
    )
    with pytest.raises(InferenceError, match="requires non-empty grammar text"):
        provider._build_payload(req, mock_model_def)


@patch("urllib.request.urlopen")
def test_bounded_response_size_exceeded_raises_provider_response_error(mock_urlopen, mock_model_def):
    """Verify that responses exceeding max_response_bytes raise ProviderResponseError."""
    provider = LlamaCppProvider(max_response_bytes=50)
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    # Simulate response stream providing 51 bytes when limit is 50
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"x" * 51
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    with pytest.raises(ProviderResponseError, match="exceeded maximum allowed size of 50 bytes"):
        provider.infer(req, mock_model_def)


@patch("urllib.request.urlopen")
def test_token_usage_absent_and_none_defaults_to_zero(mock_urlopen, mock_model_def):
    """Verify that absent or null token usage fields normalize safely to 0."""
    provider = LlamaCppProvider()
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    # Case 1: usage dict has None values
    payload1 = {
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": None, "completion_tokens": None},
    }
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(payload1).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    resp1 = provider.infer(req, mock_model_def)
    assert resp1.usage.prompt_tokens == 0
    assert resp1.usage.completion_tokens == 0
    assert resp1.usage.total_tokens == 0

    # Case 2: usage key is completely missing
    payload2 = {
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    }
    mock_resp.read.return_value = json.dumps(payload2).encode("utf-8")
    resp2 = provider.infer(req, mock_model_def)
    assert resp2.usage.prompt_tokens == 0
    assert resp2.usage.completion_tokens == 0
    assert resp2.usage.total_tokens == 0


@patch("urllib.request.urlopen")
def test_token_usage_corrupt_non_numeric_raises_provider_response_error(mock_urlopen, mock_model_def):
    """Verify that corrupt non-numeric token usage raises ProviderResponseError instead of silent coercion."""
    provider = LlamaCppProvider()
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Hello")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    mock_urlopen.return_value = mock_resp

    # Corrupt string value
    payload_corrupt_str = {
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": "corrupted_non_int"},
    }
    mock_resp.read.return_value = json.dumps(payload_corrupt_str).encode("utf-8")
    with pytest.raises(ProviderResponseError, match="Malformed non-numeric token usage"):
        provider.infer(req, mock_model_def)

    # Boolean value
    payload_bool = {
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": True},
    }
    mock_resp.read.return_value = json.dumps(payload_bool).encode("utf-8")
    with pytest.raises(ProviderResponseError, match="Invalid boolean value for token usage"):
        provider.infer(req, mock_model_def)

    # Negative integer value
    payload_negative = {
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -5},
    }
    mock_resp.read.return_value = json.dumps(payload_negative).encode("utf-8")
    with pytest.raises(ProviderResponseError, match="Negative token usage value"):
        provider.infer(req, mock_model_def)
