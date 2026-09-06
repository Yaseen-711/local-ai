"""Unit tests for LlamaCppProvider multimodal payload formatting and serialization."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.common.types import FinishReason, ModelFormat
from core.inference.providers.llama_cpp import LlamaCppProvider
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    MediaAttachment,
    Message,
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
        capabilities=ModelCapabilities(chat=True, code=True),
    )


def test_build_payload_text_only(mock_model_def):
    """Text-only messages must serialize as standard string content."""
    provider = LlamaCppProvider()
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[
            Message.system("You are a helpful assistant."),
            Message.user("Hello world"),
        ],
    )
    payload = provider._build_payload(req, mock_model_def)

    assert payload["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello world"},
    ]


def test_build_payload_with_media_attachment(tmp_path: Path, mock_model_def):
    """Messages with attachments serialize into OpenAI-compatible content lists."""
    test_img = tmp_path / "diagram.png"
    test_img.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")

    attachment = MediaAttachment.from_file(test_img)
    provider = LlamaCppProvider()

    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[
            Message.user(
                content="Analyze this diagram",
                attachments=[attachment],
            )
        ],
    )
    payload = provider._build_payload(req, mock_model_def)

    assert len(payload["messages"]) == 1
    msg = payload["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 2

    # Check text item
    assert msg["content"][0] == {"type": "text", "text": "Analyze this diagram"}

    # Check image item
    img_item = msg["content"][1]
    assert img_item["type"] == "image_url"
    assert "image_url" in img_item
    assert img_item["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_payload_image_only(tmp_path: Path, mock_model_def):
    """Messages with empty text but with attachments serialize only image blocks."""
    test_img = tmp_path / "photo.jpg"
    test_img.write_bytes(b"\xff\xd8\xfffakejpeg")

    attachment = MediaAttachment.from_file(test_img)
    provider = LlamaCppProvider()

    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[
            Message.user(
                content="",
                attachments=[attachment],
            )
        ],
    )
    payload = provider._build_payload(req, mock_model_def)

    msg = payload["messages"][0]
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 1
    assert msg["content"][0]["type"] == "image_url"


@patch("urllib.request.urlopen")
def test_infer_multimodal_request(mock_urlopen, tmp_path: Path, mock_model_def):
    """End-to-end infer call correctly sends base64 payload to HTTP endpoint."""
    test_img = tmp_path / "drawing.png"
    test_img.write_bytes(b"\x89PNGmockbytes")
    attachment = MediaAttachment.from_file(test_img)

    mock_resp_data = {
        "id": "chatcmpl-vision-test",
        "object": "chat.completion",
        "created": 1788670000,
        "model": "qwen3.5-9b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Detected pump P-101 and valve V-301 in the drawing.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 20,
            "total_tokens": 170,
        },
    }

    resp_bytes = json.dumps(mock_resp_data).encode("utf-8")
    mock_http_response = MagicMock()
    mock_http_response.read.return_value = resp_bytes
    mock_http_response.status = 200
    mock_http_response.__enter__.return_value = mock_http_response
    mock_urlopen.return_value = mock_http_response

    provider = LlamaCppProvider(base_url="http://127.0.0.1:8080")
    req = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[
            Message.user(
                content="Identify equipment tags",
                attachments=[attachment],
            )
        ],
    )

    result = provider.infer(req, mock_model_def)

    assert result.finish_reason == FinishReason.STOP
    assert "P-101" in result.message.content
    assert result.usage.prompt_tokens == 150
    assert result.usage.completion_tokens == 20

    # Verify what was sent to urlopen
    called_request = mock_urlopen.call_args[0][0]
    sent_payload = json.loads(called_request.data.decode("utf-8"))
    assert sent_payload["messages"][0]["content"][0]["text"] == "Identify equipment tags"
    assert sent_payload["messages"][0]["content"][1]["type"] == "image_url"
