"""Unit tests for FoundationPydanticAIModel multimodal message translation."""

import asyncio
import base64
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from core.common.types import FinishReason, MessageRole
from core.inference.types import InferenceResponse, Message, TokenUsage
from orchestration.capabilities.builtin.agent.model_adapter import (
    FoundationPydanticAIModel,
)
from orchestration.routing.model_selector import ModelSelectionPolicy
from orchestration.routing.types import ModelTier


class MockBinaryContent:
    """Simulate PydanticAI BinaryContent or similar media part."""
    def __init__(self, data: bytes, media_type: str):
        self.data = data
        self.media_type = media_type


class MockImageUrl:
    """Simulate PydanticAI ImageUrl part."""
    def __init__(self, url: str):
        self.url = url


def create_test_policy() -> ModelSelectionPolicy:
    mock_registry = MagicMock()
    mock_registry.is_known.side_effect = lambda m: m in {"qwen3.5-0.8b", "qwen3.5-9b"}
    mock_registry.get_model.side_effect = lambda m: MagicMock(id=m)
    return ModelSelectionPolicy(registry=mock_registry)


def test_agent_model_adapter_translates_binary_content():
    """Verify that BinaryContent parts are converted to MediaAttachment on the user message."""
    async def _run():
        mock_connector = MagicMock()
        mock_connector.infer.return_value = InferenceResponse(
            request_id="req-vision-1",
            model_id="qwen3.5-9b",
            message=Message(role=MessageRole.ASSISTANT, content="I see an engineering diagram."),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
            latency_ms=30.0,
        )

        policy = create_test_policy()
        model = FoundationPydanticAIModel(
            connector=mock_connector,
            model_policy=policy,
            default_tier=ModelTier.REASONING,
        )

        sample_bytes = b"\x89PNG\r\n\x1a\nsampledata"
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "What is in this diagram?",
                            MockBinaryContent(data=sample_bytes, media_type="image/png"),
                        ]
                    )
                ]
            )
        ]
        params = ModelRequestParameters()
        resp = await model.request(messages, None, params)

        assert resp.model_name == "qwen3.5-9b"
        assert resp.parts[0].content == "I see an engineering diagram."

        # Inspect the InferenceRequest passed to connector
        called_req = mock_connector.infer.call_args[0][0]
        assert len(called_req.messages) == 1
        user_msg = called_req.messages[0]
        assert user_msg.role == MessageRole.USER
        assert "What is in this diagram?" in user_msg.content
        assert len(user_msg.attachments) == 1
        att = user_msg.attachments[0]
        assert att.mime_type == "image/png"
        assert att.load_bytes() == sample_bytes

    asyncio.run(_run())


def test_agent_model_adapter_translates_image_data_url():
    """Verify that data: URL ImageUrl parts are decoded into MediaAttachment."""
    async def _run():
        mock_connector = MagicMock()
        mock_connector.infer.return_value = InferenceResponse(
            request_id="req-vision-2",
            model_id="qwen3.5-9b",
            message=Message(role=MessageRole.ASSISTANT, content="Identified P&ID valve V-301."),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
            latency_ms=25.0,
        )

        policy = create_test_policy()
        model = FoundationPydanticAIModel(
            connector=mock_connector,
            model_policy=policy,
            default_tier=ModelTier.REASONING,
        )

        sample_raw = b"\xff\xd8\xffsamplejpeg"
        b64_str = base64.b64encode(sample_raw).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64_str}"

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "Inspect image",
                            MockImageUrl(url=data_url),
                        ]
                    )
                ]
            )
        ]
        params = ModelRequestParameters()
        await model.request(messages, None, params)

        called_req = mock_connector.infer.call_args[0][0]
        user_msg = called_req.messages[0]
        assert len(user_msg.attachments) == 1
        att = user_msg.attachments[0]
        assert att.mime_type == "image/jpeg"
        assert att.load_bytes() == sample_raw

    asyncio.run(_run())


def test_agent_model_adapter_translates_local_file_url(tmp_path: Path):
    """Verify that file path ImageUrl parts are referenced via MediaAttachment.from_file."""
    async def _run():
        test_img = tmp_path / "equipment.png"
        test_img.write_bytes(b"\x89PNG\r\n\x1a\nlocalimage")

        mock_connector = MagicMock()
        mock_connector.infer.return_value = InferenceResponse(
            request_id="req-vision-3",
            model_id="qwen3.5-9b",
            message=Message(role=MessageRole.ASSISTANT, content="Pump P-101 detected."),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=40, completion_tokens=8, total_tokens=48),
            latency_ms=20.0,
        )

        policy = create_test_policy()
        model = FoundationPydanticAIModel(
            connector=mock_connector,
            model_policy=policy,
            default_tier=ModelTier.REASONING,
        )

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "Check this file",
                            MockImageUrl(url=f"file://{test_img.resolve()}"),
                        ]
                    )
                ]
            )
        ]
        params = ModelRequestParameters()
        await model.request(messages, None, params)

        called_req = mock_connector.infer.call_args[0][0]
        user_msg = called_req.messages[0]
        assert len(user_msg.attachments) == 1
        att = user_msg.attachments[0]
        assert att.mime_type == "image/png"
        assert att.source_path == str(test_img.resolve())
        assert att.load_bytes() == b"\x89PNG\r\n\x1a\nlocalimage"

    asyncio.run(_run())
