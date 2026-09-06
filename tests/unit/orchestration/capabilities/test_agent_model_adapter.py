"""Unit tests for FoundationPydanticAIModel adapter."""

import asyncio
from unittest.mock import MagicMock
import pytest

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from core.common.errors import ProviderResponseError
from core.common.types import FinishReason, MessageRole
from core.inference.types import InferenceRequest, InferenceResponse, Message, TokenUsage
from core.models.registry import ModelRegistry
from core.models.schema import ModelDefinition
from orchestration.capabilities.builtin.agent.model_adapter import (
    FoundationPydanticAIModel,
)
from orchestration.routing.model_selector import ModelSelectionPolicy
from orchestration.routing.types import ModelTier


def create_test_policy() -> ModelSelectionPolicy:
    mock_registry = MagicMock()
    mock_registry.is_known.side_effect = lambda m: m in {"qwen3.5-0.8b", "qwen3.5-9b"}
    mock_registry.get_model.side_effect = lambda m: MagicMock(id=m)
    return ModelSelectionPolicy(registry=mock_registry)


def test_model_adapter_request_response_translation():
    """Verify single-turn translation between PydanticAI messages and InferenceRequest/Response."""
    async def _run():
        mock_connector = MagicMock()
        mock_connector.infer.return_value = InferenceResponse(
            request_id="req-1",
            model_id="qwen3.5-9b",
            message=Message(role=MessageRole.ASSISTANT, content="Hello back!"),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=15, completion_tokens=8, total_tokens=23),
            latency_ms=45.0,
        )

        policy = create_test_policy()
        model = FoundationPydanticAIModel(
            connector=mock_connector,
            model_policy=policy,
            default_tier=ModelTier.REASONING,
        )

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello AI")]),
        ]
        params = ModelRequestParameters()
        resp = await model.request(messages, None, params)

        assert resp.model_name == "qwen3.5-9b"
        assert len(resp.parts) == 1
        assert isinstance(resp.parts[0], TextPart)
        assert resp.parts[0].content == "Hello back!"
        assert resp.usage.input_tokens == 15
        assert resp.usage.output_tokens == 8

        # Verify what was passed to mock_connector.infer
        mock_connector.infer.assert_called_once()
        inf_req: InferenceRequest = mock_connector.infer.call_args[0][0]
        assert inf_req.model_id == "qwen3.5-9b"
        assert len(inf_req.messages) == 1
        assert inf_req.messages[0].role == MessageRole.USER
        assert inf_req.messages[0].content == "Hello AI"

    asyncio.run(_run())


def test_model_adapter_multi_turn_tool_translation():
    """Verify multi-turn history with ToolCallPart and ToolReturnPart maps correctly."""
    async def _run():
        mock_connector = MagicMock()
        mock_connector.infer.return_value = InferenceResponse(
            request_id="req-2",
            model_id="qwen3.5-9b",
            message=Message(role=MessageRole.ASSISTANT, content="Final answer"),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
            latency_ms=50.0,
        )

        policy = create_test_policy()
        model = FoundationPydanticAIModel(
            connector=mock_connector,
            model_policy=policy,
            default_tier=ModelTier.REASONING,
        )

        messages = [
            ModelRequest(parts=[SystemPromptPart(content="You are helpful")]),
            ModelRequest(parts=[UserPromptPart(content="Calculate 2+2")]),
            ModelResponse(
                parts=[ToolCallPart(tool_name="add", args={"a": 2, "b": 2}, tool_call_id="call-1")]
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="add", content={"result": 4}, tool_call_id="call-1")]
            ),
        ]
        params = ModelRequestParameters()
        await model.request(messages, None, params)

        inf_req: InferenceRequest = mock_connector.infer.call_args[0][0]
        assert len(inf_req.messages) == 4
        assert inf_req.messages[0].role == MessageRole.SYSTEM
        assert inf_req.messages[0].content == "You are helpful"
        assert inf_req.messages[1].role == MessageRole.USER
        assert inf_req.messages[1].content == "Calculate 2+2"
        assert inf_req.messages[2].role == MessageRole.ASSISTANT
        assert "call-1" in inf_req.messages[2].content
        assert inf_req.messages[3].role == MessageRole.TOOL
        assert inf_req.messages[3].name == "add"
        assert "4" in inf_req.messages[3].content

    asyncio.run(_run())


def test_model_adapter_model_tier_resolution():
    """Verify abstract ModelTier resolves dynamically through ModelSelectionPolicy."""
    async def _run():
        mock_connector = MagicMock()
        mock_connector.infer.return_value = InferenceResponse(
            request_id="req-3",
            model_id="qwen3.5-0.8b",
            message=Message(role=MessageRole.ASSISTANT, content="lightweight response"),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=20.0,
        )

        policy = create_test_policy()
        model = FoundationPydanticAIModel(
            connector=mock_connector,
            model_policy=policy,
            default_tier=ModelTier.LIGHTWEIGHT,
        )

        assert model.model_name == "qwen3.5-0.8b"

        messages = [ModelRequest(parts=[UserPromptPart(content="test")])]
        await model.request(messages, None, ModelRequestParameters())

        inf_req: InferenceRequest = mock_connector.infer.call_args[0][0]
        assert inf_req.model_id == "qwen3.5-0.8b"

        # Switch tier to REASONING
        model.set_tier(ModelTier.REASONING)
        assert model.model_name == "qwen3.5-9b"

    asyncio.run(_run())


def test_model_adapter_runtime_identity_error_propagation():
    """Verify that ProviderResponseError from runtime identity mismatch propagates cleanly."""
    async def _run():
        mock_connector = MagicMock()
        mock_connector.infer.side_effect = ProviderResponseError(
            "LlamaCppProvider executed wrong model: requested 'qwen3.5-9b', but llama-server reported 'qwen3.5-0.8b'."
        )

        policy = create_test_policy()
        model = FoundationPydanticAIModel(
            connector=mock_connector,
            model_policy=policy,
            default_tier=ModelTier.REASONING,
        )

        messages = [ModelRequest(parts=[UserPromptPart(content="test")])]
        with pytest.raises(ProviderResponseError, match="LlamaCppProvider executed wrong model"):
            await model.request(messages, None, ModelRequestParameters())

    asyncio.run(_run())


def test_model_adapter_tool_call_parsing():
    """Verify parsing of tool call output from model."""
    from pydantic_ai.tools import ToolDefinition

    async def _run():
        mock_connector = MagicMock()
        mock_connector.infer.return_value = InferenceResponse(
            request_id="req-4",
            model_id="qwen3.5-9b",
            message=Message(
                role=MessageRole.ASSISTANT,
                content='{"tool": "read_file", "arguments": {"path": "test.txt"}}',
            ),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            latency_ms=30.0,
        )

        policy = create_test_policy()
        model = FoundationPydanticAIModel(
            connector=mock_connector,
            model_policy=policy,
            default_tier=ModelTier.REASONING,
        )

        messages = [ModelRequest(parts=[UserPromptPart(content="read test.txt")])]
        params = ModelRequestParameters(
            function_tools=[ToolDefinition(name="read_file", parameters_json_schema={}, description="Read file")],
        )
        resp = await model.request(messages, None, params)

        assert len(resp.parts) == 1
        assert isinstance(resp.parts[0], ToolCallPart)
        assert resp.parts[0].tool_name == "read_file"
        assert resp.parts[0].args == {"path": "test.txt"}

    asyncio.run(_run())
