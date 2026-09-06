"""Unit tests for AgentCapability lifecycle, budget, timeout, cancellation, and provenance."""

import asyncio
import time
from unittest.mock import MagicMock, patch
import pytest

from core.common.types import FinishReason, MessageRole
from core.inference.types import InferenceResponse, Message, TokenUsage
from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.builtin.agent.capability import AgentCapability
from orchestration.capabilities.builtin.agent.model_adapter import (
    FoundationPydanticAIModel,
)
from orchestration.capabilities.builtin.agent.policy import (
    AgentExecutionPolicy,
)
from orchestration.capabilities.builtin.agent.tool_adapter import (
    CapabilityToolAdapter,
)
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskResult
from orchestration.errors import CapabilityUnavailableError
from orchestration.routing.model_selector import ModelSelectionPolicy
from orchestration.routing.types import ModelTier


def create_mock_policy() -> ModelSelectionPolicy:
    mock_reg = MagicMock()
    mock_reg.is_known.side_effect = lambda m: m in {"qwen3.5-0.8b", "qwen3.5-9b"}
    mock_reg.get_model.side_effect = lambda m: MagicMock(id=m)
    return ModelSelectionPolicy(registry=mock_reg)


class DummyToolCapability:
    def __init__(self, cid: str):
        self._cid = cid
        self.invoked = False

    @property
    def capability_id(self) -> str:
        return self._cid

    def execute(self, parameters, inputs, context):
        self.invoked = True
        return TaskResult(
            output={"processed": inputs.get("text", "ok")},
            references=[DataReference(key="ref_key", uri="file:///tmp/ref.txt")],
            artifacts=[
                ArtifactReference(
                    artifact_id="art-test",
                    name="test.xlsx",
                    uri="file:///tmp/test.xlsx",
                    mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            ],
        )


def test_agent_capability_protocol_compliance():
    """Verify AgentCapability conforms to the Capability structural protocol."""
    mock_connector = MagicMock()
    policy = create_mock_policy()
    model_adapter = FoundationPydanticAIModel(
        connector=mock_connector,
        model_policy=policy,
        default_tier=ModelTier.REASONING,
    )
    registry = CapabilityRegistry()
    tool_adapter = CapabilityToolAdapter(registry=registry)

    agent_cap = AgentCapability(
        model_adapter=model_adapter,
        tool_adapter=tool_adapter,
    )

    assert isinstance(agent_cap, Capability)
    assert agent_cap.capability_id == "agent.pydantic_ai"
    assert agent_cap.is_available is True

    descriptor = agent_cap.get_descriptor()
    assert descriptor.capability_id == "agent.pydantic_ai"
    assert descriptor.is_available is True
    assert "prompt" in descriptor.input_schema


def test_agent_capability_single_turn_execution():
    """Verify straightforward prompt execution returning TaskResult."""
    mock_connector = MagicMock()
    mock_connector.infer.return_value = InferenceResponse(
        request_id="req-10",
        model_id="qwen3.5-9b",
        message=Message(role=MessageRole.ASSISTANT, content="The answer is 42."),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        latency_ms=25.0,
    )

    policy = create_mock_policy()
    model_adapter = FoundationPydanticAIModel(
        connector=mock_connector,
        model_policy=policy,
        default_tier=ModelTier.REASONING,
    )
    registry = CapabilityRegistry()
    tool_adapter = CapabilityToolAdapter(registry=registry)
    agent_cap = AgentCapability(model_adapter=model_adapter, tool_adapter=tool_adapter)

    context = CapabilityContext(execution_id="exec-turn-1")
    result: TaskResult = agent_cap.execute(
        parameters={"model_tier": "reasoning"},
        inputs={"prompt": "What is the answer?"},
        context=context,
    )

    assert isinstance(result, TaskResult)
    assert result.output["response"] == "The answer is 42."
    assert result.output["finish_reason"] == "stop"
    assert result.metadata["execution_id"] == "exec-turn-1"
    assert result.metadata["model_id"] == "qwen3.5-9b"


def test_agent_capability_preserves_child_provenance():
    """Verify that references and artifacts from child tool executions are aggregated in TaskResult."""
    mock_connector = MagicMock()
    turns = [0]

    def mock_infer(req):
        turns[0] += 1
        if turns[0] == 1:
            # Model calls text_analysis tool
            return InferenceResponse(
                request_id="req-tool-1",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content='{"tool": "text_analysis", "arguments": {"text": "Hello world"}}',
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                latency_ms=30.0,
            )
        else:
            # Model returns final answer
            return InferenceResponse(
                request_id="req-tool-2",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="Analyzed text successfully.",
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(prompt_tokens=25, completion_tokens=5, total_tokens=30),
                latency_ms=20.0,
            )

    mock_connector.infer.side_effect = mock_infer

    policy = create_mock_policy()
    model_adapter = FoundationPydanticAIModel(
        connector=mock_connector,
        model_policy=policy,
        default_tier=ModelTier.REASONING,
    )
    registry = CapabilityRegistry()
    dummy_cap = DummyToolCapability("workflow.text_analysis")
    registry.register(dummy_cap)

    tool_adapter = CapabilityToolAdapter(registry=registry)
    agent_cap = AgentCapability(model_adapter=model_adapter, tool_adapter=tool_adapter)

    context = CapabilityContext(execution_id="exec-prov-1")
    result: TaskResult = agent_cap.execute(
        parameters={"allowed_capabilities": ["workflow.text_analysis"]},
        inputs={"prompt": "Please analyze Hello world"},
        context=context,
    )

    assert dummy_cap.invoked is True
    assert result.output["response"] == "Analyzed text successfully."
    assert len(result.references) == 1
    assert result.references[0].key == "ref_key"
    assert len(result.artifacts) == 1
    assert result.artifacts[0].name == "test.xlsx"
    assert len(result.output["tool_calls"]) == 1
    assert result.output["tool_calls"][0]["tool_name"] == "workflow.text_analysis"


def test_agent_capability_cancellation():
    """Verify cancellation flag in context prevents execution."""
    mock_connector = MagicMock()
    policy = create_mock_policy()
    model_adapter = FoundationPydanticAIModel(
        connector=mock_connector,
        model_policy=policy,
        default_tier=ModelTier.REASONING,
    )
    registry = CapabilityRegistry()
    tool_adapter = CapabilityToolAdapter(registry=registry)
    agent_cap = AgentCapability(model_adapter=model_adapter, tool_adapter=tool_adapter)

    context = CapabilityContext(
        execution_id="exec-cancel-1",
        metadata={"cancelled": True},
    )
    result = agent_cap.execute(
        parameters={},
        inputs={"prompt": "Do work"},
        context=context,
    )

    assert result.output["finish_reason"] == "cancelled"
    mock_connector.infer.assert_not_called()


def test_agent_capability_timeout():
    """Verify timeout_seconds triggers TimeoutError."""
    mock_connector = MagicMock()

    def slow_infer(req):
        time.sleep(0.5)
        return InferenceResponse(
            request_id="req-slow",
            model_id="qwen3.5-9b",
            message=Message(role=MessageRole.ASSISTANT, content="Too late"),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(1, 1, 2),
            latency_ms=500.0,
        )

    mock_connector.infer.side_effect = slow_infer

    policy = create_mock_policy()
    model_adapter = FoundationPydanticAIModel(
        connector=mock_connector,
        model_policy=policy,
        default_tier=ModelTier.REASONING,
    )
    registry = CapabilityRegistry()
    tool_adapter = CapabilityToolAdapter(registry=registry)
    agent_cap = AgentCapability(model_adapter=model_adapter, tool_adapter=tool_adapter)

    context = CapabilityContext(execution_id="exec-timeout-1")
    with pytest.raises(TimeoutError, match="timed out after 0.1 seconds"):
        agent_cap.execute(
            parameters={"timeout_seconds": 0.1},
            inputs={"prompt": "Wait a while"},
            context=context,
        )


def test_agent_capability_truthful_missing_dependency():
    """Verify truthful error when pydantic_ai dependency is unavailable."""
    mock_connector = MagicMock()
    policy = create_mock_policy()
    model_adapter = FoundationPydanticAIModel(
        connector=mock_connector,
        model_policy=policy,
    )
    registry = CapabilityRegistry()
    tool_adapter = CapabilityToolAdapter(registry=registry)
    agent_cap = AgentCapability(model_adapter=model_adapter, tool_adapter=tool_adapter)

    with patch(
        "orchestration.capabilities.builtin.agent.capability._PYDANTIC_AI_AVAILABLE",
        False,
    ):
        assert agent_cap.is_available is False
        with pytest.raises(CapabilityUnavailableError, match="pydantic_ai is not installed"):
            agent_cap.execute(
                parameters={},
                inputs={"prompt": "test"},
                context=CapabilityContext(execution_id="exec-unavail"),
            )


def test_agent_capability_replan_proposal_extension():
    """Verify advisory replanning proposal extracted and returned in output without mutating DAG."""
    mock_connector = MagicMock()
    proposal_content = (
        '{"proposal": {"type": "replanning", "reason": "Need extra data extraction", '
        '"suggested_tasks": [{"task_id": "step-2", "capability": "document.understand"}]}}'
    )
    mock_connector.infer.return_value = InferenceResponse(
        request_id="req-prop",
        model_id="qwen3.5-9b",
        message=Message(role=MessageRole.ASSISTANT, content=proposal_content),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(10, 20, 30),
        latency_ms=20.0,
    )

    policy = create_mock_policy()
    model_adapter = FoundationPydanticAIModel(
        connector=mock_connector,
        model_policy=policy,
        default_tier=ModelTier.REASONING,
    )
    registry = CapabilityRegistry()
    tool_adapter = CapabilityToolAdapter(registry=registry)
    agent_cap = AgentCapability(model_adapter=model_adapter, tool_adapter=tool_adapter)

    context = CapabilityContext(execution_id="exec-prop-1")
    result = agent_cap.execute(
        parameters={},
        inputs={"prompt": "Plan needed"},
        context=context,
    )

    assert result.output["proposal"] is not None
    assert result.output["proposal"]["type"] == "replanning"
    assert result.output["proposal"]["reason"] == "Need extra data extraction"
    assert len(result.output["proposal"]["suggested_tasks"]) == 1
