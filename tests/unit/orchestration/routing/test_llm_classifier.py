"""Focused unit tests for LLMIntentClassifier hardening and staged escalation.

Verifies:
  - Issue 2: ModelSelectionPolicy wires model_tier -> concrete model_id (lightweight vs reasoning).
  - Issue 3: Structured diagnostics distinguishing inference failure, parsing failure,
    invalid route, and genuine no-match, allowing safe degradation.
  - Issue 4: RouteDefinition.strategy is authoritative; LLM cannot override strategy in JSON.
  - Staged escalation: Stage 3 (lightweight) -> Stage 4 (reasoning) -> fallback.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.common.types import FinishReason, MessageRole, ModelFormat
from core.inference.types import InferenceResponse, Message, TokenUsage
from core.models.registry import ModelRegistry
from core.models.schema import ModelCapabilities, ModelDefinition
from orchestration.domain.goals import Goal
from orchestration.routing import (
    ExecutionStrategy,
    LLMIntentClassifier,
    ModelSelectionPolicy,
    ModelTier,
    RouteDefinition,
    StagedEscalationRouter,
)


def _make_mock_registry(tmp_path: Path) -> ModelRegistry:
    registry = ModelRegistry(configs_dir=tmp_path, auto_load=False)
    m_light = ModelDefinition(
        id="qwen-light",
        display_name="Qwen Light",
        format=ModelFormat.GGUF,
        relative_path=Path("dummy/light"),
        supported_providers=["llama_cpp"],
        aliases=["light"],
        capabilities=ModelCapabilities(),
    )
    m_reasoning = ModelDefinition(
        id="qwen-reasoning",
        display_name="Qwen Reasoning",
        format=ModelFormat.GGUF,
        relative_path=Path("dummy/reasoning"),
        supported_providers=["llama_cpp"],
        aliases=["reasoning"],
        capabilities=ModelCapabilities(),
    )
    with registry._lock:
        registry._models["qwen-light"] = m_light
        registry._models["qwen-reasoning"] = m_reasoning
        registry._alias_map["qwen-light"] = "qwen-light"
        registry._alias_map["qwen-reasoning"] = "qwen-reasoning"
        registry._alias_map["default"] = "qwen-light"
    return registry


def _make_sample_routes() -> list[RouteDefinition]:
    return [
        RouteDefinition(
            name="direct_status",
            strategy=ExecutionStrategy.DIRECT_DETERMINISTIC,
            description="Check system status",
        ),
        RouteDefinition(
            name="direct_echo",
            strategy=ExecutionStrategy.DIRECT_CAPABILITY,
            target_capability_id="test.echo",
            description="Echo message",
        ),
        RouteDefinition(
            name="deep_plan",
            strategy=ExecutionStrategy.PLAN_REQUIRED,
            description="Multi-step planning required",
        ),
    ]


def test_issue_2_model_selection_policy_resolves_different_tiers(tmp_path):
    """Issue 2: Verify Stage 3 resolves lightweight model and Stage 4 resolves reasoning model."""
    registry = _make_mock_registry(tmp_path)
    policy = ModelSelectionPolicy(
        registry=registry,
        tier_mapping={
            ModelTier.LIGHTWEIGHT: "qwen-light",
            ModelTier.REASONING: "qwen-reasoning",
        },
    )

    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = InferenceResponse(
        request_id="req-1",
        model_id="qwen-light",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=json.dumps({"route_name": "direct_status", "confidence": 0.95}),
        ),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        latency_ms=5.0,
    )

    classifier = LLMIntentClassifier(
        connector=mock_connector,
        routes=_make_sample_routes(),
        model_selection_policy=policy,
    )

    goal = Goal(goal_id="g1", description="check system health")

    # 1. Stage 3 invocation (LIGHTWEIGHT)
    res_light = classifier.classify(goal, model_tier=ModelTier.LIGHTWEIGHT)
    assert res_light is not None
    assert res_light.route_name == "direct_status"
    assert res_light.stage_resolved == "llm_classifier"
    assert res_light.metadata["resolved_model_id"] == "qwen-light"
    assert mock_connector.infer_prompt.call_args.kwargs["model_id"] == "qwen-light"

    # 2. Stage 4 invocation (REASONING)
    res_reasoning = classifier.classify(goal, model_tier=ModelTier.REASONING)
    assert res_reasoning is not None
    assert res_reasoning.stage_resolved == "escalated"
    assert res_reasoning.metadata["resolved_model_id"] == "qwen-reasoning"
    assert mock_connector.infer_prompt.call_args.kwargs["model_id"] == "qwen-reasoning"


def test_issue_4_route_definition_strategy_is_authoritative(tmp_path):
    """Issue 4: LLM cannot override RouteDefinition.strategy via its JSON output."""
    registry = _make_mock_registry(tmp_path)
    policy = ModelSelectionPolicy(registry=registry)

    # Route 'direct_echo' has strategy DIRECT_CAPABILITY.
    # LLM attempts to output strategy 'plan_required' to hijack execution strategy.
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = InferenceResponse(
        request_id="req-hijack",
        model_id="default",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=json.dumps({
                "route_name": "direct_echo",
                "strategy": "plan_required",  # LLM tries to override!
                "confidence": 0.90,
            }),
        ),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        latency_ms=5.0,
    )

    classifier = LLMIntentClassifier(
        connector=mock_connector,
        routes=_make_sample_routes(),
        model_selection_policy=policy,
    )

    goal = Goal(goal_id="g2", description="echo test")
    result = classifier.classify(goal, model_tier=ModelTier.LIGHTWEIGHT)

    assert result is not None
    assert result.route_name == "direct_echo"
    # Must remain DIRECT_CAPABILITY as declared in RouteDefinition, NOT plan_required!
    assert result.strategy == ExecutionStrategy.DIRECT_CAPABILITY
    assert result.target_capability_id == "test.echo"


def test_issue_3_structured_diagnostics_on_inference_failure(tmp_path, caplog):
    """Issue 3: Infrastructure / inference exception logs warning and returns None (safe degradation)."""
    registry = _make_mock_registry(tmp_path)
    policy = ModelSelectionPolicy(registry=registry)

    mock_connector = MagicMock()
    mock_connector.infer_prompt.side_effect = RuntimeError("Inference backend connection refused")

    classifier = LLMIntentClassifier(
        connector=mock_connector,
        routes=_make_sample_routes(),
        model_selection_policy=policy,
    )

    goal = Goal(goal_id="g3", description="do something")
    with caplog.at_level(logging.WARNING):
        result = classifier.classify(goal, model_tier=ModelTier.LIGHTWEIGHT)

    assert result is None
    assert "inference failure" in caplog.text
    assert "RuntimeError" in caplog.text


def test_issue_3_structured_diagnostics_on_json_parsing_failure(tmp_path, caplog):
    """Issue 3: Malformed JSON output logs warning and returns None."""
    registry = _make_mock_registry(tmp_path)
    policy = ModelSelectionPolicy(registry=registry)

    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = InferenceResponse(
        request_id="req-malformed",
        model_id="default",
        message=Message(role=MessageRole.ASSISTANT, content="Here is your answer: {not valid json..."),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        latency_ms=5.0,
    )

    classifier = LLMIntentClassifier(
        connector=mock_connector,
        routes=_make_sample_routes(),
        model_selection_policy=policy,
    )

    goal = Goal(goal_id="g4", description="parse fail")
    with caplog.at_level(logging.WARNING):
        result = classifier.classify(goal, model_tier=ModelTier.LIGHTWEIGHT)

    assert result is None
    assert "JSON parsing failure" in caplog.text


def test_issue_3_structured_diagnostics_on_unrecognized_route(tmp_path, caplog):
    """Issue 3: LLM selecting unknown route logs info and returns None (no-match)."""
    registry = _make_mock_registry(tmp_path)
    policy = ModelSelectionPolicy(registry=registry)

    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = InferenceResponse(
        request_id="req-unknown",
        model_id="default",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=json.dumps({"route_name": "non_existent_route", "confidence": 0.88}),
        ),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        latency_ms=5.0,
    )

    classifier = LLMIntentClassifier(
        connector=mock_connector,
        routes=_make_sample_routes(),
        model_selection_policy=policy,
    )

    goal = Goal(goal_id="g5", description="unknown route test")
    with caplog.at_level(logging.INFO):
        result = classifier.classify(goal, model_tier=ModelTier.LIGHTWEIGHT)

    assert result is None
    assert "unrecognized route 'non_existent_route'" in caplog.text


def test_staged_escalation_with_llm_classifier(tmp_path):
    """Test full StagedEscalationRouter: low confidence in Stage 3 escalates to Stage 4."""
    registry = _make_mock_registry(tmp_path)
    policy = ModelSelectionPolicy(
        registry=registry,
        tier_mapping={
            ModelTier.LIGHTWEIGHT: "qwen-light",
            ModelTier.REASONING: "qwen-reasoning",
        },
    )

    routes = _make_sample_routes()

    # Stage 3 returns low confidence (0.55 < 0.70)
    # Stage 4 returns high confidence (0.92)
    mock_connector = MagicMock()
    mock_connector.infer_prompt.side_effect = [
        # Call 1: Stage 3 (lightweight)
        InferenceResponse(
            request_id="req-stage3",
            model_id="qwen-light",
            message=Message(
                role=MessageRole.ASSISTANT,
                content=json.dumps({"route_name": "deep_plan", "confidence": 0.55}),
            ),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            latency_ms=5.0,
        ),
        # Call 2: Stage 4 (reasoning)
        InferenceResponse(
            request_id="req-stage4",
            model_id="qwen-reasoning",
            message=Message(
                role=MessageRole.ASSISTANT,
                content=json.dumps({"route_name": "deep_plan", "confidence": 0.92}),
            ),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            latency_ms=10.0,
        ),
    ]

    classifier = LLMIntentClassifier(
        connector=mock_connector,
        routes=routes,
        model_selection_policy=policy,
    )

    staged_router = StagedEscalationRouter(
        routes=routes,
        llm_classifier=classifier,
    )

    goal = Goal(goal_id="g_escalate", description="complex multi-stage task needing reasoning")
    result = staged_router.route(goal)

    assert result.stage_resolved == "escalated"
    assert result.route_name == "deep_plan"
    assert result.confidence == 0.92
    assert mock_connector.infer_prompt.call_count == 2
    assert mock_connector.infer_prompt.call_args_list[0].kwargs["model_id"] == "qwen-light"
    assert mock_connector.infer_prompt.call_args_list[1].kwargs["model_id"] == "qwen-reasoning"
