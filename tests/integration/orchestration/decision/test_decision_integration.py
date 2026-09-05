"""Integration tests for Decision & Planning layer wired via AppContext."""

from unittest.mock import MagicMock
import pytest

from apps.context import AppContext
from core.common.types import FinishReason, MessageRole
from core.inference.types import InferenceResponse, Message, TokenUsage
from orchestration.domain.goals import Goal
from orchestration.domain.types import GoalStatus
from orchestration.routing.types import ExecutionStrategy


@pytest.fixture
def mock_app_context():
    mock_core = MagicMock()
    mock_inference = MagicMock()

    # Configure mock responses for inference connector
    mock_inference.infer_prompt.return_value = InferenceResponse(
        request_id="req-int-1",
        model_id="default",
        message=Message(
            role=MessageRole.ASSISTANT,
            content='{"summary": "Test summary", "key_points": ["Point 1", "Point 2"]}',
        ),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        latency_ms=10.0,
    )

    return AppContext(core=mock_core, inference=mock_inference)


def test_decision_engine_via_app_context_direct_deterministic(mock_app_context):
    engine = mock_app_context.create_decision_engine()

    goal = Goal(goal_id="g_int_ping", description="ping status check")
    decision = engine.process_goal(goal)

    assert decision.decision_type == ExecutionStrategy.DIRECT_DETERMINISTIC
    assert decision.direct_result is not None
    assert decision.direct_result.result is not None
    assert goal.status == GoalStatus.COMPLETED
    assert "status" not in goal.context


def test_decision_engine_via_app_context_direct_capability(mock_app_context):
    engine = mock_app_context.create_decision_engine()

    goal = Goal(
        goal_id="g_int_analysis",
        description="analyze text document",
        context={"parameters": {"text": "A quick brown fox jumps over the lazy dog."}},
    )
    decision = engine.process_goal(goal)

    assert decision.decision_type == ExecutionStrategy.DIRECT_CAPABILITY
    assert decision.direct_result is not None
    assert decision.direct_result.error is None
    assert decision.direct_result.result is not None
    assert goal.status == GoalStatus.COMPLETED
    assert "summary" not in goal.context


def test_decision_engine_via_app_context_plan_required(mock_app_context):
    import json

    # Mock inference to return valid plan JSON on planner invocation
    plan_json = {
        "title": "Analysis Plan",
        "tasks": [
            {
                "task_id": "step1",
                "title": "Analyze Text",
                "capability_id": "workflow.text_analysis",
                "description": "Run text analysis",
                "parameters": {"text": "Document content to analyze"},
            }
        ],
    }

    # First call: planner inference returns plan_json
    # Subsequent calls (during task execution): workflow analysis output
    mock_app_context.inference.infer_prompt.side_effect = [
        InferenceResponse(
            request_id="plan-req",
            model_id="default",
            message=Message(role=MessageRole.ASSISTANT, content=json.dumps(plan_json)),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            latency_ms=15.0,
        ),
        InferenceResponse(
            request_id="task-req",
            model_id="default",
            message=Message(
                role=MessageRole.ASSISTANT,
                content='{"summary": "Analysis complete", "key_points": ["Insight 1"]}',
            ),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=15, completion_tokens=15, total_tokens=30),
            latency_ms=12.0,
        ),
    ]

    engine = mock_app_context.create_decision_engine()

    goal = Goal(
        goal_id="g_int_pipeline",
        description="multi-step pipeline text processing",
    )
    decision = engine.process_goal(goal)

    assert decision.decision_type == ExecutionStrategy.PLAN_REQUIRED
    assert decision.validation_result is not None
    assert decision.validation_result.is_valid is True
    assert decision.plan_id is not None
    assert goal.status == GoalStatus.COMPLETED
