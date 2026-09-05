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


def test_end_to_end_aurelio_semantic_route_to_decision_engine(mock_app_context):
    """Verify that a Goal missing Stage 1 deterministic rules matches Stage 2 Aurelio Semantic Router,

    produces a normalized RouteResult, and executes through DecisionEngine without Aurelio
    overstepping its intent recognition boundary.
    """
    router = mock_app_context.create_intent_router()
    engine = mock_app_context.create_decision_engine(router=router)

    # Goal misses Stage 1 deterministic prefix ('ping', 'health') but matches Stage 2 Aurelio route 'text_analysis'
    goal = Goal(
        goal_id="g_e2e_aurelio",
        description="extract key points from document",
        context={"parameters": {"text": "Q3 Financial Report content."}},
    )

    # 1. Verify router isolation and RouteResult semantics
    route_res = router.route(goal)
    assert route_res is not None
    assert route_res.stage_resolved == "semantic_router"
    assert route_res.route_name == "text_analysis"
    assert route_res.strategy == ExecutionStrategy.DIRECT_CAPABILITY
    assert route_res.target_capability_id == "workflow.text_analysis"
    assert route_res.confidence >= 0.60
    assert route_res.metadata.get("engine") == "aurelio_semantic_router"

    # 2. Verify downstream DecisionEngine consumes RouteResult and executes capability
    decision = engine.process_goal(goal)

    assert decision.decision_type == ExecutionStrategy.DIRECT_CAPABILITY
    assert decision.direct_result is not None
    assert decision.direct_result.error is None
    assert decision.direct_result.result is not None
    assert decision.direct_result.result.output.summary == "Test summary"
    assert decision.direct_result.result.output.key_points == ["Point 1", "Point 2"]
    assert goal.status == GoalStatus.COMPLETED


def test_end_to_end_aurelio_unresolved_escalation_to_llm_classifier_and_decision_engine(mock_app_context):
    """Verify that a Goal unresolved by Stage 1 and Stage 2 Aurelio escalates to Stage 3 LLM classifier

    via ModelSelectionPolicy, produces a normalized RouteResult, and executes cleanly through DecisionEngine.
    """
    # Configure mock registry so ModelSelectionPolicy resolves ModelTier.LIGHTWEIGHT
    mock_app_context.core.model_registry.is_known.return_value = True
    mock_app_context.core.model_registry.get_model.return_value.id = "qwen3.5-0.8b"

    router = mock_app_context.create_intent_router(enable_llm=True)
    engine = mock_app_context.create_decision_engine(router=router)

    # Goal is completely out of vocabulary for deterministic and Aurelio Stage 2 (unresolved)
    goal = Goal(
        goal_id="g_e2e_escalate",
        description="perform complex multidimensional domain assessment of legacy contracts",
        context={"parameters": {"text": "Contract clauses and conditions."}},
    )

    # Mock inference:
    # 1st call: Stage 3 LLM Intent Classifier classification
    # 2nd call: Workflow execution for text_analysis capability
    mock_app_context.inference.infer_prompt.side_effect = [
        InferenceResponse(
            request_id="clf-req",
            model_id="qwen3.5-0.8b",
            message=Message(
                role=MessageRole.ASSISTANT,
                content='{"route_name": "text_analysis", "confidence": 0.88}',
            ),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            latency_ms=10.0,
        ),
        InferenceResponse(
            request_id="wf-req",
            model_id="default",
            message=Message(
                role=MessageRole.ASSISTANT,
                content='{"summary": "Classified and analyzed", "key_points": ["Key finding"]}',
            ),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=15, completion_tokens=15, total_tokens=30),
            latency_ms=12.0,
        ),
    ]

    # Process goal end-to-end through DecisionEngine with staged router
    decision = engine.process_goal(goal)

    assert decision.decision_type == ExecutionStrategy.DIRECT_CAPABILITY
    assert decision.direct_result is not None
    assert decision.direct_result.error is None
    assert decision.direct_result.result is not None
    assert decision.direct_result.result.output.summary == "Classified and analyzed"
    assert decision.direct_result.result.output.key_points == ["Key finding"]
    assert goal.status == GoalStatus.COMPLETED

