"""Unit tests for TemplatePlanner and LLMPlanner."""

import json
from unittest.mock import MagicMock
import pytest

from core.common.types import FinishReason, MessageRole
from core.inference.types import InferenceResponse, Message, TokenUsage
from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.types import PlanStatus
from orchestration.planning import (
    CandidatePlan,
    CandidateTask,
    LLMPlanner,
    PlanningContext,
    TemplatePlanner,
)


def test_template_planner_basic():
    planner = TemplatePlanner()

    def my_template(ctx: PlanningContext) -> CandidatePlan:
        return CandidatePlan(
            plan_id="p_template",
            goal_id=ctx.goal.goal_id,
            title="Two-Step Pipeline",
            tasks=[
                CandidateTask(
                    task_id="t1",
                    title="Step 1",
                    capability_id="transform",
                ),
                CandidateTask(
                    task_id="t2",
                    title="Step 2",
                    capability_id="summarize",
                    dependencies=[Dependency("t1", "t2")],
                ),
            ],
            dependencies=[Dependency("t1", "t2")],
        )

    planner.register_template("two_step", my_template)

    goal = Goal(goal_id="g1", description="Run pipeline", context={"template": "two_step"})
    ctx = PlanningContext(goal=goal)

    candidate = planner.plan(ctx)
    assert candidate.plan_id == "p_template"
    assert len(candidate.tasks) == 2
    assert candidate.tasks[1].dependencies[0].upstream_task_id == "t1"

    # Convert to domain Plan
    domain_plan = candidate.to_plan()
    assert domain_plan.status == PlanStatus.DRAFT
    assert len(domain_plan.tasks) == 2
    assert domain_plan.tasks["t2"].dependencies[0].upstream_task_id == "t1"


def test_llm_planner_structured_inference():
    mock_connector = MagicMock()
    mock_json = {
        "title": "LLM Generated Plan",
        "tasks": [
            {
                "task_id": "extract",
                "title": "Extract Data",
                "capability_id": "text_extract",
                "description": "Extract entities",
                "parameters": {"format": "entities"},
                "input_references": [],
                "dependencies": [],
            },
            {
                "task_id": "synthesize",
                "title": "Synthesize",
                "capability_id": "summarize",
                "description": "Summarize extracted",
                "parameters": {"max_len": 100},
                "input_references": [
                    {"key": "extracted_entities", "source_task_id": "extract"}
                ],
                "dependencies": ["extract"],
            },
        ],
    }

    mock_connector.infer_prompt.return_value = InferenceResponse(
        request_id="req-1",
        model_id="default",
        message=Message(role=MessageRole.ASSISTANT, content=json.dumps(mock_json)),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        latency_ms=15.0,
    )

    registry = CapabilityRegistry()
    registry.register_descriptor(
        CapabilityDescriptor(capability_id="text_extract", description="Extract text")
    )
    registry.register_descriptor(
        CapabilityDescriptor(capability_id="summarize", description="Summarize text")
    )

    planner = LLMPlanner(
        connector=mock_connector,
        capability_registry=registry,
        model_id="qwen-capable",
    )

    goal = Goal(goal_id="g_llm", description="Analyze customer feedback")
    ctx = PlanningContext(goal=goal)

    candidate = planner.plan(ctx)
    assert candidate.title == "LLM Generated Plan"
    assert len(candidate.tasks) == 2
    assert candidate.tasks[0].task_id == "extract"
    assert candidate.tasks[1].task_id == "synthesize"
    assert "extracted_entities" in candidate.tasks[1].input_references
    assert candidate.tasks[1].input_references["extracted_entities"].source_task_id == "extract"
    assert len(candidate.tasks[1].dependencies) == 1
    assert candidate.tasks[1].dependencies[0].upstream_task_id == "extract"
