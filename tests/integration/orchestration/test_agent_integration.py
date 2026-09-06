"""End-to-end integration tests for PydanticAI Agent capability."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from apps.context import AppContext
from core.common.types import FinishReason, MessageRole
from core.inference.types import InferenceResponse, Message, TokenUsage
from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.agent import (
    AgentCapability,
    AgentExecutionPolicy,
    CapabilityToolAdapter,
    FoundationPydanticAIModel,
    UnauthorizedCapabilityError,
)
from orchestration.capabilities.builtin.code.base import WorkspaceExecutor
from orchestration.capabilities.builtin.code.capability import WorkspaceCodingCapability
from orchestration.capabilities.builtin.code.types import (
    WorkspaceCommandResponse,
    WorkspaceFileRead,
)
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.results import TaskResult
from orchestration.domain.tasks import Task
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.execution import InProcessPlanRunner
from orchestration.execution.temporal import TaskExecutionActivity
from orchestration.routing.model_selector import ModelSelectionPolicy
from orchestration.routing.types import ExecutionStrategy, ModelTier


class MockWorkspaceExecutor(WorkspaceExecutor):
    """Deterministic in-memory/isolated mock workspace executor."""

    def __init__(self):
        self.files = {"main.py": "print('hello from workspace')"}
        self.commands = []

    def read_file(self, path: str, start_line=None, end_line=None):
        content = self.files.get(path, "")
        return WorkspaceFileRead(
            path=path,
            content=content,
            start_line=start_line or 1,
            end_line=end_line or len(content.splitlines()),
            total_lines=len(content.splitlines()),
        )

    def write_file(self, path: str, content: str, overwrite: bool = True):
        self.files[path] = content

    def edit_file(self, path: str, target: str, replacement: str):
        if path in self.files:
            self.files[path] = self.files[path].replace(target, replacement)

    def list_dir(self, path: str = ".", recursive: bool = False, max_depth: int = 3):
        return list(self.files.keys())

    def run_command(self, request):
        self.commands.append(request.command)
        return WorkspaceCommandResponse(
            command=request.command,
            exit_code=0,
            stdout="command output ok",
            stderr="",
            execution_time_ms=10.0,
            success=True,
        )


@pytest.fixture
def mock_app_context():
    mock_core = MagicMock()
    mock_inference = MagicMock()

    # Configure mock responses for core
    mock_reg = MagicMock()
    mock_reg.is_known.side_effect = lambda m: m in {"qwen3.5-0.8b", "qwen3.5-9b", "default"}
    mock_reg.get_model.side_effect = lambda m: MagicMock(id=m)
    mock_core.registry = mock_reg
    mock_core.settings.database.url = "sqlite:///:memory:"

    mock_inference.infer_prompt.return_value = InferenceResponse(
        request_id="req-prompt-1",
        model_id="default",
        message=Message(
            role=MessageRole.ASSISTANT,
            content='{"summary": "Quarterly revenue reached expectations.", "key_points": ["revenue ok"]}',
        ),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(10, 10, 20),
        latency_ms=10.0,
    )

    return AppContext(core=mock_core, inference=mock_inference)


def test_direct_goal_to_agent_execution(mock_app_context):
    """Test Goal -> DecisionEngine -> GoalOrchestrator -> AgentCapability execution."""
    turns = [0]

    def mock_infer(req):
        turns[0] += 1
        if turns[0] == 1:
            # First turn: call workflow.text_analysis tool
            return InferenceResponse(
                request_id="req-agent-1",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content='{"tool": "text_analysis", "arguments": {"text": "Quarterly financial results"}}',
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                latency_ms=20.0,
            )
        else:
            # Second turn: final response
            return InferenceResponse(
                request_id="req-agent-2",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="Summary: Financial results reviewed successfully.",
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(prompt_tokens=25, completion_tokens=10, total_tokens=35),
                latency_ms=20.0,
            )

    mock_app_context.inference.infer.side_effect = mock_infer
    engine = mock_app_context.create_decision_engine()

    goal = Goal(
        goal_id="g_agent_direct",
        description="run agent task",
        context={
            "parameters": {
                "allowed_capabilities": ["workflow.text_analysis"],
            },
            "inputs": {
                "prompt": "Analyze quarterly results using tools",
            },
        },
    )

    decision = engine.process_goal(goal)

    assert decision.decision_type == ExecutionStrategy.DIRECT_CAPABILITY
    assert decision.direct_result is not None
    assert decision.direct_result.error is None
    assert goal.status == GoalStatus.COMPLETED

    task_result: TaskResult = decision.direct_result.result
    assert task_result.output["response"] == "Summary: Financial results reviewed successfully."
    assert task_result.output["finish_reason"] == "stop"
    assert len(task_result.output["tool_calls"]) == 1
    assert task_result.output["tool_calls"][0]["tool_name"] == "workflow.text_analysis"


def test_plan_task_to_agent_in_process_runner(mock_app_context):
    """Test Plan -> Task(agent.pydantic_ai) -> InProcessPlanRunner execution."""
    registry = mock_app_context.create_base_capability_registry()

    # Wire a mock workspace coding capability
    mock_executor = MockWorkspaceExecutor()
    ws_cap = WorkspaceCodingCapability(executor=mock_executor)
    registry.register_descriptor(ws_cap)  # or re-register
    # Update capability in registry
    registry._capabilities["code.workspace"] = ws_cap

    agent_cap = mock_app_context.create_agent_capability(registry=registry)
    registry.register(agent_cap)

    # Mock inference to call code_workspace then respond
    turns = [0]
    def mock_infer(req):
        turns[0] += 1
        if turns[0] == 1:
            return InferenceResponse(
                request_id="req-code-1",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content='{"tool": "code_workspace", "arguments": {"action": "read_file", "path": "main.py"}}',
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(10, 10, 20),
                latency_ms=15.0,
            )
        else:
            return InferenceResponse(
                request_id="req-code-2",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="Inspection finished: main.py prints hello.",
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(20, 10, 30),
                latency_ms=15.0,
            )

    mock_app_context.inference.infer.side_effect = mock_infer

    runner = InProcessPlanRunner(registry=registry)
    task = Task(
        task_id="t_agent_code",
        plan_id="p_agent_code",
        title="Agent Code Inspection",
        capability_id="agent.pydantic_ai",
        parameters={
            "allowed_capabilities": ["code.workspace"],
            "prompt": "Inspect main.py in sandbox",
        },
    )
    plan = Plan(plan_id="p_agent_code", goal_id="g_test", title="Agent Code Plan")
    plan.add_task(task)

    executed_plan = runner.run(plan)

    assert executed_plan.status == PlanStatus.COMPLETED
    assert task.status == TaskStatus.COMPLETED
    assert task.result is not None
    assert task.result.output["response"] == "Inspection finished: main.py prints hello."
    assert len(task.result.output["tool_calls"]) == 1
    assert task.result.output["tool_calls"][0]["tool_name"] == "code.workspace"


def test_plan_task_to_agent_temporal_activity_compatibility(mock_app_context):
    """Test Temporal TaskExecutionActivity executes AgentCapability cleanly."""
    from orchestration.execution.temporal.types import TaskActivityInput

    registry = mock_app_context.create_capability_registry()

    mock_app_context.inference.infer.return_value = InferenceResponse(
        request_id="req-temp-1",
        model_id="qwen3.5-9b",
        message=Message(role=MessageRole.ASSISTANT, content="Temporal agent result."),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(10, 5, 15),
        latency_ms=10.0,
    )

    activity = TaskExecutionActivity(registry=registry)
    act_input = TaskActivityInput(
        task_id="t-temp-agent",
        attempt_id="att-temp-1",
        capability_id="agent.pydantic_ai",
        parameters={"model_tier": "reasoning"},
        inputs={"prompt": "Run via Temporal activity"},
    )

    res = asyncio.run(activity.execute_task(act_input))

    assert res.status == "COMPLETED"
    assert res.error_message is None
    assert res.output["response"] == "Temporal agent result."


def test_agent_multi_capability_document_and_artifact_provenance(mock_app_context, tmp_path):
    """Test agent invoking document.understand and artifact.generate with full provenance."""
    # Create sample doc on disk
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text("Revenue: $100M\nProfit: $20M")

    from orchestration.capabilities.builtin.artifact import ArtifactGenerationCapability

    registry = mock_app_context.create_base_capability_registry()
    registry._capabilities["artifact.generate"] = ArtifactGenerationCapability(output_dir=tmp_path)
    agent_cap = mock_app_context.create_agent_capability(registry=registry)
    registry.register(agent_cap)

    turns = [0]
    def mock_infer(req):
        turns[0] += 1
        if turns[0] == 1:
            # Call document.understand
            return InferenceResponse(
                request_id="req-multi-1",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=f'{{"tool": "document_understand", "arguments": {{"file_path": "{doc_path}"}}}}',
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(10, 10, 20),
                latency_ms=10.0,
            )
        elif turns[0] == 2:
            # Call artifact.generate
            return InferenceResponse(
                request_id="req-multi-2",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content='{"tool": "artifact_generate", "arguments": {"artifact_type": "pdf", "title": "Financial Report", "content": "Revenue: $100M"}}',
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(20, 10, 30),
                latency_ms=10.0,
            )
        else:
            # Final response
            return InferenceResponse(
                request_id="req-multi-3",
                model_id="qwen3.5-9b",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="Document parsed and PDF report generated.",
                ),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(30, 5, 35),
                latency_ms=10.0,
            )

    mock_app_context.inference.infer.side_effect = mock_infer

    result: TaskResult = agent_cap.execute(
        parameters={"allowed_capabilities": ["document.understand", "artifact.generate"]},
        inputs={"prompt": "Parse doc and make pdf report"},
        context=CapabilityContext(execution_id="exec-multi-prov"),
    )

    assert result.output["response"] == "Document parsed and PDF report generated."
    assert len(result.output["tool_calls"]) == 2
    # Verify provenance: artifact reference is present
    assert len(result.artifacts) >= 1
    pdf_art = [a for a in result.artifacts if a.mime_type == "application/pdf"][0]
    assert pdf_art.name.endswith(".pdf")


def test_agent_per_call_authorization_denial(mock_app_context):
    """Test that unauthorized capability call proposal is rejected at the policy boundary."""
    registry = mock_app_context.create_capability_registry()
    agent_cap = mock_app_context.create_agent_capability(registry=registry)

    # Model attempts to call code.workspace when only text_analysis was approved
    mock_app_context.inference.infer.return_value = InferenceResponse(
        request_id="req-unauth-1",
        model_id="qwen3.5-9b",
        message=Message(
            role=MessageRole.ASSISTANT,
            content='{"tool": "code_workspace", "arguments": {"action": "run_command", "command": "rm -rf /"}}',
        ),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(10, 10, 20),
        latency_ms=10.0,
    )

    # Allowed only workflow.text_analysis
    with pytest.raises(Exception):
        agent_cap.execute(
            parameters={"allowed_capabilities": ["workflow.text_analysis"]},
            inputs={"prompt": "Try calling unapproved code tool"},
            context=CapabilityContext(execution_id="exec-unauth"),
        )
