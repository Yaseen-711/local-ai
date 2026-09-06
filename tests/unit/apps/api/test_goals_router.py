"""Unit tests for /api/v1/goals lifecycle and orchestration endpoints."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
import httpx
import pytest

from apps.api.app import create_app
from apps.api.dependencies import get_event_bus, set_app_context
from apps.context import AppContext
from core.common.types import FinishReason
from core.inference.types import InferenceResponse, Message, TokenUsage


def _make_mock_response(text: str = "Mock model output") -> InferenceResponse:
    return InferenceResponse(
        request_id="req-mock-goals",
        model_id="test-model",
        message=Message.assistant(text),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        latency_ms=10.0,
    )


@pytest.fixture
def mock_context(tmp_path: Path) -> AppContext:
    mock_core = MagicMock()
    mock_core.repo_root = tmp_path
    mock_connector = MagicMock()
    mock_connector.infer.return_value = _make_mock_response()
    mock_connector.infer_prompt.return_value = _make_mock_response()

    ctx = AppContext(core=mock_core, inference=mock_connector)
    set_app_context(ctx)
    return ctx


@pytest.fixture
def api_client(mock_context: AppContext) -> TestClient:
    app = create_app(app_context=mock_context)
    return TestClient(app)


def test_create_goal_lifecycle(api_client: TestClient):
    """Verify Goal creation returns 201 with proper links."""
    payload = {
        "title": "Review crude distillation P&ID",
        "description": "Inspect equipment tags and generate checklist",
        "inputs": {"file_id": "file-123"},
    }
    resp = api_client.post("/api/v1/goals", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Review crude distillation P&ID"
    assert data["status"] == "pending"
    assert "goal_id" in data
    assert "links" in data
    assert data["links"]["decide"].startswith("/api/v1/goals/")
    assert data["links"]["execute"].startswith("/api/v1/goals/")
    assert data["links"]["events"].startswith("/api/v1/goals/")


def test_get_goal_detail(api_client: TestClient):
    """Verify retrieving goal detail by ID."""
    create_resp = api_client.post("/api/v1/goals", json={"title": "Test Goal Detail"})
    goal_id = create_resp.json()["goal_id"]

    get_resp = api_client.get(f"/api/v1/goals/{goal_id}")
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["goal"]["goal_id"] == goal_id
    assert detail["goal"]["status"] == "pending"


def test_cancel_goal(api_client: TestClient):
    """Verify cancelling an active or pending goal."""
    create_resp = api_client.post("/api/v1/goals", json={"title": "Goal to Cancel"})
    goal_id = create_resp.json()["goal_id"]

    cancel_resp = api_client.post(f"/api/v1/goals/{goal_id}/cancel")
    assert cancel_resp.status_code == 200
    cancel_data = cancel_resp.json()
    assert cancel_data["goal_id"] == goal_id
    assert cancel_data["status"] == "cancelled"

    # Verify state is updated
    get_resp = api_client.get(f"/api/v1/goals/{goal_id}")
    assert get_resp.json()["goal"]["status"] == "cancelled"


def test_decide_goal_deterministic_route(api_client: TestClient):
    """Verify decide endpoint processes through intent router."""
    create_resp = api_client.post(
        "/api/v1/goals",
        json={"title": "ping system health", "description": "health check"},
    )
    goal_id = create_resp.json()["goal_id"]

    decide_resp = api_client.post(f"/api/v1/goals/{goal_id}/decide")
    assert decide_resp.status_code == 200
    decision = decide_resp.json()
    assert decision["goal_id"] == goal_id
    assert decision["strategy"] in ("direct_deterministic", "direct_capability", "plan_required")


def test_goal_sse_stream_connect(mock_context: AppContext):
    """Verify SSE streaming connection connects and returns events cleanly."""
    async def _test():
        app = create_app(app_context=mock_context)
        event_bus = get_event_bus()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            create_resp = await client.post("/api/v1/goals", json={"title": "Goal for SSE"})
            assert create_resp.status_code == 201
            goal_id = create_resp.json()["goal_id"]

            async def _publish_terminal():
                await asyncio.sleep(0.05)
                await event_bus.publish(
                    goal_id=goal_id,
                    event_type="goal.completed",
                    data={"status": "completed"},
                )

            asyncio.create_task(_publish_terminal())

            resp = await client.get(f"/api/v1/goals/{goal_id}/events")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = resp.text
            assert "event: stream.connected" in body
            assert f'"goal_id": "{goal_id}"' in body
            assert "event: goal.completed" in body
            assert '"status": "completed"' in body

    asyncio.run(_test())


def test_decide_goal_multi_task_dag_dependency_serialization(api_client: TestClient, mock_context: AppContext):
    """Regression Test for Defect 1: Verify multi-task DAG dependencies serialize cleanly to string IDs."""
    from unittest.mock import patch
    from orchestration.decision.types import DecisionResult
    from orchestration.domain.dependencies import Dependency
    from orchestration.planning.types import CandidatePlan, CandidateTask
    from orchestration.routing.types import ExecutionStrategy, RouteResult
    from orchestration.validation.types import ValidationResult

    dep1 = Dependency("task-1", "task-2")
    dep2 = Dependency("task-2", "task-3")
    cand = CandidatePlan(
        plan_id="plan-dag-123",
        goal_id="goal-dag-test",
        title="Three-step pipeline",
        tasks=[
            CandidateTask(task_id="task-1", title="Step 1", capability_id="code.workspace"),
            CandidateTask(task_id="task-2", title="Step 2", capability_id="artifact.generate", dependencies=[dep1]),
            CandidateTask(task_id="task-3", title="Step 3", capability_id="document.understand", dependencies=[dep2]),
        ],
        dependencies=[dep1, dep2],
    )

    mock_dec_engine = MagicMock()
    mock_dec_engine.process_goal_async = AsyncMock(return_value=DecisionResult(
        decision_type=ExecutionStrategy.PLAN_REQUIRED,
        goal_id="goal-dag-test",
        plan_id="plan-dag-123",
        route_result=RouteResult(
            route_name="complex_workflow",
            strategy=ExecutionStrategy.PLAN_REQUIRED,
            confidence=1.0,
            stage_resolved="stage1_rules",
        ),
        candidate_plan=cand,
        validation_result=ValidationResult(is_valid=True, errors=[]),
    ))

    with patch.object(AppContext, "create_decision_engine", return_value=mock_dec_engine):
        # 1. Create goal
        create_resp = api_client.post("/api/v1/goals", json={"title": "Run DAG pipeline"})
        assert create_resp.status_code == 201
        goal_id = create_resp.json()["goal_id"]

        # 2. Decide goal
        decide_resp = api_client.post(f"/api/v1/goals/{goal_id}/decide")
        assert decide_resp.status_code == 200
        plan_data = decide_resp.json()
        assert plan_data["is_valid"] is True
        assert len(plan_data["tasks"]) == 3

        # Check dependency string serialization
        assert plan_data["tasks"][0]["dependencies"] == []
        assert plan_data["tasks"][1]["dependencies"] == ["task-1"]
        assert plan_data["tasks"][2]["dependencies"] == ["task-2"]


def test_goal_execution_preserves_results_and_artifacts(api_client: TestClient, mock_context: AppContext, tmp_path: Path):
    """Regression Test for Defect 2: Verify direct execution results and artifacts are preserved in GET /goals/{id}."""
    from unittest.mock import patch
    from orchestration.decision.types import DecisionResult
    from orchestration.domain.references import ArtifactReference
    from orchestration.domain.results import TaskResult
    from orchestration.domain.types import GoalStatus
    from orchestration.orchestrator import DirectGoalResult
    from orchestration.routing.types import ExecutionStrategy

    dummy_art_file = tmp_path / "equipment_register.xlsx"
    dummy_art_file.write_text("dummy xlsx content", encoding="utf-8")

    art_ref = ArtifactReference(
        artifact_id="art-test-999",
        name="equipment_register.xlsx",
        uri=dummy_art_file.as_uri(),
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        size_bytes=dummy_art_file.stat().st_size,
    )
    task_res = TaskResult(
        output={"status": "completed", "row_count": 42},
        artifacts=[art_ref],
    )

    async def _mock_process(goal, execute=True):
        if execute:
            goal.status = GoalStatus.COMPLETED
            return DecisionResult(
                decision_type=ExecutionStrategy.DIRECT_CAPABILITY,
                goal_id=goal.goal_id,
                direct_result=DirectGoalResult(goal=goal, result=task_res),
            )
        return DecisionResult(
            decision_type=ExecutionStrategy.DIRECT_CAPABILITY,
            goal_id=goal.goal_id,
        )

    mock_dec_engine = MagicMock()
    mock_dec_engine.process_goal_async = _mock_process

    with patch.object(AppContext, "create_decision_engine", return_value=mock_dec_engine):
        # 1. Create goal
        create_resp = api_client.post("/api/v1/goals", json={"title": "Generate register"})
        goal_id = create_resp.json()["goal_id"]

        # 2. Execute goal
        exec_resp = api_client.post(f"/api/v1/goals/{goal_id}/execute")
        assert exec_resp.status_code == 202

        # Give background task a moment to run
        import time
        time.sleep(0.1)

        # 3. Retrieve goal detail
        detail_resp = api_client.get(f"/api/v1/goals/{goal_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()

        assert detail["goal"]["status"] == "completed"
        assert "direct" in detail["results"]
        assert detail["results"]["direct"]["row_count"] == 42
        assert len(detail["artifacts"]) == 1
        assert detail["artifacts"][0]["artifact_id"] == "art-test-999"
        assert detail["artifacts"][0]["download_url"] == "/api/v1/artifacts/art-test-999/download"


