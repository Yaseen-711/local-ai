"""Unit tests for /api/v1/direct endpoints."""

from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from apps.api.app import create_app
from apps.api.dependencies import set_app_context
from apps.context import AppContext
from core.common.types import FinishReason
from core.inference.types import InferenceResponse, Message, TokenUsage


def _make_mock_response(text: str = "Mock capability output") -> InferenceResponse:
    return InferenceResponse(
        request_id="req-mock-dir",
        model_id="test-model",
        message=Message.assistant(text),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        latency_ms=12.0,
    )


@pytest.fixture
def mock_context(tmp_path: Path) -> AppContext:
    """Create AppContext with mock inference connector."""
    mock_core = MagicMock()
    mock_core.repo_root = tmp_path
    mock_connector = MagicMock()
    mock_connector.infer.return_value = _make_mock_response("P&ID Tag P-101 detected")
    mock_connector.infer_prompt.return_value = _make_mock_response("Analysis summary: all operational.")

    ctx = AppContext(core=mock_core, inference=mock_connector)
    set_app_context(ctx)
    return ctx


@pytest.fixture
def api_client(mock_context: AppContext) -> TestClient:
    app = create_app(app_context=mock_context)
    return TestClient(app)


def test_direct_artifact_generation(api_client: TestClient, tmp_path: Path):
    """Verify direct synchronous artifact generation creates valid artifact."""
    payload = {
        "artifact_type": "xlsx",
        "filename": "direct_test.xlsx",
        "title": "Direct Test",
        "data": {
            "Sheet1": [["Header1", "Header2"], ["Val1", "Val2"]]
        },
    }
    resp = api_client.post("/api/v1/direct/artifact", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["capability_id"] == "artifact.generate"
    assert data["status"] == "completed"
    assert len(data["artifacts"]) == 1
    assert data["artifacts"][0]["name"] == "direct_test.xlsx"
    assert data["artifacts"][0]["download_url"] is not None


def test_direct_text_analysis(api_client: TestClient):
    """Verify direct synchronous text analysis workflow execution."""
    payload = {
        "text": "Refinery inspection completed on crude distillation unit. All valves operational.",
        "depth": "quick",
    }
    resp = api_client.post("/api/v1/direct/text-analysis", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["capability_id"] == "workflow.text_analysis"
    assert data["status"] == "completed"
    assert "output" in data


def test_direct_vision_inspection_with_file_path(api_client: TestClient, tmp_path: Path):
    """Verify direct vision inspection using file_path."""
    sample_img = tmp_path / "drawing.png"
    sample_img.write_bytes(b"\x89PNG\r\n\x1a\nsample")

    payload = {
        "file_path": str(sample_img),
        "query": "Identify pumps and valves",
    }
    resp = api_client.post("/api/v1/direct/vision", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["capability_id"] == "vision.inspect"
    assert "P-101" in data["output"]
    assert len(data["references"]) == 1
    assert data["references"][0]["key"] == "inspected_image"
