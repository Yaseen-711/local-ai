"""Unit tests for /api/v1/artifacts and /api/v1/health & telemetry endpoints."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from apps.api.app import create_app
from apps.api.dependencies import get_artifacts_dir, set_app_context
from apps.context import AppContext
from core.common.types import ModelFormat, RuntimeState
from core.models.schema import ModelCapabilities, ModelDefinition


@pytest.fixture
def mock_context(tmp_path: Path) -> AppContext:
    mock_core = MagicMock()
    mock_core.repo_root = tmp_path

    # Provider mock
    mock_provider = MagicMock()
    mock_provider.check_health.return_value = RuntimeState.READY
    mock_provider.base_url = "http://127.0.0.1:9999"
    mock_core.provider_manager.get_provider.return_value = mock_provider

    # Registry mock
    mock_mdef = ModelDefinition(
        id="qwen3.5-9b",
        display_name="Qwen 3.5 9B",
        format=ModelFormat.GGUF,
        relative_path=Path("models/qwen.gguf"),
        supported_providers=["llama_cpp"],
        aliases=["default"],
        capabilities=ModelCapabilities(chat=True, code=True, reasoning=True, structured_output=True),
    )
    mock_core.registry.list_models.return_value = [mock_mdef]
    mock_core.registry.get_model.return_value = mock_mdef

    mock_connector = MagicMock()
    ctx = AppContext(core=mock_core, inference=mock_connector)
    set_app_context(ctx)
    return ctx


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    return art_dir


@pytest.fixture
def api_client(mock_context: AppContext, artifacts_dir: Path) -> TestClient:
    app = create_app(app_context=mock_context)
    app.dependency_overrides[get_artifacts_dir] = lambda: artifacts_dir
    return TestClient(app)


def test_health_endpoint(api_client: TestClient):
    """Verify /api/v1/health reports status healthy and runtime ready."""
    resp = api_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["runtime"] == "ready"


def test_telemetry_models_endpoint(api_client: TestClient):
    """Verify /api/v1/telemetry/models returns configured models."""
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"data": [{"id": "qwen3.5-9b"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_url.return_value = mock_resp

        resp = api_client.get("/api/v1/telemetry/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "configured_models" in data
        assert len(data["configured_models"]) == 1
        assert data["configured_models"][0]["id"] == "qwen3.5-9b"
        assert len(data["runtime_models"]) == 1


def test_artifact_metadata_and_download(api_client: TestClient, artifacts_dir: Path):
    """Verify metadata retrieval and binary file download for artifacts."""
    sample_file = artifacts_dir / "art_1234_report.md"
    sample_file.write_text("# Industrial Report\nP-101 verified.", encoding="utf-8")

    # 1. Metadata
    resp = api_client.get("/api/v1/artifacts/1234")
    assert resp.status_code == 200
    meta = resp.json()
    assert meta["artifact_id"] == "1234"
    assert meta["name"] == "art_1234_report.md"
    assert meta["mime_type"] == "text/markdown"
    assert meta["size_bytes"] > 0
    assert "sha256" in meta["metadata"]

    # 2. Download
    dl_resp = api_client.get("/api/v1/artifacts/1234/download")
    assert dl_resp.status_code == 200
    assert "ETag" in dl_resp.headers
    assert "P-101 verified." in dl_resp.text


def test_artifact_not_found(api_client: TestClient):
    """Verify 404 on missing artifact."""
    resp = api_client.get("/api/v1/artifacts/nonexistent_id")
    assert resp.status_code == 404
