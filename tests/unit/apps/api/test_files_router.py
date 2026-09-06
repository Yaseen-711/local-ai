"""Unit tests for /api/v1/files endpoints."""

import io
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from apps.api.app import create_app
from apps.api.dependencies import set_app_context
from apps.context import AppContext


@pytest.fixture
def api_client(tmp_path: Path) -> TestClient:
    """Create test client with isolated AppContext and staging directory."""
    # Build minimal tmp environment for AppContext
    configs_dir = tmp_path / "configs" / "models"
    models_dir = tmp_path / "models" / "gguf"
    staging_dir = tmp_path / ".staging" / "uploads"
    configs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    staging_dir.mkdir(parents=True)

    (models_dir / "test-model.gguf").write_bytes(b"mock-weights")
    (configs_dir / "test-model.toml").write_bytes(b"""\
[model]
id = "test-model"
format = "gguf"
path = "models/gguf/test-model.gguf"
supported_providers = ["llama_cpp"]
""")

    settings_file = tmp_path / "settings.toml"
    settings_file.write_bytes(b"""\
[foundation]
environment = "unit-test"
models_dir = "models"
configs_dir = "configs/models"

[providers.llama_cpp]
base_url = "http://127.0.0.1:8080"
""")

    ctx = AppContext.create(
        repo_root=tmp_path,
        configs_dir=configs_dir,
        settings_path=settings_file,
    )
    set_app_context(ctx)
    app = create_app(app_context=ctx)
    return TestClient(app)


def test_upload_file_success(api_client: TestClient):
    """Verify standard file upload, streaming, and metadata hashing."""
    sample_content = b"\x89PNG\r\n\x1a\nfakeimagecontentforuploadtest"
    file_tuple = ("pid_drawing.png", io.BytesIO(sample_content), "image/png")

    resp = api_client.post("/api/v1/files/upload", files={"file": file_tuple})
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "pid_drawing.png"
    assert data["mime_type"] == "image/png"
    assert data["size_bytes"] == len(sample_content)
    assert len(data["sha256"]) == 64
    assert data["file_id"].startswith("file-")

    # Verify retrieval
    meta_resp = api_client.get(f"/api/v1/files/{data['file_id']}")
    assert meta_resp.status_code == 200
    meta_data = meta_resp.json()
    assert meta_data["sha256"] == data["sha256"]
    assert meta_data["exists"] is True


def test_upload_empty_file_rejected(api_client: TestClient):
    """Empty files must be rejected with 400 Bad Request."""
    empty_tuple = ("empty.png", io.BytesIO(b""), "image/png")
    resp = api_client.post("/api/v1/files/upload", files={"file": empty_tuple})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_upload_oversized_file_rejected(api_client: TestClient, monkeypatch):
    """Files exceeding size limit must be rejected with 413."""
    # Temporarily monkeypatch DEFAULT_MAX_ATTACHMENT_BYTES in files router to small limit
    from apps.api.routers import files
    monkeypatch.setattr(files, "DEFAULT_MAX_ATTACHMENT_BYTES", 64)

    oversized_data = b"x" * 128
    file_tuple = ("large.png", io.BytesIO(oversized_data), "image/png")
    resp = api_client.post("/api/v1/files/upload", files={"file": file_tuple})
    assert resp.status_code == 413
    assert "exceeds allowed limit" in resp.json()["detail"]


def test_get_file_not_found(api_client: TestClient):
    """Non-existent file_id returns 404."""
    resp = api_client.get("/api/v1/files/file-nonexistent999")
    assert resp.status_code == 404
