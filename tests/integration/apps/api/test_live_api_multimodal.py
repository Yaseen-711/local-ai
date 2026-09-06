"""Live integration tests for the MRPL FastAPI application with multimodal inference.

Validates end-to-end API workflows against running llama-server on 127.0.0.1:8080:
1. Health and telemetry reporting live llama-server readiness.
2. File upload of real P&ID diagram through /api/v1/files/upload.
3. Direct multimodal vision inspection through /api/v1/direct/vision.
4. Grounded tag detection (P-101, V-301, E-204) via the FastAPI delivery layer.
"""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from apps.api.app import create_app
from apps.context import AppContext
from core.common.types import RuntimeState


def _get_test_image_path() -> Path:
    """Resolve the test P&ID drawing path."""
    repo_root = Path(__file__).resolve().parents[4]
    fixture_path = repo_root / "tests" / "fixtures" / "pid_sample.png"
    if fixture_path.is_file():
        return fixture_path

    alt_path = Path(
        "/home/yaseen/.gemini/antigravity/brain/e178fae7-9de0-4c7f-b11e-8043112a74a0/.user_uploaded/media_1788679058860.png"
    )
    if alt_path.is_file():
        return alt_path

    pytest.skip("Test P&ID image not found on filesystem.")


@pytest.fixture(scope="module")
def live_context() -> AppContext:
    """Initialize real AppContext connected to live llama-server."""
    repo_root = Path(__file__).resolve().parents[4]
    ctx = AppContext.create(repo_root=repo_root)
    provider = ctx.core.provider_manager.get_provider("llama_cpp")
    health = provider.check_health()
    if health != RuntimeState.READY:
        pytest.skip(
            f"llama-server is not running at {provider.base_url} (state: {health.value}). "
            f"Start via scripts/start_llama_server.sh to run live integration tests."
        )

    model_def = ctx.core.registry.get_model("qwen3.5-9b")
    if not provider.is_model_loaded(model_def):
        pytest.skip(f"Model '{model_def.id}' is not loaded by llama-server.")

    return ctx


@pytest.fixture(scope="module")
def live_client(live_context: AppContext) -> TestClient:
    """Create FastAPI TestClient wired to live AppContext."""
    app = create_app(app_context=live_context)
    return TestClient(app)


@pytest.mark.integration
def test_live_api_health_and_telemetry(live_client: TestClient):
    """Verify live API health and model telemetry reflect real llama-server."""
    # 1. Health
    health_resp = live_client.get("/api/v1/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["runtime"] == "ready"

    # 2. Telemetry
    telemetry_resp = live_client.get("/api/v1/telemetry/models")
    assert telemetry_resp.status_code == 200
    telemetry_data = telemetry_resp.json()
    assert "configured_models" in telemetry_data
    assert "runtime_models" in telemetry_data
    assert any(m["id"] == "qwen3.5-9b" for m in telemetry_data["configured_models"])
    assert len(telemetry_data["runtime_models"]) > 0


@pytest.mark.integration
def test_live_api_multimodal_vision_inspection(live_client: TestClient):
    """End-to-end test: upload P&ID drawing and run direct vision inspection via API."""
    image_path = _get_test_image_path()

    # Step 1: Upload image file via /api/v1/files/upload
    with open(image_path, "rb") as f:
        upload_resp = live_client.post(
            "/api/v1/files/upload",
            files={"file": ("pid_sample.png", f, "image/png")},
        )

    assert upload_resp.status_code == 201
    file_info = upload_resp.json()
    assert "file_id" in file_info
    assert file_info["filename"] == "pid_sample.png"
    assert file_info["mime_type"] == "image/png"
    assert file_info["size_bytes"] == image_path.stat().st_size
    assert len(file_info["sha256"]) == 64

    file_id = file_info["file_id"]

    # Step 2: Retrieve file metadata via /api/v1/files/{file_id}
    get_file_resp = live_client.get(f"/api/v1/files/{file_id}")
    assert get_file_resp.status_code == 200
    get_file_info = get_file_resp.json()
    assert get_file_info["file_id"] == file_id
    assert get_file_info["sha256"] == file_info["sha256"]

    # Step 3: Run direct vision inspection via /api/v1/direct/vision using file_id
    vision_payload = {
        "file_id": file_id,
        "prompt": (
            "You are inspecting an industrial P&ID diagram. Identify the primary equipment tags "
            "shown in the drawing, especially looking for equipment like pumps, vessels, and heat exchangers "
            "(e.g., tags matching patterns like P-101, V-301, E-204). List each tag found."
        ),
        "temperature": 0.1,
        "max_tokens": 512,
    }

    vision_resp = live_client.post("/api/v1/direct/vision", json=vision_payload)
    assert vision_resp.status_code == 200
    result = vision_resp.json()

    assert result["status"] == "completed"
    assert result["capability_id"] == "vision.inspect"
    assert result["output"] is not None

    output_text = result["output"].upper()
    # Verify visual grounding: tags detected from P&ID drawing
    assert "P-101" in output_text or "P101" in output_text, f"Expected P-101 in vision output, got:\n{result['output']}"
    assert "V-301" in output_text or "V301" in output_text, f"Expected V-301 in vision output, got:\n{result['output']}"
    assert "E-204" in output_text or "E204" in output_text, f"Expected E-204 in vision output, got:\n{result['output']}"
