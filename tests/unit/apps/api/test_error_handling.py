"""Unit tests for RFC 7807 exception handling in MRPL API."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from apps.api.app import create_app
from core.common.errors import (
    InferenceError,
    ModelNotFoundError,
    ModelUnavailableError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from orchestration.errors import (
    CapabilityNotFoundError,
    OrchestrationError,
    PlanValidationError,
)


@pytest.fixture
def test_client() -> TestClient:
    """Create test client with custom route triggers to test global exception handlers."""
    app = create_app()

    @app.get("/test/plan-validation-error")
    def trigger_plan_val():
        raise PlanValidationError("Cyclic dependency detected in task DAG: task-1 -> task-2 -> task-1")

    @app.get("/test/capability-not-found")
    def trigger_cap_not_found():
        raise CapabilityNotFoundError("Capability 'unknown.tool' not found.")

    @app.get("/test/model-not-found")
    def trigger_model_not_found():
        raise ModelNotFoundError("Model 'nonexistent-model' is not configured.")

    @app.get("/test/provider-unavailable")
    def trigger_provider_unavailable():
        raise ProviderUnavailableError("llama-server connection refused on 127.0.0.1:8080.")

    @app.get("/test/inference-error")
    def trigger_inference_error():
        raise InferenceError("llama-server context window exceeded.")

    @app.get("/test/file-not-found")
    def trigger_file_not_found():
        raise FileNotFoundError("Image file /data/drawing.png does not exist.")

    @app.get("/test/value-error")
    def trigger_value_error():
        raise ValueError("Invalid temperature: 3.5 exceeds maximum allowed 2.0.")

    return TestClient(app, raise_server_exceptions=False)


def test_plan_validation_error_maps_to_422(test_client: TestClient):
    resp = test_client.get("/test/plan-validation-error")
    assert resp.status_code == 422
    assert resp.headers["content-type"] == "application/problem+json"
    data = resp.json()
    assert data["title"] == "Plan Validation Failed"
    assert data["status"] == 422
    assert "Cyclic dependency" in data["detail"]


def test_capability_not_found_maps_to_404(test_client: TestClient):
    resp = test_client.get("/test/capability-not-found")
    assert resp.status_code == 404
    data = resp.json()
    assert data["title"] == "Capability Not Found"
    assert data["status"] == 404


def test_model_not_found_maps_to_404(test_client: TestClient):
    resp = test_client.get("/test/model-not-found")
    assert resp.status_code == 404
    data = resp.json()
    assert data["title"] == "Model Not Found"


def test_provider_unavailable_maps_to_503(test_client: TestClient):
    resp = test_client.get("/test/provider-unavailable")
    assert resp.status_code == 503
    data = resp.json()
    assert data["title"] == "Runtime Provider Offline"
    assert data["status"] == 503


def test_inference_error_maps_to_502(test_client: TestClient):
    resp = test_client.get("/test/inference-error")
    assert resp.status_code == 502
    data = resp.json()
    assert data["title"] == "Inference Error"


def test_file_not_found_maps_to_404(test_client: TestClient):
    resp = test_client.get("/test/file-not-found")
    assert resp.status_code == 404
    data = resp.json()
    assert data["title"] == "File Not Found"


def test_value_error_maps_to_400(test_client: TestClient):
    resp = test_client.get("/test/value-error")
    assert resp.status_code == 400
    data = resp.json()
    assert data["title"] == "Invalid Request Parameter"
