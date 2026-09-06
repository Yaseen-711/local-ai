"""Integration tests for Artifacts Delivery API and Industrial Templates.

Verifies:
1. POST /api/v1/direct/artifact creates generic PPTX.
2. POST /api/v1/direct/artifact creates DOCX Technical Approval Note via template.
3. POST /api/v1/direct/artifact creates XLSX Calculation Workbook via template.
4. POST /api/v1/direct/artifact creates PPTX Executive Presentation via template.
5. GET /api/v1/artifacts/{id} returns accurate metadata and SHA-256 digest.
6. GET /api/v1/artifacts/{id}/download serves binary content matching the SHA-256 ETag.
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from apps.api.app import create_app
from apps.api.dependencies import set_app_context
from apps.context import AppContext
from core.common.types import FinishReason
from core.inference.types import InferenceResponse, Message, TokenUsage


@pytest.fixture
def test_app_context(tmp_path: Path) -> AppContext:
    mock_core = MagicMock()
    mock_core.repo_root = tmp_path
    mock_connector = MagicMock()
    ctx = AppContext(core=mock_core, inference=mock_connector)
    set_app_context(ctx)
    return ctx


@pytest.fixture
def client(test_app_context: AppContext) -> TestClient:
    app = create_app(app_context=test_app_context)
    return TestClient(app)


def test_api_direct_generic_pptx_and_download(client: TestClient):
    """Verify creating and downloading a generic PPTX presentation via the API."""
    payload = {
        "artifact_type": "pptx",
        "filename": "quarterly_deck.pptx",
        "title": "Q3 Operations Deck",
        "content": "## Slide 1\n- High throughput achieved\n- Safety score 100%",
        "data": [
            ["Metric", "Value"],
            ["Availability", "99.4%"],
        ],
    }
    resp = client.post("/api/v1/direct/artifact", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["capability_id"] == "artifact.generate"
    assert data["status"] == "completed"
    assert len(data["artifacts"]) == 1

    art_info = data["artifacts"][0]
    art_id = art_info["artifact_id"]
    assert art_info["name"] == "quarterly_deck.pptx"
    assert art_info["mime_type"] == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    assert art_info["download_url"] == f"/api/v1/artifacts/{art_id}/download"

    sha256 = art_info["metadata"]["sha256"]
    assert len(sha256) == 64

    # 1. Inspect metadata endpoint
    meta_resp = client.get(f"/api/v1/artifacts/{art_id}")
    assert meta_resp.status_code == 200
    meta_data = meta_resp.json()
    assert meta_data["artifact_id"] == art_id
    assert meta_data["metadata"]["sha256"] == sha256

    # 2. Download binary
    dl_resp = client.get(f"/api/v1/artifacts/{art_id}/download")
    assert dl_resp.status_code == 200
    assert dl_resp.headers["etag"] == f'"{sha256}"'
    assert dl_resp.headers["content-type"] == "application/vnd.openxmlformats-officedocument.presentationml.presentation"

    # Confirm cryptographic hash of downloaded bytes strictly matches ETag
    dl_bytes = dl_resp.content
    assert hashlib.sha256(dl_bytes).hexdigest() == sha256


def test_api_direct_docx_approval_note_template(client: TestClient):
    """Verify creating and downloading a Technical Approval Note DOCX via API."""
    payload = {
        "artifact_type": "docx",
        "template": "approval_note",
        "filename": "moc_approval_note.docx",
    }
    resp = client.post("/api/v1/direct/artifact", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    art_info = data["artifacts"][0]
    art_id = art_info["artifact_id"]
    assert art_info["name"] == "moc_approval_note.docx"
    assert art_info["mime_type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    dl_resp = client.get(f"/api/v1/artifacts/{art_id}/download")
    assert dl_resp.status_code == 200
    assert len(dl_resp.content) > 0
    assert hashlib.sha256(dl_resp.content).hexdigest() == art_info["metadata"]["sha256"]


def test_api_direct_xlsx_calculation_workbook_template(client: TestClient):
    """Verify creating and downloading an Engineering Calculation Workbook XLSX via API."""
    payload = {
        "artifact_type": "xlsx",
        "template": "calculation_workbook",
        "filename": "relief_verification.xlsx",
    }
    resp = client.post("/api/v1/direct/artifact", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    art_info = data["artifacts"][0]
    art_id = art_info["artifact_id"]
    assert art_info["name"] == "relief_verification.xlsx"
    assert art_info["mime_type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    dl_resp = client.get(f"/api/v1/artifacts/{art_id}/download")
    assert dl_resp.status_code == 200
    assert hashlib.sha256(dl_resp.content).hexdigest() == art_info["metadata"]["sha256"]


def test_api_direct_pptx_executive_presentation_template(client: TestClient):
    """Verify creating and downloading an Executive Presentation PPTX via API."""
    payload = {
        "artifact_type": "pptx",
        "template": "executive_presentation",
        "filename": "board_briefing.pptx",
    }
    resp = client.post("/api/v1/direct/artifact", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    art_info = data["artifacts"][0]
    art_id = art_info["artifact_id"]
    assert art_info["name"] == "board_briefing.pptx"
    assert art_info["mime_type"] == "application/vnd.openxmlformats-officedocument.presentationml.presentation"

    dl_resp = client.get(f"/api/v1/artifacts/{art_id}/download")
    assert dl_resp.status_code == 200
    assert hashlib.sha256(dl_resp.content).hexdigest() == art_info["metadata"]["sha256"]
