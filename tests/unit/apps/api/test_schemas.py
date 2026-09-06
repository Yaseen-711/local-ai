"""Unit tests for MRPL API Pydantic v2 schemas."""

from datetime import datetime, timezone
import pytest

from apps.api.schemas.common import ArtifactReferenceSchema, DataReferenceSchema, ProblemDetail
from apps.api.schemas.direct import (
    DirectArtifactRequest,
    DirectCapabilityResponse,
    DirectDocumentRequest,
    DirectTextAnalysisRequest,
    DirectVisionRequest,
)
from apps.api.schemas.files import FileMetadataResponse, FileUploadResponse
from apps.api.schemas.goals import (
    CandidatePlanResponse,
    CancelGoalResponse,
    CreateGoalRequest,
    GoalDetailResponse,
    GoalExecutionResponse,
    GoalResponse,
    TaskSchema,
)


def test_problem_detail_schema():
    """Verify RFC 7807 ProblemDetail serialization."""
    problem = ProblemDetail(
        title="Invalid Request",
        status=400,
        detail="The parameter 'model_id' is unknown.",
        instance="/api/v1/direct/vision",
    )
    dumped = problem.model_dump(exclude_none=True)
    assert dumped["status"] == 400
    assert dumped["title"] == "Invalid Request"
    assert dumped["detail"] == "The parameter 'model_id' is unknown."
    assert dumped["instance"] == "/api/v1/direct/vision"


def test_file_upload_schema():
    """Verify FileUploadResponse schema."""
    resp = FileUploadResponse(
        file_id="file-12345",
        filename="drawing.png",
        mime_type="image/png",
        size_bytes=1024,
        sha256="abcdef1234567890",
        uri="file:///path/to/file",
        created_at=datetime.now(timezone.utc),
    )
    assert resp.file_id == "file-12345"
    assert resp.size_bytes == 1024


def test_direct_vision_request_defaults():
    """Verify DirectVisionRequest defaults and validation."""
    req = DirectVisionRequest(file_id="file-123")
    assert req.file_id == "file-123"
    assert req.model_id == "qwen3.5-9b"
    assert req.temperature == 0.1
    assert req.max_tokens == 512


def test_create_goal_request_schema():
    """Verify CreateGoalRequest schema."""
    req = CreateGoalRequest(
        title="Review P&ID",
        description="Inspect all valves",
        inputs={"file_id": "file-123"},
        parameters={"priority": "high"},
    )
    assert req.title == "Review P&ID"
    assert req.inputs["file_id"] == "file-123"
    assert req.parameters["priority"] == "high"


def test_direct_capability_response_mapping():
    """Verify DirectCapabilityResponse with references and artifacts."""
    data_ref = DataReferenceSchema(
        key="image_provenance",
        uri="file:///path/to/img.png",
        mime_type="image/png",
        metadata={"sha256": "digest123"},
    )
    art_ref = ArtifactReferenceSchema(
        artifact_id="art-456",
        name="report.xlsx",
        uri="file:///path/to/report.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        size_bytes=4096,
        download_url="/api/v1/artifacts/art-456/download",
    )
    resp = DirectCapabilityResponse(
        capability_id="vision.inspect",
        status="completed",
        output="Inspection finished.",
        metadata={"latency_ms": 50.0},
        references=[data_ref],
        artifacts=[art_ref],
    )
    assert resp.capability_id == "vision.inspect"
    assert len(resp.references) == 1
    assert len(resp.artifacts) == 1
    assert resp.artifacts[0].download_url == "/api/v1/artifacts/art-456/download"
