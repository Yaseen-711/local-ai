"""Unit tests for VisionInspectionCapability."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.common.types import FinishReason
from core.inference.types import InferenceRequest, InferenceResponse, Message, TokenUsage
from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.vision import VisionInspectionCapability
from orchestration.capabilities.registry import CapabilityRegistry


def test_vision_capability_id():
    """VisionInspectionCapability returns canonical id 'vision.inspect'."""
    cap = VisionInspectionCapability(connector=MagicMock())
    assert cap.capability_id == "vision.inspect"


def test_vision_capability_missing_image_path():
    """Execution without image_path raises ValueError."""
    cap = VisionInspectionCapability(connector=MagicMock())
    ctx = CapabilityContext(execution_id="exec-1")

    with pytest.raises(ValueError, match="requires a valid 'image_path'"):
        cap.execute(parameters={}, inputs={}, context=ctx)


def test_vision_capability_file_not_found():
    """Execution with non-existent file raises FileNotFoundError."""
    cap = VisionInspectionCapability(connector=MagicMock())
    ctx = CapabilityContext(execution_id="exec-2")

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        cap.execute(parameters={"image_path": "/non/existent/file.png"}, inputs={}, context=ctx)


def test_vision_capability_successful_execution(tmp_path: Path):
    """Successful execution formats request, attaches media, and returns TaskResult with provenance."""
    sample_img = tmp_path / "pid_drawing.png"
    sample_img.write_bytes(b"\x89PNG\r\n\x1a\nsample_drawing_bytes")

    mock_connector = MagicMock()
    mock_connector.infer.return_value = InferenceResponse(
        request_id="exec-3",
        model_id="qwen3.5-9b",
        message=Message.assistant("Detected Centrifugal Pump P-101 and Gate Valve V-301."),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=120, completion_tokens=15, total_tokens=135),
        latency_ms=42.0,
    )

    cap = VisionInspectionCapability(connector=mock_connector)
    ctx = CapabilityContext(execution_id="exec-3")

    result = cap.execute(
        parameters={"model_id": "qwen3.5-9b", "temperature": 0.2},
        inputs={"image_path": str(sample_img), "query": "List all valves and pumps"},
        context=ctx,
    )

    # Verify inference connector was invoked correctly
    assert mock_connector.infer.called
    infer_req = mock_connector.infer.call_args[0][0]
    assert infer_req.model_id == "qwen3.5-9b"
    assert infer_req.options.temperature == 0.2
    assert len(infer_req.messages) == 1
    user_msg = infer_req.messages[0]
    assert user_msg.content == "List all valves and pumps"
    assert len(user_msg.attachments) == 1
    assert user_msg.attachments[0].source_path == str(sample_img.resolve())

    # Verify TaskResult structure and provenance references
    assert result.output == "Detected Centrifugal Pump P-101 and Gate Valve V-301."
    assert result.metadata["model_id"] == "qwen3.5-9b"
    assert result.metadata["tokens_used"] == 135
    assert result.metadata["latency_ms"] == 42.0
    assert len(result.references) == 1

    ref = result.references[0]
    assert ref.key == "inspected_image"
    assert ref.uri == f"file://{sample_img.resolve()}"
    assert ref.mime_type == "image/png"
    assert "sha256" in ref.metadata
    assert ref.metadata["size_bytes"] == len(b"\x89PNG\r\n\x1a\nsample_drawing_bytes")


def test_vision_capability_registry_integration():
    """VisionInspectionCapability can be registered and retrieved from CapabilityRegistry."""
    mock_connector = MagicMock()
    cap = VisionInspectionCapability(connector=mock_connector)
    registry = CapabilityRegistry()
    registry.register(cap)

    retrieved = registry.get("vision.inspect")
    assert retrieved is cap
