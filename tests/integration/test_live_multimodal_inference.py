"""Live integration tests for multimodal inference and vision inspection capability.

Validates end-to-end multimodal inference against running llama-server on 127.0.0.1:8080.
Verifies that Qwen3.5-9B + mmproj actually parses and visually grounds equipment tags
(P-101, V-301, E-204) from a real engineering P&ID drawing.
"""

from pathlib import Path
import pytest

from apps import AppContext
from core.common.types import FinishReason, RuntimeState
from core.foundation import FoundationCore
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    MediaAttachment,
    Message,
)
from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.vision import VisionInspectionCapability
from orchestration.domain.results import TaskResult


def _get_test_image_path() -> Path:
    """Resolve the test P&ID drawing path."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    fixture_path = repo_root / "tests" / "fixtures" / "pid_sample.png"
    if fixture_path.is_file():
        return fixture_path

    # Fallback to user-uploaded conversation artifact
    alt_path = Path(
        "/home/yaseen/.gemini/antigravity/brain/e178fae7-9de0-4c7f-b11e-8043112a74a0/.user_uploaded/media_1788679058860.png"
    )
    if alt_path.is_file():
        return alt_path

    pytest.skip("Test P&ID image not found on filesystem.")


def _ensure_live_server_with_multimodal(core: FoundationCore):
    """Ensure llama-server is healthy and qwen3.5-9b is available."""
    provider = core.provider_manager.get_provider("llama_cpp")
    health = provider.check_health()
    if health != RuntimeState.READY:
        pytest.skip(
            f"llama-server is not running at {provider.base_url} (state: {health.value}). "
            f"Start via scripts/start_llama_server.sh to run live integration tests."
        )

    model_def = core.registry.get_model("qwen3.5-9b")
    if not provider.is_model_loaded(model_def):
        pytest.skip(f"Model '{model_def.id}' is not loaded by llama-server.")


@pytest.mark.integration
def test_live_multimodal_inference_visual_grounding():
    """Verify live Qwen3.5-9B multimodal inference correctly identifies P&ID tags."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)
    _ensure_live_server_with_multimodal(core)

    image_path = _get_test_image_path()
    attachment = MediaAttachment.from_file(image_path)

    prompt = (
        "You are an expert industrial engineer. Inspect this Piping & Instrumentation Diagram (P&ID). "
        "List all equipment tags (e.g. pumps, exchangers) and valve tags visible in the drawing. "
        "Be concise and exact."
    )

    request = InferenceRequest(
        model_id="qwen3.5-9b",
        messages=[
            Message.user(
                content=prompt,
                attachments=[attachment],
            )
        ],
        options=GenerationOptions(temperature=0.1, max_tokens=512),
    )

    response = core.infer(request)

    assert response.finish_reason == FinishReason.STOP
    assert response.model_id == "qwen3.5-9b"
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0

    text = response.text.upper()
    # Must demonstrate genuine visual understanding of the diagram
    # Specifically tags P-101 (feed pump), V-301 (valve), or E-204 (heat exchanger)
    has_pump = "P-101" in text or "P101" in text
    has_valve = "V-301" in text or "V301" in text
    has_exchanger = "E-204" in text or "E204" in text

    grounded_matches = [tag for tag, found in [("P-101", has_pump), ("V-301", has_valve), ("E-204", has_exchanger)] if found]
    assert len(grounded_matches) >= 2, (
        f"Multimodal inference failed visual grounding on P&ID. "
        f"Expected at least 2 of [P-101, V-301, E-204], found {grounded_matches}. "
        f"Response text: {response.text}"
    )


@pytest.mark.integration
def test_live_vision_inspection_capability():
    """Verify VisionInspectionCapability produces grounded TaskResult and DataReference provenance."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    ctx = AppContext.create(repo_root=repo_root)
    _ensure_live_server_with_multimodal(ctx.core)

    image_path = _get_test_image_path()
    cap = ctx.create_vision_inspection_capability()
    cap_ctx = CapabilityContext(execution_id="live-vision-exec-001")

    result = cap.execute(
        parameters={"model_id": "qwen3.5-9b", "temperature": 0.1, "max_tokens": 512},
        inputs={
            "image_path": str(image_path),
            "query": "Identify the main pump tag and isolation valve tag in this diagram.",
        },
        context=cap_ctx,
    )

    assert isinstance(result, TaskResult)
    assert result.output is not None
    assert len(result.references) == 1
    ref = result.references[0]
    assert ref.key == "inspected_image"
    assert ref.mime_type == "image/png"
    assert "sha256" in ref.metadata

    output_upper = result.output.upper()
    has_p101 = "P-101" in output_upper or "P101" in output_upper
    has_v301 = "V-301" in output_upper or "V301" in output_upper
    assert has_p101 or has_v301, (
        f"Vision capability failed visual grounding. "
        f"Expected P-101 or V-301 in output. Output: {result.output}"
    )
