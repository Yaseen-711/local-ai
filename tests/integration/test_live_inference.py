"""Opt-in live integration test against running llama-server.

This test requires llama-server to be running (e.g. via scripts/start_llama_server.sh).
If the server is offline, this test is skipped automatically.
"""

from pathlib import Path
import pytest

from core.common.types import FinishReason, RuntimeState
from core.foundation import FoundationCore
from core.inference.providers.llama_cpp import LlamaCppProvider
from core.inference.types import GenerationOptions, InferenceRequest


@pytest.mark.integration
def test_live_llama_server_inference():
    """Verify end-to-end inference when llama-server is active."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)

    # Check if llama.cpp server is reachable
    provider = core.provider_manager.get_provider("llama_cpp")
    health = provider.check_health()
    if health != RuntimeState.READY:
        pytest.skip(
            f"llama-server is not running at {provider.base_url} (state: {health.value}). "
            f"Start via scripts/start_llama_server.sh to run live integration test."
        )

    # Check if qwen3.5-9b is available and loaded
    model_def = core.registry.get_model("qwen3.5-9b")
    if not provider.is_model_loaded(model_def):
        pytest.skip(f"Model '{model_def.id}' is not reported as loaded by llama-server.")

    # Execute inference request
    req = InferenceRequest.from_prompt(
        model_id="qwen3.5-9b",
        prompt="Reply with exactly one short sentence: Local inference is working.",
        options=GenerationOptions(temperature=0.1, max_tokens=64),
    )

    response = core.infer(req)

    assert response.model_id == "qwen3.5-9b"
    assert len(response.text.strip()) > 0
    assert response.finish_reason == FinishReason.STOP
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.latency_ms > 0
