"""Live multi-model switching integration test against running llama-server in router mode.

Verifies:
1. ModelRegistry discovers and validates both qwen3.5-9b and qwen3.5-0.8b.
2. ModelSelectionPolicy resolves ModelTier.LIGHTWEIGHT -> qwen3.5-0.8b and ModelTier.REASONING -> qwen3.5-9b.
3. Live A -> B -> A model switching against llama-server in native router mode.
4. Correctness assertions check model identity in both normalized response and raw_response["model"].
   (Zero token-generation speed assertions).
5. Router rejects unknown model names with HTTP 400 Bad Request.
6. LlamaCppProvider raises ProviderResponseError on runtime model identity mismatch.
"""

import json
from pathlib import Path
import urllib.error
import urllib.request
import pytest

from core.common.errors import ProviderResponseError
from core.common.types import FinishReason, RuntimeState
from core.foundation import FoundationCore
from core.inference.providers.llama_cpp import LlamaCppProvider
from core.inference.types import GenerationOptions, InferenceRequest
from orchestration.routing.model_selector import ModelSelectionPolicy
from orchestration.routing.types import ModelTier


@pytest.mark.integration
def test_registry_discovers_both_models():
    """Verify ModelRegistry discovers both 9B and 0.8B models and reports advisory availability."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)

    model_9b = core.registry.get_model("qwen3.5-9b")
    assert model_9b.id == "qwen3.5-9b"
    assert "default" in model_9b.aliases
    avail_9b = core.registry.get_availability("qwen3.5-9b")
    assert avail_9b.is_available, f"9B model file not found: {avail_9b.error_message}"

    model_08b = core.registry.get_model("qwen3.5-0.8b")
    assert model_08b.id == "qwen3.5-0.8b"
    assert "fast" in model_08b.aliases
    avail_08b = core.registry.get_availability("qwen3.5-0.8b")
    assert avail_08b.is_available, f"0.8B model file not found: {avail_08b.error_message}"


def test_model_selection_policy_resolves_tiers():
    """Verify ModelSelectionPolicy explicitly maps LIGHTWEIGHT -> 0.8B and REASONING -> 9B."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)

    policy = ModelSelectionPolicy(registry=core.registry)
    assert policy.resolve_model_id(ModelTier.LIGHTWEIGHT) == "qwen3.5-0.8b"
    assert policy.resolve_model_id(ModelTier.REASONING) == "qwen3.5-9b"


@pytest.mark.integration
def test_live_model_switching_end_to_end():
    """Verify live A -> B -> A model switching through FoundationCore and llama-server router."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)

    provider = core.provider_manager.get_provider("llama_cpp")
    health = provider.check_health()
    if health != RuntimeState.READY:
        pytest.skip(
            f"llama-server is not running at {provider.base_url} (state: {health.value}). "
            f"Start via scripts/start_llama_server.sh to run live integration test."
        )

    model_def_9b = core.registry.get_model("qwen3.5-9b")
    model_def_08b = core.registry.get_model("qwen3.5-0.8b")

    if not provider.is_model_loaded(model_def_9b):
        pytest.skip("Model 'qwen3.5-9b' is not reported as loaded by llama-server router.")
    if not provider.is_model_loaded(model_def_08b):
        pytest.skip("Model 'qwen3.5-0.8b' is not reported as loaded by llama-server router.")

    # 1. Execute inference against Model A (qwen3.5-9b)
    req_a1 = InferenceRequest.from_prompt(
        model_id="qwen3.5-9b",
        prompt="Reply with the single word: Alpha",
        options=GenerationOptions(temperature=0.0, max_tokens=16),
    )
    resp_a1 = core.infer(req_a1)

    assert resp_a1.model_id == "qwen3.5-9b"
    assert resp_a1.raw_response is not None
    assert resp_a1.raw_response.get("model") == "qwen3.5-9b"
    assert resp_a1.finish_reason == FinishReason.STOP
    assert len(resp_a1.text.strip()) > 0
    assert resp_a1.usage.prompt_tokens > 0
    assert resp_a1.usage.completion_tokens > 0

    # 2. Execute inference against Model B (qwen3.5-0.8b)
    req_b = InferenceRequest.from_prompt(
        model_id="qwen3.5-0.8b",
        prompt="Reply with the single word: Beta",
        options=GenerationOptions(temperature=0.0, max_tokens=16),
    )
    resp_b = core.infer(req_b)

    assert resp_b.model_id == "qwen3.5-0.8b"
    assert resp_b.raw_response is not None
    assert resp_b.raw_response.get("model") == "qwen3.5-0.8b"
    assert resp_b.finish_reason == FinishReason.STOP
    assert len(resp_b.text.strip()) > 0
    assert resp_b.usage.prompt_tokens > 0
    assert resp_b.usage.completion_tokens > 0

    # 3. Execute inference back to Model A (qwen3.5-9b)
    req_a2 = InferenceRequest.from_prompt(
        model_id="qwen3.5-9b",
        prompt="Reply with the single word: Gamma",
        options=GenerationOptions(temperature=0.0, max_tokens=16),
    )
    resp_a2 = core.infer(req_a2)

    assert resp_a2.model_id == "qwen3.5-9b"
    assert resp_a2.raw_response is not None
    assert resp_a2.raw_response.get("model") == "qwen3.5-9b"
    assert resp_a2.finish_reason == FinishReason.STOP
    assert len(resp_a2.text.strip()) > 0
    assert resp_a2.usage.prompt_tokens > 0
    assert resp_a2.usage.completion_tokens > 0


@pytest.mark.integration
def test_live_unknown_model_rejected_by_router():
    """Verify router rejects unknown model identifiers with HTTP 400 Bad Request."""
    provider = LlamaCppProvider()
    if provider.check_health() != RuntimeState.READY:
        pytest.skip("llama-server router is not running.")

    url = f"{provider.base_url}/v1/chat/completions"
    payload = {
        "model": "non-existent-model-xyz",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5.0)

    assert exc_info.value.code == 400, f"Expected HTTP 400, got {exc_info.value.code}"


def test_hard_mismatch_failure_on_spoofed_response():
    """Verify provider raises hard ProviderResponseError when returned model doesn't match requested."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)
    provider = core.provider_manager.get_provider("llama_cpp")

    model_9b = core.registry.get_model("qwen3.5-9b")
    req = InferenceRequest.from_prompt(model_id="qwen3.5-9b", prompt="Test")

    # Spoofed response claiming 0.8B when 9B was requested
    spoofed_resp = {
        "id": "chatcmpl-spoof",
        "model": "qwen3.5-0.8b",
        "choices": [{"message": {"role": "assistant", "content": "Spoofed output"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    with pytest.raises(ProviderResponseError, match="LlamaCppProvider executed wrong model"):
        provider._normalize_response(spoofed_resp, req, model_9b, latency_ms=5.0)
