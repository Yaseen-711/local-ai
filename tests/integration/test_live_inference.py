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
from connectors import FoundationInferenceConnector
from workflows import AnalysisDepth, AnalysisOptions, TextAnalysisWorkflow


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


@pytest.mark.integration
def test_live_llama_server_infer_prompt_with_alias():
    """Verify end-to-end inference using infer_prompt() and a Foundation alias."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)

    provider = core.provider_manager.get_provider("llama_cpp")
    health = provider.check_health()
    if health != RuntimeState.READY:
        pytest.skip(f"llama-server is not running at {provider.base_url}")

    model_def = core.registry.get_model("default")
    if not provider.is_model_loaded(model_def):
        pytest.skip(f"Model '{model_def.id}' is not reported as loaded by llama-server.")

    response = core.infer_prompt(
        model_id="default",
        prompt="Reply with: Alias routing works.",
        options=GenerationOptions(temperature=0.1, max_tokens=32),
    )

    assert response.model_id == "qwen3.5-9b"
    assert len(response.text.strip()) > 0
    assert response.finish_reason == FinishReason.STOP
    assert response.usage.total_tokens > 0
    assert response.latency_ms > 0


@pytest.mark.integration
def test_live_text_analysis_workflow_quick():
    """Verify live single-pass TextAnalysisWorkflow against running llama-server."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)

    provider = core.provider_manager.get_provider("llama_cpp")
    if provider.check_health() != RuntimeState.READY:
        pytest.skip("llama-server is not running")

    model_def = core.registry.get_model("default")
    if not provider.is_model_loaded(model_def):
        pytest.skip(f"Model '{model_def.id}' is not loaded by llama-server.")

    connector = FoundationInferenceConnector(core=core)
    workflow = TextAnalysisWorkflow(inference=connector)

    sample_text = (
        "PostgreSQL 16 introduced substantial improvements to query parallelism and logical replication. "
        "Engineers observed up to a 30% reduction in CPU overhead for high-concurrency analytical queries."
    )

    result = workflow.analyze(
        sample_text,
        options=AnalysisOptions(
            depth=AnalysisDepth.QUICK,
            focus="performance metrics",
            max_tokens=128,
        ),
    )

    assert result.model_id == "qwen3.5-9b"
    assert len(result.output.summary) > 0
    assert result.output.word_count == len(sample_text.split())
    assert result.output.depth == AnalysisDepth.QUICK
    assert result.metadata["steps_executed"] == 1
    assert result.metadata["total_tokens"] > 0
    assert result.metadata["total_inference_latency_ms"] > 0


@pytest.mark.integration
def test_live_text_analysis_workflow_detailed():
    """Verify live two-pass DETAILED TextAnalysisWorkflow against running llama-server."""
    repo_root = Path(__file__).parent.parent.parent.resolve()
    core = FoundationCore.create(repo_root=repo_root)

    provider = core.provider_manager.get_provider("llama_cpp")
    if provider.check_health() != RuntimeState.READY:
        pytest.skip("llama-server is not running")

    model_def = core.registry.get_model("default")
    if not provider.is_model_loaded(model_def):
        pytest.skip(f"Model '{model_def.id}' is not loaded by llama-server.")

    connector = FoundationInferenceConnector(core=core)
    workflow = TextAnalysisWorkflow(inference=connector)

    sample_text = (
        "Database migrations failed due to lock contention during index creation. "
        "The operations team resolved this by scheduling concurrent index builds during off-peak hours, "
        "reducing incident downtime from 45 minutes to zero."
    )

    result = workflow.analyze(
        sample_text,
        options=AnalysisOptions(
            depth=AnalysisDepth.DETAILED,
            focus="resolution steps",
            max_tokens=128,
        ),
    )

    assert result.model_id == "qwen3.5-9b"
    assert len(result.output.summary) > 0
    assert len(result.output.key_points) > 0
    assert result.output.depth == AnalysisDepth.DETAILED
    assert result.metadata["steps_executed"] == 2
    assert result.metadata["total_tokens"] > 0
    assert result.metadata["total_inference_latency_ms"] > 0
    assert "extraction" in result.metadata["phase_inference_latencies_ms"]
    assert "synthesis" in result.metadata["phase_inference_latencies_ms"]
