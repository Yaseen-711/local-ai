"""Unit tests for FoundationCore coordinator and factory."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.common.types import FinishReason, RuntimeState
from core.foundation import FoundationCore
from core.inference.types import InferenceRequest, InferenceResponse, Message, TokenUsage


def test_foundation_core_factory_and_infer(tmp_path: Path):
    """Verify FoundationCore factory initialization and end-to-end routing with mocks."""
    configs_dir = tmp_path / "configs" / "models"
    models_dir = tmp_path / "models" / "gguf"
    configs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Create mock model file
    model_file = models_dir / "qwen.gguf"
    model_file.write_bytes(b"mock data")

    # Configs
    (configs_dir / "qwen.toml").write_bytes(b"""
    [model]
    id = "qwen-test"
    format = "gguf"
    path = "models/gguf/qwen.gguf"
    supported_providers = ["llama_cpp"]
    """)

    settings_file = tmp_path / "settings.toml"
    settings_file.write_bytes(b"""
    [foundation]
    environment = "unit-test"
    models_dir = "models"
    configs_dir = "configs/models"

    [providers.llama_cpp]
    base_url = "http://127.0.0.1:8080"
    """)

    core = FoundationCore.create(
        repo_root=tmp_path,
        configs_dir=configs_dir,
        settings_path=settings_file,
    )

    assert core.registry.is_known("qwen-test")
    assert core.provider_manager.get_provider("llama_cpp") is not None

    # Mock the llama_cpp provider infer call
    llama_prov = core.provider_manager.get_provider("llama_cpp")
    mock_resp = InferenceResponse(
        request_id="req-1",
        model_id="qwen-test",
        message=Message.assistant("Test response output"),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        latency_ms=12.0,
    )
    llama_prov.infer = MagicMock(return_value=mock_resp)

    req = InferenceRequest.from_prompt(model_id="qwen-test", prompt="Hello")
    resp = core.infer(req)

    assert resp.model_id == "qwen-test"
    assert resp.text == "Test response output"
    assert resp.usage.total_tokens == 15


def test_foundation_core_infer_prompt(tmp_path: Path):
    """Verify FoundationCore.infer_prompt convenience method."""
    configs_dir = tmp_path / "configs" / "models"
    models_dir = tmp_path / "models" / "gguf"
    configs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    model_file = models_dir / "qwen.gguf"
    model_file.write_bytes(b"mock data")

    (configs_dir / "qwen.toml").write_bytes(b"""
    [model]
    id = "qwen-test"
    format = "gguf"
    path = "models/gguf/qwen.gguf"
    supported_providers = ["llama_cpp"]
    """)

    settings_file = tmp_path / "settings.toml"
    settings_file.write_bytes(b"""
    [foundation]
    environment = "unit-test"
    models_dir = "models"
    configs_dir = "configs/models"

    [providers.llama_cpp]
    base_url = "http://127.0.0.1:8080"
    """)

    core = FoundationCore.create(
        repo_root=tmp_path,
        configs_dir=configs_dir,
        settings_path=settings_file,
    )

    llama_prov = core.provider_manager.get_provider("llama_cpp")
    mock_resp = InferenceResponse(
        request_id="req-2",
        model_id="qwen-test",
        message=Message.assistant("Single prompt response"),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=4, completion_tokens=8, total_tokens=12),
        latency_ms=10.5,
    )
    llama_prov.infer = MagicMock(return_value=mock_resp)

    resp = core.infer_prompt(
        model_id="qwen-test",
        prompt="Explain quicksort",
        system_prompt="You are a teacher.",
    )

    assert resp.model_id == "qwen-test"
    assert resp.text == "Single prompt response"
    assert resp.usage.total_tokens == 12

    # Verify infer was called with correctly constructed normalized request
    llama_prov.infer.assert_called_once()
    called_request, called_model_def = llama_prov.infer.call_args[0]
    assert called_request.model_id == "qwen-test"
    assert len(called_request.messages) == 2
    assert called_request.messages[0].content == "You are a teacher."
    assert called_request.messages[1].content == "Explain quicksort"
    assert called_model_def.id == "qwen-test"
