"""Unit tests for ProviderManager and provider routing."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.common.errors import (
    ModelNotFoundError,
    ProviderNotFoundError,
)
from core.common.types import FinishReason, MessageRole, ModelFormat, RuntimeState
from core.inference.manager import ProviderManager
from core.inference.provider import BaseProvider
from core.inference.types import (
    InferenceRequest,
    InferenceResponse,
    Message,
    TokenUsage,
)
from core.models.registry import ModelRegistry
from core.models.schema import ModelCapabilities, ModelDefinition


class DummyProvider(BaseProvider):
    def __init__(self, name: str = "dummy_provider", state: RuntimeState = RuntimeState.READY):
        self._name = name
        self._state = state
        self.last_infer_args = None

    @property
    def provider_name(self) -> str:
        return self._name

    def check_health(self) -> RuntimeState:
        return self._state

    def is_model_loaded(self, model_def: ModelDefinition) -> bool:
        return True

    def infer(self, request: InferenceRequest, model_def: ModelDefinition) -> InferenceResponse:
        self.last_infer_args = (request, model_def)
        return InferenceResponse(
            request_id=request.request_id or "dummy-id",
            model_id=model_def.id,
            message=Message.assistant(f"Echo from {self._name}"),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            latency_ms=15.0,
        )


@pytest.fixture
def mock_registry(tmp_path: Path):
    configs_dir = tmp_path / "configs"
    models_dir = tmp_path / "models"
    configs_dir.mkdir()
    models_dir.mkdir()

    # Create dummy model file on disk
    dummy_model_file = models_dir / "valid.gguf"
    dummy_model_file.write_bytes(b"dummy")

    (configs_dir / "valid.toml").write_bytes(b"""
    [model]
    id = "valid-model"
    format = "gguf"
    path = "models/valid.gguf"
    supported_providers = ["dummy_provider", "llama_cpp"]
    """)

    (configs_dir / "missing.toml").write_bytes(b"""
    [model]
    id = "missing-model"
    format = "gguf"
    path = "models/non_existent.gguf"
    supported_providers = ["dummy_provider"]
    """)

    return ModelRegistry(configs_dir=configs_dir, repo_root=tmp_path)


def test_provider_registration(mock_registry):
    manager = ProviderManager(mock_registry)
    provider1 = DummyProvider("dummy_1")
    provider2 = DummyProvider("dummy_2")

    manager.register_provider(provider1)
    manager.register_provider(provider2)

    assert set(manager.list_providers()) == {"dummy_1", "dummy_2"}
    assert manager.get_provider("dummy_1") is provider1

    manager.unregister_provider("dummy_1")
    assert manager.list_providers() == ["dummy_2"]

    with pytest.raises(ProviderNotFoundError):
        manager.get_provider("dummy_1")


def test_get_provider_for_model(mock_registry):
    manager = ProviderManager(mock_registry)
    provider = DummyProvider("dummy_provider")
    manager.register_provider(provider)

    model_def = mock_registry.get_model("valid-model")
    selected_prov = manager.get_provider_for_model(model_def)
    assert selected_prov is provider


def test_get_provider_for_model_not_found(mock_registry):
    manager = ProviderManager(mock_registry)
    # Register an unrelated provider
    manager.register_provider(DummyProvider("unrelated_provider"))

    model_def = mock_registry.get_model("valid-model")
    with pytest.raises(ProviderNotFoundError, match="No registered provider found"):
        manager.get_provider_for_model(model_def)


def test_get_runtime_state(mock_registry):
    manager = ProviderManager(mock_registry)
    manager.register_provider(DummyProvider("ready_prov", RuntimeState.READY))
    manager.register_provider(DummyProvider("unavail_prov", RuntimeState.UNAVAILABLE))

    all_states = manager.get_runtime_state()
    assert all_states["ready_prov"] == RuntimeState.READY
    assert all_states["unavail_prov"] == RuntimeState.UNAVAILABLE

    single_state = manager.get_runtime_state("ready_prov")
    assert single_state == {"ready_prov": RuntimeState.READY}


def test_execute_inference_success(mock_registry):
    manager = ProviderManager(mock_registry)
    dummy_prov = DummyProvider("dummy_provider")
    manager.register_provider(dummy_prov)

    req = InferenceRequest.from_prompt(model_id="valid-model", prompt="Hello test")
    resp = manager.execute_inference(req)

    assert resp.model_id == "valid-model"
    assert resp.text == "Echo from dummy_provider"
    assert resp.usage.total_tokens == 10
    assert dummy_prov.last_infer_args is not None


def test_execute_inference_unknown_model(mock_registry):
    manager = ProviderManager(mock_registry)
    manager.register_provider(DummyProvider("dummy_provider"))

    req = InferenceRequest.from_prompt(model_id="unknown-model", prompt="Hello test")
    with pytest.raises(ModelNotFoundError):
        manager.execute_inference(req)


def test_execute_inference_advisory_unavailable_still_dispatches(mock_registry):
    """Verify that a model with advisory is_available=False still dispatches to provider.
    
    The registry availability is advisory; the provider/runtime remains the
    execution authority.
    """
    manager = ProviderManager(mock_registry)
    dummy_prov = DummyProvider("dummy_provider")
    manager.register_provider(dummy_prov)

    # 'missing-model' file does not exist on disk, so registry reports is_available=False
    avail_info = mock_registry.get_availability("missing-model")
    assert avail_info.is_available is False

    # ProviderManager must still dispatch to provider rather than blocking
    req = InferenceRequest.from_prompt(model_id="missing-model", prompt="Hello test")
    resp = manager.execute_inference(req)

    assert resp.model_id == "missing-model"
    assert resp.text == "Echo from dummy_provider"
    assert dummy_prov.last_infer_args is not None


def test_execute_inference_provider_error_propagates(mock_registry):
    """Verify that when the provider/runtime fails, the runtime error is authoritative."""
    from core.common.errors import ProviderUnavailableError

    class FailingProvider(BaseProvider):
        @property
        def provider_name(self) -> str:
            return "dummy_provider"

        def check_health(self) -> RuntimeState:
            return RuntimeState.UNAVAILABLE

        def is_model_loaded(self, model_def: ModelDefinition) -> bool:
            return False

        def infer(self, request: InferenceRequest, model_def: ModelDefinition) -> InferenceResponse:
            raise ProviderUnavailableError("Backend server is offline")

    manager = ProviderManager(mock_registry)
    manager.register_provider(FailingProvider())

    req = InferenceRequest.from_prompt(model_id="valid-model", prompt="Hello test")
    with pytest.raises(ProviderUnavailableError, match="Backend server is offline"):
        manager.execute_inference(req)
