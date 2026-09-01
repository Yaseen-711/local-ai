"""Unit tests for Core data contracts and schemas."""

from pathlib import Path
import pytest

from core.common.types import (
    FinishReason,
    MessageRole,
    ModelFormat,
    ModelRole,
    RuntimeState,
)
from core.common.errors import (
    ConfigurationError,
    FoundationError,
    InferenceError,
    LifecycleConflictError,
    ModelNotFoundError,
    ModelRegistryError,
    ModelUnavailableError,
    ProviderError,
    ProviderNotFoundError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from core.models.schema import (
    AvailabilityInfo,
    ModelCapabilities,
    ModelDefinition,
)
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
    Message,
    TokenUsage,
)


def test_model_enums():
    """Verify core domain enum representations."""
    assert ModelFormat.GGUF == "gguf"
    assert ModelFormat.SAFETENSORS == "safetensors"
    assert ModelRole.CODING == "coding"
    assert ModelRole.GENERAL == "general"
    assert RuntimeState.READY == "ready"
    assert RuntimeState.UNAVAILABLE == "unavailable"
    assert RuntimeState.UNKNOWN == "unknown"
    assert RuntimeState.ERROR == "error"
    assert MessageRole.USER == "user"
    assert FinishReason.STOP == "stop"


def test_error_hierarchy():
    """Verify domain error inheritance hierarchy."""
    assert issubclass(ConfigurationError, FoundationError)
    assert issubclass(ModelRegistryError, FoundationError)
    assert issubclass(ModelNotFoundError, ModelRegistryError)
    assert issubclass(ModelUnavailableError, ModelRegistryError)
    assert issubclass(ProviderError, FoundationError)
    assert issubclass(ProviderNotFoundError, ProviderError)
    assert issubclass(ProviderUnavailableError, ProviderError)
    assert issubclass(ProviderResponseError, ProviderError)
    assert issubclass(InferenceError, ProviderError)
    assert issubclass(LifecycleConflictError, ProviderError)


def test_model_definition():
    """Verify ModelDefinition creation and identifier matching."""
    caps = ModelCapabilities(chat=True, code=True, reasoning=False, context_window=4096)
    model = ModelDefinition(
        id="qwen3.5-9b",
        display_name="Qwen 3.5 9B Q4_K_M",
        format=ModelFormat.GGUF,
        relative_path=Path("models/gguf/Qwen3.5-9B-Q4_K_M.gguf"),
        supported_providers=["llama_cpp"],
        aliases=["qwen3.5", "default"],
        roles=[ModelRole.GENERAL, ModelRole.CODING],
        capabilities=caps,
        metadata={"quantization": "Q4_K_M"},
    )

    assert model.id == "qwen3.5-9b"
    assert model.matches_identifier("qwen3.5-9b")
    assert model.matches_identifier("qwen3.5")
    assert model.matches_identifier("default")
    assert not model.matches_identifier("non-existent")
    assert model.capabilities.code is True
    assert model.metadata["quantization"] == "Q4_K_M"


def test_availability_info():
    """Verify AvailabilityInfo properties."""
    avail = AvailabilityInfo(
        is_available=True,
        resolved_path=Path("/tmp/models/test.gguf"),
        size_bytes=5300000000,
    )
    assert avail.is_available is True
    assert avail.resolved_path == Path("/tmp/models/test.gguf")
    assert avail.size_bytes == 5300000000

    unavail = AvailabilityInfo(is_available=False, error_message="File not found")
    assert unavail.is_available is False
    assert unavail.error_message == "File not found"


def test_message_creation():
    """Verify Message dataclass and convenience helpers."""
    sys_msg = Message.system("You are a helpful assistant.")
    user_msg = Message.user("Review this code.")
    asst_msg = Message.assistant("Looks good!")

    assert sys_msg.role == MessageRole.SYSTEM
    assert sys_msg.content == "You are a helpful assistant."
    assert user_msg.role == MessageRole.USER
    assert asst_msg.role == MessageRole.ASSISTANT


def test_inference_request_from_prompt():
    """Verify InferenceRequest construction and from_prompt factory."""
    req = InferenceRequest.from_prompt(
        model_id="qwen3.5-9b",
        prompt="Explain quicksort in Python",
        system_prompt="You are an expert software engineer.",
        options=GenerationOptions(temperature=0.2, max_tokens=512),
        request_id="req-123",
    )

    assert req.model_id == "qwen3.5-9b"
    assert req.request_id == "req-123"
    assert len(req.messages) == 2
    assert req.messages[0].role == MessageRole.SYSTEM
    assert req.messages[0].content == "You are an expert software engineer."
    assert req.messages[1].role == MessageRole.USER
    assert req.messages[1].content == "Explain quicksort in Python"
    assert req.options.temperature == 0.2
    assert req.options.max_tokens == 512


def test_inference_response():
    """Verify InferenceResponse properties and raw_response isolation."""
    usage = TokenUsage(prompt_tokens=25, completion_tokens=50, total_tokens=75)
    response = InferenceResponse(
        request_id="req-123",
        model_id="qwen3.5-9b",
        message=Message.assistant("Quicksort is a divide-and-conquer algorithm."),
        finish_reason=FinishReason.STOP,
        usage=usage,
        latency_ms=124.5,
        raw_response={"id": "chatcmpl-123", "object": "chat.completion"},
    )

    assert response.request_id == "req-123"
    assert response.model_id == "qwen3.5-9b"
    assert response.text == "Quicksort is a divide-and-conquer algorithm."
    assert response.finish_reason == FinishReason.STOP
    assert response.usage.total_tokens == 75
    assert response.latency_ms == 124.5
    assert response.raw_response["id"] == "chatcmpl-123"
