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
    OutputConstraint,
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


def test_output_constraint_explicit_constructors():
    """Verify explicit factory constructors for OutputConstraint."""
    json_constraint = OutputConstraint.json()
    assert json_constraint.format == "json"
    assert json_constraint.grammar is None

    grammar_constraint = OutputConstraint.from_grammar('root ::= "true" | "false"')
    assert grammar_constraint.format == "grammar"
    assert grammar_constraint.grammar == 'root ::= "true" | "false"'

    direct_constraint = OutputConstraint(format="yaml")
    assert direct_constraint.format == "yaml"
    assert direct_constraint.grammar is None


def test_output_constraint_requires_explicit_format():
    """Verify that bare OutputConstraint() without format raises TypeError."""
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'format'|missing required argument 'format'"):
        OutputConstraint()  # type: ignore[call-arg]


def test_generation_options_validation_valid():
    """Verify default and valid custom GenerationOptions construct cleanly."""
    default_opts = GenerationOptions()
    assert default_opts.temperature == 0.7
    assert default_opts.top_p == 0.95
    assert default_opts.max_tokens == 1024
    assert default_opts.seed is None

    custom_opts = GenerationOptions(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        seed=-42,
    )
    assert custom_opts.temperature == 0.0
    assert custom_opts.top_p == 1.0
    assert custom_opts.max_tokens == 1
    assert custom_opts.seed == -42


def test_generation_options_invalid_temperature():
    """Verify negative, NaN, Inf, and boolean temperatures raise ValueError."""
    with pytest.raises(ValueError, match="temperature must be a finite non-negative number"):
        GenerationOptions(temperature=-0.1)

    with pytest.raises(ValueError, match="temperature must be a finite non-negative number"):
        GenerationOptions(temperature=float("nan"))

    with pytest.raises(ValueError, match="temperature must be a finite non-negative number"):
        GenerationOptions(temperature=float("inf"))

    with pytest.raises(ValueError, match="temperature must be a finite non-negative number"):
        GenerationOptions(temperature=True)  # type: ignore[arg-type]


def test_generation_options_invalid_top_p():
    """Verify top_p out of [0, 1], NaN, Inf, and booleans raise ValueError."""
    with pytest.raises(ValueError, match="top_p must be a finite number between 0.0 and 1.0"):
        GenerationOptions(top_p=-0.01)

    with pytest.raises(ValueError, match="top_p must be a finite number between 0.0 and 1.0"):
        GenerationOptions(top_p=1.01)

    with pytest.raises(ValueError, match="top_p must be a finite number between 0.0 and 1.0"):
        GenerationOptions(top_p=float("nan"))

    with pytest.raises(ValueError, match="top_p must be a finite number between 0.0 and 1.0"):
        GenerationOptions(top_p=False)  # type: ignore[arg-type]


def test_generation_options_invalid_max_tokens():
    """Verify max_tokens <= 0 and booleans raise ValueError."""
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GenerationOptions(max_tokens=0)

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GenerationOptions(max_tokens=-10)

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        GenerationOptions(max_tokens=True)  # type: ignore[arg-type]


def test_generation_options_seed_validation():
    """Verify seed accepts positive and negative signed integers, but rejects non-integers and booleans."""
    opts_neg = GenerationOptions(seed=-999)
    assert opts_neg.seed == -999

    opts_pos = GenerationOptions(seed=123456)
    assert opts_pos.seed == 123456

    with pytest.raises(ValueError, match="seed must be an integer"):
        GenerationOptions(seed=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed must be an integer"):
        GenerationOptions(seed="123")  # type: ignore[arg-type]
