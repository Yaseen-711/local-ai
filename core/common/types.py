"""Core domain enums and basic types for Local AI Foundation."""

from enum import Enum


class ModelFormat(str, Enum):
    """Supported model packaging/weight formats."""
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    OTHER = "other"


class ModelRole(str, Enum):
    """Functional role or primary intended task of a model."""
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    VISION = "vision"


class RuntimeState(str, Enum):
    """Authoritative provider and runtime health state.
    
    Minimal state model:
    - UNKNOWN: State has not yet been assessed.
    - UNAVAILABLE: Backend runtime is offline, unreachable, or refused connection.
    - READY: Runtime is online, responsive, and able to execute inference.
    - ERROR: Runtime is reachable but in a faulted or unrecoverable error state.
    """
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    READY = "ready"
    ERROR = "error"


class MessageRole(str, Enum):
    """Message roles for structured conversation/inference."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reason why the generation finished."""
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"
