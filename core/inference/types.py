"""Normalized inference request, response, and message contracts.

These contracts provide a stable, provider-agnostic representation of text-oriented
inference tasks (code review, data analysis, document extraction, structured output,
reports, agent tasks, and chat).
"""

import base64
import hashlib
import math
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.common.types import FinishReason, MessageRole


SUPPORTED_IMAGE_MIME_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
})
DEFAULT_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 20 MiB


@dataclass(frozen=True)
class MediaAttachment:
    """Normalized media attachment bound to an inference message.
    
    Stores reference-oriented representation (path, MIME, SHA-256, size)
    to prevent memory bloat, loading raw bytes only when needed.
    """
    mime_type: str
    source_path: Optional[str] = None
    data_bytes: Optional[bytes] = field(default=None, repr=False)
    sha256: str = ""
    size_bytes: int = 0
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.mime_type or not isinstance(self.mime_type, str):
            raise ValueError("mime_type must be a non-empty string")
        if self.mime_type.lower() not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(
                f"Unsupported mime_type '{self.mime_type}'. Supported: {sorted(SUPPORTED_IMAGE_MIME_TYPES)}"
            )
        if not self.source_path and self.data_bytes is None:
            raise ValueError("MediaAttachment requires either source_path or data_bytes")

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        mime_type: Optional[str] = None,
        max_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
        allowed_root: Optional[Union[str, Path]] = None,
    ) -> "MediaAttachment":
        """Construct a reference-oriented attachment from a local file path."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Media attachment file not found: {p}")
        
        if allowed_root:
            root = Path(allowed_root).resolve()
            try:
                p.relative_to(root)
            except ValueError as e:
                raise ValueError(f"File path {p} is outside allowed root directory {root}") from e

        size = p.stat().st_size
        if size <= 0:
            raise ValueError(f"Media attachment file is empty: {p}")
        if size > max_bytes:
            raise ValueError(
                f"Media attachment file size ({size} bytes) exceeds limit of {max_bytes} bytes"
            )

        detected_mime = mime_type or mimetypes.guess_type(p)[0]
        if not detected_mime:
            suffix = p.suffix.lower()
            if suffix == ".png":
                detected_mime = "image/png"
            elif suffix in (".jpg", ".jpeg"):
                detected_mime = "image/jpeg"
            elif suffix == ".webp":
                detected_mime = "image/webp"
            else:
                detected_mime = "image/png"

        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        digest = h.hexdigest()

        return cls(
            mime_type=detected_mime.lower(),
            source_path=str(p),
            data_bytes=None,
            sha256=digest,
            size_bytes=size,
            name=p.name,
        )

    @classmethod
    def from_bytes(
        cls,
        data: Union[bytes, bytearray],
        mime_type: str,
        name: Optional[str] = None,
        max_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
    ) -> "MediaAttachment":
        """Construct an in-memory attachment from raw bytes."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"data must be bytes or bytearray, got {type(data).__name__}")
        size = len(data)
        if size <= 0:
            raise ValueError("data cannot be empty")
        if size > max_bytes:
            raise ValueError(
                f"Media attachment size ({size} bytes) exceeds limit of {max_bytes} bytes"
            )
        digest = hashlib.sha256(data).hexdigest()
        return cls(
            mime_type=mime_type.lower(),
            source_path=None,
            data_bytes=bytes(data),
            sha256=digest,
            size_bytes=size,
            name=name,
        )

    def load_bytes(self) -> bytes:
        """Load raw attachment bytes on demand."""
        if self.data_bytes is not None:
            return self.data_bytes
        if self.source_path:
            p = Path(self.source_path)
            if not p.is_file():
                raise FileNotFoundError(f"Media attachment file missing at load time: {p}")
            return p.read_bytes()
        raise ValueError("MediaAttachment has neither data_bytes nor source_path")

    def to_base64_data_uri(self) -> str:
        """Encode to base64 Data URI for OpenAI-compatible payload serialization."""
        raw = self.load_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{self.mime_type};base64,{b64}"


@dataclass(frozen=True)
class Message:
    """A single normalized message in an inference conversation or workflow."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    attachments: Tuple[MediaAttachment, ...] = field(default_factory=tuple)

    @classmethod
    def system(cls, content: str) -> "Message":
        """Convenience constructor for a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(
        cls,
        content: str,
        name: Optional[str] = None,
        attachments: Sequence[MediaAttachment] = (),
    ) -> "Message":
        """Convenience constructor for a user message."""
        return cls(
            role=MessageRole.USER,
            content=content,
            name=name,
            attachments=tuple(attachments),
        )

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Convenience constructor for an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str, name: str) -> "Message":
        """Convenience constructor for a tool message."""
        return cls(role=MessageRole.TOOL, content=content, name=name)


@dataclass(frozen=True)
class OutputConstraint:
    """Declarative constraint on model token generation.

    Decouples the generation-time structural constraint (e.g. JSON mode, grammar)
    from downstream domain semantics.
    """
    format: str
    grammar: Optional[str] = None


    @classmethod
    def json(cls) -> "OutputConstraint":
        """Convenience constructor for JSON-constrained generation."""
        return cls(format="json")

    @classmethod
    def from_grammar(cls, grammar_text: str) -> "OutputConstraint":
        """Convenience constructor for grammar-constrained generation."""
        return cls(format="grammar", grammar=grammar_text)



@dataclass(frozen=True)
class GenerationOptions:
    """Normalized generation options for inference execution."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    constraint: Optional[OutputConstraint] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or math.isnan(self.temperature)
            or math.isinf(self.temperature)
            or self.temperature < 0.0
        ):
            raise ValueError(
                f"temperature must be a finite non-negative number, got {self.temperature!r}"
            )

        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or math.isnan(self.top_p)
            or math.isinf(self.top_p)
            or not (0.0 <= self.top_p <= 1.0)
        ):
            raise ValueError(
                f"top_p must be a finite number between 0.0 and 1.0, got {self.top_p!r}"
            )

        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError(
                f"max_tokens must be a positive integer, got {self.max_tokens!r}"
            )

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ValueError(f"seed must be an integer, got {self.seed!r}")




@dataclass(frozen=True)
class InferenceRequest:
    """Normalized inference request submitted to the Foundation Core."""
    model_id: str
    messages: List[Message]
    options: GenerationOptions = field(default_factory=GenerationOptions)
    request_id: Optional[str] = None

    @classmethod
    def from_prompt(
        cls,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        request_id: Optional[str] = None,
    ) -> "InferenceRequest":
        """Convenience factory for non-conversational single prompt execution."""
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message.system(system_prompt))
        messages.append(Message.user(prompt))
        return cls(
            model_id=model_id,
            messages=messages,
            options=options or GenerationOptions(),
            request_id=request_id,
        )


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token accounting for an inference request."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class InferenceResponse:
    """Normalized inference response returned by the Foundation Core.
    
    Consumers must rely strictly on normalized fields (message, finish_reason, usage, latency_ms).
    The raw_response dictionary is retained purely for diagnostics/debugging.
    """
    request_id: Optional[str]
    model_id: str
    message: Message
    finish_reason: FinishReason
    usage: TokenUsage
    latency_ms: float
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def text(self) -> str:
        """Convenience accessor for generated text content."""
        return self.message.content
