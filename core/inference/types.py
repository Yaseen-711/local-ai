"""Normalized inference request, response, and message contracts.

These contracts provide a stable, provider-agnostic representation of text-oriented
inference tasks (code review, data analysis, document extraction, structured output,
reports, agent tasks, and chat).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.common.types import FinishReason, MessageRole



@dataclass(frozen=True)
class Message:
    """A single normalized message in an inference conversation or workflow."""
    role: MessageRole
    content: str
    name: Optional[str] = None

    @classmethod
    def system(cls, content: str) -> "Message":
        """Convenience constructor for a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Convenience constructor for a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Convenience constructor for an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


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
