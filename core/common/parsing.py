"""Generic syntax parsing utilities for model outputs.

Decouples low-level string and syntax decoding (such as stripping markdown fences
and parsing JSON) from domain-level validation and business object creation.
"""

import json
import re
from typing import Any, Union

from core.common.errors import SyntaxParsingError

_MARKDOWN_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_MULTI_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)
_UNCLOSED_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*)$",
    re.DOTALL | re.IGNORECASE,
)


def parse_json_payload(text: str) -> Union[dict, list]:
    """Parse raw text from model output into a Python dictionary or list.

    Strips markdown code fences (```json ... ``` or ``` ... ```) if present.
    When multiple code blocks exist (e.g. example schema followed by actual payload),
    evaluates candidates deterministically from last to first. Note that this is a
    deterministic syntactic heuristic, not semantic identification of the 'correct'
    domain payload; semantic validation remains the caller's responsibility.
    Does NOT perform domain validation or schema mapping.

    Args:
        text: Raw text string emitted by model inference.

    Returns:
        Parsed JSON payload (dict or list).

    Raises:
        SyntaxParsingError: If text is empty or cannot be parsed as valid JSON.
            Preserves the original raw text for debugging.
    """
    if not text or not text.strip():
        raise SyntaxParsingError("Model output is empty or whitespace-only.", raw_text=text)

    cleaned = text.strip()

    # 1. Whole-string single enclosing fence match
    fence_match = _MARKDOWN_FENCE_PATTERN.match(cleaned)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    # 2. Multi-codeblock deterministic heuristic (evaluate from last to first)
    fenced_blocks = _MULTI_FENCE_PATTERN.findall(cleaned)
    if fenced_blocks:
        for block in reversed(fenced_blocks):
            candidate = block.strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue

    # 3. Unclosed opening fence check (e.g. truncated output)
    unclosed_match = _UNCLOSED_FENCE_PATTERN.match(cleaned)
    if unclosed_match:
        candidate = unclosed_match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    # 4. Direct JSON parsing fallback
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SyntaxParsingError(
            f"Failed to parse JSON syntax from model output: {exc}",
            raw_text=text,
            details=str(exc),
        ) from exc

    if not isinstance(parsed, (dict, list)):
        raise SyntaxParsingError(
            f"Parsed JSON payload must be a dict or list, got {type(parsed).__name__}.",
            raw_text=text,
        )

    return parsed
