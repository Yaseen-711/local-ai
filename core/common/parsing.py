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


def parse_json_payload(text: str) -> Union[dict, list]:
    """Parse raw text from model output into a Python dictionary or list.

    Strips markdown code fences (```json ... ``` or ``` ... ```) if present.
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

    fence_match = _MARKDOWN_FENCE_PATTERN.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Also handle cases where there is conversational text before or after a fenced block
        search_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL | re.IGNORECASE)
        if search_match:
            candidate = search_match.group(1).strip()
            if candidate.startswith(("{", "[")):
                cleaned = candidate

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
