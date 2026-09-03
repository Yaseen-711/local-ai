"""Unit tests for generic syntax parsing utilities in core.common.parsing."""

import pytest

from core.common.errors import SyntaxParsingError
from core.common.parsing import parse_json_payload


def test_parse_clean_json_dict():
    """Verify parsing a clean, unfenced JSON object string."""
    raw = '{"summary": "Executive summary.", "key_points": ["Point 1", "Point 2"]}'
    parsed = parse_json_payload(raw)

    assert isinstance(parsed, dict)
    assert parsed["summary"] == "Executive summary."
    assert parsed["key_points"] == ["Point 1", "Point 2"]


def test_parse_fenced_json_dict():
    """Verify stripping ```json markdown fences."""
    raw = """```json
{
  "summary": "Fenced summary.",
  "count": 42
}
```"""
    parsed = parse_json_payload(raw)

    assert isinstance(parsed, dict)
    assert parsed["summary"] == "Fenced summary."
    assert parsed["count"] == 42


def test_parse_fenced_generic_codeblock():
    """Verify stripping ``` markdown fences without language tag."""
    raw = """```
{"status": "ok", "items": [1, 2, 3]}
```"""
    parsed = parse_json_payload(raw)

    assert isinstance(parsed, dict)
    assert parsed["status"] == "ok"
    assert parsed["items"] == [1, 2, 3]


def test_parse_json_with_surrounding_prose():
    """Verify extracting JSON when model emits conversational text around a fenced block."""
    raw = """Here is the structured analysis you requested:
```json
{
  "summary": "Found in middle.",
  "key_points": ["A"]
}
```
I hope this helps!"""
    parsed = parse_json_payload(raw)

    assert isinstance(parsed, dict)
    assert parsed["summary"] == "Found in middle."
    assert parsed["key_points"] == ["A"]


def test_parse_json_array():
    """Verify parsing a JSON array payload."""
    raw = '[{"name": "alpha"}, {"name": "beta"}]'
    parsed = parse_json_payload(raw)

    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "alpha"
    assert parsed[1]["name"] == "beta"


def test_parse_empty_or_whitespace_raises_syntax_parsing_error():
    """Verify empty or whitespace strings raise SyntaxParsingError and preserve raw_text."""
    with pytest.raises(SyntaxParsingError) as exc_info:
        parse_json_payload("")
    assert exc_info.value.raw_text == ""
    assert "empty" in str(exc_info.value)

    with pytest.raises(SyntaxParsingError) as exc_info2:
        parse_json_payload("   \n\t  ")
    assert exc_info2.value.raw_text == "   \n\t  "


def test_parse_malformed_json_raises_syntax_parsing_error():
    """Verify malformed JSON raises SyntaxParsingError, preserving raw_text and error details."""
    raw = '{"summary": "incomplete, unclosed...'
    with pytest.raises(SyntaxParsingError) as exc_info:
        parse_json_payload(raw)

    err = exc_info.value
    assert err.raw_text == raw
    assert "Failed to parse JSON syntax" in str(err)
    assert len(err.details) > 0


def test_parse_scalar_json_raises_syntax_parsing_error():
    """Verify scalar values (not dict or list) are rejected by parse_json_payload."""
    with pytest.raises(SyntaxParsingError, match="must be a dict or list"):
        parse_json_payload("12345")

    with pytest.raises(SyntaxParsingError, match="must be a dict or list"):
        parse_json_payload('"just a plain string"')


def test_parse_multiple_codeblocks_selects_valid_payload():
    """Verify that multi-codeblock output evaluates candidates and selects valid JSON payload."""
    raw = """Here is an example schema:
```json
{ "example": "not the real output" }
```
And here is the final structured response:
```json
{
  "summary": "Actual final payload",
  "key_points": ["Point A", "Point B"]
}
```
Thank you!"""
    parsed = parse_json_payload(raw)
    assert isinstance(parsed, dict)
    assert parsed["summary"] == "Actual final payload"
    assert parsed["key_points"] == ["Point A", "Point B"]


def test_parse_unclosed_opening_fence():
    """Verify that truncated responses with an unclosed opening fence parse cleanly."""
    raw = """```json
{
  "summary": "Truncated fence without closing backticks",
  "key_points": ["Item 1"]
}"""
    parsed = parse_json_payload(raw)
    assert isinstance(parsed, dict)
    assert parsed["summary"] == "Truncated fence without closing backticks"
    assert parsed["key_points"] == ["Item 1"]
