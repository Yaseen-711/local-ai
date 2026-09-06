"""Unit tests for multimodal inference contracts (MediaAttachment and Message).

Verifies validation, SHA-256 calculation, size enforcement, allowed root restriction,
MIME detection, and immutability.
"""

import hashlib
from pathlib import Path
import pytest

from core.common.types import MessageRole
from core.inference.types import (
    DEFAULT_MAX_ATTACHMENT_BYTES,
    SUPPORTED_IMAGE_MIME_TYPES,
    MediaAttachment,
    Message,
)


def test_media_attachment_from_file(tmp_path: Path):
    """MediaAttachment.from_file accurately computes metadata and reads bytes."""
    test_file = tmp_path / "sample.png"
    sample_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtestdata"
    test_file.write_bytes(sample_data)

    expected_sha256 = hashlib.sha256(sample_data).hexdigest()

    attachment = MediaAttachment.from_file(test_file)
    assert attachment.mime_type == "image/png"
    assert attachment.source_path == str(test_file.resolve())
    assert attachment.data_bytes is None
    assert attachment.sha256 == expected_sha256
    assert attachment.size_bytes == len(sample_data)
    assert attachment.name == "sample.png"

    # Bytes loaded on demand
    assert attachment.load_bytes() == sample_data

    # Base64 data URI
    data_uri = attachment.to_base64_data_uri()
    assert data_uri.startswith("data:image/png;base64,")


def test_media_attachment_from_bytes():
    """MediaAttachment.from_bytes stores in-memory data and computes digest."""
    sample_data = b"raw_jpeg_data_sample"
    expected_sha256 = hashlib.sha256(sample_data).hexdigest()

    attachment = MediaAttachment.from_bytes(
        data=sample_data,
        mime_type="image/jpeg",
        name="test.jpg",
    )
    assert attachment.mime_type == "image/jpeg"
    assert attachment.source_path is None
    assert attachment.data_bytes == sample_data
    assert attachment.sha256 == expected_sha256
    assert attachment.size_bytes == len(sample_data)
    assert attachment.name == "test.jpg"
    assert attachment.load_bytes() == sample_data
    assert attachment.to_base64_data_uri().startswith("data:image/jpeg;base64,")


def test_media_attachment_unsupported_mime(tmp_path: Path):
    """Unsupported MIME type raises ValueError."""
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(b"hello world")

    with pytest.raises(ValueError, match="Unsupported mime_type"):
        MediaAttachment.from_file(test_file, mime_type="text/plain")


def test_media_attachment_empty_file_error(tmp_path: Path):
    """Empty file raises ValueError."""
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        MediaAttachment.from_file(empty_file)


def test_media_attachment_empty_bytes_error():
    """Empty bytes raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        MediaAttachment.from_bytes(data=b"", mime_type="image/png")


def test_media_attachment_file_not_found(tmp_path: Path):
    """Missing file raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError):
        MediaAttachment.from_file(missing)


def test_media_attachment_size_limit(tmp_path: Path):
    """Exceeding max_bytes limit raises ValueError."""
    test_file = tmp_path / "big.png"
    test_file.write_bytes(b"1234567890")

    with pytest.raises(ValueError, match="exceeds limit"):
        MediaAttachment.from_file(test_file, max_bytes=5)

    with pytest.raises(ValueError, match="exceeds limit"):
        MediaAttachment.from_bytes(data=b"1234567890", mime_type="image/png", max_bytes=5)


def test_media_attachment_allowed_root_enforcement(tmp_path: Path):
    """Paths outside allowed_root raise ValueError."""
    allowed_dir = tmp_path / "allowed"
    outside_dir = tmp_path / "outside"
    allowed_dir.mkdir()
    outside_dir.mkdir()

    outside_file = outside_dir / "secret.png"
    outside_file.write_bytes(b"secret_data")

    with pytest.raises(ValueError, match="outside allowed root"):
        MediaAttachment.from_file(outside_file, allowed_root=allowed_dir)


def test_media_attachment_is_frozen(tmp_path: Path):
    """MediaAttachment is immutable."""
    test_file = tmp_path / "sample.png"
    test_file.write_bytes(b"data")
    att = MediaAttachment.from_file(test_file)

    with pytest.raises((AttributeError, TypeError)):
        att.mime_type = "image/jpeg"  # type: ignore


def test_message_attachments_backward_compatibility():
    """Message defaults to empty attachments and preserves string content."""
    msg = Message.user("Hello world")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello world"
    assert msg.attachments == ()
    assert isinstance(msg.attachments, tuple)


def test_message_with_attachments(tmp_path: Path):
    """Message.user accepts attachments and converts them to tuple."""
    test_file = tmp_path / "drawing.png"
    test_file.write_bytes(b"drawing_bytes")
    att = MediaAttachment.from_file(test_file)

    msg = Message.user("Inspect this P&ID", attachments=[att])
    assert msg.role == MessageRole.USER
    assert msg.content == "Inspect this P&ID"
    assert len(msg.attachments) == 1
    assert msg.attachments[0] is att
    assert isinstance(msg.attachments, tuple)
