"""Unit tests for DataReference and ArtifactReference value objects."""

import pytest

from orchestration.domain.references import ArtifactReference, DataReference


# ---------------------------------------------------------------------------
# DataReference
# ---------------------------------------------------------------------------

def test_data_reference_creation_defaults():
    """Verify DataReference minimal creation."""
    ref = DataReference(key="input_text")
    assert ref.key == "input_text"
    assert ref.source_task_id is None
    assert ref.uri is None
    assert ref.mime_type == "application/json"
    assert ref.metadata == {}


def test_data_reference_full():
    """Verify DataReference with all fields populated."""
    ref = DataReference(
        key="analysis_output",
        source_task_id="task-1",
        uri="mem://results/analysis",
        mime_type="text/plain",
        metadata={"encoding": "utf-8"},
    )
    assert ref.key == "analysis_output"
    assert ref.source_task_id == "task-1"
    assert ref.uri == "mem://results/analysis"
    assert ref.mime_type == "text/plain"
    assert ref.metadata["encoding"] == "utf-8"


def test_data_reference_is_frozen():
    """Verify DataReference is immutable."""
    ref = DataReference(key="x")
    with pytest.raises(AttributeError):
        ref.key = "y"  # type: ignore[misc]


def test_data_reference_equality():
    """Verify two DataReferences with the same fields are equal."""
    ref1 = DataReference(key="x", source_task_id="t-1")
    ref2 = DataReference(key="x", source_task_id="t-1")
    assert ref1 == ref2


# ---------------------------------------------------------------------------
# ArtifactReference
# ---------------------------------------------------------------------------

def test_artifact_reference_creation():
    """Verify ArtifactReference stores artifact descriptor."""
    art = ArtifactReference(
        artifact_id="art-1",
        name="report.docx",
        uri="file:///output/report.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert art.artifact_id == "art-1"
    assert art.name == "report.docx"
    assert art.uri == "file:///output/report.docx"
    assert art.size_bytes is None
    assert art.metadata == {}


def test_artifact_reference_with_size():
    """Verify ArtifactReference stores file size."""
    art = ArtifactReference(
        artifact_id="art-2",
        name="data.csv",
        uri="file:///output/data.csv",
        mime_type="text/csv",
        size_bytes=4096,
    )
    assert art.size_bytes == 4096


def test_artifact_reference_is_frozen():
    """Verify ArtifactReference is immutable."""
    art = ArtifactReference(
        artifact_id="art-1",
        name="x",
        uri="file:///x",
        mime_type="text/plain",
    )
    with pytest.raises(AttributeError):
        art.name = "y"  # type: ignore[misc]
