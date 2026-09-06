"""Lightweight reference value objects for orchestration data flow.

DataReference and ArtifactReference represent logical pointers to data
without coupling to physical storage, file systems, or databases.
Storage resolution is a future architectural concern.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DataReference:
    """Logical reference to an input or output data payload.

    Used by tasks to declare data dependencies and outputs without
    embedding the actual data or coupling to a storage backend.

    Attributes:
        key: Logical name for this reference within a task's context.
        source_task_id: Task that produced this data, if applicable.
        uri: Optional location hint (opaque string; resolution is deferred).
        mime_type: Advisory content type for downstream consumers.
        metadata: Arbitrary metadata for reference-specific context.
    """
    key: str
    source_task_id: Optional[str] = None
    uri: Optional[str] = None
    mime_type: str = "application/json"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactReference:
    """Descriptor for a large generated artifact stored externally.

    Represents outputs such as PDF documents, DOCX files, datasets, or
    log files that are too large to embed directly in a TaskResult.
    Actual storage and retrieval are deferred to future architecture.

    Attributes:
        artifact_id: Unique identifier for this artifact.
        name: Human-readable artifact name.
        uri: Location of the artifact (opaque; resolution is deferred).
        mime_type: Content type of the artifact.
        size_bytes: Advisory file size, if known.
        metadata: Arbitrary metadata for artifact-specific context.
    """
    artifact_id: str
    name: str
    uri: str
    mime_type: str
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
