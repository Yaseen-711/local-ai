"""Local AI Foundation Connector Layer.

Provides capability-specific integration boundaries for orchestration
and workflow layers.
"""

from connectors.inference import (
    FoundationInferenceConnector,
    InferenceConnector,
)

__all__ = [
    "InferenceConnector",
    "FoundationInferenceConnector",
]
