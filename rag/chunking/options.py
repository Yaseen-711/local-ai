"""Configurable parameters for RAG document chunking strategies.

Note: Chunking parameters in this project are experimental and intentionally
configurable. These default values provide an initial structural baseline and
can be tuned or replaced per application or experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FallbackSplitStrategy(str, Enum):
    """Strategy used when a single structural element exceeds max_chunk_size."""

    PARAGRAPH_OR_SENTENCE = "paragraph_or_sentence"
    LINE = "line"
    CHARACTER = "character"


@dataclass(frozen=True)
class ChunkingOptions:
    """Configuration options for document chunking.

    Attributes:
        max_chunk_size: Target maximum character length for a single chunk before
            splitting or flushing (default: 1200 characters).
        min_chunk_size: Minimum character length below which small adjacent elements
            are merged into the same chunk if compatible (default: 150 characters).
        overlap_size: Overlap in characters when splitting oversized elements (default: 150).
        include_heading_context: Whether to inject parent heading context into chunk
            content and metadata (default: True).
        preserve_tables: Whether to preserve tables as coherent, dedicated chunks
            wherever practical (default: True).
        preserve_lists: Whether to group adjacent list items into coherent list chunks (default: True).
        fallback_strategy: Fallback strategy when a single element exceeds max_chunk_size
            (default: FallbackSplitStrategy.PARAGRAPH_OR_SENTENCE).
    """

    max_chunk_size: int = 1200
    min_chunk_size: int = 150
    overlap_size: int = 150
    include_heading_context: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True
    fallback_strategy: FallbackSplitStrategy = FallbackSplitStrategy.PARAGRAPH_OR_SENTENCE

    def __post_init__(self) -> None:
        if self.max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {self.max_chunk_size}")
        if self.min_chunk_size < 0:
            raise ValueError(f"min_chunk_size must be non-negative, got {self.min_chunk_size}")
        if self.min_chunk_size > self.max_chunk_size:
            object.__setattr__(self, "min_chunk_size", self.max_chunk_size)

        if self.overlap_size < 0:
            raise ValueError(f"overlap_size must be non-negative, got {self.overlap_size}")
        if self.overlap_size >= self.max_chunk_size:
            # Safely clamp overlap_size to max_chunk_size - 1
            object.__setattr__(self, "overlap_size", max(0, self.max_chunk_size - 1))

