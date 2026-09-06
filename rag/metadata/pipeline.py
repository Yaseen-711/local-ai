"""Metadata pipeline for extracting, propagating, and enriching RAG metadata.

Connects document ingestion, normalization, and chunking into a coherent,
traceable provenance pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.domain.models import Chunk, Document
from rag.metadata.models import (
    ChunkMetadata,
    DocumentMetadata,
    ElementMetadata,
    ProvenanceMetadata,
)
from rag.normalization.models import NormalizedDocument, NormalizedElement


class MetadataPipeline:
    """Pipelines metadata flow from Document to NormalizedElement to Chunk."""

    @staticmethod
    def extract_document_metadata(document: NormalizedDocument) -> DocumentMetadata:
        """Extract typed DocumentMetadata from a NormalizedDocument."""
        meta = dict(document.metadata)
        source_path = str(document.file_path) if document.file_path else meta.get("source_path")
        file_name = Path(source_path).name if source_path else meta.get("file_name")

        # Discover document title
        title = meta.get("title")
        if not title:
            for elem in document.elements:
                if elem.element_type.value == "title" and elem.content.strip():
                    title = elem.content.strip()
                    break

        return DocumentMetadata(
            document_id=document.document_id,
            source_path=source_path,
            file_name=file_name,
            format=document.format or meta.get("format", ""),
            file_size_bytes=meta.get("file_size_bytes"),
            title=title,
            page_count=document.page_count,
            element_count=len(document.elements),
            custom={
                k: v
                for k, v in meta.items()
                if k not in {
                    "document_id",
                    "source_path",
                    "file_name",
                    "format",
                    "file_size_bytes",
                    "title",
                    "page_count",
                    "element_count",
                }
            },
        )

    @staticmethod
    def extract_element_metadata(element: NormalizedElement) -> ElementMetadata:
        """Extract typed ElementMetadata from a NormalizedElement."""
        emeta = dict(element.metadata)
        return ElementMetadata(
            index=element.index,
            element_type=element.element_type.value,
            page_number=element.page_number,
            heading_level=element.heading_level,
            parent_heading=element.parent_heading,
            table_rows=emeta.get("num_rows") or emeta.get("table_rows"),
            table_cols=emeta.get("num_cols") or emeta.get("table_cols"),
            custom={
                k: v
                for k, v in emeta.items()
                if k not in {"num_rows", "table_rows", "num_cols", "table_cols"}
            },
        )

    @staticmethod
    def get_chunk_provenance(chunk: Chunk) -> ProvenanceMetadata:
        """Extract provenance metadata from a domain Chunk."""
        chunk_meta = ChunkMetadata.from_chunk(chunk)
        return chunk_meta.get_provenance()

    @staticmethod
    def enrich_chunk(
        chunk: Chunk,
        document_metadata: Optional[DocumentMetadata] = None,
    ) -> Chunk:
        """Enrich a Chunk's metadata with document-level context if missing."""
        chunk_meta = ChunkMetadata.from_chunk(chunk)
        if document_metadata:
            updated_dict = chunk_meta.to_dict()
            if not updated_dict.get("source_path") and document_metadata.source_path:
                updated_dict["source_path"] = document_metadata.source_path
            if not updated_dict.get("file_name") and document_metadata.file_name:
                updated_dict["file_name"] = document_metadata.file_name
            if not updated_dict.get("format") and document_metadata.format:
                updated_dict["format"] = document_metadata.format
            if not updated_dict.get("document_title") and document_metadata.title:
                updated_dict["document_title"] = document_metadata.title

            new_meta = ChunkMetadata.from_dict(updated_dict)
            return Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=new_meta.to_dict(),
            )
        return chunk
