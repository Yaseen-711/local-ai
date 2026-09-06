"""Comprehensive unit and integration tests for the RAG metadata pipeline.

Verifies:
1. Document metadata propagation
2. Element metadata propagation
3. Page number propagation
4. Heading and heading_path propagation
5. Multiple element indices in a chunk
6. Multiple page numbers in a chunk
7. Element type aggregation
8. Table metadata preservation
9. List metadata preservation
10. Split chunk provenance
11. Deterministic chunk metadata
12. Missing/optional metadata handling
13. No fabricated metadata
14. Content vs metadata separation (content never polluted with metadata)
15. Realistic multi-page document provenance tracing
"""

from __future__ import annotations

import unittest
from pathlib import Path

from rag.chunking.options import ChunkingOptions, FallbackSplitStrategy
from rag.chunking.structural import StructuralChunker
from rag.domain.models import Chunk
from rag.metadata.models import (
    ChunkMetadata,
    DocumentMetadata,
    ElementMetadata,
    ProvenanceMetadata,
)
from rag.metadata.pipeline import MetadataPipeline
from rag.normalization.models import (
    NormalizedDocument,
    NormalizedElement,
    NormalizedElementType,
)


class TestMetadataPipeline(unittest.TestCase):
    """Test suite for RAG metadata models and propagation pipeline."""

    def test_document_metadata_propagation(self) -> None:
        """1. Verify DocumentMetadata extracts and propagates document-level fields."""
        doc = NormalizedDocument(
            document_id="doc_meta_01",
            file_path=Path("/docs/architecture_overview.pdf"),
            format="pdf",
            elements=[
                NormalizedElement(0, NormalizedElementType.TITLE, "Architecture Overview"),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "Intro text.", page_number=1),
            ],
            metadata={
                "file_size_bytes": 1048576,
                "page_count": 5,
                "author": "Engineering Team",
            },
        )
        doc_meta = MetadataPipeline.extract_document_metadata(doc)
        self.assertEqual(doc_meta.document_id, "doc_meta_01")
        self.assertEqual(doc_meta.source_path, "/docs/architecture_overview.pdf")
        self.assertEqual(doc_meta.file_name, "architecture_overview.pdf")
        self.assertEqual(doc_meta.format, "pdf")
        self.assertEqual(doc_meta.file_size_bytes, 1048576)
        self.assertEqual(doc_meta.title, "Architecture Overview")
        self.assertEqual(doc_meta.page_count, 5)
        self.assertEqual(doc_meta.element_count, 2)
        self.assertEqual(doc_meta.custom.get("author"), "Engineering Team")

        d = doc_meta.to_dict()
        reconstructed = DocumentMetadata.from_dict(d)
        self.assertEqual(reconstructed.document_id, doc_meta.document_id)
        self.assertEqual(reconstructed.title, doc_meta.title)

    def test_element_metadata_propagation(self) -> None:
        """2. Verify ElementMetadata preserves structural attributes and table dimensions."""
        elem = NormalizedElement(
            index=3,
            element_type=NormalizedElementType.TABLE,
            content="| A | B |\n|---|---|\n| 1 | 2 |",
            page_number=2,
            heading_level=None,
            parent_heading="Performance Metrics",
            metadata={"num_rows": 1, "num_cols": 2, "source_tag": "tbl_01"},
        )
        elem_meta = MetadataPipeline.extract_element_metadata(elem)
        self.assertEqual(elem_meta.index, 3)
        self.assertEqual(elem_meta.element_type, "table")
        self.assertEqual(elem_meta.page_number, 2)
        self.assertEqual(elem_meta.parent_heading, "Performance Metrics")
        self.assertEqual(elem_meta.table_rows, 1)
        self.assertEqual(elem_meta.table_cols, 2)
        self.assertEqual(elem_meta.custom.get("source_tag"), "tbl_01")

        d = elem_meta.to_dict()
        reconstructed = ElementMetadata.from_dict(d)
        self.assertEqual(reconstructed.index, 3)
        self.assertEqual(reconstructed.table_rows, 1)

    def test_page_number_propagation_single_and_derived(self) -> None:
        """3. Verify single page number and derived primary_page and page_range."""
        doc = NormalizedDocument(
            document_id="doc_p1",
            file_path=Path("guide.md"),
            format="md",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Single page paragraph.", page_number=4),
            ],
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        meta = ChunkMetadata.from_chunk(chunks[0])
        self.assertEqual(meta.page_numbers, [4])
        self.assertEqual(meta.primary_page, 4)
        self.assertEqual(meta.page_range, "4")

    def test_heading_and_heading_path_propagation(self) -> None:
        """4. Verify heading breadcrumb and complete ancestral heading_path list."""
        doc = NormalizedDocument(
            document_id="doc_hierarchy",
            elements=[
                NormalizedElement(0, NormalizedElementType.TITLE, "Chapter 1: Foundations"),
                NormalizedElement(1, NormalizedElementType.SECTION_HEADER, "1.1 Storage Layer"),
                NormalizedElement(2, NormalizedElementType.PARAGRAPH, "Data stored in pgvector."),
            ],
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        meta = ChunkMetadata.from_chunk(chunks[0])
        self.assertEqual(meta.heading, "Chapter 1: Foundations > 1.1 Storage Layer")
        self.assertEqual(meta.heading_path, ["Chapter 1: Foundations", "1.1 Storage Layer"])

    def test_multiple_element_indices_in_chunk(self) -> None:
        """5. Verify chunks containing multiple elements preserve all source indices in order."""
        doc = NormalizedDocument(
            document_id="doc_multi_elem",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "P0."),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "P1."),
                NormalizedElement(2, NormalizedElementType.PARAGRAPH, "P2."),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(max_chunk_size=1000))
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        meta = ChunkMetadata.from_chunk(chunks[0])
        self.assertEqual(meta.element_indices, [0, 1, 2])

    def test_multiple_page_numbers_in_chunk(self) -> None:
        """6. Verify multi-page chunks aggregate sorted pages and format page_range cleanly."""
        doc = NormalizedDocument(
            document_id="doc_multi_page",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Page 2 content.", page_number=2),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "Page 3 content.", page_number=3),
                NormalizedElement(2, NormalizedElementType.PARAGRAPH, "Page 4 content.", page_number=4),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(max_chunk_size=1000))
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        meta = ChunkMetadata.from_chunk(chunks[0])
        self.assertEqual(meta.page_numbers, [2, 3, 4])
        self.assertEqual(meta.primary_page, 2)
        self.assertEqual(meta.page_range, "2-4")

    def test_element_type_aggregation(self) -> None:
        """7. Verify element_types aggregates unique types and derives primary_element_type."""
        doc = NormalizedDocument(
            document_id="doc_types",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Code explanation:"),
                NormalizedElement(1, NormalizedElementType.CODE, "SELECT * FROM chunks;"),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(max_chunk_size=1000))
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        meta = ChunkMetadata.from_chunk(chunks[0])
        self.assertEqual(meta.element_types, ["paragraph", "code"])
        self.assertTrue(meta.has_code)
        self.assertEqual(meta.primary_element_type, "code")

    def test_table_metadata_preservation(self) -> None:
        """8. Verify table metadata carries dimensional properties and flags."""
        doc = NormalizedDocument(
            document_id="doc_tbl",
            elements=[
                NormalizedElement(
                    0,
                    NormalizedElementType.TABLE,
                    "| Col1 | Col2 |\n|---|---|\n| V1 | V2 |\n| V3 | V4 |",
                    metadata={"num_rows": 2, "num_cols": 2},
                ),
            ],
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        meta = ChunkMetadata.from_chunk(chunks[0])
        self.assertTrue(meta.is_table)
        self.assertTrue(meta.has_table)
        self.assertEqual(meta.table_rows, 2)
        self.assertEqual(meta.table_cols, 2)
        self.assertEqual(meta.primary_element_type, "table")
        # Ensure backward-compatible dictionary keys
        self.assertEqual(chunks[0].metadata["num_rows"], 2)
        self.assertEqual(chunks[0].metadata["num_cols"], 2)

    def test_list_metadata_preservation(self) -> None:
        """9. Verify list chunks derive has_list and primary_element_type."""
        doc = NormalizedDocument(
            document_id="doc_lst",
            elements=[
                NormalizedElement(0, NormalizedElementType.LIST_ITEM, "- First"),
                NormalizedElement(1, NormalizedElementType.LIST_ITEM, "- Second"),
            ],
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        meta = ChunkMetadata.from_chunk(chunks[0])
        self.assertTrue(meta.has_list)
        self.assertFalse(meta.has_table)
        self.assertEqual(meta.primary_element_type, "list_item")

    def test_split_chunk_provenance(self) -> None:
        """10. Verify split chunks retain parent element indices and split parts."""
        huge_text = "Sentence number one. " * 30  # ~630 chars
        doc = NormalizedDocument(
            document_id="doc_split",
            elements=[NormalizedElement(7, NormalizedElementType.PARAGRAPH, huge_text, page_number=3)],
        )
        chunker = StructuralChunker(
            ChunkingOptions(
                max_chunk_size=150,
                overlap_size=30,
                fallback_strategy=FallbackSplitStrategy.PARAGRAPH_OR_SENTENCE,
            )
        )
        chunks = chunker.chunk(doc)
        self.assertGreater(len(chunks), 1)

        total = len(chunks)
        for i, c in enumerate(chunks):
            meta = ChunkMetadata.from_chunk(c)
            self.assertTrue(meta.is_split)
            self.assertEqual(meta.split_part, i + 1)
            self.assertEqual(meta.total_parts, total)
            self.assertEqual(meta.element_indices, [7])
            self.assertEqual(meta.page_numbers, [3])

    def test_deterministic_chunk_metadata(self) -> None:
        """11. Verify chunk metadata is 100% deterministic across multiple invocations."""
        doc = NormalizedDocument(
            document_id="doc_det_meta",
            file_path=Path("/tmp/sample.txt"),
            format="txt",
            elements=[
                NormalizedElement(0, NormalizedElementType.TITLE, "Title"),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "Text paragraph.", page_number=1),
            ],
        )
        chunker = StructuralChunker()
        run1 = chunker.chunk(doc)[0].metadata
        run2 = chunker.chunk(doc)[0].metadata

        self.assertEqual(run1, run2)

    def test_missing_optional_metadata_handling(self) -> None:
        """12. Verify documents with missing file paths or pages handle optionals cleanly."""
        doc = NormalizedDocument(
            document_id="doc_minimal",
            elements=[NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Bare paragraph.")],
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        meta = ChunkMetadata.from_chunk(chunks[0])
        self.assertIsNone(meta.source_path)
        self.assertIsNone(meta.file_name)
        self.assertEqual(meta.page_numbers, [])
        self.assertIsNone(meta.primary_page)
        self.assertIsNone(meta.page_range)
        self.assertIsNone(meta.heading)
        self.assertEqual(meta.heading_path, [])
        self.assertFalse(meta.is_split)
        self.assertIn("doc_minimal", meta.citation)

    def test_no_fabricated_metadata(self) -> None:
        """13. Verify that metadata fields are never populated with fabricated default values."""
        doc = NormalizedDocument(
            document_id="doc_nofab",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "No page specified.", page_number=None),
            ],
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)
        meta = ChunkMetadata.from_chunk(chunks[0])

        # Page numbers must be empty list, NOT [1] or [0]
        self.assertEqual(meta.page_numbers, [])
        self.assertIsNone(meta.primary_page)
        self.assertIsNone(meta.page_range)
        self.assertIsNone(meta.table_rows)
        self.assertIsNone(meta.table_cols)

    def test_metadata_does_not_alter_chunk_content_unexpectedly(self) -> None:
        """14. Verify metadata fields (file, page, doc_id) are never injected into chunk.content."""
        doc = NormalizedDocument(
            document_id="doc_pure_content",
            file_path=Path("/secret/path/to/private_financial_data.pdf"),
            format="pdf",
            elements=[
                NormalizedElement(
                    0,
                    NormalizedElementType.PARAGRAPH,
                    "Semantic content for embedding.",
                    page_number=42,
                )
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(include_heading_context=False))
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)

        chunk = chunks[0]
        self.assertEqual(chunk.content, "Semantic content for embedding.")
        # Content must NOT contain metadata strings
        self.assertNotIn("doc_pure_content", chunk.content)
        self.assertNotIn("private_financial_data", chunk.content)
        self.assertNotIn("42", chunk.content)
        self.assertNotIn("page", chunk.content)

    def test_realistic_document_provenance_tracing(self) -> None:
        """15. End-to-end realistic document: Title -> Section -> Subsection -> P -> List -> Table -> P on page 2."""
        elements = [
            NormalizedElement(0, NormalizedElementType.TITLE, "System Architecture Guide", page_number=1),
            NormalizedElement(1, NormalizedElementType.SECTION_HEADER, "1. Vector Indexing", page_number=1, heading_level=2),
            NormalizedElement(2, NormalizedElementType.SECTION_HEADER, "1.1 Storage Engine", page_number=1, heading_level=3),
            NormalizedElement(3, NormalizedElementType.PARAGRAPH, "PostgreSQL stores 768-dimensional vectors.", page_number=1),
            NormalizedElement(4, NormalizedElementType.LIST_ITEM, "- HNSW indexing", page_number=1),
            NormalizedElement(5, NormalizedElementType.LIST_ITEM, "- IVFFlat indexing", page_number=1),
            NormalizedElement(
                6,
                NormalizedElementType.TABLE,
                "| Index | Build Time |\n|---|---|\n| HNSW | Fast |\n| IVFFlat | Medium |",
                page_number=1,
                metadata={"num_rows": 2, "num_cols": 2},
            ),
            NormalizedElement(7, NormalizedElementType.SECTION_HEADER, "2. Query Processing", page_number=2, heading_level=2),
            NormalizedElement(8, NormalizedElementType.PARAGRAPH, "Queries are normalized and embedded.", page_number=2),
        ]

        doc = NormalizedDocument(
            document_id="doc_arch_guide",
            file_path=Path("/srv/docs/architecture_guide.md"),
            format="md",
            elements=elements,
            metadata={"title": "System Architecture Guide", "page_count": 2},
        )

        chunker = StructuralChunker(ChunkingOptions(min_chunk_size=0, preserve_tables=True, preserve_lists=True))
        chunks = chunker.chunk(doc)

        self.assertGreater(len(chunks), 3)

        # 1. Inspect table chunk provenance
        table_chunks = [c for c in chunks if c.metadata.get("is_table")]
        self.assertEqual(len(table_chunks), 1)
        tbl_meta = ChunkMetadata.from_chunk(table_chunks[0])
        self.assertEqual(tbl_meta.table_rows, 2)
        self.assertEqual(tbl_meta.table_cols, 2)
        self.assertEqual(tbl_meta.element_indices, [6])
        self.assertEqual(tbl_meta.page_numbers, [1])
        self.assertIn("1. Vector Indexing", tbl_meta.heading_path)
        self.assertIn("1.1 Storage Engine", tbl_meta.heading_path)

        # 2. Inspect query processing chunk on page 2
        page2_chunks = [c for c in chunks if 2 in c.metadata.get("page_numbers", [])]
        self.assertEqual(len(page2_chunks), 1)
        p2_meta = ChunkMetadata.from_chunk(page2_chunks[0])
        self.assertEqual(p2_meta.primary_page, 2)
        self.assertEqual(p2_meta.heading, "System Architecture Guide > 2. Query Processing")
        self.assertEqual(p2_meta.heading_path, ["System Architecture Guide", "2. Query Processing"])
        self.assertEqual(p2_meta.element_indices, [8])

        # 3. Verify provenance tracing
        prov = p2_meta.get_provenance()
        self.assertIsInstance(prov, ProvenanceMetadata)
        self.assertEqual(prov.file_name, "architecture_guide.md")
        self.assertEqual(prov.source_path, "/srv/docs/architecture_guide.md")
        self.assertEqual(prov.page_range, "2")
        self.assertIn("architecture_guide.md p. 2", prov.citation)


if __name__ == "__main__":
    unittest.main()
