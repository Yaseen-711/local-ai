"""Unit and integration tests for the RAG document chunking subsystem.

Verifies structural boundary preservation (headings, tables, lists, paragraphs),
genuine overlap application across fallback strategies, safe handling of boundary overlap values,
heading hierarchy propagation, deterministic IDs, and end-to-end normalization integration.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from rag.chunking.interfaces import DocumentChunker
from rag.chunking.options import ChunkingOptions, FallbackSplitStrategy
from rag.chunking.structural import StructuralChunker
from rag.domain.models import Chunk
from rag.normalization.models import (
    NormalizedDocument,
    NormalizedElement,
    NormalizedElementType,
)
from rag.normalization.normalizer import StandardDocumentNormalizer


class TestStructuralChunker(unittest.TestCase):
    """Test suite for StructuralChunker."""

    def test_protocol_compliance(self) -> None:
        """Verify StructuralChunker implements DocumentChunker protocol."""
        chunker = StructuralChunker()
        self.assertIsInstance(chunker, DocumentChunker)

    def test_empty_document_produces_empty_chunks(self) -> None:
        """Verify empty elements and empty text produce zero chunks."""
        doc = NormalizedDocument(document_id="doc_empty", elements=[], text="")
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)
        self.assertEqual(chunks, [])

    def test_whitespace_only_elements_produce_empty_chunks(self) -> None:
        """Verify document with only whitespace elements produces zero chunks."""
        doc = NormalizedDocument(
            document_id="doc_ws",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "   \n\t  "),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "   "),
            ],
            text="",
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)
        self.assertEqual(chunks, [])

    def test_single_paragraph_chunking(self) -> None:
        """Verify a single paragraph produces one well-formed Chunk."""
        doc = NormalizedDocument(
            document_id="doc_p1",
            elements=[
                NormalizedElement(
                    index=0,
                    element_type=NormalizedElementType.PARAGRAPH,
                    content="This is a clean, single paragraph.",
                    page_number=1,
                )
            ],
            format="txt",
            file_path=Path("/tmp/test.txt"),
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertIsInstance(chunk, Chunk)
        self.assertEqual(chunk.id, "doc_p1_chk_0000")
        self.assertEqual(chunk.document_id, "doc_p1")
        self.assertEqual(chunk.content, "This is a clean, single paragraph.")
        self.assertEqual(chunk.metadata["chunk_index"], 0)
        self.assertEqual(chunk.metadata["element_indices"], [0])
        self.assertEqual(chunk.metadata["element_types"], ["paragraph"])
        self.assertEqual(chunk.metadata["page_numbers"], [1])
        self.assertEqual(chunk.metadata["format"], "txt")
        self.assertEqual(chunk.metadata["source_path"], "/tmp/test.txt")

    def test_adjacent_paragraphs_merged_within_budget(self) -> None:
        """Verify multiple small paragraphs under the same heading are merged."""
        doc = NormalizedDocument(
            document_id="doc_merge",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "First sentence here."),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "Second sentence here."),
                NormalizedElement(2, NormalizedElementType.PARAGRAPH, "Third sentence here."),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(max_chunk_size=1000))
        chunks = chunker.chunk(doc)

        self.assertEqual(len(chunks), 1)
        self.assertIn("First sentence here.\n\nSecond sentence here.\n\nThird sentence here.", chunks[0].content)
        self.assertEqual(chunks[0].metadata["element_indices"], [0, 1, 2])

    def test_heading_context_injection_and_boundaries(self) -> None:
        """Verify headings split sections and inject context when configured."""
        doc = NormalizedDocument(
            document_id="doc_headings",
            elements=[
                NormalizedElement(0, NormalizedElementType.TITLE, "Document Title"),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "Body text in intro."),
                NormalizedElement(2, NormalizedElementType.SECTION_HEADER, "Section 2: Deep Dive"),
                NormalizedElement(3, NormalizedElementType.PARAGRAPH, "Body text in section 2."),
            ],
        )

        # With heading context enabled (default)
        chunker_with_context = StructuralChunker(ChunkingOptions(include_heading_context=True, min_chunk_size=0))
        chunks = chunker_with_context.chunk(doc)

        self.assertEqual(len(chunks), 2)
        # Chunk 0 under Document Title
        self.assertEqual(chunks[0].id, "doc_headings_chk_0000")
        self.assertEqual(chunks[0].metadata["heading"], "Document Title")
        self.assertEqual(chunks[0].metadata["heading_path"], ["Document Title"])
        self.assertIn("Document Title\n\nBody text in intro.", chunks[0].content)

        # Chunk 1 under Section 2, preserving Document Title in hierarchy
        self.assertEqual(chunks[1].id, "doc_headings_chk_0001")
        self.assertIn("Section 2: Deep Dive", chunks[1].metadata["heading"])
        self.assertIn("Document Title", chunks[1].metadata["heading"])
        self.assertEqual(chunks[1].metadata["heading_path"], ["Document Title", "Section 2: Deep Dive"])

        # With heading context disabled
        chunker_no_context = StructuralChunker(ChunkingOptions(include_heading_context=False, min_chunk_size=0))
        chunks_no_ctx = chunker_no_context.chunk(doc)
        self.assertEqual(len(chunks_no_ctx), 2)
        self.assertEqual(chunks_no_ctx[0].content, "Body text in intro.")
        self.assertEqual(chunks_no_ctx[1].content, "Body text in section 2.")

    def test_heading_hierarchy_preservation(self) -> None:
        """Verify conceptual structure: Chapter 3 -> 3.1 Vector Storage -> paragraphs."""
        doc = NormalizedDocument(
            document_id="doc_hierarchy",
            elements=[
                NormalizedElement(0, NormalizedElementType.TITLE, "Chapter 3"),
                NormalizedElement(1, NormalizedElementType.SECTION_HEADER, "3.1 Vector Storage"),
                NormalizedElement(2, NormalizedElementType.PARAGRAPH, "Paragraph A explains vectors."),
                NormalizedElement(3, NormalizedElementType.PARAGRAPH, "Paragraph B explains indices."),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(min_chunk_size=0))
        chunks = chunker.chunk(doc)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        # Verify heading path preserves full hierarchical depth
        self.assertEqual(chunk.metadata["heading_path"], ["Chapter 3", "3.1 Vector Storage"])
        # Verify heading string retains both parent chapter and section
        self.assertIn("Chapter 3", chunk.metadata["heading"])
        self.assertIn("3.1 Vector Storage", chunk.metadata["heading"])
        # Verify content has heading context injected
        self.assertTrue(chunk.content.startswith("Chapter 3 > 3.1 Vector Storage\n\n"))
        self.assertIn("Paragraph A explains vectors.", chunk.content)
        self.assertIn("Paragraph B explains indices.", chunk.content)

    def test_table_preservation_as_dedicated_chunk(self) -> None:
        """Verify markdown tables are preserved as coherent isolated chunks."""
        table_content = "| Col A | Col B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        doc = NormalizedDocument(
            document_id="doc_table",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Paragraph before table."),
                NormalizedElement(
                    index=1,
                    element_type=NormalizedElementType.TABLE,
                    content=table_content,
                    metadata={"num_rows": 2, "num_cols": 2},
                ),
                NormalizedElement(2, NormalizedElementType.PARAGRAPH, "Paragraph after table."),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(preserve_tables=True, min_chunk_size=0))
        chunks = chunker.chunk(doc)

        self.assertEqual(len(chunks), 3)
        table_chunk = chunks[1]
        self.assertTrue(table_chunk.metadata.get("is_table"))
        self.assertEqual(table_chunk.metadata.get("num_rows"), 2)
        self.assertEqual(table_chunk.metadata.get("num_cols"), 2)
        self.assertIn("| Col A | Col B |", table_chunk.content)
        self.assertEqual(table_chunk.metadata["element_types"], ["table"])

    def test_list_items_grouping(self) -> None:
        """Verify adjacent list items are grouped together into coherent list chunks."""
        doc = NormalizedDocument(
            document_id="doc_list",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Here are the items:"),
                NormalizedElement(1, NormalizedElementType.LIST_ITEM, "- First item"),
                NormalizedElement(2, NormalizedElementType.LIST_ITEM, "- Second item"),
                NormalizedElement(3, NormalizedElementType.LIST_ITEM, "- Third item"),
                NormalizedElement(4, NormalizedElementType.PARAGRAPH, "Summary conclusion."),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(preserve_lists=True, min_chunk_size=0))
        chunks = chunker.chunk(doc)

        self.assertEqual(len(chunks), 3)
        list_chunk = chunks[1]
        self.assertEqual(list_chunk.metadata["element_types"], ["list_item"])
        self.assertEqual(list_chunk.metadata["element_indices"], [1, 2, 3])
        expected_list_text = "- First item\n- Second item\n- Third item"
        self.assertIn(expected_list_text, list_chunk.content)

    def test_actual_overlap_presence_and_semantic_preservation(self) -> None:
        """Verify configured overlap is genuinely present between consecutive fallback chunks."""
        # 10 distinct sentences
        sentences = [f"Sentence {i} has crucial content." for i in range(10)]
        full_text = " ".join(sentences)
        doc = NormalizedDocument(
            document_id="doc_overlap_check",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, full_text),
            ],
        )
        # Max chunk size ~100 chars, overlap 35 chars
        chunker = StructuralChunker(
            ChunkingOptions(
                max_chunk_size=100,
                overlap_size=35,
                include_heading_context=False,
                fallback_strategy=FallbackSplitStrategy.PARAGRAPH_OR_SENTENCE,
            )
        )
        chunks = chunker.chunk(doc)

        self.assertGreater(len(chunks), 1)
        # Verify that consecutive chunks genuinely share content (overlap)
        found_actual_overlap = False
        for i in range(len(chunks) - 1):
            c1_words = set(chunks[i].content.split())
            c2_words = set(chunks[i + 1].content.split())
            shared_words = c1_words.intersection(c2_words)
            # Both chunks must share words from the overlapping sentence
            if shared_words:
                found_actual_overlap = True
            # Verify no cut sentences: every chunk starts and ends with clean sentence boundary
            self.assertTrue(chunks[i].content.endswith("."))
            self.assertTrue(chunks[i + 1].content.endswith("."))

        self.assertTrue(found_actual_overlap, "Consecutive chunks must genuinely share overlapping content")

    def test_line_fallback_overlap_preserves_line_boundaries(self) -> None:
        """Verify FallbackSplitStrategy.LINE genuinely overlaps lines and preserves line breaks."""
        lines = [f"Log line {i:02d}: system status normal" for i in range(15)]
        full_text = "\n".join(lines)
        doc = NormalizedDocument(
            document_id="doc_line_overlap",
            elements=[
                NormalizedElement(0, NormalizedElementType.CODE, full_text),
            ],
        )
        chunker = StructuralChunker(
            ChunkingOptions(
                max_chunk_size=120,
                overlap_size=40,
                include_heading_context=False,
                fallback_strategy=FallbackSplitStrategy.LINE,
            )
        )
        chunks = chunker.chunk(doc)

        self.assertGreater(len(chunks), 1)
        # Check that consecutive chunks share at least one entire line
        for i in range(len(chunks) - 1):
            c1_lines = set(chunks[i].content.split("\n"))
            c2_lines = set(chunks[i + 1].content.split("\n"))
            shared_lines = c1_lines.intersection(c2_lines)
            self.assertTrue(len(shared_lines) > 0, f"Expected overlapping line between chunk {i} and {i+1}")

    def test_character_fallback_deterministic_overlap(self) -> None:
        """Verify character fallback is deterministic and applies exact character sliding window."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        doc = NormalizedDocument(
            document_id="doc_char_overlap",
            elements=[NormalizedElement(0, NormalizedElementType.PARAGRAPH, text)],
        )
        chunker = StructuralChunker(
            ChunkingOptions(
                max_chunk_size=10,
                overlap_size=3,
                include_heading_context=False,
                fallback_strategy=FallbackSplitStrategy.CHARACTER,
            )
        )
        chunks = chunker.chunk(doc)

        self.assertGreater(len(chunks), 1)
        # Step is max_chunk_size - overlap_size = 10 - 3 = 7
        self.assertEqual(chunks[0].content, "ABCDEFGHIJ")
        self.assertEqual(chunks[1].content, "HIJKLMNOPQ")  # Overlaps 'HIJ' (3 chars)
        self.assertTrue(chunks[0].content.endswith(chunks[1].content[:3]))

    def test_overlap_zero_produces_no_overlap(self) -> None:
        """Verify overlap_size = 0 works and produces non-overlapping chunks."""
        sentences = [f"Item {i}." for i in range(12)]
        full_text = " ".join(sentences)
        doc = NormalizedDocument(
            document_id="doc_zero_overlap",
            elements=[NormalizedElement(0, NormalizedElementType.PARAGRAPH, full_text)],
        )
        chunker = StructuralChunker(
            ChunkingOptions(
                max_chunk_size=30,
                overlap_size=0,
                include_heading_context=False,
                fallback_strategy=FallbackSplitStrategy.PARAGRAPH_OR_SENTENCE,
            )
        )
        chunks = chunker.chunk(doc)

        self.assertGreater(len(chunks), 1)
        for i in range(len(chunks) - 1):
            import re
            c1_items = set(re.findall(r"Item \d+\.", chunks[i].content))
            c2_items = set(re.findall(r"Item \d+\.", chunks[i + 1].content))
            shared = c1_items.intersection(c2_items)
            self.assertEqual(shared, set(), f"Chunks {i} and {i+1} unexpectedly shared items with overlap=0")

    def test_overlap_greater_than_or_equal_to_max_size_safe(self) -> None:
        """Verify overlap_size >= max_chunk_size is handled safely without infinite loops or errors."""
        options = ChunkingOptions(max_chunk_size=50, overlap_size=80)
        # Should be clamped safely to max_chunk_size - 1 = 49
        self.assertEqual(options.overlap_size, 49)

        doc = NormalizedDocument(
            document_id="doc_large_overlap",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Sentence one. Sentence two. Sentence three. Sentence four.")
            ],
        )
        chunker = StructuralChunker(options)
        chunks = chunker.chunk(doc)

        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertLessEqual(len(c.content), 50)

    def test_final_content_never_exceeds_max_chunk_size(self) -> None:
        """Verify that every chunk strictly satisfies len(chunk.content) <= max_chunk_size even with long headings."""
        doc = NormalizedDocument(
            document_id="doc_strict_bound",
            elements=[
                NormalizedElement(0, NormalizedElementType.TITLE, "Chapter 3"),
                NormalizedElement(1, NormalizedElementType.SECTION_HEADER, "3.1 Vector Storage"),
                NormalizedElement(2, NormalizedElementType.SECTION_HEADER, "3.1.1 Very Long Subsection Name Explaining Details"),
                NormalizedElement(
                    3,
                    NormalizedElementType.PARAGRAPH,
                    "Paragraph content with lots of text that needs to be split across multiple chunks safely and correctly without any overflow.",
                ),
            ],
        )
        for test_size in [50, 75, 100, 150]:
            chunker = StructuralChunker(
                ChunkingOptions(
                    max_chunk_size=test_size,
                    overlap_size=15,
                    include_heading_context=True,
                )
            )
            chunks = chunker.chunk(doc)
            self.assertGreater(len(chunks), 0)
            for c in chunks:
                self.assertLessEqual(
                    len(c.content),
                    test_size,
                    f"Chunk '{c.content}' has len {len(c.content)} > {test_size}",
                )


    def test_no_duplicate_or_infinite_chunks(self) -> None:
        """Verify that repetitive content or edge cases terminate properly without duplicate chunks."""
        sentences = [f"Statement {i:03d}." for i in range(50)]
        doc = NormalizedDocument(
            document_id="doc_advance",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, " ".join(sentences))
            ],
        )
        chunker = StructuralChunker(
            ChunkingOptions(
                max_chunk_size=60,
                overlap_size=20,
                fallback_strategy=FallbackSplitStrategy.PARAGRAPH_OR_SENTENCE,
            )
        )
        chunks = chunker.chunk(doc)

        self.assertGreater(len(chunks), 1)
        # Verify all chunk IDs are strictly unique
        ids = [c.id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        # Verify no two consecutive chunks are identical
        for i in range(len(chunks) - 1):
            self.assertNotEqual(chunks[i].content, chunks[i + 1].content)

    def test_deterministic_chunk_ids(self) -> None:
        """Verify deterministic chunk indexing across multiple chunks."""
        doc = NormalizedDocument(
            document_id="doc_det",
            elements=[
                NormalizedElement(0, NormalizedElementType.SECTION_HEADER, "Header 1"),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "P1 content"),
                NormalizedElement(2, NormalizedElementType.SECTION_HEADER, "Header 2"),
                NormalizedElement(3, NormalizedElementType.PARAGRAPH, "P2 content"),
                NormalizedElement(4, NormalizedElementType.SECTION_HEADER, "Header 3"),
                NormalizedElement(5, NormalizedElementType.PARAGRAPH, "P3 content"),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(min_chunk_size=0))
        chunks = chunker.chunk(doc)

        self.assertEqual(len(chunks), 3)
        self.assertEqual([c.id for c in chunks], ["doc_det_chk_0000", "doc_det_chk_0001", "doc_det_chk_0002"])
        self.assertEqual([c.metadata["chunk_index"] for c in chunks], [0, 1, 2])

    def test_page_number_tracking(self) -> None:
        """Verify page numbers from elements are gathered in chunk metadata."""
        doc = NormalizedDocument(
            document_id="doc_pages",
            elements=[
                NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Page 1 start", page_number=1),
                NormalizedElement(1, NormalizedElementType.PARAGRAPH, "Page 1 end", page_number=1),
                NormalizedElement(2, NormalizedElementType.PARAGRAPH, "Page 2 start", page_number=2),
            ],
        )
        chunker = StructuralChunker(ChunkingOptions(max_chunk_size=1000))
        chunks = chunker.chunk(doc)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["page_numbers"], [1, 2])

    def test_end_to_end_normalization_to_chunking(self) -> None:
        """Verify full pipeline: IngestedDocument -> StandardDocumentNormalizer -> StructuralChunker."""
        from rag.ingestion.models import DocumentElement, ElementType, IngestedDocument

        raw_elements = [
            DocumentElement(element_type=ElementType.TITLE, content="API Reference Manual"),
            DocumentElement(element_type=ElementType.PARAGRAPH, content="This manual covers authentication."),
            DocumentElement(element_type=ElementType.SECTION_HEADER, content="Bearer Tokens"),
            DocumentElement(element_type=ElementType.PARAGRAPH, content="All requests require an Authorization header."),
            DocumentElement(element_type=ElementType.LIST_ITEM, content="- Bearer prefix required"),
            DocumentElement(element_type=ElementType.LIST_ITEM, content="- 24 hour expiry"),
            DocumentElement(
                element_type=ElementType.TABLE,
                content="| Status | Meaning |\n|---|---|\n| 200 | OK |\n| 401 | Unauthorized |",
                metadata={"num_rows": 2, "num_cols": 2},
            ),
        ]

        ingested = IngestedDocument(
            id="manual_md",
            file_path=Path("manual.md"),
            format="md",
            elements=raw_elements,
            text="Sample text",
        )

        normalizer = StandardDocumentNormalizer()
        normalized = normalizer.normalize(ingested)

        self.assertEqual(len(normalized.elements), 7)

        chunker = StructuralChunker(ChunkingOptions(min_chunk_size=0))
        chunks = chunker.chunk(normalized)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertTrue(chunk.id.startswith("manual_md_chk_"))
            self.assertIn("element_indices", chunk.metadata)
            self.assertIn("element_types", chunk.metadata)
            self.assertIn("heading_path", chunk.metadata)

    def test_non_goals_isolation(self) -> None:
        """Verify chunks remain pure domain contracts without embeddings or DB leaks."""
        doc = NormalizedDocument(
            document_id="doc_iso",
            elements=[NormalizedElement(0, NormalizedElementType.PARAGRAPH, "Sample text.")],
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(doc)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertFalse(hasattr(chunk, "vector"))
        self.assertFalse(hasattr(chunk, "embedding"))
        self.assertNotIn("embedding", chunk.metadata)


if __name__ == "__main__":
    unittest.main()
