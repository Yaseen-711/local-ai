"""Unit and boundary tests for the RAG document normalization layer.

Verifies:
1. NormalizedDocument and NormalizedElement instantiation.
2. Whitespace and unicode normalization rules.
3. Empty/whitespace-only element filtering.
4. Structural element preservation (headings, paragraphs, lists, tables).
5. Heading hierarchy tracking (parent_heading).
6. Page numbers and ordering preservation.
7. Determinism (same input -> same output).
8. Independence from Docling internals.
9. Non-goals: absence of chunks, embeddings, and database operations.
10. End-to-end pipeline: IngestedDocument -> Normalizer -> NormalizedDocument.
"""

from pathlib import Path
import tempfile
import unittest

from rag.domain.models import Document
from rag.ingestion.docling import DoclingDocumentIngester
from rag.ingestion.models import DocumentElement, ElementType, IngestedDocument
from rag.normalization import (
    DocumentNormalizer,
    NormalizedDocument,
    NormalizedElement,
    NormalizedElementType,
    StandardDocumentNormalizer,
    clean_table_content,
    clean_text_whitespace,
)


class TestDocumentNormalization(unittest.TestCase):
    """Test suite for RAG document normalization."""

    def setUp(self) -> None:
        self.normalizer = StandardDocumentNormalizer()

    def test_protocol_compliance(self) -> None:
        """Verify StandardDocumentNormalizer satisfies DocumentNormalizer protocol."""
        self.assertIsInstance(self.normalizer, DocumentNormalizer)
        self.assertTrue(issubclass(StandardDocumentNormalizer, DocumentNormalizer))

    def test_clean_text_whitespace(self) -> None:
        """Verify whitespace and unicode normalization rules on text blocks."""
        # Multiple spaces and tabs
        raw = "  This   is    a \t  sentence   with    spaces.  "
        cleaned = clean_text_whitespace(raw)
        self.assertEqual(cleaned, "This is a sentence with spaces.")

        # Line breaks preserved, multiple vertical breaks collapsed
        multiline = "Paragraph 1.\n\n\n\n\nParagraph 2.\nLine continuation."
        cleaned_multi = clean_text_whitespace(multiline)
        self.assertEqual(cleaned_multi, "Paragraph 1.\n\nParagraph 2.\nLine continuation.")

        # Unicode non-breaking space (U+00A0) normalization
        nbsp_text = "Word1\u00a0Word2\u00a0Word3"
        cleaned_nbsp = clean_text_whitespace(nbsp_text)
        self.assertEqual(cleaned_nbsp, "Word1 Word2 Word3")

    def test_clean_table_content(self) -> None:
        """Verify table content cleaning preserves rows and columns without collapsing structure."""
        raw_table = """
|   Col A   |   Col B   |
|-----------|-----------|
|   Val 1   |   Val 2   |

|   Val 3   |   Val 4   |
"""
        cleaned = clean_table_content(raw_table)
        lines = cleaned.split("\n")
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("| Col A"))
        self.assertTrue(lines[2].startswith("| Val 1"))

    def test_empty_and_whitespace_element_filtering(self) -> None:
        """Verify elements containing only whitespace are stripped."""
        ingested = IngestedDocument(
            id="doc_empty_test",
            file_path=Path("/tmp/test.md"),
            format="md",
            elements=[
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Valid paragraph"),
                DocumentElement(element_type=ElementType.PARAGRAPH, content="   \n\t  "),
                DocumentElement(element_type=ElementType.PARAGRAPH, content=""),
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Another valid paragraph"),
            ],
            text="Raw text",
        )

        norm_doc = self.normalizer.normalize(ingested)
        self.assertEqual(len(norm_doc.elements), 2)
        self.assertEqual(norm_doc.elements[0].content, "Valid paragraph")
        self.assertEqual(norm_doc.elements[1].content, "Another valid paragraph")

    def test_heading_hierarchy_and_parent_tracking(self) -> None:
        """Verify headings preserve level and propagate parent_heading context to children."""
        ingested = IngestedDocument(
            id="doc_hierarchy_test",
            file_path=Path("/tmp/test.md"),
            format="md",
            elements=[
                DocumentElement(element_type=ElementType.TITLE, content="Main Title", heading_level=1),
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Introductory paragraph"),
                DocumentElement(element_type=ElementType.SECTION_HEADER, content="Section 1: Setup", heading_level=2),
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Setup step details"),
                DocumentElement(element_type=ElementType.TABLE, content="| Key | Val |\n|---|---|"),
                DocumentElement(element_type=ElementType.SECTION_HEADER, content="Section 2: Run", heading_level=2),
                DocumentElement(element_type=ElementType.LIST_ITEM, content="Command to execute"),
            ],
            text="Raw text",
        )

        norm_doc = self.normalizer.normalize(ingested)
        elems = norm_doc.elements

        self.assertEqual(len(elems), 7)

        # Main Title (heading itself has no parent)
        self.assertEqual(elems[0].element_type, NormalizedElementType.TITLE)
        self.assertIsNone(elems[0].parent_heading)

        # Intro paragraph has "Main Title" as parent
        self.assertEqual(elems[1].element_type, NormalizedElementType.PARAGRAPH)
        self.assertEqual(elems[1].parent_heading, "Main Title")

        # Section 1 Header
        self.assertEqual(elems[2].element_type, NormalizedElementType.SECTION_HEADER)
        self.assertIsNone(elems[2].parent_heading)
        self.assertEqual(elems[2].heading_level, 2)

        # Children under Section 1 have "Section 1: Setup" as parent
        self.assertEqual(elems[3].parent_heading, "Section 1: Setup")
        self.assertEqual(elems[4].parent_heading, "Section 1: Setup")

        # Section 2 Header switches the active parent
        self.assertEqual(elems[5].element_type, NormalizedElementType.SECTION_HEADER)
        self.assertEqual(elems[6].parent_heading, "Section 2: Run")

    def test_table_metadata_and_structure_preservation(self) -> None:
        """Verify tables preserve rows, columns metadata and content."""
        table_meta = {"num_rows": 5, "num_cols": 3}
        table_md = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"

        ingested = IngestedDocument(
            id="doc_table_test",
            file_path=Path("/tmp/table.md"),
            format="md",
            elements=[
                DocumentElement(
                    element_type=ElementType.TABLE,
                    content=table_md,
                    page_number=2,
                    metadata=table_meta,
                ),
            ],
            text=table_md,
        )

        norm_doc = self.normalizer.normalize(ingested)
        self.assertEqual(len(norm_doc.tables), 1)

        table_elem = norm_doc.tables[0]
        self.assertEqual(table_elem.element_type, NormalizedElementType.TABLE)
        self.assertEqual(table_elem.metadata["num_rows"], 5)
        self.assertEqual(table_elem.metadata["num_cols"], 3)
        self.assertEqual(table_elem.page_number, 2)
        self.assertIn("| A | B | C |", table_elem.content)

    def test_page_number_and_ordering_preservation(self) -> None:
        """Verify 0-indexed element positions and 1-indexed page numbers are preserved."""
        ingested = IngestedDocument(
            id="doc_page_test",
            file_path=Path("/tmp/pages.pdf"),
            format="pdf",
            elements=[
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Page 1 item", page_number=1),
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Page 2 item", page_number=2),
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Page 3 item", page_number=3),
            ],
            text="Text",
        )

        norm_doc = self.normalizer.normalize(ingested)
        self.assertEqual(norm_doc.elements[0].index, 0)
        self.assertEqual(norm_doc.elements[0].page_number, 1)
        self.assertEqual(norm_doc.elements[1].index, 1)
        self.assertEqual(norm_doc.elements[1].page_number, 2)
        self.assertEqual(norm_doc.elements[2].index, 2)
        self.assertEqual(norm_doc.elements[2].page_number, 3)
        self.assertEqual(norm_doc.page_count, 3)

    def test_determinism(self) -> None:
        """Verify that identical input always produces identical normalized output."""
        ingested = IngestedDocument(
            id="doc_det_test",
            file_path=Path("/tmp/det.md"),
            format="md",
            elements=[
                DocumentElement(element_type=ElementType.TITLE, content="Title   1"),
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Text with   irregular   spaces."),
            ],
            text="Raw",
        )

        res1 = self.normalizer.normalize(ingested)
        res2 = self.normalizer.normalize(ingested)

        self.assertEqual(res1.text, res2.text)
        self.assertEqual(len(res1.elements), len(res2.elements))
        for e1, e2 in zip(res1.elements, res2.elements):
            self.assertEqual(e1, e2)

    def test_to_domain_document(self) -> None:
        """Verify conversion from NormalizedDocument to domain Document."""
        ingested = IngestedDocument(
            id="doc_domain_test",
            file_path=Path("/tmp/domain.md"),
            format="md",
            elements=[
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Test content"),
            ],
            text="Test content",
        )

        norm_doc = self.normalizer.normalize(ingested)
        domain_doc = norm_doc.to_domain_document()

        self.assertIsInstance(domain_doc, Document)
        self.assertEqual(domain_doc.id, "doc_domain_test")
        self.assertEqual(domain_doc.content, "Test content")
        self.assertEqual(domain_doc.metadata["element_count"], 1)
        self.assertTrue(domain_doc.metadata["normalized"])

    def test_non_goals_not_performed(self) -> None:
        """Verify normalization does NOT perform chunking, embedding, or database operations."""
        ingested = IngestedDocument(
            id="doc_nongoals",
            file_path=Path("/tmp/test.md"),
            format="md",
            elements=[
                DocumentElement(element_type=ElementType.PARAGRAPH, content="Non-goals check"),
            ],
            text="Non-goals check",
        )

        norm_doc = self.normalizer.normalize(ingested)

        # No chunks: Result is a single document with elements
        self.assertFalse(hasattr(norm_doc, "chunks"))
        self.assertFalse(hasattr(norm_doc, "chunk_size"))
        self.assertNotEqual(type(norm_doc).__name__, "Chunk")

        # No embeddings
        self.assertFalse(hasattr(norm_doc, "embedding"))
        for elem in norm_doc.elements:
            self.assertFalse(hasattr(elem, "embedding"))

        # No database
        self.assertFalse(hasattr(self.normalizer, "db"))
        self.assertFalse(hasattr(self.normalizer, "session"))

    def test_end_to_end_ingestion_to_normalization_pipeline(self) -> None:
        """Verify end-to-end flow from real Docling ingestion to normalization."""
        sample_md = """# Pipeline Document Title

This is an introductory paragraph with   redundant    whitespace.

## Key Specifications

| Spec | Value |
|------|-------|
| Dim  | 768   |
| DB   | pgvector |

- Item one
- Item two
"""
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(sample_md)
            tmp_path = Path(f.name)

        try:
            # 1. Ingestion
            ingester = DoclingDocumentIngester(do_ocr=False)
            ingested_doc = ingester.ingest(tmp_path)
            self.assertIsInstance(ingested_doc, IngestedDocument)

            # 2. Normalization
            normalizer = StandardDocumentNormalizer()
            normalized_doc = normalizer.normalize(ingested_doc)
            self.assertIsInstance(normalized_doc, NormalizedDocument)

            # Verify structural preservation
            self.assertEqual(len(normalized_doc.headings), 2)
            self.assertEqual(normalized_doc.headings[0].content, "Pipeline Document Title")
            self.assertEqual(normalized_doc.headings[1].content, "Key Specifications")

            # Verify table preservation
            self.assertEqual(len(normalized_doc.tables), 1)
            self.assertIn("| Dim", normalized_doc.tables[0].content)

            # Verify whitespace cleaning
            intro_p = [p for p in normalized_doc.paragraphs if "introductory" in p.content][0]
            self.assertEqual(intro_p.content, "This is an introductory paragraph with redundant whitespace.")
            self.assertEqual(intro_p.parent_heading, "Pipeline Document Title")

            # Verify Docling classes are completely absent from normalized output
            docling_types = {"DoclingDocument", "TitleItem", "TextItem", "TableItem", "ListItem"}
            elem_types = {type(elem).__name__ for elem in normalized_doc.elements}
            self.assertTrue(docling_types.isdisjoint(elem_types))

        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
