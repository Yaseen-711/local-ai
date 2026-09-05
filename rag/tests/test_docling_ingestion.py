"""Tests for Docling document ingestion layer and DocumentIngester protocol.

Verifies:
1. DocumentIngester protocol compliance.
2. Ingestion error handling (missing files, directories, unsupported extensions).
3. Real document ingestion using local test fixtures.
4. Structural element preservation (titles, headings, paragraphs, tables, lists).
5. Conversion to domain Document model.
6. Absence of chunking, embedding, or database operations.
"""

import tempfile
import unittest
from pathlib import Path

from rag.domain.models import Document
from rag.ingestion.docling import (
    DoclingDocumentIngester,
    SUPPORTED_EXTENSIONS,
)
from rag.ingestion.errors import (
    IngestionError,
    UnsupportedDocumentError,
)
from rag.ingestion.interfaces import DocumentIngester
from rag.ingestion.models import (
    DocumentElement,
    ElementType,
    IngestedDocument,
)


class TestDoclingIngestion(unittest.TestCase):
    """Test suite for Docling document ingestion."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create shared sample markdown and text test fixtures."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Markdown fixture with rich structure
        cls.md_content = """# Local AI Foundation Overview

The Local AI Foundation provides a modular local AI infrastructure stack.

## Architecture and Core

The core coordinates model registration and execution providers.

| Component | Responsibility | Status |
|-----------|----------------|--------|
| Registry  | Model metadata | Active |
| Storage   | pgvector       | Active |

Key capabilities:
- Native GGUF inference
- Isolated RAG subsystem
- Standardized interfaces
"""
        cls.md_file = cls.temp_path / "sample_architecture.md"
        cls.md_file.write_text(cls.md_content, encoding="utf-8")

        # Plain text fixture
        cls.txt_file = cls.temp_path / "simple_notes.txt"
        cls.txt_file.write_text("Simple plain text document notes.", encoding="utf-8")

        # Unsupported file
        cls.unsupported_file = cls.temp_path / "data.bin"
        cls.unsupported_file.write_bytes(b"\x00\x01\x02\x03")

        # Real ingester
        cls.ingester = DoclingDocumentIngester(do_ocr=False)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test fixtures."""
        cls.temp_dir.cleanup()

    def test_protocol_compliance(self) -> None:
        """Verify DoclingDocumentIngester implements DocumentIngester protocol."""
        self.assertIsInstance(self.ingester, DocumentIngester)
        self.assertTrue(issubclass(DoclingDocumentIngester, DocumentIngester))

    def test_supports_format(self) -> None:
        """Verify format support checking."""
        self.assertTrue(self.ingester.supports_format(self.md_file))
        self.assertTrue(self.ingester.supports_format(self.txt_file))
        self.assertTrue(self.ingester.supports_format("report.pdf"))
        self.assertTrue(self.ingester.supports_format("document.docx"))
        self.assertFalse(self.ingester.supports_format(self.unsupported_file))
        self.assertFalse(self.ingester.supports_format("archive.zip"))

    def test_missing_file_raises_file_not_found(self) -> None:
        """Verify missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.ingester.ingest(self.temp_path / "non_existent_file.md")

    def test_directory_path_raises_value_error(self) -> None:
        """Verify directory path raises ValueError."""
        with self.assertRaises(ValueError):
            self.ingester.ingest(self.temp_path)

    def test_unsupported_format_raises_error(self) -> None:
        """Verify unsupported file extension raises UnsupportedDocumentError."""
        with self.assertRaises(UnsupportedDocumentError):
            self.ingester.ingest(self.unsupported_file)

    def test_real_markdown_ingestion_and_structure_preservation(self) -> None:
        """Verify real markdown document is parsed into structured elements."""
        result = self.ingester.ingest(self.md_file)

        # 1. Output type check
        self.assertIsInstance(result, IngestedDocument)
        self.assertEqual(result.format, "md")
        self.assertEqual(result.file_path, self.md_file.resolve())
        self.assertTrue(result.id.startswith("doc_sample_architecture_"))
        self.assertGreater(len(result.text), 50)

        # 2. Structural preservation
        headings = result.headings
        self.assertGreaterEqual(len(headings), 2)
        self.assertEqual(headings[0].element_type, ElementType.TITLE)
        self.assertIn("Local AI Foundation Overview", headings[0].content)
        self.assertEqual(headings[1].element_type, ElementType.SECTION_HEADER)
        self.assertIn("Architecture and Core", headings[1].content)

        # 3. Table preservation
        tables = result.tables
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].element_type, ElementType.TABLE)
        self.assertIn("Registry", tables[0].content)
        self.assertIn("Storage", tables[0].content)
        self.assertEqual(tables[0].metadata.get("num_cols"), 3)

        # 4. Paragraphs and lists preservation
        paragraphs = result.paragraphs
        self.assertGreaterEqual(len(paragraphs), 1)
        self.assertTrue(any("modular local AI infrastructure" in p.content for p in paragraphs))

        list_items = result.list_items
        self.assertGreaterEqual(len(list_items), 2)
        self.assertTrue(any("Native GGUF inference" in item.content for item in list_items))

    def test_real_plain_text_ingestion(self) -> None:
        """Verify plain text file ingestion."""
        result = self.ingester.ingest(self.txt_file)
        self.assertIsInstance(result, IngestedDocument)
        self.assertEqual(result.format, "txt")
        self.assertIn("Simple plain text document notes", result.text)

    def test_conversion_to_domain_document(self) -> None:
        """Verify IngestedDocument can convert cleanly to domain Document."""
        ingested = self.ingester.ingest(self.md_file)
        domain_doc = ingested.to_domain_document()

        self.assertIsInstance(domain_doc, Document)
        self.assertEqual(domain_doc.id, ingested.id)
        self.assertEqual(domain_doc.content, ingested.text)
        self.assertIn("source_path", domain_doc.metadata)
        self.assertEqual(domain_doc.metadata["format"], "md")
        self.assertGreater(domain_doc.metadata["element_count"], 0)

    def test_non_goals_not_performed(self) -> None:
        """Verify ingestion does NOT perform chunking, embedding, or indexing."""
        result = self.ingester.ingest(self.md_file)

        # No chunking: Result is a single IngestedDocument with elements, not chunks
        self.assertNotEqual(type(result).__name__, "Chunk")
        self.assertFalse(hasattr(result, "chunk_index"))
        self.assertFalse(hasattr(result, "chunks"))

        # No embeddings: No vector coordinates exist on elements or document
        self.assertFalse(hasattr(result, "embedding"))
        self.assertFalse(hasattr(result, "embeddings"))
        for elem in result.elements:
            self.assertFalse(hasattr(elem, "embedding"))

        # No database connection: No DB engine or session is touched
        self.assertFalse(hasattr(self.ingester, "db"))
        self.assertFalse(hasattr(self.ingester, "session"))


if __name__ == "__main__":
    unittest.main()
