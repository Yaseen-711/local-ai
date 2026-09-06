"""Unit and integration tests for the Local RAG Developer / Manual Test Harness (rag/cli/).

Covers:
1. File path validation (existent, non-existent, unsupported extension, directories)
2. Harness initialization and dependency injection
3. Ingestion workflow and IngestionStats / IngestionTimings generation
4. Query execution and QueryResult / QueryTimings generation
5. Rank shift computation logic
6. Document listing and chunk inspection (list_documents, list_chunks, get_chunk_details)
7. Test data cleanup (clear_data)
8. CLI argument parser verification (--ingest, --query, --top-k, --top-n, --debug)
9. Console formatting and summary verification
10. End-to-end integration test with database and harness
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
from typing import List, Tuple
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import delete

from rag.cli.console import (
    RAGConsoleApp,
    build_arg_parser,
    display_documents,
    display_ingestion_stats,
    display_query_results,
)
from rag.cli.harness import (
    ChunkDetails,
    DocumentSummary,
    IngestionStats,
    IngestionTimings,
    QueryResult,
    QueryTimings,
    RAGTestHarness,
)
from rag.domain.models import Chunk
from rag.embedding.models import EmbeddingResult
from rag.embedding.nomic import EMBEDDING_DIMENSION
from rag.indexing.indexer import PgVectorIndexer
from rag.metadata.models import ChunkMetadata
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.reranking.models import RankedChunk
from rag.retrieval.models import RetrievedChunk
from rag.retrieval.retriever import PgVectorRetriever
from rag.storage.database import DatabaseManager
from rag.storage.models import ChunkModel, DocumentModel


def fake_reranker_backend(pairs: List[Tuple[str, str]]) -> List[float]:
    """Deterministic mock scoring for test harness."""
    scores: List[float] = []
    for query, content in pairs:
        q_words = set(query.lower().split())
        c_words = set(content.lower().split())
        overlap = len(q_words & c_words)
        scores.append(float(overlap * 2.0 - 0.5))
    return scores


class TestRAGTestHarnessUnit(unittest.TestCase):
    """Unit tests for RAGTestHarness components and validation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_validate_file_path_valid(self) -> None:
        """Valid file paths (pdf, md, txt) resolve correctly."""
        for ext in [".pdf", ".md", ".txt"]:
            file_path = Path(self.temp_dir.name) / f"test{ext}"
            file_path.write_text("sample content")
            resolved = RAGTestHarness.validate_file_path(file_path)
            self.assertEqual(resolved, file_path.resolve())

    def test_validate_file_path_nonexistent(self) -> None:
        """Nonexistent file raises FileNotFoundError."""
        missing = Path(self.temp_dir.name) / "does_not_exist.pdf"
        with self.assertRaises(FileNotFoundError):
            RAGTestHarness.validate_file_path(missing)

    def test_validate_file_path_directory(self) -> None:
        """Directory path raises ValueError."""
        dir_path = Path(self.temp_dir.name) / "somedir"
        dir_path.mkdir()
        with self.assertRaises(ValueError) as ctx:
            RAGTestHarness.validate_file_path(dir_path)
        self.assertIn("not a regular file", str(ctx.exception).lower())

    def test_validate_file_path_unsupported_ext(self) -> None:
        """Unsupported file extension raises ValueError."""
        bad_file = Path(self.temp_dir.name) / "data.xyz"
        bad_file.write_text("binary data")
        with self.assertRaises(ValueError) as ctx:
            RAGTestHarness.validate_file_path(bad_file)
        self.assertIn("unsupported file format", str(ctx.exception).lower())

    def test_query_validation_empty_query(self) -> None:
        """Empty or whitespace queries raise ValueError."""
        mock_db = MagicMock()
        harness = RAGTestHarness(db_manager=mock_db)
        with self.assertRaises(ValueError):
            harness.query("")
        with self.assertRaises(ValueError):
            harness.query("   \n\t  ")

    def test_rank_shift_calculation(self) -> None:
        """Rank shifts calculate promotion, demotion, and equality correctly."""
        retrieved_chunk_1 = RetrievedChunk(
            chunk_id="c1",
            document_id="doc1",
            content="Content 1",
            metadata={"source_path": "/test.pdf", "page_numbers": [1]},
            similarity_score=0.9,
            rank=1,
            chunk_index=0,
        )
        retrieved_chunk_2 = RetrievedChunk(
            chunk_id="c2",
            document_id="doc1",
            content="Content 2",
            metadata={"source_path": "/test.pdf", "page_numbers": [2]},
            similarity_score=0.8,
            rank=2,
            chunk_index=1,
        )

        # Invert order in reranking
        ranked_chunk_2 = RankedChunk.from_retrieved_chunk(
            candidate=retrieved_chunk_2,
            reranking_score=3.5,
            rerank_rank=1,
        )
        ranked_chunk_1 = RankedChunk.from_retrieved_chunk(
            candidate=retrieved_chunk_1,
            reranking_score=1.2,
            rerank_rank=2,
        )

        # ranked_chunk_2 was original_retrieval_rank=2, now rerank_rank=1 -> shift = 2 - 1 = +1
        shift_2 = ranked_chunk_2.original_retrieval_rank - ranked_chunk_2.rerank_rank
        self.assertEqual(shift_2, 1)

        # ranked_chunk_1 was original_retrieval_rank=1, now rerank_rank=2 -> shift = 1 - 2 = -1
        shift_1 = ranked_chunk_1.original_retrieval_rank - ranked_chunk_1.rerank_rank
        self.assertEqual(shift_1, -1)

    def test_harness_mocked_ingestion(self) -> None:
        """Ingestion flow generates correct statistics and timings with mocked components."""
        mock_db = MagicMock()
        mock_session_ctx = MagicMock()
        mock_session = MagicMock()
        mock_session.scalar.return_value = None  # No existing doc
        mock_session_ctx.__enter__.return_value = mock_session
        mock_session_ctx.__exit__.return_value = None
        mock_db.session.return_value = mock_session_ctx

        mock_ingester = MagicMock()
        mock_normalizer = MagicMock()
        mock_chunker = MagicMock()
        mock_embed_service = MagicMock()
        mock_indexer = MagicMock()

        # Mock outputs
        mock_ingested = MagicMock()
        mock_ingester.ingest.return_value = mock_ingested

        mock_normalized = MagicMock()
        mock_normalized.document_id = "test-doc"
        mock_normalized.format = "markdown"
        mock_normalized.page_count = 1
        mock_normalized.elements = [MagicMock()]
        mock_normalizer.normalize.return_value = mock_normalized

        c1 = Chunk(
            id="c1",
            document_id="test-doc",
            content="Heading 1\nBody 1",
            metadata={
                "chunk_index": 0,
                "heading": "Heading 1",
                "heading_path": ["Heading 1"],
            },
        )
        mock_chunker.chunk.return_value = [c1]

        mock_embed_res = EmbeddingResult(
            chunk_id="c1",
            vector=[0.1] * EMBEDDING_DIMENSION,
            model_name="nomic-mock",
            dimension=EMBEDDING_DIMENSION,
        )
        mock_embed_service.embed_chunks.return_value = [mock_embed_res]
        mock_indexer.index_chunks.return_value = ["c1"]

        harness = RAGTestHarness(
            db_manager=mock_db,
            ingester=mock_ingester,
            normalizer=mock_normalizer,
            chunker=mock_chunker,
            embedding_service=mock_embed_service,
            indexer=mock_indexer,
        )

        dummy_file = Path(self.temp_dir.name) / "test.md"
        dummy_file.write_text("# Heading 1\nBody 1")

        progress_calls = []

        def on_progress(step_name: str, step_idx: int, total_steps: int = 6) -> None:
            progress_calls.append((step_name, step_idx, total_steps))

        stats = harness.ingest_document(dummy_file, progress_callback=on_progress)

        self.assertEqual(len(progress_calls), 6)
        self.assertEqual(stats.document_id, "test-doc")
        self.assertEqual(stats.chunk_count, 1)
        self.assertEqual(stats.embedding_count, 1)
        self.assertGreaterEqual(stats.timings.total_sec, 0.0)


class TestRAGConsoleApp(unittest.TestCase):
    """Tests for the RAG CLI console user interface and arguments."""

    def test_cli_argument_parsing(self) -> None:
        """CLI arguments are correctly parsed."""
        parser = build_arg_parser()

        args = parser.parse_args([
            "--ingest", "/path/to/doc.pdf",
            "--query", "What is RAG?",
            "--top-k", "10",
            "--top-n", "3",
            "--debug",
        ])
        self.assertEqual(args.ingest, "/path/to/doc.pdf")
        self.assertEqual(args.query, "What is RAG?")
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.top_n, 3)
        self.assertTrue(args.debug)

    def test_console_display_query_results(self) -> None:
        """Console formats query results clearly including rank shifts and provenance."""
        retrieved_chunk = RetrievedChunk(
            chunk_id="chunk-abc-123",
            document_id="doc-test",
            content="Deep Learning for Search and RAG systems.",
            metadata={
                "file_name": "guide.pdf",
                "page_numbers": [3, 4],
                "heading_path": "Introduction > Background",
            },
            similarity_score=0.8875,
            rank=1,
            chunk_index=0,
        )
        ranked_chunk = RankedChunk.from_retrieved_chunk(
            candidate=retrieved_chunk,
            reranking_score=4.25,
            rerank_rank=1,
        )
        query_result = QueryResult(
            query="What is RAG?",
            query_vector_dim=768,
            retrieved_candidates=[retrieved_chunk],
            ranked_candidates=[ranked_chunk],
            timings=QueryTimings(
                query_embedding_sec=0.012,
                retrieval_sec=0.025,
                reranking_sec=0.035,
                total_sec=0.072,
            ),
            top_k=1,
            top_n=1,
        )

        output_buffer = io.StringIO()
        with patch("sys.stdout", output_buffer):
            display_query_results(query_result)

        rendered = output_buffer.getvalue()
        self.assertIn("STAGE 1: VECTOR RETRIEVAL", rendered)
        self.assertIn("STAGE 2: CROSS-ENCODER RERANKING", rendered)
        self.assertIn("Introduction > Background", rendered)
        self.assertIn("0.8875", rendered)
        self.assertIn("4.2500", rendered)
        self.assertIn("Shift: =", rendered)  # Rank shift
        self.assertIn("LLM Generation:      NOT IMPLEMENTED", rendered)

    def test_display_documents_never_truncates_long_id(self) -> None:
        """Verify that long document IDs (e.g. 43+ characters) are displayed in full without truncation."""
        long_doc_id = "doc_KPRL-SAFETY-HANDBOOK-2020_08f62889a3c6"
        docs = [
            DocumentSummary(
                document_id=long_doc_id,
                file_name="KPRL-SAFETY-HANDBOOK-2020.pdf",
                format="PDF",
                page_count=50,
                chunk_count=120,
                created_at="2026-09-06 12:00:00",
            )
        ]
        output_buffer = io.StringIO()
        with patch("sys.stdout", output_buffer):
            display_documents(docs)

        rendered = output_buffer.getvalue()
        # Full long document ID must appear intact in output
        self.assertIn(long_doc_id, rendered)

    def test_handle_inspect_with_exact_id_and_row_number(self) -> None:
        """Test handle_inspect resolving both exact document ID and row number."""
        long_doc_id = "doc_KPRL-SAFETY-HANDBOOK-2020_08f62889a3c6"
        mock_harness = MagicMock()
        mock_harness.list_documents.return_value = [
            DocumentSummary(
                document_id=long_doc_id,
                file_name="safety.pdf",
                format="pdf",
                page_count=10,
                chunk_count=5,
                created_at="2026-09-06",
            )
        ]
        mock_harness.list_chunks_for_document.return_value = [
            {"chunk_id": "c1", "chunk_index": 0, "page": 1, "heading": "Intro", "preview": "sample"}
        ]

        app = RAGConsoleApp(harness=mock_harness)

        # Case A: User enters/pastes the exact displayed document ID
        with patch("builtins.input", side_effect=["1", long_doc_id]):
            app.handle_inspect()
        mock_harness.list_chunks_for_document.assert_called_with(long_doc_id)

        # Case B: User types row number '1'
        mock_harness.reset_mock()
        mock_harness.list_documents.return_value = [
            DocumentSummary(
                document_id=long_doc_id,
                file_name="safety.pdf",
                format="pdf",
                page_count=10,
                chunk_count=5,
                created_at="2026-09-06",
            )
        ]
        mock_harness.list_chunks_for_document.return_value = [
            {"chunk_id": "c1", "chunk_index": 0, "page": 1, "heading": "Intro", "preview": "sample"}
        ]
        with patch("builtins.input", side_effect=["1", "1"]):
            app.handle_inspect()
        mock_harness.list_chunks_for_document.assert_called_with(long_doc_id)


class TestRAGTestHarnessIntegration(unittest.TestCase):
    """Integration test with database and mock models for fast regression testing."""

    TEST_DOC_ID = "integration-doc-1"

    def setUp(self) -> None:
        self.db = DatabaseManager()
        self.db.init_db()
        self._cleanup()

        self.retriever = PgVectorRetriever(self.db)
        self.reranker = CrossEncoderReranker(backend=fake_reranker_backend)
        self.harness = RAGTestHarness(
            db_manager=self.db,
            retriever=self.retriever,
            reranker=self.reranker,
        )

    def tearDown(self) -> None:
        self._cleanup()
        self.db.close()

    def _cleanup(self) -> None:
        with self.db.session() as session:
            session.execute(
                delete(DocumentModel).where(DocumentModel.id == self.TEST_DOC_ID)
            )

    def test_harness_integration_query_and_inspection(self) -> None:
        """Verify harness query, list_documents, and inspect chunks against actual database."""
        indexer = PgVectorIndexer(self.db)

        # Index two sample chunks
        c1 = Chunk(
            id=f"{self.TEST_DOC_ID}-c1",
            document_id=self.TEST_DOC_ID,
            content="Vector databases provide high-performance similarity search.",
            metadata={
                "document_id": self.TEST_DOC_ID,
                "chunk_index": 0,
                "heading_path": "Databases > Vector",
                "file_name": "vectors.pdf",
                "primary_page": 1,
            },
        )
        c2 = Chunk(
            id=f"{self.TEST_DOC_ID}-c2",
            document_id=self.TEST_DOC_ID,
            content="Cross encoders evaluate query and document pairs jointly.",
            metadata={
                "document_id": self.TEST_DOC_ID,
                "chunk_index": 1,
                "heading_path": "Models > CrossEncoder",
                "file_name": "vectors.pdf",
                "primary_page": 2,
            },
        )

        emb1 = EmbeddingResult(chunk_id=c1.id, vector=[0.1] * EMBEDDING_DIMENSION, model_name="nomic-mock", dimension=EMBEDDING_DIMENSION)
        emb2 = EmbeddingResult(chunk_id=c2.id, vector=[0.2] * EMBEDDING_DIMENSION, model_name="nomic-mock", dimension=EMBEDDING_DIMENSION)

        indexer.index_chunks([c1, c2], [emb1, emb2])

        # Test document listing
        docs = self.harness.list_documents()
        matching_docs = [d for d in docs if d.document_id == self.TEST_DOC_ID]
        self.assertEqual(len(matching_docs), 1)
        self.assertEqual(matching_docs[0].document_id, self.TEST_DOC_ID)
        self.assertEqual(matching_docs[0].chunk_count, 2)

        # Test chunk listing
        chunks = self.harness.list_chunks_for_document(self.TEST_DOC_ID)
        self.assertEqual(len(chunks), 2)

        # Test chunk details inspection
        details = self.harness.get_chunk_details(c1.id)
        self.assertIsNotNone(details)
        assert details is not None
        self.assertEqual(details.chunk_id, c1.id)
        self.assertEqual(details.metadata.get("heading_path"), "Databases > Vector")
        self.assertIn("similarity search", details.content)

        # Test mock query execution
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * EMBEDDING_DIMENSION
        self.harness.embedding_service.model = mock_embedder

        result = self.harness.query(question="similarity search vector", top_k=2, top_n=2)
        self.assertGreaterEqual(len(result.retrieved_candidates), 1)
        self.assertGreaterEqual(len(result.ranked_candidates), 1)

        # Clear test data for this specific document
        deleted_count = self.harness.clear_data(document_id=self.TEST_DOC_ID)
        self.assertEqual(deleted_count, 1)
        docs_after = [d for d in self.harness.list_documents() if d.document_id == self.TEST_DOC_ID]
        self.assertEqual(len(docs_after), 0)

    def test_displayed_document_id_inspection_regression(self) -> None:
        """Regression test: prove a displayed document ID can be passed back into inspection and returns chunks."""
        indexer = PgVectorIndexer(self.db)
        long_doc_id = "doc_KPRL-SAFETY-HANDBOOK-2020_08f62889a3c6"

        with self.db.session() as session:
            session.execute(delete(DocumentModel).where(DocumentModel.id == long_doc_id))

        c1 = Chunk(
            id=f"{long_doc_id}-c0",
            document_id=long_doc_id,
            content="Emergency shutdown protocols and evacuation routes.",
            metadata={
                "document_id": long_doc_id,
                "chunk_index": 0,
                "heading_path": "Safety > Protocols",
                "primary_page": 5,
            },
        )
        emb1 = EmbeddingResult(
            chunk_id=c1.id,
            vector=[0.1] * EMBEDDING_DIMENSION,
            model_name="nomic-mock",
            dimension=EMBEDDING_DIMENSION,
        )
        indexer.index_chunks([c1], [emb1])

        # 1. Fetch documents via harness
        docs = self.harness.list_documents()
        matching = [d for d in docs if d.document_id == long_doc_id]
        self.assertEqual(len(matching), 1)

        # 2. Render documents table and extract the displayed document ID
        output_buffer = io.StringIO()
        with patch("sys.stdout", output_buffer):
            display_documents(docs)

        rendered = output_buffer.getvalue()
        self.assertIn(long_doc_id, rendered)

        # Extract the displayed ID from the rendered table
        displayed_doc_id = None
        for line in rendered.splitlines():
            if "doc_KPRL" in line:
                tokens = line.split()
                # Line format: # Document_ID Format Pages Chunks Created_At...
                self.assertGreaterEqual(len(tokens), 2)
                displayed_doc_id = tokens[1]
                break

        self.assertIsNotNone(displayed_doc_id)
        # 3. Prove that the displayed ID exactly matches the database ID (not truncated to 24 chars)
        self.assertEqual(displayed_doc_id, long_doc_id)

        # 4. Pass the displayed document ID into list_chunks_for_document
        chunks = self.harness.list_chunks_for_document(displayed_doc_id)
        # Must return the correct chunks, NOT 0 chunks!
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], f"{long_doc_id}-c0")
        self.assertEqual(chunks[0]["heading"], "Safety > Protocols")

        # 5. Clean up
        self.harness.clear_data(document_id=long_doc_id)


if __name__ == "__main__":
    unittest.main()
