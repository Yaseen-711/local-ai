"""End-to-end integration test for PostgreSQL + pgvector RAG storage layer.

Verifies:
1. Database initialization and table creation (rag_documents, rag_chunks).
2. Document persistence using domain Document models.
3. Chunk and 768-dimensional embedding persistence using domain Chunk models.
4. Nearest-neighbor vector similarity retrieval via pgvector.
5. Result ranking, domain model conversion, and score calculation.
6. Isolated cleanup of test records.
"""

import unittest
from typing import List

from sqlalchemy import delete, inspect

from rag.domain.models import Chunk, Document, RetrievalResult
from rag.storage.database import DatabaseManager
from rag.storage.models import DocumentModel, EMBEDDING_DIMENSION
from rag.storage.vector_store import PgVectorStore


class TestPgVectorStorageIntegration(unittest.TestCase):
    """Integration test suite executing against live PostgreSQL + pgvector."""

    TEST_DOC_ID = "test-doc-integration-001"
    TEST_CHUNK_A_ID = "test-chunk-integration-a"
    TEST_CHUNK_B_ID = "test-chunk-integration-b"

    def setUp(self) -> None:
        """Initialize database manager, verify connection, and setup vector store."""
        self.db = DatabaseManager()
        self.store = PgVectorStore(self.db)

        # 1. Initialize tables
        self.db.init_db()

        # Clean up any leftover artifacts from prior failed runs
        self._cleanup_test_data()

    def tearDown(self) -> None:
        """Ensure isolated cleanup after tests run."""
        self._cleanup_test_data()
        self.db.close()

    def _cleanup_test_data(self) -> None:
        """Remove test documents and cascading chunks."""
        with self.db.session() as session:
            session.execute(
                delete(DocumentModel).where(DocumentModel.id == self.TEST_DOC_ID)
            )

    def test_end_to_end_storage_and_vector_retrieval(self) -> None:
        """Execute full integration cycle: init -> insert -> query -> verify -> cleanup."""

        # 1. Verify table creation
        inspector = inspect(self.db.engine)
        table_names = inspector.get_table_names()
        self.assertIn("rag_documents", table_names, "rag_documents table was not created")
        self.assertIn("rag_chunks", table_names, "rag_chunks table was not created")

        # 2. Insert test Document
        doc = Document(
            id=self.TEST_DOC_ID,
            content="Local AI Foundation architecture and RAG subsystem documentation.",
            metadata={"source": "integration_test", "tier": "foundation"},
        )
        self.store.add_document(doc)

        # 3. Create Chunks with deterministic 768-dimensional synthetic embeddings
        # Chunk A vector: unit vector along axis 0
        emb_a: List[float] = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
        # Chunk B vector: unit vector along axis 1 (orthogonal to chunk A)
        emb_b: List[float] = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSION - 2)
        # Query vector: exactly aligns with Chunk A
        query_emb: List[float] = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)

        self.assertEqual(len(emb_a), 768, "Embedding A dimension must be 768")
        self.assertEqual(len(emb_b), 768, "Embedding B dimension must be 768")
        self.assertEqual(len(query_emb), 768, "Query embedding dimension must be 768")

        chunk_a = Chunk(
            id=self.TEST_CHUNK_A_ID,
            document_id=self.TEST_DOC_ID,
            content="Chunk A: Vector database storage using PostgreSQL and pgvector.",
            metadata={"chunk_index": 0, "topic": "pgvector"},
        )
        chunk_b = Chunk(
            id=self.TEST_CHUNK_B_ID,
            document_id=self.TEST_DOC_ID,
            content="Chunk B: Peripheral hardware details and power supply specifications.",
            metadata={"chunk_index": 1, "topic": "hardware"},
        )

        self.store.add_chunks(chunks=[chunk_a, chunk_b], embeddings=[emb_a, emb_b])

        # 4. Retrieve top 2 matches
        results = self.store.query(query_embedding=query_emb, top_k=2)

        # 5. Verify results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2, "Expected exactly 2 retrieval results")

        # Result 1: Chunk A
        res_a = results[0]
        self.assertIsInstance(res_a, RetrievalResult)
        self.assertIsInstance(res_a.chunk, Chunk)
        self.assertEqual(res_a.chunk.id, self.TEST_CHUNK_A_ID)
        self.assertEqual(res_a.chunk.document_id, self.TEST_DOC_ID)
        self.assertEqual(
            res_a.chunk.content,
            "Chunk A: Vector database storage using PostgreSQL and pgvector.",
        )
        self.assertEqual(res_a.chunk.metadata.get("topic"), "pgvector")
        # Cosine distance = 0.0 -> score = 1.0
        self.assertAlmostEqual(res_a.score, 1.0, places=4)

        # Result 2: Chunk B
        res_b = results[1]
        self.assertIsInstance(res_b, RetrievalResult)
        self.assertIsInstance(res_b.chunk, Chunk)
        self.assertEqual(res_b.chunk.id, self.TEST_CHUNK_B_ID)
        self.assertEqual(res_b.chunk.document_id, self.TEST_DOC_ID)
        self.assertEqual(
            res_b.chunk.content,
            "Chunk B: Peripheral hardware details and power supply specifications.",
        )
        self.assertEqual(res_b.chunk.metadata.get("topic"), "hardware")
        # Cosine distance = 1.0 (orthogonal) -> score = 0.0
        self.assertAlmostEqual(res_b.score, 0.0, places=4)

        # Verify ranking: Chunk A is strictly ranked ahead of Chunk B
        self.assertGreater(
            res_a.score,
            res_b.score,
            "Chunk A must have a strictly higher similarity score than Chunk B",
        )


if __name__ == "__main__":
    unittest.main()
