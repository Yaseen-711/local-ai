"""Unit and integration tests for the RAG Vector Indexing and Persistence layer (rag/indexing/).

Covers:
A. Single chunk indexing
B. Batch indexing
C. Correct vector persistence
D. Correct vector dimension
E. Chunk ID <-> embedding ID validation
F. Metadata preservation
G. Provenance preservation
H. Content preservation
I. Duplicate/idempotent indexing
J. Re-index/update behavior
K. Invalid vector handling
L. Dimension mismatch
M. Non-finite vector values
N. Transaction rollback behavior
O. Delete behavior (single chunk, document chunks, cascade document delete)
P. Complete embedded-document indexing flow with real Nomic embedding
"""

from __future__ import annotations

import math
import unittest
from typing import List

from sqlalchemy import delete, select

from rag.domain.models import Chunk, Document
from rag.embedding.models import EmbeddingResult
from rag.embedding.nomic import EMBEDDING_DIMENSION, NomicEmbeddingModel, l2_normalize
from rag.indexing.indexer import PgVectorIndexer
from rag.indexing.interfaces import VectorIndexer
from rag.storage.database import DatabaseManager
from rag.storage.models import ChunkModel, DocumentModel


class TestVectorIndexerProtocols(unittest.TestCase):
    """Verify protocol compliance for VectorIndexer."""

    def test_protocol_conformance(self) -> None:
        db = DatabaseManager()
        indexer = PgVectorIndexer(db)
        self.assertIsInstance(indexer, VectorIndexer)
        self.assertEqual(indexer.dimension, 768)


class TestPgVectorIndexerValidation(unittest.TestCase):
    """Validation and error handling tests without requiring live database access."""

    def setUp(self) -> None:
        self.db = DatabaseManager()
        self.indexer = PgVectorIndexer(self.db, dimension=768)

    def test_chunk_embedding_id_mismatch_raises(self) -> None:
        chunk = Chunk(id="c1", document_id="d1", content="Sample content")
        emb = EmbeddingResult(
            chunk_id="c2",  # Mismatched ID!
            vector=[0.1] * 768,
            dimension=768,
            model_name="test-model",
        )
        with self.assertRaises(ValueError) as ctx:
            self.indexer.index_chunk(chunk, emb)
        self.assertIn("Identity mismatch", str(ctx.exception))

    def test_dimension_mismatch_raises(self) -> None:
        chunk = Chunk(id="c1", document_id="d1", content="Sample content")
        # 512-dim embedding against 768-dim indexer
        emb = EmbeddingResult(
            chunk_id="c1",
            vector=[0.1] * 512,
            dimension=512,
            model_name="test-model",
        )
        with self.assertRaises(ValueError) as ctx:
            self.indexer.index_chunk(chunk, emb)
        self.assertIn("does not match indexer dimension", str(ctx.exception))

    def test_non_finite_vector_values_raise(self) -> None:
        chunk = Chunk(id="c1", document_id="d1", content="Sample content")

        # NaN test
        vec_nan = [0.1] * 768
        vec_nan[10] = float("nan")
        emb_nan = EmbeddingResult(
            chunk_id="c1",
            vector=vec_nan,
            dimension=768,
            model_name="test-model",
        )
        with self.assertRaises(ValueError) as ctx:
            self.indexer.index_chunk(chunk, emb_nan)
        self.assertIn("non-finite value", str(ctx.exception))

        # Inf test
        vec_inf = [0.1] * 768
        vec_inf[20] = float("inf")
        emb_inf = EmbeddingResult(
            chunk_id="c1",
            vector=vec_inf,
            dimension=768,
            model_name="test-model",
        )
        with self.assertRaises(ValueError) as ctx:
            self.indexer.index_chunk(chunk, emb_inf)
        self.assertIn("non-finite value", str(ctx.exception))

    def test_empty_content_raises(self) -> None:
        chunk = Chunk(id="c1", document_id="d1", content="   \n\t  ")
        emb = EmbeddingResult(
            chunk_id="c1",
            vector=[0.1] * 768,
            dimension=768,
            model_name="test-model",
        )
        with self.assertRaises(ValueError) as ctx:
            self.indexer.index_chunk(chunk, emb)
        self.assertIn("empty or whitespace-only content", str(ctx.exception))

    def test_sequence_length_mismatch_raises(self) -> None:
        chunk = Chunk(id="c1", document_id="d1", content="Sample content")
        with self.assertRaises(ValueError) as ctx:
            self.indexer.index_chunks([chunk], [])
        self.assertIn("Mismatch between number of chunks", str(ctx.exception))

    def test_empty_sequence_returns_zero(self) -> None:
        indexed = self.indexer.index_chunks([], [])
        self.assertEqual(indexed, 0)


class TestPgVectorIndexerIntegration(unittest.TestCase):
    """Live integration tests executing against PostgreSQL and pgvector."""

    TEST_DOC_ID = "test-doc-idx-001"
    TEST_DOC_B_ID = "test-doc-idx-002"

    def setUp(self) -> None:
        self.db = DatabaseManager()
        self.indexer = PgVectorIndexer(self.db)
        self.db.init_db()
        self._cleanup()

    def tearDown(self) -> None:
        self._cleanup()
        self.db.close()

    def _cleanup(self) -> None:
        with self.db.session() as session:
            session.execute(
                delete(DocumentModel).where(
                    DocumentModel.id.in_([self.TEST_DOC_ID, self.TEST_DOC_B_ID])
                )
            )

    def _make_vector(self, seed: float) -> List[float]:
        vec = [seed + i * 0.001 for i in range(768)]
        return l2_normalize(vec)

    def test_single_chunk_indexing_and_persistence(self) -> None:
        """Verify A, C, D, F, G, H: Single chunk indexing, vector, content, metadata preservation."""
        chunk_id = f"{self.TEST_DOC_ID}-c0"
        vec = self._make_vector(0.5)

        chunk = Chunk(
            id=chunk_id,
            document_id=self.TEST_DOC_ID,
            content="PostgreSQL with pgvector stores 768-dim dense embeddings.",
            metadata={
                "source_path": "/docs/pgvector.md",
                "file_name": "pgvector.md",
                "format": "markdown",
                "heading_path": "Storage > pgvector",
                "page_numbers": [1],
                "primary_page": 1,
                "chunk_index": 0,
            },
        )
        emb = EmbeddingResult(
            chunk_id=chunk_id,
            vector=vec,
            dimension=768,
            model_name="nomic-ai/nomic-embed-text-v1.5",
            is_normalized=True,
            token_count=12,
        )

        self.indexer.index_chunk(chunk, emb)

        # Direct database verification using SQLAlchemy
        with self.db.session() as session:
            stored = session.scalar(
                select(ChunkModel).where(ChunkModel.id == chunk_id)
            )
            self.assertIsNotNone(stored, "Chunk was not persisted to rag_chunks")
            assert stored is not None
            self.assertEqual(stored.id, chunk_id)
            self.assertEqual(stored.document_id, self.TEST_DOC_ID)
            self.assertEqual(
                stored.content,
                "PostgreSQL with pgvector stores 768-dim dense embeddings.",
            )

            # Check vector dimension and values
            stored_vec = list(stored.embedding)
            self.assertEqual(len(stored_vec), 768)
            for expected_val, actual_val in zip(vec, stored_vec):
                self.assertAlmostEqual(expected_val, actual_val, places=4)

            # Check metadata preservation
            self.assertEqual(stored.metadata_["source_path"], "/docs/pgvector.md")
            self.assertEqual(stored.metadata_["heading_path"], "Storage > pgvector")
            self.assertEqual(stored.metadata_["embedding_model"], "nomic-ai/nomic-embed-text-v1.5")
            self.assertTrue(stored.metadata_["is_normalized"])
            self.assertEqual(stored.metadata_["token_count"], 12)

    def test_batch_indexing_and_ordering(self) -> None:
        """Verify B: Batch indexing multiple chunks atomically."""
        chunks: List[Chunk] = []
        embeddings: List[EmbeddingResult] = []

        for i in range(5):
            cid = f"{self.TEST_DOC_ID}-batch-{i}"
            vec = self._make_vector(float(i + 1))
            chunks.append(
                Chunk(
                    id=cid,
                    document_id=self.TEST_DOC_ID,
                    content=f"Batch chunk content {i}",
                    metadata={"chunk_index": i},
                )
            )
            embeddings.append(
                EmbeddingResult(
                    chunk_id=cid,
                    vector=vec,
                    dimension=768,
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                )
            )

        indexed_count = self.indexer.index_chunks(chunks, embeddings)
        self.assertEqual(indexed_count, 5)

        with self.db.session() as session:
            rows = session.scalars(
                select(ChunkModel)
                .where(ChunkModel.document_id == self.TEST_DOC_ID)
                .order_by(ChunkModel.chunk_index)
            ).all()
            self.assertEqual(len(rows), 5)
            for i, row in enumerate(rows):
                self.assertEqual(row.id, f"{self.TEST_DOC_ID}-batch-{i}")
                self.assertEqual(row.content, f"Batch chunk content {i}")
                self.assertEqual(row.chunk_index, i)

    def test_idempotent_duplicate_indexing(self) -> None:
        """Verify I: Re-indexing existing chunks runs cleanly without duplicate errors."""
        chunk_id = f"{self.TEST_DOC_ID}-idem-0"
        vec = self._make_vector(0.8)
        chunk = Chunk(
            id=chunk_id,
            document_id=self.TEST_DOC_ID,
            content="Idempotency test content",
            metadata={"chunk_index": 0},
        )
        emb = EmbeddingResult(
            chunk_id=chunk_id,
            vector=vec,
            dimension=768,
            model_name="test-model",
        )

        # First indexing
        self.indexer.index_chunk(chunk, emb)

        # Second indexing of identical chunk
        self.indexer.index_chunk(chunk, emb)

        with self.db.session() as session:
            rows = session.scalars(
                select(ChunkModel).where(ChunkModel.id == chunk_id)
            ).all()
            self.assertEqual(len(rows), 1, "Duplicate chunk row was created!")

    def test_reindex_update_behavior(self) -> None:
        """Verify J: Re-indexing updates content, metadata, and embedding in-place."""
        chunk_id = f"{self.TEST_DOC_ID}-update-0"
        vec1 = self._make_vector(0.1)
        chunk1 = Chunk(
            id=chunk_id,
            document_id=self.TEST_DOC_ID,
            content="Version 1 content",
            metadata={"version": 1},
        )
        emb1 = EmbeddingResult(
            chunk_id=chunk_id,
            vector=vec1,
            dimension=768,
            model_name="test-model-v1",
        )
        self.indexer.index_chunk(chunk1, emb1)

        # Update with version 2
        vec2 = self._make_vector(0.9)
        chunk2 = Chunk(
            id=chunk_id,
            document_id=self.TEST_DOC_ID,
            content="Version 2 updated content",
            metadata={"version": 2},
        )
        emb2 = EmbeddingResult(
            chunk_id=chunk_id,
            vector=vec2,
            dimension=768,
            model_name="test-model-v2",
        )
        self.indexer.index_chunk(chunk2, emb2)

        with self.db.session() as session:
            stored = session.scalar(
                select(ChunkModel).where(ChunkModel.id == chunk_id)
            )
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.content, "Version 2 updated content")
            self.assertEqual(stored.metadata_["version"], 2)
            self.assertEqual(stored.metadata_["embedding_model"], "test-model-v2")
            # Verify new vector was written
            stored_vec = list(stored.embedding)
            self.assertAlmostEqual(stored_vec[0], vec2[0], places=4)

    def test_transaction_rollback_behavior(self) -> None:
        """Verify N: If an error occurs in batch, entire transaction rolls back."""
        cid1 = f"{self.TEST_DOC_ID}-atomic-1"
        cid2 = f"{self.TEST_DOC_ID}-atomic-2"

        chunk1 = Chunk(id=cid1, document_id=self.TEST_DOC_ID, content="Good chunk 1")
        emb1 = EmbeddingResult(
            chunk_id=cid1,
            vector=self._make_vector(0.1),
            dimension=768,
            model_name="test",
        )

        chunk2 = Chunk(id=cid2, document_id=self.TEST_DOC_ID, content="Bad chunk 2")
        # Intentionally mismatch embedding ID on chunk 2
        emb2 = EmbeddingResult(
            chunk_id="mismatched-id",
            vector=self._make_vector(0.2),
            dimension=768,
            model_name="test",
        )

        with self.assertRaises(ValueError):
            self.indexer.index_chunks([chunk1, chunk2], [emb1, emb2])

        # Verify chunk 1 was NOT persisted due to rollback/pre-validation
        with self.db.session() as session:
            stored1 = session.scalar(
                select(ChunkModel).where(ChunkModel.id == cid1)
            )
            self.assertIsNone(stored1, "Batch failure leaked chunk1 into database!")

    def test_delete_operations(self) -> None:
        """Verify O: Single chunk delete, document chunks delete, and cascade delete."""
        cid1 = f"{self.TEST_DOC_ID}-del-1"
        cid2 = f"{self.TEST_DOC_ID}-del-2"

        chunk1 = Chunk(id=cid1, document_id=self.TEST_DOC_ID, content="Delete test 1")
        chunk2 = Chunk(id=cid2, document_id=self.TEST_DOC_ID, content="Delete test 2")
        emb1 = EmbeddingResult(
            chunk_id=cid1,
            vector=self._make_vector(0.1),
            dimension=768,
            model_name="test",
        )
        emb2 = EmbeddingResult(
            chunk_id=cid2,
            vector=self._make_vector(0.2),
            dimension=768,
            model_name="test",
        )

        self.indexer.index_chunks([chunk1, chunk2], [emb1, emb2])

        # 1. Delete single chunk
        deleted_chunk = self.indexer.delete_chunk(cid1)
        self.assertTrue(deleted_chunk)
        self.assertFalse(self.indexer.delete_chunk("non-existent-chunk"))

        with self.db.session() as session:
            self.assertIsNone(session.scalar(select(ChunkModel).where(ChunkModel.id == cid1)))
            self.assertIsNotNone(session.scalar(select(ChunkModel).where(ChunkModel.id == cid2)))

        # 2. Delete document chunks
        deleted_count = self.indexer.delete_document_chunks(self.TEST_DOC_ID)
        self.assertEqual(deleted_count, 1)

        with self.db.session() as session:
            self.assertIsNone(session.scalar(select(ChunkModel).where(ChunkModel.id == cid2)))

        # 3. Re-index and delete document (cascade delete)
        self.indexer.index_chunk(chunk1, emb1)
        deleted_doc = self.indexer.delete_document(self.TEST_DOC_ID)
        self.assertTrue(deleted_doc)

        with self.db.session() as session:
            self.assertIsNone(session.scalar(select(DocumentModel).where(DocumentModel.id == self.TEST_DOC_ID)))
            self.assertIsNone(session.scalar(select(ChunkModel).where(ChunkModel.id == cid1)))

    def test_document_indexing_flow(self) -> None:
        """Verify P: Complete document and chunk indexing flow."""
        doc = Document(
            id=self.TEST_DOC_B_ID,
            content="Full document text representation.",
            metadata={"title": "Doc B", "author": "RAG Team"},
        )
        cid = f"{self.TEST_DOC_B_ID}-c0"
        chunk = Chunk(
            id=cid,
            document_id=self.TEST_DOC_B_ID,
            content="Document B first chunk.",
            metadata={"chunk_index": 0},
        )
        emb = EmbeddingResult(
            chunk_id=cid,
            vector=self._make_vector(0.3),
            dimension=768,
            model_name="test-model",
        )

        indexed = self.indexer.index_document(doc, [chunk], [emb])
        self.assertEqual(indexed, 1)

        with self.db.session() as session:
            stored_doc = session.scalar(
                select(DocumentModel).where(DocumentModel.id == self.TEST_DOC_B_ID)
            )
            self.assertIsNotNone(stored_doc)
            assert stored_doc is not None
            self.assertEqual(stored_doc.content, "Full document text representation.")
            self.assertEqual(stored_doc.metadata_["title"], "Doc B")

            stored_chunk = session.scalar(
                select(ChunkModel).where(ChunkModel.id == cid)
            )
            self.assertIsNotNone(stored_chunk)


class TestRealNomicToPgVectorIntegration(unittest.TestCase):
    """End-to-end integration: Chunk -> Real NomicEmbedder -> PgVectorIndexer -> Read Back."""

    TEST_DOC_ID = "test-doc-real-nomic"

    def setUp(self) -> None:
        self.db = DatabaseManager()
        self.indexer = PgVectorIndexer(self.db)
        self.db.init_db()
        self._cleanup()

    def tearDown(self) -> None:
        self._cleanup()
        self.db.close()

    def _cleanup(self) -> None:
        with self.db.session() as session:
            session.execute(
                delete(DocumentModel).where(DocumentModel.id == self.TEST_DOC_ID)
            )

    def test_real_nomic_to_pgvector(self) -> None:
        """Verify real Nomic embedding vectors persist and read back from pgvector."""
        # 1. Create real chunk
        chunk_id = f"{self.TEST_DOC_ID}-real-c1"
        chunk_text = "PostgreSQL 17 with pgvector 0.8.6 enables production HNSW and IVFFlat vector indexing."
        chunk = Chunk(
            id=chunk_id,
            document_id=self.TEST_DOC_ID,
            content=chunk_text,
            metadata={"source": "real_model_test", "tier": "foundation"},
        )

        # 2. Generate real Nomic embedding
        model = NomicEmbeddingModel(normalize=True)
        raw_vector = model.embed_document(chunk.content)
        self.assertEqual(len(raw_vector), EMBEDDING_DIMENSION)

        emb = EmbeddingResult(
            chunk_id=chunk.id,
            vector=raw_vector,
            dimension=EMBEDDING_DIMENSION,
            model_name=model.model_name,
            is_normalized=model.is_normalized,
        )

        # 3. Persist chunk and embedding
        self.indexer.index_chunk(chunk, emb)

        # 4. Read back directly from PostgreSQL
        with self.db.session() as session:
            stored = session.scalar(
                select(ChunkModel).where(ChunkModel.id == chunk_id)
            )
            self.assertIsNotNone(stored, "Chunk was not found in database")
            assert stored is not None

            # 5. Verify vector properties
            stored_vec = list(stored.embedding)
            self.assertEqual(len(stored_vec), 768)

            norm = math.sqrt(sum(x * x for x in stored_vec))
            self.assertAlmostEqual(norm, 1.0, places=4)

            for expected_val, actual_val in zip(raw_vector, stored_vec):
                self.assertAlmostEqual(expected_val, actual_val, places=4)

            # 6. Verify content and metadata
            self.assertEqual(stored.content, chunk_text)
            self.assertEqual(stored.metadata_["source"], "real_model_test")
            self.assertEqual(stored.metadata_["embedding_model"], "nomic-ai/nomic-embed-text-v1.5")


if __name__ == "__main__":
    unittest.main()
