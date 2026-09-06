"""Unit and integration tests for the RAG Vector Retrieval layer (rag/retrieval/).

Covers:
A. Retriever protocol conformance
B. Single result retrieval
C. Multiple result retrieval
D. Correct similarity ordering
E. Top-K behavior
F. Empty database
G. No results after threshold
H. Similarity threshold
I. Dimension mismatch
J. Empty query vector
K. Non-finite query vector
L. Metadata preservation
M. Provenance preservation
N. Content preservation
O. Document filtering
P. Deterministic ordering for equal scores
Q. Database session behavior
R. Retrieval does not mutate stored data
S. Real PostgreSQL + pgvector integration
T. Real Nomic semantic retrieval integration test
"""

from __future__ import annotations

import math
import unittest
from typing import List

from sqlalchemy import delete, select

from rag.domain.models import Chunk
from rag.embedding.models import EmbeddingResult
from rag.embedding.nomic import EMBEDDING_DIMENSION, NomicEmbeddingModel, l2_normalize
from rag.indexing.indexer import PgVectorIndexer
from rag.retrieval.interfaces import VectorRetriever
from rag.retrieval.models import RetrievedChunk
from rag.retrieval.retriever import PgVectorRetriever
from rag.storage.database import DatabaseManager
from rag.storage.models import ChunkModel, DocumentModel


class TestRetrieverProtocolsAndValidation(unittest.TestCase):
    """Validation and protocol tests for PgVectorRetriever."""

    def setUp(self) -> None:
        self.db = DatabaseManager()
        self.retriever = PgVectorRetriever(self.db, dimension=768)

    def test_protocol_conformance(self) -> None:
        """Verify A: Protocol conformance."""
        self.assertIsInstance(self.retriever, VectorRetriever)
        self.assertEqual(self.retriever.dimension, 768)

    def test_empty_query_vector_raises(self) -> None:
        """Verify J: Empty query vector rejection."""
        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve([])
        self.assertIn("query_vector must not be empty", str(ctx.exception))

    def test_dimension_mismatch_raises(self) -> None:
        """Verify I: Dimension mismatch rejection."""
        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve([0.1] * 512)
        self.assertIn("does not match retriever dimension", str(ctx.exception))

    def test_non_finite_query_vector_raises(self) -> None:
        """Verify K: Non-finite values rejection."""
        vec_nan = [0.1] * 768
        vec_nan[5] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve(vec_nan)
        self.assertIn("non-finite value", str(ctx.exception))

        vec_inf = [0.1] * 768
        vec_inf[10] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve(vec_inf)
        self.assertIn("non-finite value", str(ctx.exception))

    def test_invalid_top_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.retrieve([0.1] * 768, top_k=0)
        with self.assertRaises(ValueError):
            self.retriever.retrieve([0.1] * 768, top_k=-5)

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.retrieve([0.1] * 768, similarity_threshold=1.5)
        with self.assertRaises(ValueError):
            self.retriever.retrieve([0.1] * 768, similarity_threshold=-1.5)

    def test_retrieved_chunk_model_validation(self) -> None:
        with self.assertRaises(ValueError):
            RetrievedChunk(
                chunk_id="",
                document_id="doc1",
                content="content",
                similarity_score=0.9,
                rank=1,
            )

        with self.assertRaises(ValueError):
            RetrievedChunk(
                chunk_id="c1",
                document_id="doc1",
                content="content",
                similarity_score=0.9,
                rank=0,  # Must be >= 1
            )


class TestPgVectorRetrieverIntegration(unittest.TestCase):
    """Integration test suite executing vector retrieval against PostgreSQL + pgvector."""

    TEST_DOC_ID = "retrieval-test-doc-001"
    TEST_DOC_B_ID = "retrieval-test-doc-002"

    def setUp(self) -> None:
        self.db = DatabaseManager()
        self.indexer = PgVectorIndexer(self.db)
        self.retriever = PgVectorRetriever(self.db)
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

    def _unit_vector(self, axis: int) -> List[float]:
        vec = [0.0] * 768
        vec[axis] = 1.0
        return vec

    def test_empty_database_returns_empty_list(self) -> None:
        """Verify F: Empty database returns empty list without error."""
        query_vec = self._unit_vector(0)
        results = self.retriever.retrieve(query_vec, top_k=5)
        self.assertEqual(results, [])

    def test_single_and_multiple_result_retrieval(self) -> None:
        """Verify B, C, D, L, M, N: Single and multiple chunk retrieval with metadata preservation."""
        # Chunk 0 aligns with axis 0 (similarity 1.0 to axis 0)
        # Chunk 1 aligns with axis 1 (orthogonal, similarity 0.0 to axis 0)
        # Chunk 2 aligns with axis 2 (orthogonal, similarity 0.0 to axis 0)
        chunks = [
            Chunk(
                id=f"{self.TEST_DOC_ID}-c0",
                document_id=self.TEST_DOC_ID,
                content="Chunk 0: Vector retrieval architecture.",
                metadata={"heading_path": "Architecture > Retrieval", "page_numbers": [1], "category": "core"},
            ),
            Chunk(
                id=f"{self.TEST_DOC_ID}-c1",
                document_id=self.TEST_DOC_ID,
                content="Chunk 1: Hardware peripherals and power supply.",
                metadata={"heading_path": "Hardware > Power", "page_numbers": [2], "category": "hardware"},
            ),
            Chunk(
                id=f"{self.TEST_DOC_ID}-c2",
                document_id=self.TEST_DOC_ID,
                content="Chunk 2: Cafeteria menu and opening hours.",
                metadata={"heading_path": "Facilities > Dining", "page_numbers": [3], "category": "facilities"},
            ),
        ]
        embeddings = [
            EmbeddingResult(chunk_id=chunks[0].id, vector=self._unit_vector(0), dimension=768, model_name="test"),
            EmbeddingResult(chunk_id=chunks[1].id, vector=self._unit_vector(1), dimension=768, model_name="test"),
            EmbeddingResult(chunk_id=chunks[2].id, vector=self._unit_vector(2), dimension=768, model_name="test"),
        ]

        self.indexer.index_chunks(chunks, embeddings)

        # 1. Single top_k=1 retrieval aligning with chunk 0
        query_vec = self._unit_vector(0)
        results_1 = self.retriever.retrieve(query_vec, top_k=1)
        self.assertEqual(len(results_1), 1)
        top = results_1[0]
        self.assertEqual(top.chunk_id, chunks[0].id)
        self.assertEqual(top.document_id, self.TEST_DOC_ID)
        self.assertEqual(top.content, "Chunk 0: Vector retrieval architecture.")
        self.assertEqual(top.rank, 1)
        self.assertAlmostEqual(top.similarity_score, 1.0, places=4)
        # Verify metadata & provenance intact
        self.assertEqual(top.metadata["heading_path"], "Architecture > Retrieval")
        self.assertEqual(top.metadata["page_numbers"], [1])

        # 2. Multiple retrieval top_k=3
        results_3 = self.retriever.retrieve(query_vec, top_k=3)
        self.assertEqual(len(results_3), 3)
        self.assertEqual(results_3[0].chunk_id, chunks[0].id)
        self.assertAlmostEqual(results_3[0].similarity_score, 1.0, places=4)
        self.assertEqual(results_3[0].rank, 1)

        # Other 2 chunks are orthogonal (similarity 0.0)
        self.assertAlmostEqual(results_3[1].similarity_score, 0.0, places=4)
        self.assertEqual(results_3[1].rank, 2)
        self.assertAlmostEqual(results_3[2].similarity_score, 0.0, places=4)
        self.assertEqual(results_3[2].rank, 3)

    def test_top_k_truncation(self) -> None:
        """Verify E: Database-level top_k limits returned items."""
        chunks = [
            Chunk(id=f"{self.TEST_DOC_ID}-tk-{i}", document_id=self.TEST_DOC_ID, content=f"Chunk {i}")
            for i in range(10)
        ]
        embeddings = [
            EmbeddingResult(chunk_id=c.id, vector=self._unit_vector(i), dimension=768, model_name="test")
            for i, c in enumerate(chunks)
        ]
        self.indexer.index_chunks(chunks, embeddings)

        query_vec = self._unit_vector(0)
        results = self.retriever.retrieve(query_vec, top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.rank for r in results], [1, 2, 3])

    def test_similarity_threshold_filtering(self) -> None:
        """Verify G, H: Similarity threshold filtering in database."""
        # Chunk 0 has similarity 1.0
        # Chunk 1 has similarity 0.5 (45 degree angle between axis 0 and axis 1)
        # Chunk 2 has similarity 0.0
        vec_0 = self._unit_vector(0)
        vec_1 = l2_normalize([1.0, 1.0] + [0.0] * 766)  # cos ~ 0.7071
        vec_2 = self._unit_vector(2)  # cos = 0.0

        chunks = [
            Chunk(id=f"{self.TEST_DOC_ID}-th-0", document_id=self.TEST_DOC_ID, content="Match 1.0"),
            Chunk(id=f"{self.TEST_DOC_ID}-th-1", document_id=self.TEST_DOC_ID, content="Match ~0.7"),
            Chunk(id=f"{self.TEST_DOC_ID}-th-2", document_id=self.TEST_DOC_ID, content="Match 0.0"),
        ]
        embeddings = [
            EmbeddingResult(chunk_id=chunks[0].id, vector=vec_0, dimension=768, model_name="test"),
            EmbeddingResult(chunk_id=chunks[1].id, vector=vec_1, dimension=768, model_name="test"),
            EmbeddingResult(chunk_id=chunks[2].id, vector=vec_2, dimension=768, model_name="test"),
        ]
        self.indexer.index_chunks(chunks, embeddings)

        # Threshold 0.8: Only Chunk 0 meets >= 0.8
        res_high = self.retriever.retrieve(vec_0, top_k=5, similarity_threshold=0.8)
        self.assertEqual(len(res_high), 1)
        self.assertEqual(res_high[0].chunk_id, chunks[0].id)

        # Threshold 0.6: Chunk 0 and Chunk 1 meet >= 0.6
        res_mid = self.retriever.retrieve(vec_0, top_k=5, similarity_threshold=0.6)
        self.assertEqual(len(res_mid), 2)
        self.assertEqual(res_mid[0].chunk_id, chunks[0].id)
        self.assertEqual(res_mid[1].chunk_id, chunks[1].id)

        # Threshold 0.5 against orthogonal vector vec_3 (all similarities are 0.0) -> empty list
        vec_3 = self._unit_vector(3)
        res_none = self.retriever.retrieve(vec_3, top_k=5, similarity_threshold=0.5)
        self.assertEqual(res_none, [])

    def test_document_scoping_and_metadata_filtering(self) -> None:
        """Verify O: Document scoping and JSONB metadata filtering."""
        c1 = Chunk(
            id=f"{self.TEST_DOC_ID}-doc1",
            document_id=self.TEST_DOC_ID,
            content="Doc 1 chunk",
            metadata={"tier": "production", "dept": "engineering"},
        )
        c2 = Chunk(
            id=f"{self.TEST_DOC_B_ID}-doc2",
            document_id=self.TEST_DOC_B_ID,
            content="Doc 2 chunk",
            metadata={"tier": "staging", "dept": "engineering"},
        )
        e1 = EmbeddingResult(chunk_id=c1.id, vector=self._unit_vector(0), dimension=768, model_name="test")
        e2 = EmbeddingResult(chunk_id=c2.id, vector=self._unit_vector(0), dimension=768, model_name="test")

        self.indexer.index_chunks([c1, c2], [e1, e2])

        # Retrieve scoped to TEST_DOC_ID
        res_doc1 = self.retriever.retrieve(self._unit_vector(0), top_k=5, document_id=self.TEST_DOC_ID)
        self.assertEqual(len(res_doc1), 1)
        self.assertEqual(res_doc1[0].chunk_id, c1.id)

        # Retrieve with metadata filter tier="staging"
        res_tier = self.retriever.retrieve(
            self._unit_vector(0),
            top_k=5,
            filters={"tier": "staging"},
        )
        self.assertEqual(len(res_tier), 1)
        self.assertEqual(res_tier[0].chunk_id, c2.id)

    def test_deterministic_ordering_for_equal_scores(self) -> None:
        """Verify P: Equal scores break ties deterministically on chunk ID."""
        c_b = Chunk(id=f"{self.TEST_DOC_ID}-tie-B", document_id=self.TEST_DOC_ID, content="Tie B")
        c_a = Chunk(id=f"{self.TEST_DOC_ID}-tie-A", document_id=self.TEST_DOC_ID, content="Tie A")

        # Identical vectors -> identical similarity score
        e_b = EmbeddingResult(chunk_id=c_b.id, vector=self._unit_vector(0), dimension=768, model_name="test")
        e_a = EmbeddingResult(chunk_id=c_a.id, vector=self._unit_vector(0), dimension=768, model_name="test")

        self.indexer.index_chunks([c_b, c_a], [e_b, e_a])

        results = self.retriever.retrieve(self._unit_vector(0), top_k=5)
        self.assertEqual(len(results), 2)
        # tie-A must precede tie-B lexicographically
        self.assertEqual(results[0].chunk_id, c_a.id)
        self.assertEqual(results[1].chunk_id, c_b.id)

    def test_read_only_retrieval_does_not_mutate(self) -> None:
        """Verify Q, R: Retrieval is read-only and does not mutate stored rows."""
        chunk = Chunk(
            id=f"{self.TEST_DOC_ID}-ro-0",
            document_id=self.TEST_DOC_ID,
            content="Original content",
            metadata={"count": 10},
        )
        emb = EmbeddingResult(chunk_id=chunk.id, vector=self._unit_vector(0), dimension=768, model_name="test")
        self.indexer.index_chunk(chunk, emb)

        # Query multiple times
        self.retriever.retrieve(self._unit_vector(0), top_k=5)
        self.retriever.retrieve(self._unit_vector(0), top_k=5)

        with self.db.session() as session:
            stored = session.scalar(select(ChunkModel).where(ChunkModel.id == chunk.id))
            assert stored is not None
            self.assertEqual(stored.content, "Original content")
            self.assertEqual(stored.metadata_["count"], 10)


class TestRealNomicSemanticRetrieval(unittest.TestCase):
    """End-to-end integration test: Real Nomic embedding model + PgVectorRetriever."""

    TEST_DOC_ID = "retrieval-nomic-semantic-doc"

    def setUp(self) -> None:
        self.db = DatabaseManager()
        self.indexer = PgVectorIndexer(self.db)
        self.retriever = PgVectorRetriever(self.db)
        self.model = NomicEmbeddingModel(normalize=True)
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

    def test_semantic_retrieval_ranking(self) -> None:
        """Verify item 20: Real Nomic semantic retrieval ranks the leave policy chunk top."""
        # 1. Three distinct chunks:
        # A: Annual leave policy
        # B: Company laptop return policy
        # C: Cafeteria hours
        chunk_leave = Chunk(
            id=f"{self.TEST_DOC_ID}-leave",
            document_id=self.TEST_DOC_ID,
            content="Employees receive 20 days of annual leave each calendar year.",
            metadata={"topic": "hr_policy"},
        )
        chunk_laptop = Chunk(
            id=f"{self.TEST_DOC_ID}-laptop",
            document_id=self.TEST_DOC_ID,
            content="Company laptops must be returned to IT upon conclusion of employment.",
            metadata={"topic": "it_asset"},
        )
        chunk_cafeteria = Chunk(
            id=f"{self.TEST_DOC_ID}-cafeteria",
            document_id=self.TEST_DOC_ID,
            content="The main cafeteria is open daily from 8 AM to 6 PM for breakfast and lunch.",
            metadata={"topic": "facilities"},
        )

        chunks = [chunk_leave, chunk_laptop, chunk_cafeteria]

        # 2. Embed chunks with document prefix
        doc_vectors = self.model.embed_documents([c.content for c in chunks])
        embeddings = [
            EmbeddingResult(
                chunk_id=c.id,
                vector=vec,
                dimension=EMBEDDING_DIMENSION,
                model_name=self.model.model_name,
                is_normalized=self.model.is_normalized,
            )
            for c, vec in zip(chunks, doc_vectors)
        ]

        # 3. Persist via indexer
        self.indexer.index_chunks(chunks, embeddings)

        # 4. Embed query with query prefix
        query_text = "How much annual vacation do employees receive?"
        query_vector = self.model.embed_query(query_text)

        # 5. Execute retrieval
        results = self.retriever.retrieve(query_vector=query_vector, top_k=3)
        self.assertEqual(len(results), 3)

        # 6. Verify ranking:
        # The leave policy chunk must be ranked #1
        top_result = results[0]
        self.assertEqual(top_result.chunk_id, chunk_leave.id)
        self.assertEqual(top_result.rank, 1)
        self.assertIn("20 days of annual leave", top_result.content)
        self.assertEqual(top_result.metadata["topic"], "hr_policy")

        # Check that top score is strictly higher than laptop and cafeteria
        self.assertGreater(top_result.similarity_score, results[1].similarity_score)
        self.assertGreater(top_result.similarity_score, results[2].similarity_score)

        print("\n[Real Nomic Semantic Retrieval Results]")
        print(f"  Query: '{query_text}'")
        for res in results:
            print(f"  Rank {res.rank}: (Score: {res.similarity_score:.4f}) '{res.content}'")


if __name__ == "__main__":
    unittest.main()
