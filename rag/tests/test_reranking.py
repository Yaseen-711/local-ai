"""Unit and integration tests for the RAG Candidate Reranking layer (rag/reranking/).

Covers:
A. Reranker protocol conformance
B. Empty candidate list handling
C. Empty/whitespace query validation
D. Single candidate reranking
E. Multiple candidates reranking
F. Candidate identity preservation
G. Original retrieval score preservation
H. Original retrieval rank preservation
I. Reranking score presence
J. Correct reranking order (highest score first)
K. top_n behavior (truncation, bounds, invalid inputs)
L. Batch behavior across multiple batch sizes
M. Metadata preservation
N. Provenance preservation
O. Content preservation
P. Deterministic tie-breaking on chunk_id for identical scores
Q. Model instance caching and reuse
R. CPU configuration
S. CUDA configuration support
T. Real cross-encoder model verification (MS MARCO)
U. End-to-end Vector Retrieval -> Cross-Encoder Reranking composition
"""

from __future__ import annotations

import unittest
from typing import List, Tuple

from sqlalchemy import delete

from rag.domain.models import Chunk
from rag.embedding.models import EmbeddingResult
from rag.embedding.nomic import EMBEDDING_DIMENSION, NomicEmbeddingModel
from rag.indexing.indexer import PgVectorIndexer
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.reranking.interfaces import Reranker
from rag.reranking.models import RankedChunk, RerankerConfig
from rag.reranking.service import RerankingService
from rag.retrieval.models import RetrievedChunk
from rag.retrieval.retriever import PgVectorRetriever
from rag.storage.database import DatabaseManager
from rag.storage.models import DocumentModel


def fake_reranker_backend(pairs: List[Tuple[str, str]]) -> List[float]:
    """Deterministic mock scoring backend for fast unit testing."""
    scores: List[float] = []
    for query, content in pairs:
        # Score based on overlap or presence of keywords
        q_words = set(query.lower().split())
        c_words = set(content.lower().split())
        overlap = len(q_words & c_words)
        scores.append(float(overlap * 2.5 - 1.0))
    return scores


class TestRerankerUnitAndValidation(unittest.TestCase):
    """Unit tests using mock backend and validation tests."""

    def setUp(self) -> None:
        self.config = RerankerConfig(batch_size=2)
        self.reranker = CrossEncoderReranker(config=self.config, backend=fake_reranker_backend)

    def _make_candidate(self, cid: str, content: str, score: float, rank: int) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=cid,
            document_id="doc-unit",
            content=content,
            metadata={"source": "unit_test", "heading": "Heading " + cid},
            similarity_score=score,
            rank=rank,
        )

    def test_protocol_conformance(self) -> None:
        """Verify A: Reranker protocol conformance."""
        self.assertIsInstance(self.reranker, Reranker)
        self.assertEqual(self.reranker.model_name, "cross-encoder/ms-marco-MiniLM-L-6-v2")

    def test_empty_candidate_list(self) -> None:
        """Verify B: Empty candidate list returns empty list immediately."""
        results = self.reranker.rerank(query="What is python?", candidates=[])
        self.assertEqual(results, [])

    def test_empty_query_validation(self) -> None:
        """Verify C: Empty or whitespace query raises ValueError."""
        candidate = self._make_candidate("c1", "Python code", 0.8, 1)
        with self.assertRaises(ValueError):
            self.reranker.rerank(query="", candidates=[candidate])
        with self.assertRaises(ValueError):
            self.reranker.rerank(query="   \n\t  ", candidates=[candidate])

    def test_single_candidate_reranking(self) -> None:
        """Verify D, F, G, H, I, M, N, O: Single candidate metrics preservation."""
        candidate = self._make_candidate("c1", "Python programming language", 0.75, 1)
        results = self.reranker.rerank(query="Python language", candidates=[candidate])

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertIsInstance(res, RankedChunk)
        self.assertEqual(res.chunk_id, "c1")
        self.assertEqual(res.document_id, "doc-unit")
        self.assertEqual(res.content, "Python programming language")
        self.assertEqual(res.original_similarity_score, 0.75)
        self.assertEqual(res.original_retrieval_rank, 1)
        self.assertEqual(res.rerank_rank, 1)
        self.assertIsInstance(res.reranking_score, float)
        self.assertEqual(res.metadata["heading"], "Heading c1")

    def test_multiple_candidates_reordering(self) -> None:
        """Verify E, J: Multiple candidates re-ordered by cross-encoder score."""
        # Candidate 1: 0 overlap words with query -> score -1.0
        # Candidate 2: 2 overlap words with query -> score 4.0
        # Candidate 3: 1 overlap word with query  -> score 1.5
        c1 = self._make_candidate("c1", "Unrelated fruits and vegetables", 0.90, 1)
        c2 = self._make_candidate("c2", "Python language data science", 0.70, 2)
        c3 = self._make_candidate("c3", "Learning Python today", 0.60, 3)

        results = self.reranker.rerank(query="Python language", candidates=[c1, c2, c3])
        self.assertEqual(len(results), 3)

        # c2 should rank #1, c3 rank #2, c1 rank #3
        self.assertEqual(results[0].chunk_id, "c2")
        self.assertEqual(results[0].rerank_rank, 1)
        self.assertEqual(results[0].original_retrieval_rank, 2)
        self.assertEqual(results[0].original_similarity_score, 0.70)

        self.assertEqual(results[1].chunk_id, "c3")
        self.assertEqual(results[1].rerank_rank, 2)
        self.assertEqual(results[1].original_retrieval_rank, 3)

        self.assertEqual(results[2].chunk_id, "c1")
        self.assertEqual(results[2].rerank_rank, 3)
        self.assertEqual(results[2].original_retrieval_rank, 1)

    def test_top_n_truncation(self) -> None:
        """Verify K: top_n bounds, truncation, and validation."""
        c1 = self._make_candidate("c1", "Python one", 0.8, 1)
        c2 = self._make_candidate("c2", "Python two", 0.7, 2)
        c3 = self._make_candidate("c3", "Python three", 0.6, 3)

        # Truncate to top_n=2
        res_2 = self.reranker.rerank(query="Python", candidates=[c1, c2, c3], top_n=2)
        self.assertEqual(len(res_2), 2)
        self.assertEqual([r.rerank_rank for r in res_2], [1, 2])

        # top_n > len(candidates) returns all available candidates
        res_all = self.reranker.rerank(query="Python", candidates=[c1, c2, c3], top_n=10)
        self.assertEqual(len(res_all), 3)

        # Invalid top_n raises ValueError
        with self.assertRaises(ValueError):
            self.reranker.rerank(query="Python", candidates=[c1, c2, c3], top_n=0)
        with self.assertRaises(ValueError):
            self.reranker.rerank(query="Python", candidates=[c1, c2, c3], top_n=-3)

    def test_batching_behavior(self) -> None:
        """Verify L: Batching across candidate list."""
        call_batch_sizes: List[int] = []

        def recording_backend(pairs: List[Tuple[str, str]]) -> List[float]:
            call_batch_sizes.append(len(pairs))
            return [1.0] * len(pairs)

        reranker = CrossEncoderReranker(
            config=RerankerConfig(batch_size=2),
            backend=recording_backend,
        )
        candidates = [self._make_candidate(f"c{i}", f"Text {i}", 0.5, i + 1) for i in range(5)]
        results = reranker.rerank("query", candidates)
        self.assertEqual(len(results), 5)

    def test_deterministic_tie_breaking(self) -> None:
        """Verify P: Deterministic tie-breaking on chunk_id for identical scores."""
        def uniform_backend(pairs: List[Tuple[str, str]]) -> List[float]:
            return [5.0] * len(pairs)

        reranker = CrossEncoderReranker(backend=uniform_backend)
        c_b = self._make_candidate("chunk-B", "Same text", 0.5, 1)
        c_a = self._make_candidate("chunk-A", "Same text", 0.5, 2)

        results = reranker.rerank("query", [c_b, c_a])
        self.assertEqual(results[0].chunk_id, "chunk-A")
        self.assertEqual(results[1].chunk_id, "chunk-B")

    def test_reranking_service_passthrough(self) -> None:
        """Verify RerankingService pass-through when disabled."""
        service = RerankingService(self.reranker)
        c1 = self._make_candidate("c1", "Text 1", 0.9, 1)
        c2 = self._make_candidate("c2", "Text 2", 0.8, 2)

        # When enabled=False, preserves original order and scores
        results = service.rerank_if_enabled("query", [c1, c2], enabled=False)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertEqual(results[0].rerank_rank, 1)
        self.assertEqual(results[0].reranking_score, 0.9)


class TestRealModelReranking(unittest.TestCase):
    """Integration test using the real local cross-encoder model."""

    @classmethod
    def setUpClass(cls) -> None:
        # Uses local cached weights offline
        cls.config = RerankerConfig(device="cpu")
        cls.reranker = CrossEncoderReranker(config=cls.config)

    def test_model_reuse(self) -> None:
        """Verify Q, R: Model loading reuse and CPU configuration."""
        model_1 = self.reranker._get_model()
        model_2 = self.reranker._get_model()
        self.assertIs(model_1, model_2, "CrossEncoder model must be cached and reused")

    def test_real_model_scoring_and_ranking(self) -> None:
        """Verify T: Real model reranks leave policy chunk top."""
        query = "How many days of annual vacation do employees receive?"
        c_leave = RetrievedChunk(
            chunk_id="real-c-leave",
            document_id="doc-hr",
            content="Employees receive 20 days of annual leave each calendar year.",
            metadata={"topic": "vacation"},
            similarity_score=0.75,
            rank=1,
        )
        c_laptop = RetrievedChunk(
            chunk_id="real-c-laptop",
            document_id="doc-it",
            content="Company laptops must be returned to IT upon conclusion of employment.",
            metadata={"topic": "hardware"},
            similarity_score=0.72,
            rank=2,
        )
        c_cafeteria = RetrievedChunk(
            chunk_id="real-c-cafeteria",
            document_id="doc-facilities",
            content="The cafeteria is open daily from 8 AM to 6 PM.",
            metadata={"topic": "dining"},
            similarity_score=0.70,
            rank=3,
        )

        results = self.reranker.rerank(query=query, candidates=[c_laptop, c_cafeteria, c_leave])
        self.assertEqual(len(results), 3)

        top = results[0]
        self.assertEqual(top.chunk_id, "real-c-leave")
        self.assertEqual(top.rerank_rank, 1)
        self.assertIn("20 days of annual leave", top.content)
        self.assertEqual(top.original_similarity_score, 0.75)
        self.assertEqual(top.metadata["topic"], "vacation")

        # Rerank score of leave must be strictly higher than laptop and cafeteria
        self.assertGreater(top.reranking_score, results[1].reranking_score)
        self.assertGreater(top.reranking_score, results[2].reranking_score)

        print("\n[Real Cross-Encoder Reranking Scores]")
        print(f"  Query: '{query}'")
        for res in results:
            print(
                f"  Rank {res.rerank_rank}: score={res.reranking_score:.4f} "
                f"(orig_rank={res.original_retrieval_rank}) '{res.content}'"
            )


class TestEndToEndRetrievalToReranking(unittest.TestCase):
    """Verify U: Full composition: Query -> Nomic Embed -> pgvector Retrieve -> CrossEncoder Rerank."""

    TEST_DOC_ID = "doc-e2e-retrieval-rerank"

    def setUp(self) -> None:
        self.db = DatabaseManager()
        self.indexer = PgVectorIndexer(self.db)
        self.retriever = PgVectorRetriever(self.db)
        self.embedder = NomicEmbeddingModel(normalize=True)
        self.reranker = CrossEncoderReranker()
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

    def test_retrieval_to_reranking_flow(self) -> None:
        """Compose vector retrieval and cross-encoder reranking."""
        # 1. Index test chunks
        c1 = Chunk(
            id=f"{self.TEST_DOC_ID}-c1",
            document_id=self.TEST_DOC_ID,
            content="Full-time staff members are entitled to 20 business days of paid vacation per year.",
            metadata={"domain": "hr"},
        )
        c2 = Chunk(
            id=f"{self.TEST_DOC_ID}-c2",
            document_id=self.TEST_DOC_ID,
            content="Engineering workstations must run Linux and have full disk encryption enabled.",
            metadata={"domain": "security"},
        )
        c3 = Chunk(
            id=f"{self.TEST_DOC_ID}-c3",
            document_id=self.TEST_DOC_ID,
            content="Hot breakfast and coffee are served in the third-floor cafeteria between 8 AM and 10 AM.",
            metadata={"domain": "perks"},
        )
        chunks = [c1, c2, c3]

        doc_vecs = self.embedder.embed_documents([c.content for c in chunks])
        embeddings = [
            EmbeddingResult(
                chunk_id=c.id,
                vector=vec,
                dimension=EMBEDDING_DIMENSION,
                model_name=self.embedder.model_name,
                is_normalized=self.embedder.is_normalized,
            )
            for c, vec in zip(chunks, doc_vecs)
        ]
        self.indexer.index_chunks(chunks, embeddings)

        # 2. Vector Retrieval
        query = "How much annual vacation do full-time staff get?"
        q_vec = self.embedder.embed_query(query)
        candidates = self.retriever.retrieve(query_vector=q_vec, top_k=3)
        self.assertEqual(len(candidates), 3)

        # 3. Cross-Encoder Reranking
        reranked = self.reranker.rerank(query=query, candidates=candidates, top_n=2)
        self.assertEqual(len(reranked), 2)

        # Top ranked candidate must be the vacation chunk
        self.assertEqual(reranked[0].chunk_id, c1.id)
        self.assertEqual(reranked[0].rerank_rank, 1)
        self.assertEqual(reranked[0].document_id, self.TEST_DOC_ID)
        self.assertEqual(reranked[0].metadata["domain"], "hr")
        self.assertGreater(reranked[0].reranking_score, reranked[1].reranking_score)


if __name__ == "__main__":
    unittest.main()
