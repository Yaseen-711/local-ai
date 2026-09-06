"""Unit and integration tests for the RAG embedding layer (rag/embedding/).

Covers:
- Protocol compliance (EmbeddingModel, Embedder)
- EmbeddingResult model validation and invariants
- NomicEmbeddingModel task prefixing and normalization
- ChunkEmbeddingService batching, error handling, and strict order preservation
- Pipeline integration from NormalizedDocument -> StructuralChunker -> MetadataPipeline -> ChunkEmbeddingService
- Real Nomic model verification using local offline weights
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import List, Sequence

from rag.chunking.options import ChunkingOptions
from rag.chunking.structural import StructuralChunker
from rag.domain.interfaces import Embedder
from rag.domain.models import Chunk
from rag.embedding.interfaces import EmbeddingModel
from rag.embedding.models import EmbeddingResult
from rag.embedding.nomic import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    QUERY_PREFIX,
    NomicEmbedder,
    NomicEmbeddingModel,
    l2_normalize,
)
from rag.embedding.service import ChunkEmbeddingService
from rag.metadata.pipeline import MetadataPipeline
from rag.normalization.models import NormalizedDocument, NormalizedElement, NormalizedElementType


class FakeEmbeddingModel:
    """Deterministic mock embedding model for fast unit testing."""

    def __init__(
        self,
        dimension: int = 768,
        model_name: str = "mock-embedder-v1",
        is_normalized: bool = True,
    ) -> None:
        self._dim = dimension
        self._model_name = model_name
        self._is_norm = is_normalized
        self.call_log: List[List[str]] = []

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_normalized(self) -> bool:
        return self._is_norm

    def _make_vector(self, text: str) -> List[float]:
        # Generate pseudo-deterministic vector based on string hash
        val = float(abs(hash(text)) % 1000 + 1)
        raw = [val + i * 0.1 for i in range(self._dim)]
        if self._is_norm:
            return l2_normalize(raw)
        return raw

    def embed_documents(self, document_texts: Sequence[str]) -> List[List[float]]:
        texts = list(document_texts)
        self.call_log.append(texts)
        return [self._make_vector(t) for t in texts]

    def embed_query(self, query_text: str) -> List[float]:
        self.call_log.append([query_text])
        return self._make_vector(query_text)


class TestEmbeddingModelProtocols(unittest.TestCase):
    """Verify protocol compliance for EmbeddingModel and domain Embedder."""

    def test_protocol_conformance(self) -> None:
        mock = FakeEmbeddingModel()
        self.assertIsInstance(mock, EmbeddingModel)

        nomic = NomicEmbeddingModel()
        self.assertIsInstance(nomic, EmbeddingModel)
        self.assertIsInstance(nomic, Embedder)

        # Alias check
        self.assertIs(NomicEmbedder, NomicEmbeddingModel)


class TestEmbeddingResultModel(unittest.TestCase):
    """Verify EmbeddingResult dataclass invariants and validations."""

    def test_valid_construction(self) -> None:
        vec = [0.1] * 768
        norm_vec = l2_normalize(vec)
        result = EmbeddingResult(
            chunk_id="chunk-001",
            vector=norm_vec,
            dimension=768,
            model_name="nomic-ai/nomic-embed-text-v1.5",
            is_normalized=True,
            token_count=42,
        )
        self.assertEqual(result.chunk_id, "chunk-001")
        self.assertEqual(result.dimension, 768)
        self.assertEqual(len(result.vector), 768)
        self.assertEqual(result.model_name, "nomic-ai/nomic-embed-text-v1.5")
        self.assertTrue(result.is_normalized)
        self.assertEqual(result.token_count, 42)

        data = result.to_dict()
        self.assertEqual(data["chunk_id"], "chunk-001")
        self.assertEqual(data["dimension"], 768)
        self.assertEqual(data["token_count"], 42)

    def test_empty_chunk_id_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            EmbeddingResult(
                chunk_id="",
                vector=[0.1] * 768,
                dimension=768,
                model_name="test-model",
            )
        self.assertIn("chunk_id", str(ctx.exception))

    def test_invalid_vector_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            EmbeddingResult(
                chunk_id="c1",
                vector="invalid-vector",  # type: ignore
                dimension=768,
                model_name="test-model",
            )

    def test_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            EmbeddingResult(
                chunk_id="c1",
                vector=[0.1, 0.2, 0.3],
                dimension=768,
                model_name="test-model",
            )
        self.assertIn("does not match declared dimension", str(ctx.exception))

    def test_non_positive_dimension_raises(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingResult(
                chunk_id="c1",
                vector=[],
                dimension=0,
                model_name="test-model",
            )

    def test_empty_model_name_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            EmbeddingResult(
                chunk_id="c1",
                vector=[0.5],
                dimension=1,
                model_name="  ",
            )
        self.assertIn("model_name", str(ctx.exception))

    def test_negative_token_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingResult(
                chunk_id="c1",
                vector=[0.5],
                dimension=1,
                model_name="test",
                token_count=-5,
            )


class TestNomicEmbeddingModelUnit(unittest.TestCase):
    """Unit tests for NomicEmbeddingModel prefixing, normalization, and backend handling."""

    def test_prefix_formatting(self) -> None:
        model = NomicEmbeddingModel()
        doc_formatted = model.format_text("Sample document text", is_query=False)
        self.assertEqual(doc_formatted, f"{DOCUMENT_PREFIX}Sample document text")

        # Does not duplicate prefix if already present
        self.assertEqual(model.format_text(doc_formatted, is_query=False), doc_formatted)

        query_formatted = model.format_text("What is pgvector?", is_query=True)
        self.assertEqual(query_formatted, f"{QUERY_PREFIX}What is pgvector?")
        self.assertEqual(model.format_text(query_formatted, is_query=True), query_formatted)

    def test_invalid_text_raises(self) -> None:
        model = NomicEmbeddingModel()
        with self.assertRaises(TypeError):
            model.format_text(123)  # type: ignore
        with self.assertRaises(ValueError):
            model.format_text("")
        with self.assertRaises(ValueError):
            model.format_text("   \n\t  ")

    def test_custom_backend_document_and_query_embedding(self) -> None:
        captured_prompts: List[str] = []

        def mock_backend(texts: List[str]) -> List[List[float]]:
            captured_prompts.extend(texts)
            return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

        model = NomicEmbeddingModel(backend=mock_backend, normalize=True)
        self.assertEqual(model.dimension, 768)
        self.assertEqual(model.model_name, "nomic-ai/nomic-embed-text-v1.5")
        self.assertTrue(model.is_normalized)

        # Document embedding
        doc_vecs = model.embed_documents(["Doc 1", "Doc 2"])
        self.assertEqual(len(doc_vecs), 2)
        self.assertEqual(len(doc_vecs[0]), 768)
        self.assertTrue(captured_prompts[0].startswith(DOCUMENT_PREFIX))
        self.assertTrue(captured_prompts[1].startswith(DOCUMENT_PREFIX))

        # Query embedding
        q_vec = model.embed_query("Query 1")
        self.assertEqual(len(q_vec), 768)
        self.assertTrue(captured_prompts[2].startswith(QUERY_PREFIX))

        # Check normalization
        norm = math.sqrt(sum(x * x for x in q_vec))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_backend_dimension_mismatch_raises(self) -> None:
        def bad_backend(texts: List[str]) -> List[List[float]]:
            return [[0.5] * 128 for _ in texts]  # Returns 128 instead of 768

        model = NomicEmbeddingModel(backend=bad_backend)
        with self.assertRaises(ValueError) as ctx:
            model.embed_documents(["Text"])
        self.assertIn("invalid dimension 128", str(ctx.exception))


class TestChunkEmbeddingService(unittest.TestCase):
    """Unit tests for ChunkEmbeddingService batching and invariant enforcement."""

    def setUp(self) -> None:
        self.mock_model = FakeEmbeddingModel(dimension=768, is_normalized=True)
        self.service = ChunkEmbeddingService(model=self.mock_model)

    def test_embed_single_chunk(self) -> None:
        chunk = Chunk(id="chunk-1", document_id="doc-1", content="Hello RAG world")
        res = self.service.embed_chunk(chunk)

        self.assertIsInstance(res, EmbeddingResult)
        self.assertEqual(res.chunk_id, "chunk-1")
        self.assertEqual(res.dimension, 768)
        self.assertEqual(len(res.vector), 768)
        self.assertEqual(res.model_name, "mock-embedder-v1")
        self.assertTrue(res.is_normalized)

    def test_embed_single_chunk_invalid_inputs(self) -> None:
        with self.assertRaises(TypeError):
            self.service.embed_chunk("not-a-chunk")  # type: ignore

        with self.assertRaises(ValueError):
            self.service.embed_chunk(Chunk(id="", document_id="doc-1", content="valid text"))

        with self.assertRaises(ValueError):
            self.service.embed_chunk(Chunk(id="c1", document_id="doc-1", content="   \n  "))

    def test_embed_chunks_empty_sequence(self) -> None:
        results = self.service.embed_chunks([])
        self.assertEqual(results, [])
        self.assertEqual(len(self.mock_model.call_log), 0)

    def test_embed_chunks_batching_and_order_preservation(self) -> None:
        chunks = [
            Chunk(id=f"c-{i:02d}", document_id="doc-A", content=f"Sentence content {i}")
            for i in range(10)
        ]

        # Batch size 3 -> batches of 3, 3, 3, 1 (total 4 model calls)
        results = self.service.embed_chunks(chunks, batch_size=3)

        self.assertEqual(len(results), 10)
        self.assertEqual(len(self.mock_model.call_log), 4)
        self.assertEqual(len(self.mock_model.call_log[0]), 3)
        self.assertEqual(len(self.mock_model.call_log[1]), 3)
        self.assertEqual(len(self.mock_model.call_log[2]), 3)
        self.assertEqual(len(self.mock_model.call_log[3]), 1)

        # Invariant check: Strict 1-to-1 order preservation
        for i, res in enumerate(results):
            self.assertEqual(res.chunk_id, chunks[i].id)
            self.assertEqual(res.dimension, 768)
            norm = math.sqrt(sum(x * x for x in res.vector))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_embed_chunks_invalid_batch_size(self) -> None:
        chunk = Chunk(id="c1", document_id="d1", content="Sample")
        with self.assertRaises(ValueError):
            self.service.embed_chunks([chunk], batch_size=0)
        with self.assertRaises(ValueError):
            self.service.embed_chunks([chunk], batch_size=-5)

    def test_embed_chunks_invalid_sequence_elements(self) -> None:
        chunks_with_bad_item = [
            Chunk(id="c1", document_id="d1", content="Valid"),
            "not a chunk",  # type: ignore
        ]
        with self.assertRaises(TypeError):
            self.service.embed_chunks(chunks_with_bad_item)  # type: ignore

        chunks_with_empty_content = [
            Chunk(id="c1", document_id="d1", content="Valid"),
            Chunk(id="c2", document_id="d1", content="   "),
        ]
        with self.assertRaises(ValueError) as ctx:
            self.service.embed_chunks(chunks_with_empty_content)
        self.assertIn("c2", str(ctx.exception))


class TestPipelineEmbeddingIntegration(unittest.TestCase):
    """Integration test connecting NormalizedDocument -> Chunker -> Metadata -> Embedding."""

    def test_full_pipeline_to_embeddings(self) -> None:
        # 1. Normalized document
        elements = [
            NormalizedElement(
                index=0,
                element_type=NormalizedElementType.TITLE,
                content="Architecture Overview",
                page_number=1,
            ),
            NormalizedElement(
                index=1,
                element_type=NormalizedElementType.SECTION_HEADER,
                content="1. Vector Storage",
                heading_level=1,
                page_number=1,
            ),
            NormalizedElement(
                index=2,
                element_type=NormalizedElementType.PARAGRAPH,
                content="PostgreSQL with pgvector provides performant similarity search over 768-dim embeddings.",
                parent_heading="1. Vector Storage",
                page_number=1,
            ),
            NormalizedElement(
                index=3,
                element_type=NormalizedElementType.PARAGRAPH,
                content="The nomic-embed-text-v1.5 model generates high quality unit-normalized dense vectors.",
                parent_heading="1. Vector Storage",
                page_number=2,
            ),
        ]
        norm_doc = NormalizedDocument(
            document_id="doc_pipe_001",
            elements=elements,
            text="\n".join(e.content for e in elements),
            file_path=Path("/data/docs/architecture.pdf"),
            format="pdf",
            metadata={"page_count": 2},
        )

        # 2. Structural chunking
        chunker = StructuralChunker(ChunkingOptions(max_chunk_size=300, overlap_size=50))
        chunks = chunker.chunk(norm_doc)
        self.assertGreater(len(chunks), 0)

        # 3. Metadata enrichment
        doc_metadata = MetadataPipeline.extract_document_metadata(norm_doc)
        enriched_chunks = [
            MetadataPipeline.enrich_chunk(c, document_metadata=doc_metadata)
            for c in chunks
        ]

        # Verify chunks have provenance metadata intact
        for chunk in enriched_chunks:
            self.assertEqual(chunk.metadata["document_id"], "doc_pipe_001")
            self.assertEqual(chunk.metadata["file_name"], "architecture.pdf")

        # 4. Batch Embedding
        mock_model = FakeEmbeddingModel(dimension=768)
        service = ChunkEmbeddingService(model=mock_model)
        embeddings = service.embed_chunks(enriched_chunks, batch_size=2)

        # Invariant checks:
        self.assertEqual(len(embeddings), len(enriched_chunks))
        for chunk, emb in zip(enriched_chunks, embeddings):
            self.assertEqual(emb.chunk_id, chunk.id)
            self.assertEqual(emb.dimension, 768)
            self.assertEqual(emb.model_name, "mock-embedder-v1")
            self.assertTrue(emb.is_normalized)
            norm = math.sqrt(sum(x * x for x in emb.vector))
            self.assertAlmostEqual(norm, 1.0, places=5)


class TestNomicRealModelEmbedding(unittest.TestCase):
    """Verification using real local nomic weights with ChunkEmbeddingService."""

    @classmethod
    def setUpClass(cls) -> None:
        # Loads cached weights offline
        cls.model = NomicEmbeddingModel(normalize=True)
        cls.service = ChunkEmbeddingService(model=cls.model)

    def test_real_model_chunk_embedding(self) -> None:
        chunks = [
            Chunk(
                id="real-chunk-1",
                document_id="doc-real",
                content="pgvector extends PostgreSQL with native vector storage and indexing capabilities.",
            ),
            Chunk(
                id="real-chunk-2",
                document_id="doc-real",
                content="Nomic embeddings map semantic text into a 768-dimensional latent vector space.",
            ),
        ]

        results = self.service.embed_chunks(chunks, batch_size=2)
        self.assertEqual(len(results), 2)

        for i, res in enumerate(results):
            self.assertEqual(res.chunk_id, chunks[i].id)
            self.assertEqual(res.dimension, 768)
            self.assertEqual(len(res.vector), 768)
            self.assertEqual(res.model_name, "nomic-ai/nomic-embed-text-v1.5")
            self.assertTrue(res.is_normalized)
            norm = math.sqrt(sum(x * x for x in res.vector))
            self.assertAlmostEqual(norm, 1.0, places=4)

        # Semantic cosine similarity check
        similarity = sum(a * b for a, b in zip(results[0].vector, results[1].vector))
        self.assertGreater(similarity, 0.4)
        self.assertLess(similarity, 1.0)


if __name__ == "__main__":
    unittest.main()
