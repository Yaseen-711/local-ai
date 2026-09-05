"""Unit tests for NomicEmbedder and Embedder protocol compliance.

Verifies:
1. Protocol conformance (isinstance(embedder, Embedder)).
2. Document embedding generation with 768 dimensions and 'search_document: ' prefix.
3. Query embedding generation with 768 dimensions and 'search_query: ' prefix.
4. Batch embedding (embed_texts) returns exactly one embedding per input.
5. Empty and invalid input rejection (ValueError / TypeError).
6. Prefix idempotency (no double-prefixing).
7. Unit normalization (L2 norm = 1.0).
8. Separation of unit tests from live model downloading.
"""

import math
import unittest
from typing import List

from rag.domain.interfaces import Embedder
from rag.embeddings.nomic import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    NomicEmbedder,
    QUERY_PREFIX,
    l2_normalize,
)


class MockNomicBackend:
    """Deterministic mock backend recording formatted texts and emitting synthetic 768-dim vectors."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.recorded_calls: List[List[str]] = []

    def __call__(self, texts: List[str]) -> List[List[float]]:
        self.recorded_calls.append(list(texts))
        outputs: List[List[float]] = []
        for text in texts:
            # Deterministic synthetic vector based on text hash
            base_val = (hash(text) % 1000) / 1000.0 + 0.1
            vec = [base_val + (i * 0.001) for i in range(self.dimension)]
            outputs.append(vec)
        return outputs


class TestNomicEmbedder(unittest.TestCase):
    """Unit tests for NomicEmbedder."""

    def setUp(self) -> None:
        self.mock_backend = MockNomicBackend()
        self.embedder = NomicEmbedder(backend=self.mock_backend, normalize=True)

    def test_implements_embedder_protocol(self) -> None:
        """Verify NomicEmbedder satisfies the domain Embedder protocol."""
        self.assertTrue(isinstance(self.embedder, Embedder))
        self.assertTrue(issubclass(NomicEmbedder, Embedder))

    def test_document_embedding_768_dimensions_and_prefix(self) -> None:
        """Verify document embedding produces 768-dim vector and uses 'search_document: ' prefix."""
        text = "This is a document about RAG architecture."
        emb = self.embedder.embed_document(text)

        # Dimension check
        self.assertEqual(len(emb), 768)

        # Prefix check
        last_batch = self.mock_backend.recorded_calls[-1]
        self.assertEqual(len(last_batch), 1)
        self.assertEqual(last_batch[0], f"{DOCUMENT_PREFIX}{text}")

    def test_query_embedding_768_dimensions_and_prefix(self) -> None:
        """Verify query embedding produces 768-dim vector and uses 'search_query: ' prefix."""
        query = "What is RAG?"
        emb = self.embedder.embed_query(query)

        # Dimension check
        self.assertEqual(len(emb), 768)

        # Prefix check
        last_batch = self.mock_backend.recorded_calls[-1]
        self.assertEqual(len(last_batch), 1)
        self.assertEqual(last_batch[0], f"{QUERY_PREFIX}{query}")

    def test_embed_text_default_is_document(self) -> None:
        """Verify embed_text defaults to document embedding."""
        text = "Sample chunk text"
        emb = self.embedder.embed_text(text)
        self.assertEqual(len(emb), 768)
        self.assertEqual(
            self.mock_backend.recorded_calls[-1][0],
            f"{DOCUMENT_PREFIX}{text}",
        )

    def test_embed_texts_returns_one_per_input(self) -> None:
        """Verify embed_texts returns exactly one embedding per input item."""
        inputs = [
            "Chunk 1 text",
            "Chunk 2 text",
            "Chunk 3 text",
        ]
        embeddings = self.embedder.embed_texts(inputs)

        self.assertEqual(len(embeddings), 3)
        for emb in embeddings:
            self.assertEqual(len(emb), 768)

        recorded = self.mock_backend.recorded_calls[-1]
        self.assertEqual(len(recorded), 3)
        for original, passed in zip(inputs, recorded):
            self.assertEqual(passed, f"{DOCUMENT_PREFIX}{original}")

    def test_empty_input_behavior(self) -> None:
        """Verify empty strings and whitespace-only strings raise ValueError."""
        with self.assertRaises(ValueError):
            self.embedder.embed_text("")

        with self.assertRaises(ValueError):
            self.embedder.embed_text("   \n\t  ")

        with self.assertRaises(ValueError):
            self.embedder.embed_texts(["valid text", ""])

        # Empty sequence returns empty list
        self.assertEqual(self.embedder.embed_texts([]), [])

    def test_invalid_type_input_behavior(self) -> None:
        """Verify non-string inputs raise TypeError."""
        with self.assertRaises(TypeError):
            self.embedder.embed_text(None)  # type: ignore

        with self.assertRaises(TypeError):
            self.embedder.embed_text(123)  # type: ignore

        with self.assertRaises(TypeError):
            self.embedder.embed_texts(123)  # type: ignore

    def test_prefix_handling_no_double_prefix(self) -> None:
        """Verify already prefixed text is not prefixed again."""
        already_doc = "search_document: Already has doc prefix"
        self.embedder.embed_document(already_doc)
        self.assertEqual(self.mock_backend.recorded_calls[-1][0], already_doc)

        already_query = "search_query: Already has query prefix"
        self.embedder.embed_query(already_query)
        self.assertEqual(self.mock_backend.recorded_calls[-1][0], already_query)

    def test_l2_normalization(self) -> None:
        """Verify generated embeddings have unit L2 norm when normalize=True."""
        emb = self.embedder.embed_document("Testing normalization")
        l2_norm = math.sqrt(sum(x * x for x in emb))
        self.assertAlmostEqual(l2_norm, 1.0, places=5)

    def test_invalid_dimension_raises_value_error(self) -> None:
        """Verify backend returning incorrect dimension raises ValueError."""
        bad_backend = MockNomicBackend(dimension=512)
        bad_embedder = NomicEmbedder(backend=bad_backend)
        with self.assertRaises(ValueError) as ctx:
            bad_embedder.embed_text("Some text")
        self.assertIn("Expected 768 dimensions", str(ctx.exception))



if __name__ == "__main__":
    unittest.main()
