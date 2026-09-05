"""Real model verification test for nomic-ai/nomic-embed-text-v1.5.

This test is strictly isolated from the unit test suite. It executes live model
inference using the weights downloaded to the local HuggingFace cache.
"""

import math
import unittest

from rag.domain.interfaces import Embedder
from rag.embeddings.nomic import EMBEDDING_DIMENSION, NomicEmbedder


class TestNomicRealModelVerification(unittest.TestCase):
    """Live model verification suite using the real nomic-ai/nomic-embed-text-v1.5 model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the real model once for the test class."""
        print("\n--- Initializing Real NomicEmbedder (nomic-ai/nomic-embed-text-v1.5) ---")
        cls.embedder = NomicEmbedder(normalize=True)

    def test_embedder_protocol_compliance(self) -> None:
        """Verify the real model instance complies with the domain Embedder protocol."""
        self.assertIsInstance(self.embedder, Embedder)

    def test_single_document_and_query_embeddings(self) -> None:
        """Verify single document and query embeddings, dimension, and L2 normalization."""
        doc_text = "The Local AI Foundation provides modular, self-hosted AI infrastructure."
        query_text = "What is the Local AI Foundation?"

        # 1. Document embedding
        doc_vector = self.embedder.embed_document(doc_text)
        self.assertEqual(
            len(doc_vector),
            EMBEDDING_DIMENSION,
            f"Expected {EMBEDDING_DIMENSION} dimensions, got {len(doc_vector)}",
        )
        doc_norm = math.sqrt(sum(x * x for x in doc_vector))
        self.assertAlmostEqual(doc_norm, 1.0, places=4)

        # 2. Query embedding
        query_vector = self.embedder.embed_query(query_text)
        self.assertEqual(
            len(query_vector),
            EMBEDDING_DIMENSION,
            f"Expected {EMBEDDING_DIMENSION} dimensions, got {len(query_vector)}",
        )
        query_norm = math.sqrt(sum(x * x for x in query_vector))
        self.assertAlmostEqual(query_norm, 1.0, places=4)

        # 3. Compute cosine similarity (dot product of L2 normalized vectors)
        similarity = sum(d * q for d, q in zip(doc_vector, query_vector))

        print("\n[Real Model Single Inferences]")
        print(f"  Document Text: '{doc_text}'")
        print(f"  Document Vector: dimension={len(doc_vector)}, L2 norm={doc_norm:.6f}, sample={doc_vector[:3]}")
        print(f"  Query Text:    '{query_text}'")
        print(f"  Query Vector:    dimension={len(query_vector)}, L2 norm={query_norm:.6f}, sample={query_vector[:3]}")
        print(f"  Cosine Similarity: {similarity:.4f}")

    def test_batch_document_embeddings(self) -> None:
        """Verify embed_texts with 3 real inputs."""
        texts = [
            "PostgreSQL with pgvector is used for efficient vector similarity search.",
            "Nomic Embed Text v1.5 maps text to 768-dimensional dense vectors.",
            "Fresh apples and oranges were picked from the organic orchard.",
        ]

        batch_vectors = self.embedder.embed_texts(texts)
        self.assertEqual(len(batch_vectors), 3)

        print("\n[Real Model Batch Inferences]")
        for i, (txt, vec) in enumerate(zip(texts, batch_vectors)):
            norm = math.sqrt(sum(x * x for x in vec))
            self.assertEqual(len(vec), EMBEDDING_DIMENSION)
            self.assertAlmostEqual(norm, 1.0, places=4)
            print(f"  Item {i+1}: '{txt[:45]}...' -> dim={len(vec)}, norm={norm:.6f}")

        # Semantic check: Item 1 and Item 2 (both about AI/DB tech) should be more similar
        # to each other than Item 1 and Item 3 (fruit orchard)
        sim_tech = sum(a * b for a, b in zip(batch_vectors[0], batch_vectors[1]))
        sim_orchard = sum(a * b for a, b in zip(batch_vectors[0], batch_vectors[2]))

        print(f"  Similarity (PostgreSQL vs Nomic tech): {sim_tech:.4f}")
        print(f"  Similarity (PostgreSQL vs Orchard):    {sim_orchard:.4f}")
        self.assertGreater(sim_tech, sim_orchard)


if __name__ == "__main__":
    unittest.main()
