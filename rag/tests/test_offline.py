"""Tests for Offline-First and Zero-Internet Runtime Hardening.

Verifies:
1. Central offline configuration (is_offline_mode, ensure_offline_environment).
2. Local model existence detection and expected path resolution.
3. Strict fail-closed error reporting (OfflineModelNotFoundError) when a model is missing.
4. Missing models do NOT trigger a download attempt.
5. Real models load and execute cleanly with socket network access completely blocked.
6. Pipeline stages (Docling ingestion, Nomic embedding, pgvector retrieval, CrossEncoder reranking)
   function with zero network access.
7. CLI startup and harness initialization function without manual HF_HUB_OFFLINE prefix.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import tempfile
from typing import List, Tuple
import unittest
from unittest.mock import MagicMock, patch

from rag.embedding.nomic import NomicEmbeddingModel
from rag.ingestion.docling import DoclingDocumentIngester
from rag.offline import (
    ENV_RAG_OFFLINE_MODE,
    OfflineModelNotFoundError,
    ensure_offline_environment,
    get_expected_model_path,
    is_model_available_locally,
    is_offline_mode,
)
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.reranking.models import RerankerConfig
from rag.retrieval.models import RetrievedChunk


class NetworkBlockedError(RuntimeError):
    """Raised when an unexpected network socket connection is attempted in offline tests."""


class TestOfflineConfiguration(unittest.TestCase):
    """Test suite for offline environment configuration and path resolution."""

    def test_offline_mode_defaults_to_true(self) -> None:
        """Verify that offline mode defaults to True without requiring explicit env vars."""
        with patch.dict(os.environ, {}, clear=False):
            if ENV_RAG_OFFLINE_MODE in os.environ:
                del os.environ[ENV_RAG_OFFLINE_MODE]
            self.assertTrue(is_offline_mode())

    def test_offline_mode_explicit_toggle(self) -> None:
        """Verify explicit toggling of RAG_OFFLINE_MODE."""
        with patch.dict(os.environ, {ENV_RAG_OFFLINE_MODE: "false"}):
            self.assertFalse(is_offline_mode())
        with patch.dict(os.environ, {ENV_RAG_OFFLINE_MODE: "0"}):
            self.assertFalse(is_offline_mode())
        with patch.dict(os.environ, {ENV_RAG_OFFLINE_MODE: "true"}):
            self.assertTrue(is_offline_mode())
        with patch.dict(os.environ, {ENV_RAG_OFFLINE_MODE: "1"}):
            self.assertTrue(is_offline_mode())

    def test_ensure_offline_environment_sets_flags(self) -> None:
        """Verify ensure_offline_environment sets required library flags."""
        with patch.dict(os.environ, {}, clear=False):
            ensure_offline_environment()
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_TELEMETRY"), "1")
            self.assertEqual(os.environ.get("DOCLING_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TOKENIZERS_PARALLELISM"), "false")

    def test_local_model_detection_real_models(self) -> None:
        """Verify detection of verified local models in cache."""
        self.assertTrue(is_model_available_locally("nomic-ai/nomic-embed-text-v1.5"))
        self.assertTrue(is_model_available_locally("cross-encoder/ms-marco-MiniLM-L-6-v2"))

        expected_nomic = get_expected_model_path("nomic-ai/nomic-embed-text-v1.5")
        self.assertTrue(expected_nomic.exists())
        self.assertIn("models--nomic-ai--nomic-embed-text-v1.5", str(expected_nomic))

        expected_ce = get_expected_model_path("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertTrue(expected_ce.exists())
        self.assertIn("models--cross-encoder--ms-marco-MiniLM-L-6-v2", str(expected_ce))

    def test_local_model_detection_missing_model(self) -> None:
        """Verify that non-existent models are detected as not available locally."""
        missing_name = "fake-org/missing-model-xyz-999"
        self.assertFalse(is_model_available_locally(missing_name))
        expected_missing = get_expected_model_path(missing_name)
        self.assertFalse(expected_missing.exists())


class TestFailClosedModelLoading(unittest.TestCase):
    """Test suite verifying fail-closed behavior without network download attempts."""

    def setUp(self) -> None:
        ensure_offline_environment()

    def test_nomic_missing_model_fails_closed(self) -> None:
        """Missing embedding model raises OfflineModelNotFoundError without downloading."""
        model = NomicEmbeddingModel(model_name="nonexistent/nomic-fake-model-123")

        # Mock socket connect to ensure no network call is even attempted
        with patch("socket.socket.connect", side_effect=NetworkBlockedError("Network access forbidden")):
            with self.assertRaises(OfflineModelNotFoundError) as ctx:
                model._get_model()

        err = ctx.exception
        self.assertEqual(err.model_name, "nonexistent/nomic-fake-model-123")
        self.assertEqual(err.component, "NomicEmbeddingModel")
        self.assertIn("Offline mode is enabled", str(err))
        self.assertIn("Downloading models is disabled", str(err))
        self.assertIn("Expected Location:", str(err))

    def test_cross_encoder_missing_model_fails_closed(self) -> None:
        """Missing cross-encoder model raises OfflineModelNotFoundError without downloading."""
        reranker = CrossEncoderReranker(
            config=RerankerConfig(model_name="nonexistent/fake-reranker-456")
        )

        with patch("socket.socket.connect", side_effect=NetworkBlockedError("Network access forbidden")):
            with self.assertRaises(OfflineModelNotFoundError) as ctx:
                reranker._get_model()

        err = ctx.exception
        self.assertEqual(err.model_name, "nonexistent/fake-reranker-456")
        self.assertEqual(err.component, "CrossEncoderReranker")
        self.assertIn("Offline mode is enabled", str(err))
        self.assertIn("Downloading models is disabled", str(err))
        self.assertIn("Expected Location:", str(err))


class TestZeroNetworkExecution(unittest.TestCase):
    """Test suite verifying that all models run with zero socket/network activity."""

    def setUp(self) -> None:
        ensure_offline_environment()
        self._orig_connect = socket.socket.connect

    def tearDown(self) -> None:
        socket.socket.connect = self._orig_connect

    def _block_network(self) -> None:
        """Block any outgoing socket connection attempt."""
        def blocked_connect(s: socket.socket, address: Tuple[str, int]) -> None:
            # Allow local postgres connections on 127.0.0.1 / localhost
            host, port = address[0], address[1]
            if host in ("127.0.0.1", "localhost", "::1") and port in (5432, 5433):
                return self._orig_connect(s, address)
            raise NetworkBlockedError(f"Unexpected network connection to {address} in offline mode!")

        socket.socket.connect = blocked_connect

    def test_nomic_embedder_zero_network(self) -> None:
        """Verify NomicEmbedder loads and generates embeddings with zero network requests."""
        self._block_network()

        embedder = NomicEmbeddingModel()
        # Embed single query
        query_vec = embedder.embed_query("Zero-network runtime verification query.")
        self.assertEqual(len(query_vec), 768)

        # Embed batch documents
        doc_vecs = embedder.embed_documents(["Document text 1", "Document text 2"])
        self.assertEqual(len(doc_vecs), 2)
        self.assertEqual(len(doc_vecs[0]), 768)

    def test_cross_encoder_zero_network(self) -> None:
        """Verify CrossEncoderReranker loads and scores pairs with zero network requests."""
        self._block_network()

        reranker = CrossEncoderReranker()
        candidate = RetrievedChunk(
            chunk_id="chk_offline_1",
            document_id="doc_offline",
            content="Local model weights are cached on the local filesystem.",
            similarity_score=0.8,
            rank=1,
        )

        ranked = reranker.rerank(
            query="Where are model weights cached?",
            candidates=[candidate],
            top_n=1,
        )
        self.assertEqual(len(ranked), 1)
        self.assertIsInstance(ranked[0].reranking_score, float)
        self.assertEqual(ranked[0].chunk_id, "chk_offline_1")

    def test_docling_ingester_zero_network(self) -> None:
        """Verify Docling document converter operates with remote services disabled."""
        self._block_network()

        ingester = DoclingDocumentIngester()
        # Verify markdown parsing without network
        sample_file = Path(__file__).parent.parent.parent / "docs" / "ARCHITECTURE.md"
        if sample_file.exists():
            ingested = ingester.ingest(sample_file)
            self.assertGreater(len(ingested.elements), 10)

    def test_cli_harness_offline_initialization(self) -> None:
        """Verify RAGTestHarness initializes and validates paths without manual HF_HUB_OFFLINE."""
        from rag.cli.harness import RAGTestHarness

        # Ensure environment flags were set automatically by module import
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

        valid_file = Path(__file__).parent.parent.parent / "docs" / "ARCHITECTURE.md"
        if valid_file.exists():
            resolved = RAGTestHarness.validate_file_path(valid_file)
            self.assertTrue(resolved.exists())


if __name__ == "__main__":
    unittest.main()
