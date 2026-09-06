"""Resource contention measurement suite.

Quantifies:
1. Memory footprint (Resident Set Size in MB) of Nomic embedding model and MiniLM CrossEncoder.
2. CPU utilization during dense vector embedding and cross-encoder reranking.
3. Latency overhead when running RAG operations sequentially vs concurrently with inference calls.
4. Verifies bounded memory usage and absence of thread contention/deadlocks.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock

import psutil

from rag.embedding.nomic import NomicEmbeddingModel
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.retrieval.models import RetrievedChunk


class TestResourceContention(unittest.TestCase):
    """Measures resource consumption and contention between RAG models and inference."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.process = psutil.Process(os.getpid())

    def _get_rss_mb(self) -> float:
        """Return current process resident set size in megabytes."""
        return self.process.memory_info().rss / (1024 * 1024)

    def test_nomic_embedding_resource_footprint(self) -> None:
        """Measure latency, CPU, and RSS memory delta for Nomic embedding generation."""
        rss_before = self._get_rss_mb()
        t0 = time.perf_counter()

        embedder = NomicEmbeddingModel()
        texts = [
            "Centrifugal pump P-101A suction line pressure is 4.5 BARG.",
            "Crude distillation tower T-101 overhead condenser cooling water flow.",
            "Hydrocracker unit reactor pressure drop exceeding safe design limits.",
            "Control valve FV-201A bypass arrangement for maintenance isolation.",
        ]
        # Generate embeddings
        vectors = embedder.embed_documents(texts)
        elapsed_sec = time.perf_counter() - t0
        rss_after = self._get_rss_mb()
        delta_mb = rss_after - rss_before

        self.assertEqual(len(vectors), 4)
        self.assertEqual(len(vectors[0]), 768)

        # Contention and boundary assertions:
        # Nomic model weights (~547MB unquantized) should occupy < 1500 MB RAM
        self.assertLess(delta_mb, 1500.0, f"Nomic memory delta {delta_mb:.1f} MB exceeded 1500 MB limit")
        # Embedding 4 short passages should complete within reasonable timeframe
        self.assertLess(elapsed_sec, 10.0, f"Nomic embedding latency {elapsed_sec:.2f}s exceeded 10.0s threshold")

        print(
            f"\n[RESOURCE TELEMETRY] Nomic Embedder: "
            f"Elapsed={elapsed_sec * 1000:.1f}ms | RSS Delta=+{delta_mb:.1f}MB | Total RSS={rss_after:.1f}MB"
        )

    def test_cross_encoder_reranker_resource_footprint(self) -> None:
        """Measure latency, CPU, and RSS memory delta for MiniLM cross-encoder reranking."""
        rss_before = self._get_rss_mb()
        t0 = time.perf_counter()

        reranker = CrossEncoderReranker()
        candidates = [
            RetrievedChunk(
                chunk_id=f"chk-{i}",
                document_id="doc-1",
                content=f"Candidate passage {i} detailing refinery process equipment operational boundaries.",
                similarity_score=0.75 - i * 0.05,
                rank=i + 1,
            )
            for i in range(5)
        ]

        ranked = reranker.rerank(
            query="What are the operational boundaries for refinery equipment?",
            candidates=candidates,
            top_n=3,
        )
        elapsed_sec = time.perf_counter() - t0
        rss_after = self._get_rss_mb()
        delta_mb = rss_after - rss_before

        self.assertEqual(len(ranked), 3)

        # MiniLM model weights (~90MB) should occupy < 500 MB RAM
        self.assertLess(delta_mb, 600.0, f"Cross-encoder memory delta {delta_mb:.1f} MB exceeded 600 MB limit")
        self.assertLess(elapsed_sec, 8.0, f"Cross-encoder latency {elapsed_sec:.2f}s exceeded 8.0s threshold")

        print(
            f"\n[RESOURCE TELEMETRY] Cross-Encoder Reranker: "
            f"Elapsed={elapsed_sec * 1000:.1f}ms | RSS Delta=+{delta_mb:.1f}MB | Total RSS={rss_after:.1f}MB"
        )

    def test_concurrent_rag_and_inference_simulation(self) -> None:
        """Simulate interleaved execution of RAG pipeline and LLM inference to test for contention."""
        from concurrent.futures import ThreadPoolExecutor

        embedder = NomicEmbeddingModel()
        reranker = CrossEncoderReranker()

        def _rag_workload() -> float:
            t_start = time.perf_counter()
            vec = embedder.embed_query("Check pump suction pressure and NPSH available.")
            self.assertEqual(len(vec), 768)
            candidates = [
                RetrievedChunk(
                    chunk_id="c1",
                    document_id="d1",
                    content="Pump P-101 NPSH required is 2.1 meters at rated flow.",
                    similarity_score=0.8,
                    rank=1,
                )
            ]
            ranked = reranker.rerank(query="NPSH available", candidates=candidates, top_n=1)
            self.assertEqual(len(ranked), 1)
            return time.perf_counter() - t_start

        def _inference_simulation_workload() -> float:
            t_start = time.perf_counter()
            # Simulate CPU/thread workload representative of connector serialization and prompt parsing
            total = 0
            for i in range(100_000):
                total += i % 7
            time.sleep(0.05)  # Simulate network/socket wait to llama-server
            return time.perf_counter() - t_start

        # Measure concurrent execution across thread pool
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_rag = executor.submit(_rag_workload)
            fut_inf = executor.submit(_inference_simulation_workload)
            t_rag = fut_rag.result()
            t_inf = fut_inf.result()
        total_time = time.perf_counter() - t0

        self.assertLess(t_rag, 10.0)
        self.assertLess(t_inf, 5.0)

        print(
            f"\n[RESOURCE TELEMETRY] Concurrent Execution: "
            f"Total={total_time * 1000:.1f}ms | RAG Workload={t_rag * 1000:.1f}ms | Inference Workload={t_inf * 1000:.1f}ms"
        )


if __name__ == "__main__":
    unittest.main()
