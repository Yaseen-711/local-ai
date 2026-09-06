"""Unit tests for FastAPI RAG router endpoints (/api/v1/rag/*)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from apps.api.app import create_app
from apps.api.dependencies import get_app_context, get_staging_dir
from apps.context import AppContext
from orchestration.capabilities.builtin.rag import RagRetrievalCapability
from orchestration.domain.results import TaskResult
from rag.cli.harness import DocumentSummary, IngestionStats, IngestionTimings


class TestRagRouter(unittest.TestCase):
    """Test suite for RAG API endpoints."""

    def setUp(self) -> None:
        self.mock_context = MagicMock(spec=AppContext)
        self.mock_core = MagicMock()
        self.mock_core.repo_root = Path("/tmp/mock_repo")
        self.mock_context.core = self.mock_core
        self.mock_staging_dir = Path("/tmp/staging")

        self.app = create_app(app_context=self.mock_context)
        self.app.dependency_overrides[get_app_context] = lambda: self.mock_context
        self.app.dependency_overrides[get_staging_dir] = lambda: self.mock_staging_dir
        self.client = TestClient(self.app)

    def test_search_endpoint_returns_candidates(self) -> None:
        """Verify POST /api/v1/rag/search dispatches search operation and returns candidates."""
        mock_cap = MagicMock()
        mock_cap.execute.return_value = TaskResult(
            output={
                "operation": "search",
                "query": "pipeline hydraulics",
                "count": 1,
                "candidates": [
                    {
                        "chunk_id": "chk-01",
                        "document_id": "doc-01",
                        "content": "Darcy friction factor calculation for 300mm line.",
                        "similarity_score": 0.85,
                        "vector_rank": 1,
                        "rerank_score": 3.2,
                        "rerank_rank": 1,
                        "chunk_index": 0,
                        "heading_path": ["Calculations", "Hydraulics"],
                        "page_numbers": [5],
                        "file_name": "hydraulics.pdf",
                        "metadata": {},
                    }
                ],
                "timings": {"total_sec": 0.04},
            },
            references=[],
        )
        self.mock_context.create_rag_capability.return_value = mock_cap

        resp = self.client.post(
            "/api/v1/rag/search",
            json={"query": "pipeline hydraulics", "top_k": 5, "top_n": 2},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["query"], "pipeline hydraulics")
        self.assertEqual(data["count"], 1)
        self.assertEqual(len(data["candidates"]), 1)
        self.assertEqual(data["candidates"][0]["chunk_id"], "chk-01")
        self.assertEqual(data["candidates"][0]["rerank_score"], 3.2)

        # Assert operation was 'search'
        call_kwargs = mock_cap.execute.call_args.kwargs
        self.assertEqual(call_kwargs["parameters"]["operation"], "search")

    def test_qa_endpoint_returns_grounded_answer(self) -> None:
        """Verify POST /api/v1/rag/qa dispatches qa operation and returns synthesized answer."""
        mock_cap = MagicMock()
        mock_cap.execute.return_value = TaskResult(
            output={
                "operation": "qa",
                "query": "What is the design margin for line L-104?",
                "answer": "The design margin is 15% exceeding minimum ASME threshold.",
                "count": 1,
                "candidates": [
                    {
                        "chunk_id": "chk-02",
                        "document_id": "doc-02",
                        "content": "Design margin is 15%.",
                        "similarity_score": 0.9,
                        "vector_rank": 1,
                        "rerank_score": 4.5,
                        "rerank_rank": 1,
                        "chunk_index": 1,
                        "heading_path": ["Lines"],
                        "page_numbers": [2],
                        "file_name": "lines.pdf",
                        "metadata": {},
                    }
                ],
                "timings": {"total_sec": 0.15},
            },
            references=[],
        )
        self.mock_context.create_rag_capability.return_value = mock_cap

        resp = self.client.post(
            "/api/v1/rag/qa",
            json={"query": "What is the design margin for line L-104?", "temperature": 0.1},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["answer"], "The design margin is 15% exceeding minimum ASME threshold.")
        self.assertEqual(len(data["candidates"]), 1)

        # Assert operation was 'qa'
        call_kwargs = mock_cap.execute.call_args.kwargs
        self.assertEqual(call_kwargs["parameters"]["operation"], "qa")

    def test_list_documents_endpoint(self) -> None:
        """Verify GET /api/v1/rag/documents lists stored documents."""
        mock_harness = MagicMock()
        mock_harness.list_documents.return_value = [
            DocumentSummary(
                document_id="doc-1",
                file_name="manual.pdf",
                format="pdf",
                page_count=20,
                chunk_count=65,
                created_at="2026-09-06T12:00:00Z",
            )
        ]
        self.mock_context.create_rag_harness.return_value = mock_harness

        resp = self.client.get("/api/v1/rag/documents")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["documents"][0]["document_id"], "doc-1")
        self.assertEqual(data["documents"][0]["chunk_count"], 65)

    def test_delete_document_endpoint(self) -> None:
        """Verify DELETE /api/v1/rag/documents/{id} deletes from indexer."""
        mock_harness = MagicMock()
        mock_harness.indexer.delete_document.return_value = True
        self.mock_context.create_rag_harness.return_value = mock_harness

        resp = self.client.delete("/api/v1/rag/documents/doc-to-delete")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "deleted")
        self.assertEqual(data["document_id"], "doc-to-delete")

        # Not found case
        mock_harness.indexer.delete_document.return_value = False
        resp_404 = self.client.delete("/api/v1/rag/documents/nonexistent")
        self.assertEqual(resp_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
