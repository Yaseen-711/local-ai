"""Unit tests for RagRetrievalCapability ('retrieval.rag').

Verifies:
1. Protocol conformance and capability_id
2. Operation 'search' (vector retrieval + reranking, zero LLM calls)
3. Operation 'qa' (retrieval + rerank + LLM prompt synthesis)
4. Reranker scores treated as ranking signals (negative logits preserved in order)
5. Optional min_score threshold filtering when explicitly supplied
6. Error handling for invalid operations and empty queries
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import MagicMock

from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.builtin.rag import RagRetrievalCapability
from rag.cli.harness import QueryResult, QueryTimings
from rag.reranking.models import RankedChunk
from rag.retrieval.models import RetrievedChunk


def _make_dummy_query_result(candidates: List[RankedChunk]) -> QueryResult:
    return QueryResult(
        query="test query",
        query_vector_dim=768,
        retrieved_candidates=[
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content,
                similarity_score=c.original_similarity_score,
                rank=c.original_retrieval_rank,
            )
            for c in candidates
        ],
        ranked_candidates=candidates,
        timings=QueryTimings(
            query_embedding_sec=0.01,
            retrieval_sec=0.005,
            reranking_sec=0.015,
            total_sec=0.03,
        ),
        top_k=10,
        top_n=len(candidates),
    )


class TestRagRetrievalCapability(unittest.TestCase):
    """Test suite for RagRetrievalCapability."""

    def setUp(self) -> None:
        self.mock_harness = MagicMock()
        self.mock_connector = MagicMock()
        self.capability = RagRetrievalCapability(
            harness=self.mock_harness,
            connector=self.mock_connector,
        )
        self.context = CapabilityContext(execution_id="exec-test-rag-1234")

    def test_protocol_conformance(self) -> None:
        """Verify capability conforms to Capability protocol and exports 'retrieval.rag'."""
        self.assertIsInstance(self.capability, Capability)
        self.assertEqual(self.capability.capability_id, "retrieval.rag")
        schema = self.capability.describe()
        self.assertEqual(schema["name"], "retrieval.rag")
        self.assertIn("search", schema["parameters"]["properties"]["operation"]["enum"])
        self.assertIn("qa", schema["parameters"]["properties"]["operation"]["enum"])

    def test_search_operation_returns_candidates_without_llm(self) -> None:
        """Verify 'search' operation executes retrieval and reranking with ZERO LLM calls."""
        ranked_chunk = RankedChunk(
            chunk_id="chk-001",
            document_id="doc-refinery-01",
            content="Crude Distillation Unit CDU-1 operates at 360 deg C.",
            original_similarity_score=0.82,
            original_retrieval_rank=1,
            reranking_score=4.15,
            rerank_rank=1,
            metadata={
                "heading_path": ["Refinery Units", "CDU-1"],
                "page_numbers": [12],
                "file_name": "refinery_spec.pdf",
            },
        )
        self.mock_harness.query.return_value = _make_dummy_query_result([ranked_chunk])

        result = self.capability.execute(
            parameters={"operation": "search", "top_k": 5, "top_n": 2},
            inputs={"query": "What is CDU-1 operating temperature?"},
            context=self.context,
        )

        # Assert zero LLM calls
        self.mock_connector.infer_prompt.assert_not_called()

        # Assert output payload
        out = result.output
        self.assertEqual(out["operation"], "search")
        self.assertEqual(out["count"], 1)
        self.assertEqual(len(out["candidates"]), 1)
        cand = out["candidates"][0]
        self.assertEqual(cand["chunk_id"], "chk-001")
        self.assertEqual(cand["rerank_score"], 4.15)
        self.assertEqual(cand["heading_path"], ["Refinery Units", "CDU-1"])
        self.assertEqual(cand["page_numbers"], [12])

        # Assert DataReference was emitted
        self.assertEqual(len(result.references), 1)
        ref = result.references[0]
        self.assertTrue(ref.key.startswith("rag_search_"))
        self.assertEqual(ref.mime_type, "application/json")

    def test_qa_operation_synthesizes_grounded_answer(self) -> None:
        """Verify 'qa' operation builds formatted context and calls LLM connector."""
        ranked_chunk = RankedChunk(
            chunk_id="chk-002",
            document_id="doc-valves",
            content="Tag FV-201A design pressure is 45.0 BARG and type is GLOBE.",
            original_similarity_score=0.79,
            original_retrieval_rank=2,
            reranking_score=3.88,
            rerank_rank=1,
            metadata={
                "heading_path": ["Valve Schedules"],
                "page_numbers": [4],
                "file_name": "valves.pdf",
            },
        )
        self.mock_harness.query.return_value = _make_dummy_query_result([ranked_chunk])

        mock_llm_response = MagicMock()
        mock_llm_response.message.content = "Tag FV-201A has a design pressure of 45.0 BARG (valves.pdf, Page 4)."
        self.mock_connector.infer_prompt.return_value = mock_llm_response

        result = self.capability.execute(
            parameters={"operation": "qa", "top_k": 5, "top_n": 1, "temperature": 0.1, "max_tokens": 256},
            inputs={"query": "What is the design pressure for FV-201A?"},
            context=self.context,
        )

        # Assert LLM was called with grounded context
        self.mock_connector.infer_prompt.assert_called_once()
        call_kwargs = self.mock_connector.infer_prompt.call_args.kwargs
        self.assertIn("FV-201A design pressure is 45.0 BARG", call_kwargs["prompt"])
        self.assertIn("valves.pdf", call_kwargs["prompt"])
        self.assertEqual(call_kwargs["temperature"], 0.1)
        self.assertEqual(call_kwargs["max_tokens"], 256)

        out = result.output
        self.assertEqual(out["operation"], "qa")
        self.assertEqual(out["answer"], "Tag FV-201A has a design pressure of 45.0 BARG (valves.pdf, Page 4).")
        self.assertEqual(out["count"], 1)

    def test_reranker_scores_used_as_ranking_signals_not_thresholds(self) -> None:
        """Verify negative reranking logits are preserved as ranking signals and NOT discarded."""
        chunks = [
            RankedChunk(
                chunk_id="chk-pos",
                document_id="doc-1",
                content="Positive relevance chunk.",
                original_similarity_score=0.70,
                reranking_score=1.2,
                rerank_rank=1,
            ),
            RankedChunk(
                chunk_id="chk-neg",
                document_id="doc-1",
                content="Marginal but top-N chunk with negative logit.",
                original_similarity_score=0.65,
                reranking_score=-0.8,  # Negative logit
                rerank_rank=2,
            ),
        ]
        self.mock_harness.query.return_value = _make_dummy_query_result(chunks)

        # No min_score specified -> both chunks returned
        result = self.capability.execute(
            parameters={"operation": "search"},
            inputs={"query": "sample query"},
            context=self.context,
        )
        out = result.output
        self.assertEqual(len(out["candidates"]), 2)
        self.assertEqual(out["candidates"][0]["chunk_id"], "chk-pos")
        self.assertEqual(out["candidates"][1]["chunk_id"], "chk-neg")

        # Caller explicitly specifies min_score = 0.0 -> negative chunk filtered
        result_filtered = self.capability.execute(
            parameters={"operation": "search", "min_score": 0.0},
            inputs={"query": "sample query"},
            context=self.context,
        )
        self.assertEqual(len(result_filtered.output["candidates"]), 1)
        self.assertEqual(result_filtered.output["candidates"][0]["chunk_id"], "chk-pos")

    def test_invalid_operation_raises_value_error(self) -> None:
        """Verify unsupported operations raise descriptive ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.capability.execute(
                parameters={"operation": "unsupported_mode"},
                inputs={"query": "test"},
                context=self.context,
            )
        self.assertIn("Invalid operation 'unsupported_mode'", str(ctx.exception))

    def test_empty_query_raises_value_error(self) -> None:
        """Verify empty or whitespace-only queries raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.capability.execute(
                parameters={"operation": "search"},
                inputs={"query": "   "},
                context=self.context,
            )
        self.assertIn("must be a non-empty string", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
