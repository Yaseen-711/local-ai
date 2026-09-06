"""Grounding validation test suite for RAG retrieval and QA synthesis.

Verifies:
1. Positive In-Corpus Grounding: Exact engineering values and tags are passed into context
   and preserved in the synthesized response.
2. Out-of-Corpus Grounding: Queries concerning entities absent from the corpus trigger explicit
   admissions of insufficient information, preventing hallucination.
3. Empty-Context Grounding: Zero retrieved candidates return immediate rejection message without
   unnecessary LLM invocation.
4. Citation and Section Trail Integrity: Page ranges and heading breadcrumbs are properly formatted.
"""

from __future__ import annotations

from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.rag import RagRetrievalCapability
from rag.cli.harness import QueryResult, QueryTimings
from rag.reranking.models import RankedChunk


def _make_result(candidates: List[RankedChunk]) -> QueryResult:
    return QueryResult(
        query="grounding test",
        query_vector_dim=768,
        retrieved_candidates=[],
        ranked_candidates=candidates,
        timings=QueryTimings(0.01, 0.01, 0.01, 0.03),
        top_k=5,
        top_n=len(candidates),
    )


class TestRagGrounding(unittest.TestCase):
    """Grounding test suite for RagRetrievalCapability."""

    def setUp(self) -> None:
        self.mock_harness = MagicMock()
        self.mock_connector = MagicMock()
        self.capability = RagRetrievalCapability(
            harness=self.mock_harness,
            connector=self.mock_connector,
        )
        self.context = CapabilityContext(execution_id="grounding-test-001")

    def test_positive_in_corpus_grounding(self) -> None:
        """Verify in-corpus question preserves exact tags and engineering limits."""
        chunk = RankedChunk(
            chunk_id="chk-refinery-102",
            document_id="doc-heat-exchangers",
            content="Heat Exchanger E-102 shell design pressure is 28.5 BARG; tube metallurgy is 316L Stainless Steel.",
            original_similarity_score=0.88,
            reranking_score=5.2,
            rerank_rank=1,
            metadata={
                "file_name": "E102_datasheet.pdf",
                "page_numbers": [3],
                "heading_path": ["Static Equipment", "Heat Exchangers", "E-102"],
            },
        )
        self.mock_harness.query.return_value = _make_result([chunk])

        expected_answer = (
            "Heat Exchanger E-102 shell design pressure is 28.5 BARG with 316L Stainless Steel tube metallurgy "
            "(Source: E102_datasheet.pdf, Page 3)."
        )
        mock_resp = MagicMock()
        mock_resp.message.content = expected_answer
        self.mock_connector.infer_prompt.return_value = mock_resp

        result = self.capability.execute(
            parameters={"operation": "qa"},
            inputs={"query": "What is the design pressure and tube metallurgy for Exchanger E-102?"},
            context=self.context,
        )

        out = result.output
        self.assertEqual(out["operation"], "qa")
        self.assertEqual(out["answer"], expected_answer)
        self.assertIn("28.5 BARG", out["answer"])
        self.assertIn("316L Stainless Steel", out["answer"])

        # Check prompt formatting contains exact citations
        call_kwargs = self.mock_connector.infer_prompt.call_args.kwargs
        prompt = call_kwargs["prompt"]
        self.assertIn("[Source 1: E102_datasheet.pdf | Page(s) 3 | Section: Static Equipment > Heat Exchangers > E-102]", prompt)
        self.assertIn("Heat Exchanger E-102 shell design pressure is 28.5 BARG", prompt)

    def test_out_of_corpus_grounding_rejects_hallucination(self) -> None:
        """Verify query on absent equipment produces strict refusal, preventing hallucinated answers."""
        # Chunk contains information about boiler B-401 only
        chunk = RankedChunk(
            chunk_id="chk-boiler-401",
            document_id="doc-boilers",
            content="Utility Boiler B-401 steam generation capacity is 120 TPH at 65 BARG.",
            original_similarity_score=0.35,
            reranking_score=-4.5,  # strongly negative logit
            rerank_rank=1,
            metadata={
                "file_name": "boiler_ops.pdf",
                "page_numbers": [1],
                "heading_path": ["Utilities", "Boilers"],
            },
        )
        self.mock_harness.query.return_value = _make_result([chunk])

        # LLM correctly adheres to system prompt constraint:
        refusal_answer = "The provided context does not contain sufficient information to answer this question."
        mock_resp = MagicMock()
        mock_resp.message.content = refusal_answer
        self.mock_connector.infer_prompt.return_value = mock_resp

        result = self.capability.execute(
            parameters={"operation": "qa"},
            inputs={"query": "What is the maintenance interval for Gas Turbine GT-901?"},
            context=self.context,
        )

        out = result.output
        self.assertEqual(out["answer"], refusal_answer)

        # Assert system prompt enforces non-hallucination constraint
        call_kwargs = self.mock_connector.infer_prompt.call_args.kwargs
        system_prompt = call_kwargs["system_prompt"]
        self.assertIn("If the answer cannot be determined from the provided context, state clearly:", system_prompt)
        self.assertIn("Do not guess or hallucinate.", system_prompt)

    def test_empty_candidates_returns_immediate_rejection_without_llm(self) -> None:
        """Verify when zero candidates are returned, immediate rejection is returned with no LLM call."""
        self.mock_harness.query.return_value = _make_result([])

        result = self.capability.execute(
            parameters={"operation": "qa"},
            inputs={"query": "What is the lubrication schedule for pump P-999?"},
            context=self.context,
        )

        # Zero LLM calls made when context is completely empty
        self.mock_connector.infer_prompt.assert_not_called()

        out = result.output
        self.assertEqual(out["count"], 0)
        self.assertIn("does not contain sufficient information", out["answer"])


if __name__ == "__main__":
    unittest.main()
