"""RAG Retrieval and Grounded Synthesis capability for the MRPL Foundation Workbench.

Provides a unified capability ('retrieval.rag') for:
1. 'search': Dense vector retrieval from pgvector + cross-encoder reranking without LLM calls.
2. 'qa': End-to-end grounded industrial question answering using reranked passages and local LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from connectors import InferenceConnector
from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskResult

logger = logging.getLogger(__name__)


class RagRetrievalCapability:
    """Capability implementing dense vector retrieval, cross-encoder reranking, and grounded QA.

    Conforms to AbstractCapability protocol with canonical ID 'retrieval.rag'.
    """

    def __init__(
        self,
        harness: Optional[Any] = None,
        connector: Optional[InferenceConnector] = None,
        db_manager: Optional[Any] = None,
    ) -> None:
        """Initialize RagRetrievalCapability.

        Args:
            harness: Optional pre-configured RAGTestHarness instance.
            connector: Optional InferenceConnector for grounded LLM synthesis.
            db_manager: Optional DatabaseManager instance targeting local_ai_rag.
        """
        self._harness = harness
        self._connector = connector
        self._db_manager = db_manager

    @property
    def capability_id(self) -> str:
        """Canonical identifier for the capability."""
        return "retrieval.rag"

    def _get_harness(self) -> Any:
        """Lazily initialize the RAG test harness if not provided."""
        if self._harness is None:
            from rag.cli.harness import RAGTestHarness
            from rag.storage.database import DatabaseConfig, DatabaseManager

            db = self._db_manager or DatabaseManager(DatabaseConfig(database="local_ai_rag"))
            self._harness = RAGTestHarness(db_manager=db)
        return self._harness

    def describe(self) -> Dict[str, Any]:
        """Return declarative schema for tool registration and agent routing."""
        return {
            "name": self.capability_id,
            "description": (
                "Search and query the local MRPL sovereign document knowledge base using dense vector "
                "embeddings and cross-encoder reranking. Supports 'search' (retrieval + rerank only) "
                "and 'qa' (grounded engineering answer synthesis with page citations)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["search", "qa"],
                        "description": "Operation mode: 'search' returns ranked chunks; 'qa' synthesizes a grounded answer.",
                        "default": "qa",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of initial vector candidates retrieved from pgvector.",
                        "default": 10,
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of final candidates returned after cross-encoder reranking.",
                        "default": 3,
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Optional specific document ID to scope retrieval.",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Optional minimum reranker score threshold. If omitted, reranker scores are used strictly as ranking signals.",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature for LLM answer synthesis (QA mode).",
                        "default": 0.1,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens generated in synthesized answer.",
                        "default": 512,
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional custom system instructions for grounding.",
                    },
                },
            },
            "inputs": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or technical question.",
                    },
                },
                "required": ["query"],
            },
        }

    def get_descriptor(self) -> CapabilityDescriptor:
        """Declarative catalog descriptor for this capability."""
        return CapabilityDescriptor(
            capability_id=self.capability_id,
            description=(
                "Search and query the local MRPL sovereign document knowledge base using dense vector "
                "embeddings and cross-encoder reranking. Supports 'search' (retrieval + rerank only) "
                "and 'qa' (grounded engineering answer synthesis with page citations)."
            ),
            parameter_schema={
                "operation": {"type": "string", "enum": ["search", "qa"], "default": "qa"},
                "top_k": {"type": "integer", "default": 10},
                "top_n": {"type": "integer", "default": 3},
                "document_id": {"type": "string"},
                "min_score": {"type": "number"},
                "temperature": {"type": "number", "default": 0.1},
                "max_tokens": {"type": "integer", "default": 512},
                "system_prompt": {"type": "string"},
            },
            input_schema={
                "query": {"type": "string", "required": True},
            },
            output_schema={
                "candidates": {"type": "array"},
                "answer": {"type": "string"},
                "count": {"type": "integer"},
                "provenance": {"type": "array"},
            },
            is_available=True,
        )

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        """Execute search or grounded QA over the RAG store.

        Args:
            parameters: Execution parameters (operation, top_k, top_n, document_id, min_score, etc.).
            inputs: Execution inputs (query).
            context: Invocation context containing execution_id and trace metadata.

        Returns:
            TaskResult containing structured candidates, optional answer, and provenance references.
        """
        raw_query = (
            inputs.get("query")
            or inputs.get("prompt")
            or inputs.get("text")
            or inputs.get("description")
            or parameters.get("query")
            or parameters.get("prompt")
            or parameters.get("system_prompt")
            or parameters.get("text")
            or parameters.get("description")
            or (next((v for v in inputs.values() if isinstance(v, str) and v.strip()), None) if inputs else None)
            or (context.metadata.get("goal_description") if context and context.metadata else None)
            or (context.metadata.get("task_description") if context and context.metadata else None)
            or (context.metadata.get("task_title") if context and context.metadata else None)
        )
        if not raw_query or not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError("Parameter 'inputs.query' must be a non-empty string.")
        query = raw_query.strip()

        operation = str(parameters.get("operation", "qa")).strip().lower()
        if operation not in ("search", "qa"):
            raise ValueError(f"Invalid operation '{operation}'. Supported operations: 'search', 'qa'.")

        top_k = int(parameters.get("top_k", 10))
        top_n = int(parameters.get("top_n", 3))
        document_id = parameters.get("document_id")
        min_score = parameters.get("min_score")
        if min_score is not None:
            min_score = float(min_score)

        harness = self._get_harness()
        query_result = harness.query(
            question=query,
            top_k=top_k,
            top_n=top_n,
            document_id=document_id,
        )

        # Apply reranking score threshold ONLY if caller explicitly requested min_score;
        # otherwise treat reranker scores purely as ranking signals.
        candidates = query_result.ranked_candidates
        if min_score is not None:
            candidates = [c for c in candidates if c.reranking_score >= min_score]

        candidate_dicts: List[Dict[str, Any]] = []
        for c in candidates:
            meta = dict(c.metadata or {})
            candidate_dicts.append({
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "content": c.content,
                "similarity_score": c.original_similarity_score,
                "vector_rank": c.original_retrieval_rank,
                "rerank_score": c.reranking_score,
                "rerank_rank": c.rerank_rank,
                "chunk_index": c.chunk_index,
                "heading_path": meta.get("heading_path", []),
                "page_numbers": meta.get("page_numbers", []),
                "file_name": meta.get("file_name", ""),
                "metadata": meta,
            })

        timings_dict = {
            "query_embedding_sec": query_result.timings.query_embedding_sec,
            "retrieval_sec": query_result.timings.retrieval_sec,
            "reranking_sec": query_result.timings.reranking_sec,
            "total_sec": query_result.timings.total_sec,
        }

        # ---------------------------------------------------------
        # Operation: 'search' (Retrieve & Rerank Only, No LLM Call)
        # ---------------------------------------------------------
        if operation == "search":
            data_ref = DataReference(
                key=f"rag_search_{context.execution_id[:8]}",
                uri=f"rag://query/{query[:32]}",
                mime_type="application/json",
                metadata={
                    "query": query,
                    "candidate_count": len(candidate_dicts),
                    "execution_id": context.execution_id,
                },
            )
            return TaskResult(
                output={
                    "operation": "search",
                    "query": query,
                    "count": len(candidate_dicts),
                    "candidates": candidate_dicts,
                    "timings": timings_dict,
                },
                references=[data_ref],
            )

        # ---------------------------------------------------------
        # Operation: 'qa' (Grounded Answer Synthesis with LLM)
        # ---------------------------------------------------------
        if not candidate_dicts:
            answer = "The provided context does not contain sufficient information to answer this question."
        else:
            context_blocks = []
            for i, c in enumerate(candidate_dicts, 1):
                heading = c.get("heading_path")
                heading_str = " > ".join(heading) if isinstance(heading, list) else str(heading or "General")
                pages = c.get("page_numbers", [])
                page_str = f"Page(s) {', '.join(map(str, pages))}" if pages else "Page: unknown"
                doc_name = c.get("file_name") or c.get("document_id", "doc")
                citation_header = f"[Source {i}: {doc_name} | {page_str} | Section: {heading_str}]"
                context_blocks.append(f"{citation_header}\n{c['content']}")

            formatted_context = "\n\n---\n\n".join(context_blocks)
            default_system = (
                "You are an industrial refinery engineering assistant for Mangalore Refinery and Petrochemicals Limited (MRPL). "
                "Answer the user's question strictly grounded in the provided document context below.\n"
                "Rules:\n"
                "1. Only state facts, numbers, equipment tags, limits, and procedures explicitly present in the context.\n"
                "2. If the answer cannot be determined from the provided context, state clearly: "
                "'The provided context does not contain sufficient information to answer this question.' Do not guess or hallucinate.\n"
                "3. When asserting facts, cite the source section or document."
            )
            system_prompt = parameters.get("system_prompt") or default_system
            user_prompt = f"Document Context:\n{formatted_context}\n\nQuestion: {query}"

            temperature = float(parameters.get("temperature", 0.1))
            max_tokens = int(parameters.get("max_tokens", 512))

            if self._connector is None:
                raise RuntimeError(
                    "RagRetrievalCapability requires an InferenceConnector for 'qa' operation. "
                    "Ensure AppContext provides a wired InferenceConnector."
                )

            llm_response = self._connector.infer_prompt(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = (
                llm_response.message.content
                if hasattr(llm_response, "message") and hasattr(llm_response.message, "content")
                else str(llm_response)
            )

        data_ref = DataReference(
            key=f"rag_qa_{context.execution_id[:8]}",
            uri=f"rag://qa/{query[:32]}",
            mime_type="application/json",
            metadata={
                "query": query,
                "candidate_count": len(candidate_dicts),
                "execution_id": context.execution_id,
            },
        )

        return TaskResult(
            output={
                "operation": "qa",
                "query": query,
                "answer": answer,
                "count": len(candidate_dicts),
                "candidates": candidate_dicts,
                "timings": timings_dict,
            },
            references=[data_ref],
        )
