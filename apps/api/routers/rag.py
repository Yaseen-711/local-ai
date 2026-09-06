"""Router for persistent RAG knowledge base operations (ingest, search, qa, document management)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import get_app_context, get_staging_dir
from apps.api.schemas.rag import (
    RagCandidateSchema,
    RagDocumentListResponse,
    RagDocumentSummarySchema,
    RagIngestRequest,
    RagIngestResponse,
    RagQARequest,
    RagQAResponse,
    RagSearchRequest,
    RagSearchResponse,
)
from apps.context import AppContext
from orchestration.capabilities.base import CapabilityContext
from orchestration.domain.results import TaskResult

router = APIRouter(prefix="/rag", tags=["RAG Knowledge Base"])


def _resolve_file(file_id: Optional[str], file_path: Optional[str], staging_dir: Path, repo_root: Path) -> Path:
    """Safely resolve an uploaded file ID or relative path within authorized roots."""
    if file_id:
        matches = list(staging_dir.glob(f"{file_id}_*"))
        if not matches or not matches[0].is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Uploaded file '{file_id}' not found in staging.",
            )
        return matches[0].resolve()

    if file_path:
        p = Path(file_path).resolve()
        allowed_roots = [staging_dir.resolve(), repo_root.resolve()]
        is_safe = any(p.is_relative_to(root) for root in allowed_roots)
        if not is_safe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Access denied: path '{file_path}' is outside authorized directory roots.",
            )
        if not p.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{file_path}' not found.",
            )
        return p

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Either 'file_id' or 'file_path' must be provided.",
    )


@router.post("/ingest", response_model=RagIngestResponse)
async def ingest_document(
    req: RagIngestRequest,
    context: AppContext = Depends(get_app_context),
    staging_dir: Path = Depends(get_staging_dir),
) -> RagIngestResponse:
    """Ingest, normalize, chunk, embed, and index a document into the persistent RAG store."""
    repo_root = getattr(context.core, "repo_root", Path.cwd())
    resolved_path = _resolve_file(req.file_id, req.file_path, staging_dir, repo_root)

    harness = context.create_rag_harness()

    try:
        stats = await asyncio.to_thread(harness.ingest_document, resolved_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document '{resolved_path.name}': {str(exc)}",
        ) from exc

    return RagIngestResponse(
        document_id=stats.document_id,
        file_name=stats.file_name,
        format=stats.format,
        page_count=stats.page_count,
        chunk_count=stats.chunk_count,
        action=stats.action,
        timings={
            "ingestion_sec": stats.timings.ingestion_sec,
            "normalization_sec": stats.timings.normalization_sec,
            "chunking_sec": stats.timings.chunking_sec,
            "metadata_sec": stats.timings.metadata_sec,
            "embedding_sec": stats.timings.embedding_sec,
            "indexing_sec": stats.timings.indexing_sec,
            "total_sec": stats.timings.total_sec,
        },
    )


@router.post("/search", response_model=RagSearchResponse)
async def search_rag(
    req: RagSearchRequest,
    context: AppContext = Depends(get_app_context),
) -> RagSearchResponse:
    """Explicit search operation: vector retrieval and cross-encoder reranking without LLM calls."""
    cap = context.create_rag_capability()
    cap_ctx = CapabilityContext(execution_id=f"rag-search-{uuid.uuid4().hex[:8]}")

    parameters: Dict[str, Any] = {
        "operation": "search",
        "top_k": req.top_k,
        "top_n": req.top_n,
    }
    if req.document_id:
        parameters["document_id"] = req.document_id
    if req.min_score is not None:
        parameters["min_score"] = req.min_score

    inputs = {"query": req.query}

    result: TaskResult = await asyncio.to_thread(
        cap.execute,
        parameters=parameters,
        inputs=inputs,
        context=cap_ctx,
    )

    out = result.output or {}
    candidates = [
        RagCandidateSchema(**c) for c in out.get("candidates", [])
    ]
    return RagSearchResponse(
        query=req.query,
        count=out.get("count", len(candidates)),
        candidates=candidates,
        timings=out.get("timings", {}),
    )


@router.post("/qa", response_model=RagQAResponse)
async def qa_rag(
    req: RagQARequest,
    context: AppContext = Depends(get_app_context),
) -> RagQAResponse:
    """Explicit grounded QA operation: vector retrieval, reranking, and local LLM answer synthesis."""
    cap = context.create_rag_capability()
    cap_ctx = CapabilityContext(execution_id=f"rag-qa-{uuid.uuid4().hex[:8]}")

    parameters: Dict[str, Any] = {
        "operation": "qa",
        "top_k": req.top_k,
        "top_n": req.top_n,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.document_id:
        parameters["document_id"] = req.document_id
    if req.min_score is not None:
        parameters["min_score"] = req.min_score
    if req.system_prompt:
        parameters["system_prompt"] = req.system_prompt

    inputs = {"query": req.query}

    result: TaskResult = await asyncio.to_thread(
        cap.execute,
        parameters=parameters,
        inputs=inputs,
        context=cap_ctx,
    )

    out = result.output or {}
    candidates = [
        RagCandidateSchema(**c) for c in out.get("candidates", [])
    ]
    return RagQAResponse(
        query=req.query,
        answer=out.get("answer", ""),
        count=out.get("count", len(candidates)),
        candidates=candidates,
        timings=out.get("timings", {}),
    )


@router.get("/documents", response_model=RagDocumentListResponse)
async def list_documents(
    context: AppContext = Depends(get_app_context),
) -> RagDocumentListResponse:
    """List all indexed documents and their chunk counts in the persistent RAG store."""
    harness = context.create_rag_harness()
    summaries = await asyncio.to_thread(harness.list_documents)

    doc_list = [
        RagDocumentSummarySchema(
            document_id=s.document_id,
            file_name=s.file_name,
            format=s.format,
            page_count=s.page_count,
            chunk_count=s.chunk_count,
            created_at=s.created_at,
        )
        for s in summaries
    ]
    return RagDocumentListResponse(count=len(doc_list), documents=doc_list)


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    context: AppContext = Depends(get_app_context),
) -> Dict[str, Any]:
    """Delete a document and all its chunks from the persistent RAG store."""
    harness = context.create_rag_harness()

    def _delete() -> bool:
        return harness.indexer.delete_document(document_id)

    deleted = await asyncio.to_thread(_delete)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in RAG store.",
        )
    return {"status": "deleted", "document_id": document_id}
