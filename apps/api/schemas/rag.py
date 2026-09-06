"""Pydantic schemas for the RAG knowledge base endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RagIngestRequest(BaseModel):
    """Request payload to ingest a document into the persistent RAG store."""

    file_id: Optional[str] = Field(None, description="Staged file ID from /api/v1/files/upload")
    file_path: Optional[str] = Field(None, description="Relative path within allowed workspace or staging roots")


class RagIngestResponse(BaseModel):
    """Response returned upon successful document ingestion and indexing."""

    document_id: str = Field(..., description="Unique document ID in RAG store")
    file_name: str = Field(..., description="Base filename of ingested document")
    format: str = Field(..., description="Document format (pdf, docx, md, txt)")
    page_count: int = Field(..., description="Total pages in document")
    chunk_count: int = Field(..., description="Total chunks extracted and embedded")
    action: str = Field(..., description="'created' or 'updated'")
    timings: Dict[str, float] = Field(default_factory=dict, description="Per-stage latency breakdown in seconds")


class RagCandidateSchema(BaseModel):
    """Schema for an individual retrieved and reranked passage."""

    chunk_id: str
    document_id: str
    content: str
    similarity_score: float = Field(0.0, description="Vector cosine similarity score [-1.0, 1.0]")
    vector_rank: int = Field(1, description="Rank from initial pgvector retrieval")
    rerank_score: float = Field(0.0, description="Cross-encoder logit relevance score")
    rerank_rank: int = Field(1, description="Rank after cross-encoder scoring")
    chunk_index: int = 0
    heading_path: List[str] = Field(default_factory=list, description="Breadcrumb section trail")
    page_numbers: List[int] = Field(default_factory=list, description="1-indexed source pages spanned")
    file_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RagSearchRequest(BaseModel):
    """Request payload for explicit search operation (no LLM call)."""

    query: str = Field(..., min_length=1, description="Technical search query")
    top_k: int = Field(10, ge=1, le=50, description="Vector candidate pool size")
    top_n: int = Field(3, ge=1, le=20, description="Reranked candidate pool size")
    document_id: Optional[str] = Field(None, description="Optional document ID to constrain search scope")
    min_score: Optional[float] = Field(None, description="Optional minimum reranker logit threshold")


class RagSearchResponse(BaseModel):
    """Response schema for explicit search operation."""

    query: str
    count: int
    candidates: List[RagCandidateSchema]
    timings: Dict[str, float] = Field(default_factory=dict)


class RagQARequest(BaseModel):
    """Request payload for grounded QA operation (retrieval + rerank + LLM answer)."""

    query: str = Field(..., min_length=1, description="Technical question to answer")
    top_k: int = Field(10, ge=1, le=50, description="Vector candidate pool size")
    top_n: int = Field(3, ge=1, le=20, description="Reranked candidate pool size")
    document_id: Optional[str] = Field(None, description="Optional document ID to constrain search scope")
    min_score: Optional[float] = Field(None, description="Optional minimum reranker logit threshold")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="LLM sampling temperature")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens generated in answer")
    system_prompt: Optional[str] = Field(None, description="Optional custom system grounding prompt")


class RagQAResponse(BaseModel):
    """Response schema for grounded QA operation."""

    query: str
    answer: str
    count: int
    candidates: List[RagCandidateSchema]
    timings: Dict[str, float] = Field(default_factory=dict)


class RagDocumentSummarySchema(BaseModel):
    """Catalog summary for an indexed document in the RAG store."""

    document_id: str
    file_name: str
    format: str
    page_count: int
    chunk_count: int
    created_at: str


class RagDocumentListResponse(BaseModel):
    """List response of all indexed documents in the RAG store."""

    count: int
    documents: List[RagDocumentSummarySchema]
