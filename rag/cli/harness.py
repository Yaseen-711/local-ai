"""Core orchestration engine for the RAG Developer Test Harness.

Coordinates the end-to-end RAG pipeline from ingestion to reranking, capturing
granular timings, statistics, and domain models without presentation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from sqlalchemy import func, select

from rag.chunking.options import ChunkingOptions
from rag.chunking.structural import StructuralChunker
from rag.domain.models import Chunk, Document
from rag.embedding.models import EmbeddingResult
from rag.embedding.nomic import NomicEmbeddingModel
from rag.embedding.service import ChunkEmbeddingService
from rag.indexing.indexer import PgVectorIndexer
from rag.ingestion.docling import SUPPORTED_EXTENSIONS, DoclingDocumentIngester
from rag.metadata.pipeline import MetadataPipeline
from rag.normalization.normalizer import StandardDocumentNormalizer
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.reranking.models import RankedChunk, RerankerConfig
from rag.retrieval.models import RetrievedChunk
from rag.retrieval.retriever import PgVectorRetriever
from rag.storage.database import DatabaseManager
from rag.storage.models import ChunkModel, DocumentModel


@dataclass(frozen=True)
class IngestionTimings:
    """Execution timings for each stage of the ingestion pipeline."""

    ingestion_sec: float
    normalization_sec: float
    chunking_sec: float
    metadata_sec: float
    embedding_sec: float
    indexing_sec: float
    total_sec: float


@dataclass(frozen=True)
class IngestionStats:
    """Statistics collected during document ingestion and indexing."""

    document_id: str
    file_name: str
    file_path: str
    format: str
    page_count: int
    element_count: int
    chunk_count: int
    embedding_count: int
    indexed_count: int
    timings: IngestionTimings
    action: str  # "created" or "updated"


@dataclass(frozen=True)
class QueryTimings:
    """Execution timings for query processing and retrieval stages."""

    query_embedding_sec: float
    retrieval_sec: float
    reranking_sec: float
    total_sec: float


@dataclass(frozen=True)
class QueryResult:
    """Results from vector retrieval and optional cross-encoder reranking."""

    query: str
    query_vector_dim: int
    retrieved_candidates: List[RetrievedChunk]
    ranked_candidates: List[RankedChunk]
    timings: QueryTimings
    top_k: int
    top_n: int
    document_id_scope: Optional[str] = None
    similarity_threshold: Optional[float] = None


@dataclass(frozen=True)
class DocumentSummary:
    """Summary of an indexed document in PostgreSQL."""

    document_id: str
    file_name: str
    format: str
    page_count: int
    chunk_count: int
    created_at: str


@dataclass(frozen=True)
class ChunkDetails:
    """Detailed view of an individual stored chunk and its embedding."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    vector_dimension: int
    vector_preview: List[float]
    vector_norm: float
    created_at: str


class RAGTestHarness:
    """Development test harness managing and executing the RAG pipeline.

    Encapsulates ingestion, normalization, chunking, metadata enrichment,
    embedding, pgvector indexing, vector retrieval, and cross-encoder reranking.
    Models and database connections are loaded once and reused across operations.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        ingester: Optional[DoclingDocumentIngester] = None,
        normalizer: Optional[StandardDocumentNormalizer] = None,
        chunker: Optional[StructuralChunker] = None,
        embedding_service: Optional[ChunkEmbeddingService] = None,
        indexer: Optional[PgVectorIndexer] = None,
        retriever: Optional[PgVectorRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.ingester = ingester or DoclingDocumentIngester()
        self.normalizer = normalizer or StandardDocumentNormalizer()
        self.chunker = chunker or StructuralChunker(ChunkingOptions())
        self.embedding_service = embedding_service or ChunkEmbeddingService()
        self.indexer = indexer or PgVectorIndexer(self.db)
        self.retriever = retriever or PgVectorRetriever(self.db)
        self.reranker = reranker or CrossEncoderReranker(RerankerConfig())

        # Ensure database tables exist
        self.db.init_db()

    @staticmethod
    def validate_file_path(path_str: Union[str, Path]) -> Path:
        """Validate that a file exists, is readable, and is a supported format."""
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        return path

    def ingest_document(
        self,
        file_path: Union[str, Path],
        progress_callback: Optional[Any] = None,
    ) -> IngestionStats:
        """Execute the full ingestion pipeline: Ingest -> Normalize -> Chunk -> Enrich -> Embed -> Index."""
        path = self.validate_file_path(file_path)

        # Check if document already exists to determine action (created vs updated)
        with self.db.session() as session:
            existing = session.scalar(
                select(DocumentModel.id).where(DocumentModel.id == str(path.stem))
            )
            action = "updated" if existing else "created"

        def _notify(step_name: str, step_idx: int, total_steps: int = 6) -> None:
            if progress_callback:
                progress_callback(step_name, step_idx, total_steps)

        # [1/6] Ingestion
        _notify("Ingesting via Docling", 1)
        t0 = time.perf_counter()
        ingested_doc = self.ingester.ingest(path)
        t_ingest = time.perf_counter() - t0

        # [2/6] Normalization
        _notify("Normalizing document structure", 2)
        t0 = time.perf_counter()
        norm_doc = self.normalizer.normalize(ingested_doc)
        t_norm = time.perf_counter() - t0

        # [3/6] Chunking
        _notify("Structural chunking", 3)
        t0 = time.perf_counter()
        raw_chunks = self.chunker.chunk(norm_doc)
        t_chunk = time.perf_counter() - t0

        # [4/6] Metadata enrichment
        _notify("Enriching metadata & provenance", 4)
        t0 = time.perf_counter()
        doc_metadata = MetadataPipeline.extract_document_metadata(norm_doc)
        enriched_chunks = [
            MetadataPipeline.enrich_chunk(c, document_metadata=doc_metadata)
            for c in raw_chunks
        ]
        t_meta = time.perf_counter() - t0

        # [5/6] Embedding
        _notify("Generating embeddings via Nomic", 5)
        t0 = time.perf_counter()
        embeddings = self.embedding_service.embed_chunks(enriched_chunks)
        t_embed = time.perf_counter() - t0

        # [6/6] Indexing
        _notify("Persisting to PostgreSQL + pgvector", 6)
        t0 = time.perf_counter()
        domain_doc = Document(
            id=norm_doc.document_id,
            content=norm_doc.text,
            metadata=doc_metadata.to_dict(),
        )
        indexed_count = self.indexer.index_document(
            document=domain_doc,
            chunks=enriched_chunks,
            embeddings=embeddings,
        )
        t_index = time.perf_counter() - t0

        total_time = t_ingest + t_norm + t_chunk + t_meta + t_embed + t_index

        timings = IngestionTimings(
            ingestion_sec=t_ingest,
            normalization_sec=t_norm,
            chunking_sec=t_chunk,
            metadata_sec=t_meta,
            embedding_sec=t_embed,
            indexing_sec=t_index,
            total_sec=total_time,
        )

        return IngestionStats(
            document_id=norm_doc.document_id,
            file_name=path.name,
            file_path=str(path),
            format=norm_doc.format,
            page_count=norm_doc.page_count,
            element_count=len(norm_doc.elements),
            chunk_count=len(enriched_chunks),
            embedding_count=len(embeddings),
            indexed_count=indexed_count,
            timings=timings,
            action=action,
        )

    def query(
        self,
        question: str,
        top_k: int = 10,
        top_n: int = 5,
        document_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> QueryResult:
        """Execute the question query flow: embed_query -> vector retrieve -> cross-encoder rerank."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        clean_q = question.strip()

        # 1. Query Embedding
        t0 = time.perf_counter()
        # NomicEmbeddingModel adheres to EmbeddingModel
        query_vector = self.embedding_service.model.embed_query(clean_q)
        t_embed = time.perf_counter() - t0

        # 2. Vector Retrieval (PgVectorRetriever)
        t0 = time.perf_counter()
        retrieved_candidates = self.retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            similarity_threshold=similarity_threshold,
        )
        t_retrieval = time.perf_counter() - t0

        # 3. Cross-Encoder Reranking
        t0 = time.perf_counter()
        ranked_candidates: List[RankedChunk] = []
        if retrieved_candidates:
            ranked_candidates = self.reranker.rerank(
                query=clean_q,
                candidates=retrieved_candidates,
                top_n=top_n,
            )
        t_rerank = time.perf_counter() - t0

        total_time = t_embed + t_retrieval + t_rerank

        timings = QueryTimings(
            query_embedding_sec=t_embed,
            retrieval_sec=t_retrieval,
            reranking_sec=t_rerank,
            total_sec=total_time,
        )

        return QueryResult(
            query=clean_q,
            query_vector_dim=len(query_vector),
            retrieved_candidates=retrieved_candidates,
            ranked_candidates=ranked_candidates,
            timings=timings,
            top_k=top_k,
            top_n=top_n,
            document_id_scope=document_id,
            similarity_threshold=similarity_threshold,
        )

    def list_documents(self) -> List[DocumentSummary]:
        """Query PostgreSQL for all stored documents and their chunk counts."""
        with self.db.session() as session:
            # Query documents and count related chunks
            stmt = (
                select(
                    DocumentModel.id,
                    DocumentModel.metadata_,
                    DocumentModel.created_at,
                    func.count(ChunkModel.id).label("chunk_count"),
                )
                .outerjoin(ChunkModel, DocumentModel.id == ChunkModel.document_id)
                .group_by(DocumentModel.id, DocumentModel.metadata_, DocumentModel.created_at)
                .order_by(DocumentModel.created_at.desc())
            )
            rows = session.execute(stmt).all()

            summaries: List[DocumentSummary] = []
            for doc_id, meta, created_at, chunk_count in rows:
                meta_dict = dict(meta or {})
                file_name = meta_dict.get("file_name", doc_id)
                fmt = meta_dict.get("format", "")
                page_count = int(meta_dict.get("page_count", 0))

                summaries.append(
                    DocumentSummary(
                        document_id=doc_id,
                        file_name=file_name,
                        format=fmt,
                        page_count=page_count,
                        chunk_count=int(chunk_count),
                        created_at=created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else "N/A",
                    )
                )

            return summaries

    def get_chunk_details(self, chunk_id: str) -> Optional[ChunkDetails]:
        """Fetch a specific chunk from the database with vector preview and metadata."""
        if not chunk_id or not str(chunk_id).strip():
            raise ValueError("chunk_id must be a non-empty string")

        with self.db.session() as session:
            chunk_row = session.scalar(
                select(ChunkModel).where(ChunkModel.id == chunk_id.strip())
            )
            if chunk_row is None:
                return None

            raw_vector = list(chunk_row.embedding)
            preview = [round(float(x), 5) for x in raw_vector[:5]]
            norm = sum(float(x) * float(x) for x in raw_vector) ** 0.5

            return ChunkDetails(
                chunk_id=chunk_row.id,
                document_id=chunk_row.document_id,
                chunk_index=chunk_row.chunk_index,
                content=chunk_row.content,
                metadata=dict(chunk_row.metadata_ or {}),
                vector_dimension=len(raw_vector),
                vector_preview=preview,
                vector_norm=round(norm, 5),
                created_at=chunk_row.created_at.strftime("%Y-%m-%d %H:%M:%S") if chunk_row.created_at else "N/A",
            )

    def list_chunks_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """List basic chunk outlines for a given document."""
        with self.db.session() as session:
            stmt = (
                select(ChunkModel.id, ChunkModel.chunk_index, ChunkModel.metadata_, ChunkModel.content)
                .where(ChunkModel.document_id == document_id)
                .order_by(ChunkModel.chunk_index.asc())
            )
            rows = session.execute(stmt).all()
            chunks_info: List[Dict[str, Any]] = []
            for r in rows:
                meta = dict(r[2] or {})
                heading_val = meta.get("heading") or meta.get("heading_path")
                if isinstance(heading_val, list):
                    heading_str = " > ".join(str(h) for h in heading_val)
                elif heading_val:
                    heading_str = str(heading_val)
                else:
                    heading_str = "N/A"

                page_val = meta.get("primary_page") or meta.get("page_numbers") or "N/A"
                chunks_info.append(
                    {
                        "chunk_id": r[0],
                        "chunk_index": r[1],
                        "heading": heading_str,
                        "page": page_val,
                        "preview": r[3][:60].replace("\n", " ") + ("..." if len(r[3]) > 60 else ""),
                    }
                )
            return chunks_info

    def clear_data(self, document_id: Optional[str] = None) -> int:
        """Clear test documents and chunks using the existing storage deletion APIs."""
        if document_id is not None:
            # Delete single document
            success = self.indexer.delete_document(document_id)
            return 1 if success else 0

        # Clear all documents
        docs = self.list_documents()
        deleted_count = 0
        for doc in docs:
            if self.indexer.delete_document(doc.document_id):
                deleted_count += 1

        return deleted_count
