"""SQLAlchemy ORM models for RAG storage in PostgreSQL with pgvector."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSION = 768


class Base(DeclarativeBase):
    """Base class for RAG storage ORM models."""
    pass


class DocumentModel(Base):
    """SQLAlchemy model representing an ingested source document."""

    __tablename__ = "rag_documents"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    chunks: Mapped[List[ChunkModel]] = relationship(
        "ChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ChunkModel.chunk_index",
    )


class ChunkModel(Base):
    """SQLAlchemy model representing a chunk with a 768-dimensional pgvector embedding."""

    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[List[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    document: Mapped[DocumentModel] = relationship(
        "DocumentModel",
        back_populates="chunks",
    )

    __table_args__ = (
        Index("ix_rag_chunks_document_index", "document_id", "chunk_index"),
    )
