"""Database connection and session infrastructure for RAG storage."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


@dataclass(frozen=True)
class DatabaseConfig:
    """Configuration for PostgreSQL connection."""

    host: str = "localhost"
    port: int = 5432
    database: str = "local_ai_rag"
    user: str = "postgres"
    password: str = "postgres"
    url: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False

    @property
    def connection_url(self) -> str:
        """Construct or return the SQLAlchemy connection URL with psycopg3 driver.

        Resolution order:
        1. Explicitly configured URL parameter.
        2. Environment variable RAG_DATABASE_URL or POSTGRES_URL.
        3. Composed URL from individual host/port/db/user/password attributes.
        """
        if self.url:
            return self.url
        env_url = os.getenv("RAG_DATABASE_URL") or os.getenv("POSTGRES_URL")
        if env_url:
            return env_url
        return (
            f"postgresql+psycopg://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class DatabaseManager:
    """Coordinates database engine, sessions, and table initialization."""

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker[Session]] = None

    @property
    def engine(self) -> Engine:
        """Lazily initialize and return the SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.config.connection_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
                echo=self.config.echo,
            )
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Return the initialized sessionmaker."""
        if self._session_factory is None:
            _ = self.engine
        assert self._session_factory is not None
        return self._session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional session scope."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def init_db(self) -> None:
        """Initialize pgvector extension and create database tables if they do not exist."""
        from rag.storage.models import Base

        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            Base.metadata.create_all(conn)

    def close(self) -> None:
        """Dispose of the engine connection pool."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
