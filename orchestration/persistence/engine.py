"""Database Engine and Session Factory for Orchestration Persistence.

Configures synchronous SQLAlchemy 2.0 Engine and Sessionmaker with Psycopg 3
support and SQLite in-memory compatibility for tests.
"""

from typing import Any, Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def create_db_engine(
    url: str,
    pool_size: Optional[int] = None,
    max_overflow: Optional[int] = None,
    echo: bool = False,
    **kwargs: Any,
) -> Engine:
    """Create a synchronous SQLAlchemy 2.0 Engine.

    Args:
        url: Database URL (e.g. postgresql+psycopg://user:pass@host:port/db
            or sqlite:///:memory:).
        pool_size: Connection pool size (ignored for SQLite memory).
        max_overflow: Max overflow connections (ignored for SQLite memory).
        echo: Whether to log generated SQL statements.
        **kwargs: Additional engine kwargs.

    Returns:
        Configured SQLAlchemy Engine.
    """
    engine_kwargs: dict[str, Any] = {"echo": echo}
    engine_kwargs.update(kwargs)

    if url.startswith("sqlite"):
        # SQLite memory or file-based engine
        if ":memory:" in url or not url.replace("sqlite:///", ""):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            engine_kwargs["poolclass"] = StaticPool
        else:
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        engine = create_engine(url, **engine_kwargs)

        # Enable foreign key constraint enforcement in SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    # PostgreSQL / Psycopg 3 engine
    if pool_size is not None:
        engine_kwargs["pool_size"] = pool_size
    if max_overflow is not None:
        engine_kwargs["max_overflow"] = max_overflow

    return create_engine(url, **engine_kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a thread-safe sessionmaker bound to the given engine.

    Args:
        engine: The SQLAlchemy Engine to bind sessions to.

    Returns:
        Configured sessionmaker.
    """
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
