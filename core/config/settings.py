"""Settings dataclasses for Local AI Foundation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FoundationSettings:
    """Core Foundation environment and path settings."""
    environment: str = "development"
    models_dir: str = "models"
    configs_dir: str = "configs/models"


@dataclass(frozen=True)
class LlamaCppProviderSettings:
    """Settings for the llama.cpp HTTP client provider."""
    base_url: str = "http://127.0.0.1:8080"
    timeout_seconds: float = 60.0
    default_alias: str = "qwen3.5-9b"


@dataclass(frozen=True)
class DatabaseSettings:
    """Settings for relational orchestration persistence."""
    url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/local_ai"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


@dataclass(frozen=True)
class Settings:
    """Global system configuration."""
    foundation: FoundationSettings = field(default_factory=FoundationSettings)
    llama_cpp: LlamaCppProviderSettings = field(default_factory=LlamaCppProviderSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
