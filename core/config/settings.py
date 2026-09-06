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
class DocumentSettings:
    """Settings for document ingestion and understanding."""
    default_parser: str = "docling"
    enable_ocr: bool = True
    ocr_engine: str = "rapidocr"
    enable_tables: bool = True
    enable_figures: bool = True
    enable_formulae: bool = True


@dataclass(frozen=True)
class ArtifactSettings:
    """Settings for deterministic artifact generation."""
    output_dir: str = "artifacts"
    enable_xlsx: bool = True
    enable_docx: bool = True
    enable_pdf: bool = True


@dataclass(frozen=True)
class WorkspaceSettings:
    """Settings for code workspace execution and sandboxing."""
    default_executor: str = "docker"
    docker_image: str = "python:3.12-slim"
    cpu_limit: float = 2.0
    mem_limit: str = "2g"
    network_mode: str = "none"
    default_timeout_seconds: float = 60.0
    base_workspaces_dir: str = ".workspaces"


@dataclass(frozen=True)
class Settings:
    """Global system configuration."""
    foundation: FoundationSettings = field(default_factory=FoundationSettings)
    llama_cpp: LlamaCppProviderSettings = field(default_factory=LlamaCppProviderSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    document: DocumentSettings = field(default_factory=DocumentSettings)
    artifact: ArtifactSettings = field(default_factory=ArtifactSettings)
    workspace: WorkspaceSettings = field(default_factory=WorkspaceSettings)
