"""Dependency injection providers for FastAPI routers."""

from pathlib import Path
from typing import Optional

from apps.context import AppContext
from apps.api.events import OrchestrationEventBus, get_event_bus

_APP_CONTEXT: Optional[AppContext] = None


def set_app_context(context: AppContext) -> None:
    """Set the process-level AppContext instance."""
    global _APP_CONTEXT
    _APP_CONTEXT = context


def get_app_context() -> AppContext:
    """FastAPI dependency resolving the current AppContext composition root."""
    global _APP_CONTEXT
    if _APP_CONTEXT is None:
        _APP_CONTEXT = AppContext.create()
    return _APP_CONTEXT


def get_staging_dir() -> Path:
    """FastAPI dependency resolving the staging upload directory."""
    ctx = get_app_context()
    repo_root = getattr(ctx.core, "repo_root", None)
    if not isinstance(repo_root, Path):
        repo_root = Path.cwd()
    staging_dir = (repo_root / ".staging" / "uploads").resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    return staging_dir


def get_artifacts_dir() -> Path:
    """FastAPI dependency resolving the generated artifacts directory."""
    ctx = get_app_context()
    art_settings = getattr(getattr(ctx.core, "settings", None), "artifact", None)
    output_dir_str = getattr(art_settings, "output_dir", None) if isinstance(getattr(art_settings, "output_dir", None), str) else "artifacts"
    repo_root = getattr(ctx.core, "repo_root", None)
    if not isinstance(repo_root, Path):
        repo_root = Path.cwd()
    artifacts_path = Path(output_dir_str)
    if not artifacts_path.is_absolute():
        artifacts_path = (repo_root / artifacts_path).resolve()
    artifacts_path.mkdir(parents=True, exist_ok=True)
    return artifacts_path
