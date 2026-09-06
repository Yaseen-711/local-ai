"""MRPL API Routers package."""

from apps.api.routers.artifacts import router as artifacts_router
from apps.api.routers.direct import router as direct_router
from apps.api.routers.files import router as files_router
from apps.api.routers.goals import router as goals_router
from apps.api.routers.telemetry import router as telemetry_router

__all__ = [
    "artifacts_router",
    "direct_router",
    "files_router",
    "goals_router",
    "telemetry_router",
]
