"""Router for system health, provider readiness, and model telemetry."""

from typing import Any, Dict, List
import urllib.request
import json

from fastapi import APIRouter, Depends

from apps.api.dependencies import get_app_context
from apps.context import AppContext
from core.common.types import RuntimeState

router = APIRouter(tags=["Telemetry & Health"])


@router.get("/health")
async def get_health(context: AppContext = Depends(get_app_context)) -> Dict[str, Any]:
    """Check system health, llama-server connectivity, and database availability."""
    # 1. Provider health
    provider_state = RuntimeState.UNKNOWN
    try:
        provider = context.core.provider_manager.get_provider("llama_cpp")
        provider_state = provider.check_health()
    except Exception:
        provider_state = RuntimeState.ERROR

    # 2. Database health
    db_status = "unconfigured"
    try:
        repo = context.create_orchestration_repository()
        if repo is not None:
            db_status = "connected"
    except Exception:
        db_status = "unavailable"

    is_healthy = provider_state == RuntimeState.READY
    return {
        "status": "healthy" if is_healthy else "degraded",
        "runtime": provider_state.value,
        "database": db_status,
        "environment": getattr(getattr(context.core, "settings", None), "foundation", None).environment if hasattr(context.core, "settings") else "development",
    }


@router.get("/telemetry/models")
async def get_model_telemetry(context: AppContext = Depends(get_app_context)) -> Dict[str, Any]:
    """Retrieve catalog of configured models, aliases, and active llama-server runtime data."""
    registry = context.core.registry
    known_models: List[Dict[str, Any]] = []

    for m_def in registry.list_models():
        known_models.append({
            "id": m_def.id,
            "display_name": m_def.display_name,
            "format": m_def.format.value,
            "aliases": m_def.aliases,
            "capabilities": {
                "chat": m_def.capabilities.chat,
                "code": m_def.capabilities.code,
                "reasoning": m_def.capabilities.reasoning,
                "structured_output": m_def.capabilities.structured_output,
            },
        })

    # Query live llama-server endpoint
    server_models: List[Dict[str, Any]] = []
    try:
        provider = context.core.provider_manager.get_provider("llama_cpp")
        url = f"{provider.base_url}/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "LocalAIFoundation/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                server_models = payload.get("data", [])
    except Exception:
        server_models = []

    return {
        "configured_models": known_models,
        "runtime_models": server_models,
    }
