"""FastAPI application factory and middleware configuration for MRPL Workbench."""

from contextlib import asynccontextmanager
import logging
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.dependencies import set_app_context
from apps.api.routers import (
    artifacts_router,
    direct_router,
    files_router,
    goals_router,
    telemetry_router,
)
from apps.api.schemas.common import ProblemDetail
from apps.context import AppContext
from core.common.errors import (
    ConfigurationError,
    FoundationError,
    InferenceError,
    ModelNotFoundError,
    ModelUnavailableError,
    ProviderError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from orchestration.errors import (
    CapabilityNotFoundError,
    CapabilityRegistryError,
    CapabilityUnavailableError,
    OrchestrationError,
    PlanValidationError,
    TaskLifecycleError,
)

logger = logging.getLogger(__name__)

# Explicit local-origin allowlist (strictly no allow_origins=["*"])
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]


def _problem_response(
    status_code: int,
    title: str,
    detail: str,
    instance: Optional[str] = None,
    errors: Optional[list] = None,
) -> JSONResponse:
    """Build standardized RFC 7807 problem response."""
    problem = ProblemDetail(
        type="about:blank",
        title=title,
        status=status_code,
        detail=detail,
        instance=instance,
        errors=errors,
    )
    return JSONResponse(
        status_code=status_code,
        content=problem.model_dump(exclude_none=True),
        headers={"Content-Type": "application/problem+json"},
    )


def create_app(app_context: Optional[AppContext] = None) -> FastAPI:
    """Create and configure the MRPL Workbench FastAPI delivery application."""
    if app_context is not None:
        set_app_context(app_context)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = app_context or AppContext.create()
        set_app_context(ctx)
        yield

    app = FastAPI(
        title="MRPL Sovereign AI Workbench API",
        description="Sovereign On-Premise Agentic AI Delivery Layer for Industrial Confidential Work",
        version="0.1.0",
        lifespan=lifespan,
    )

    if app_context is not None:
        from apps.api.dependencies import get_app_context
        app.dependency_overrides[get_app_context] = lambda: app_context

    # 1. CORS Middleware with restricted local origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Register Global RFC 7807 Exception Handlers
    @app.exception_handler(PlanValidationError)
    async def plan_validation_error_handler(request: Request, exc: PlanValidationError):
        return _problem_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Plan Validation Failed",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(CapabilityNotFoundError)
    async def capability_not_found_handler(request: Request, exc: CapabilityNotFoundError):
        return _problem_response(
            status_code=status.HTTP_404_NOT_FOUND,
            title="Capability Not Found",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return _problem_response(
            status_code=status.HTTP_404_NOT_FOUND,
            title="Model Not Found",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
        return _problem_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            title="Model Unavailable",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable_handler(request: Request, exc: ProviderUnavailableError):
        return _problem_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            title="Runtime Provider Offline",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(ProviderResponseError)
    async def provider_response_error_handler(request: Request, exc: ProviderResponseError):
        return _problem_response(
            status_code=status.HTTP_502_BAD_GATEWAY,
            title="Runtime Response Error",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(request: Request, exc: InferenceError):
        return _problem_response(
            status_code=status.HTTP_502_BAD_GATEWAY,
            title="Inference Error",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        return _problem_response(
            status_code=status.HTTP_404_NOT_FOUND,
            title="File Not Found",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return _problem_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            title="Invalid Request Parameter",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(OrchestrationError)
    async def orchestration_error_handler(request: Request, exc: OrchestrationError):
        return _problem_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            title="Orchestration Error",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(FoundationError)
    async def foundation_error_handler(request: Request, exc: FoundationError):
        logger.exception("Internal foundation error: %s", exc)
        return _problem_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Internal Foundation Error",
            detail=str(exc),
            instance=str(request.url),
        )

    # 3. Attach Routers under /api/v1
    api_v1_prefix = "/api/v1"
    app.include_router(files_router, prefix=api_v1_prefix)
    app.include_router(direct_router, prefix=api_v1_prefix)
    app.include_router(goals_router, prefix=api_v1_prefix)
    app.include_router(artifacts_router, prefix=api_v1_prefix)
    app.include_router(telemetry_router, prefix=api_v1_prefix)

    return app


# Default ASGI application instance
app = create_app()

