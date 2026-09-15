from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import analyses, health
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.logging import configure_logging, log_event
from app.core.middleware import RequestBoundary
from app.db.analyses import AnalysisRepository
from app.db.cache import LLMCache
from app.db.session import create_engine_and_session_factory
from app.schemas.errors import ErrorDetail, ErrorResponse
from app.services.artifacts import ArtifactValidationError, DataArtifacts, load_data_artifacts
from app.services.jobs import AnalysisJobService
from app.services.llm import CachedLLMClient, OpenRouterLLMClient
from app.services.model_identity import local_model_fingerprint
from app.services.pipeline import AnalysisPipeline, PipelineError
from app.services.retrieval import RetrievalError, RetrievalService, SentenceTransformerEncoder

MAX_REQUEST_BODY_BYTES = 64 * 1024
HTTP_ERROR_CODES = {
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
    status.HTTP_413_CONTENT_TOO_LARGE: "REQUEST_TOO_LARGE",
    status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    # Pass this process-wide semaphore to every OpenRouter client created by a worker.
    app.state.llm_provider_semaphore = asyncio.Semaphore(settings.llm_global_concurrency)
    app.state.llm_provider = None
    if settings.openrouter_api_key and settings.openrouter_model:
        app.state.llm_provider = OpenRouterLLMClient(
            settings, semaphore=app.state.llm_provider_semaphore
        )
    engine, session_factory = create_engine_and_session_factory(
        settings.database_url, sqlite_busy_timeout_ms=settings.sqlite_busy_timeout_ms
    )
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory
    app.state.llm_cache = (
        LLMCache(
            session_factory,
            retention_days=settings.llm_cache_retention_days,
            max_rows=settings.llm_cache_max_rows,
        )
        if settings.llm_cache_enabled
        else None
    )
    repository = AnalysisRepository(session_factory)
    app.state.database_ready = False
    app.state.analysis_jobs = AnalysisJobService(
        repository,
        _pipeline_factory(app),
        pipeline_version=settings.analysis_pipeline_version,
        max_concurrent=settings.max_concurrent_analyses,
        max_queued=settings.max_queued_analyses,
        timeout_seconds=settings.analysis_timeout_seconds,
    )
    try:
        with engine.connect() as connection:
            if connection.scalar(text("SELECT version_num FROM alembic_version")) != "20260911_03":
                raise RuntimeError("Database migrations are not current")
        app.state.analysis_jobs.reconcile_interrupted()
        app.state.database_ready = True
    except Exception:  # Database migrations may not have been applied yet.
        log_event("startup_failed", error_code="DATABASE_NOT_READY")
        app.state.database_ready = False
    app.state.data_artifacts = None
    app.state.retrieval_service = None
    try:
        artifacts, retrieval_service = initialize_retrieval(settings)
        app.state.data_artifacts = artifacts
        app.state.retrieval_service = retrieval_service
        app.state.readiness_checks = {
            "database": "ready" if app.state.database_ready else "not_ready",
            "artifacts": "ready",
            "embedding_model": "ready",
        }
    except ArtifactValidationError:
        log_event("startup_failed", error_code="ARTIFACTS_NOT_READY")
        app.state.readiness_checks = {
            "database": "ready" if app.state.database_ready else "not_ready",
            "artifacts": "not_ready",
            "embedding_model": "not_ready",
        }
    except RetrievalError:
        log_event("startup_failed", error_code="MODEL_NOT_READY")
        app.state.readiness_checks = {
            "database": "ready" if app.state.database_ready else "not_ready",
            "artifacts": "ready",
            "embedding_model": "not_ready",
        }
    yield
    if app.state.llm_provider is not None:
        await app.state.llm_provider.aclose()
    engine.dispose()
    app.state.readiness_checks = {
        "database": "not_ready",
        "artifacts": "not_ready",
        "embedding_model": "not_ready",
    }


def initialize_retrieval(settings: Settings) -> tuple[DataArtifacts, RetrievalService]:
    """Construct the only artifact/model instances used by this API process."""
    artifacts = load_data_artifacts(settings.resolved_data_dir)
    if not settings.embedding_model_path_or_id:
        raise RetrievalError("Embedding model is not configured")
    model = artifacts.manifest["embedding_model"]
    if model.get("local_sha256"):
        try:
            actual_hash = local_model_fingerprint(Path(settings.embedding_model_path_or_id))
        except (OSError, ValueError) as exc:
            raise RetrievalError("Local model identity cannot be verified") from exc
        if actual_hash != model["local_sha256"]:
            raise RetrievalError("Local model identity does not match the artifact manifest")
    encoder = SentenceTransformerEncoder(settings.embedding_model_path_or_id, model.get("revision"))
    return artifacts, RetrievalService(artifacts, encoder, max_top_k=settings.retrieval_max_top_k)


def _pipeline_factory(app: FastAPI):
    """Build fresh per-job pipeline state while retaining process-wide provider concurrency."""

    def factory(report_stage):
        settings: Settings = app.state.settings
        provider = app.state.llm_provider
        if provider is None:
            raise PipelineError(
                "LLM_NOT_CONFIGURED",
                "extracting_assumptions",
                "Analysis provider is not configured",
                internal_message="OPENROUTER_API_KEY or OPENROUTER_MODEL is unset",
            )
        llm = (
            CachedLLMClient(provider, app.state.llm_cache, prompt_version=settings.prompt_version)
            if app.state.llm_cache is not None
            else provider
        )
        artifacts = app.state.data_artifacts
        return AnalysisPipeline(
            llm,
            settings,
            retriever=app.state.retrieval_service,
            idx_benchmark_rows=artifacts.idx_benchmark_overall if artifacts is not None else (),
            report_stage=report_stage,
        )

    return factory


def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request: Request,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    response = JSONResponse(status_code=status_code, content=payload.model_dump(by_alias=True))
    response.headers["X-Request-ID"] = request_id
    if headers:
        response.headers.update(headers)
    return response


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()
    app = FastAPI(
        title="Veritas API",
        version="0.1.0",
        description=(
            "Decision intelligence API for evidence, financial stress tests, and validation plans."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestBoundary, max_bytes=MAX_REQUEST_BODY_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["Location", "X-Request-ID"],
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            request=request,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="VALIDATION_ERROR",
            message="Request validation failed",
            request=request,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = (
            exc.detail
            if isinstance(exc.detail, str) and exc.status_code < 500
            else "Request failed"
        )
        return error_response(
            status_code=exc.status_code,
            code=HTTP_ERROR_CODES.get(exc.status_code, "HTTP_ERROR"),
            message=message,
            request=request,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
        log_event("request_failed", error_code=type(_exc).__name__)
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            request=request,
        )

    app.include_router(health.router)
    app.include_router(analyses.router, prefix="/api/v1")
    return app


app = create_app()


def run() -> None:
    """Run Uvicorn with host and port sourced from application settings."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    run()
