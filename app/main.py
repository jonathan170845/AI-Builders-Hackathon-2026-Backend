from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import analyses, health
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.schemas.errors import ErrorDetail, ErrorResponse
from app.services.artifacts import ArtifactValidationError, DataArtifacts, load_data_artifacts
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
    app.state.analyses = {}
    app.state.data_artifacts = None
    app.state.retrieval_service = None
    settings: Settings = app.state.settings
    try:
        artifacts, retrieval_service = initialize_retrieval(settings)
        app.state.data_artifacts = artifacts
        app.state.retrieval_service = retrieval_service
        app.state.readiness_checks = {"artifacts": "ready", "embedding_model": "ready"}
    except ArtifactValidationError:
        app.state.readiness_checks = {"artifacts": "not_ready", "embedding_model": "not_ready"}
    except RetrievalError:
        app.state.readiness_checks = {"artifacts": "ready", "embedding_model": "not_ready"}
    yield
    app.state.readiness_checks = {"artifacts": "not_ready", "embedding_model": "not_ready"}


def initialize_retrieval(settings: Settings) -> tuple[DataArtifacts, RetrievalService]:
    """Construct the only artifact/model instances used by this API process."""
    artifacts = load_data_artifacts(settings.resolved_data_dir)
    if not settings.embedding_model_path_or_id:
        raise RetrievalError("Embedding model is not configured")
    model = artifacts.manifest["embedding_model"]
    encoder = SentenceTransformerEncoder(settings.embedding_model_path_or_id, model.get("revision"))
    return artifacts, RetrievalService(artifacts, encoder, max_top_k=settings.retrieval_max_top_k)


def error_response(*, status_code: int, code: str, message: str, request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    response = JSONResponse(status_code=status_code, content=payload.model_dump(by_alias=True))
    response.headers["X-Request-ID"] = request_id
    return response


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Veritas API",
        version="0.1.0",
        description=(
            "Decision intelligence API for evidence, financial stress tests, and validation plans."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["Location", "X-Request-ID"],
    )

    @app.middleware("http")
    async def add_request_id_and_limit_body(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
        content_length = request.headers.get("Content-Length")
        try:
            is_request_too_large = (
                content_length is not None and int(content_length) > MAX_REQUEST_BODY_BYTES
            )
        except ValueError:
            is_request_too_large = False
        if is_request_too_large:
            response = error_response(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                code="REQUEST_TOO_LARGE",
                message="Request body exceeds the allowed size",
                request=request,
            )
            return response
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(
            status_code=exc.status_code, code=exc.code, message=exc.message, request=request
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
