from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analyses, health
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.schemas.errors import ErrorDetail, ErrorResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.analyses = {}
    app.state.ready = True
    yield
    app.state.ready = False


def error_response(*, status_code: int, code: str, message: str, request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    return JSONResponse(status_code=status_code, content=payload.model_dump(by_alias=True))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Veritas API",
        version="0.1.0",
        description="Decision intelligence API for evidence, financial stress tests, and validation plans.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["Location", "X-Request-ID"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(
            status_code=exc.status_code, code=exc.code, message=exc.message, request=request
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            message="Request validation failed",
            request=request,
        )

    app.include_router(health.router)
    app.include_router(analyses.router, prefix="/api/v1")
    return app


app = create_app()
