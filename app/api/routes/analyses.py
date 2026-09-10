from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Query, Request, Response, status

from app.core.errors import ApiError
from app.db.analyses import CapacityExceededError, InvalidCursorError
from app.schemas.analysis import (
    AnalysisAccepted,
    AnalysisDetail,
    AnalysisHistory,
    CreateAnalysisRequest,
)
from app.schemas.errors import ErrorResponse

router = APIRouter(prefix="/analyses", tags=["analyses"])

ERROR_RESPONSES = {
    413: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


@router.post(
    "",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERROR_RESPONSES,
)
async def create_analysis(
    payload: CreateAnalysisRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
) -> AnalysisAccepted:
    """Persist a queued job, then schedule its single-process background execution."""
    jobs = _jobs_or_503(request)
    try:
        accepted = jobs.submit(payload)
    except CapacityExceededError as exc:
        raise ApiError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="CAPACITY_EXCEEDED",
            message="Analysis capacity is currently full",
            headers={"Retry-After": str(request.app.state.settings.analysis_retry_after_seconds)},
        ) from exc
    background_tasks.add_task(jobs.run, accepted.id)
    response.headers["Location"] = f"/api/v1/analyses/{accepted.id}"
    return accepted


@router.get("", response_model=AnalysisHistory, responses=ERROR_RESPONSES)
async def list_analyses(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=512),
) -> AnalysisHistory:
    """Return compact newest-first history using an opaque stable cursor."""
    try:
        return _jobs_or_503(request).repository.list_history(limit=limit, cursor=cursor)
    except InvalidCursorError as exc:
        raise ApiError(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="INVALID_CURSOR",
            message="Cursor is invalid",
        ) from exc


@router.get(
    "/{analysis_id}",
    response_model=AnalysisDetail,
    responses={
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_analysis(analysis_id: UUID, request: Request) -> AnalysisDetail:
    """Return the latest durable status, result, or safe job error."""
    repository = _jobs_or_503(request).repository
    record = repository.get(analysis_id)
    if record is None:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="ANALYSIS_NOT_FOUND",
            message="Analysis was not found",
        )
    return repository.to_detail(record)


def _jobs_or_503(request: Request):
    jobs = getattr(request.app.state, "analysis_jobs", None)
    if jobs is None or not getattr(request.app.state, "database_ready", False):
        raise ApiError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="DATABASE_NOT_READY",
            message="Analysis persistence is not ready",
        )
    return jobs
