from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Request, Response, status

from app.core.errors import ApiError
from app.schemas.analysis import (
    AnalysisAccepted,
    AnalysisDetail,
    AnalysisPending,
    AnalysisStage,
    AnalysisStatus,
    CreateAnalysisRequest,
)
from app.schemas.errors import ErrorResponse

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post(
    "",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses={422: {"model": ErrorResponse}},
)
async def create_analysis(payload: CreateAnalysisRequest, request: Request, response: Response) -> AnalysisAccepted:
    """Queue an analysis; background execution is introduced in Issue 005."""
    analysis_id = str(uuid4())
    created_at = datetime.now(UTC)
    item = AnalysisPending(
        id=analysis_id,
        status=AnalysisStatus.QUEUED,
        stage=AnalysisStage.QUEUED,
        created_at=created_at,
        updated_at=created_at,
    )
    request.app.state.analyses[analysis_id] = item
    response.headers["Location"] = f"/api/v1/analyses/{analysis_id}"
    return AnalysisAccepted(id=analysis_id, created_at=created_at)


@router.get(
    "/{analysis_id}",
    response_model=AnalysisDetail,
    responses={404: {"model": ErrorResponse}},
)
async def get_analysis(analysis_id: str, request: Request) -> AnalysisDetail:
    """Return the current job state from the temporary bootstrap store."""
    analysis = request.app.state.analyses.get(analysis_id)
    if analysis is None:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="ANALYSIS_NOT_FOUND",
            message="Analysis was not found",
        )
    return analysis
