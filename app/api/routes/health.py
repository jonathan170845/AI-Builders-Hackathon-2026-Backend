from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.schemas.health import LivenessResponse, ReadinessNotReadyResponse, ReadinessReadyResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LivenessResponse)
async def live() -> LivenessResponse:
    """Confirm that the HTTP process is alive."""
    return LivenessResponse()


@router.get(
    "/ready",
    response_model=ReadinessReadyResponse,
    responses={503: {"model": ReadinessNotReadyResponse}},
)
async def ready(request: Request) -> ReadinessReadyResponse | JSONResponse:
    """Confirm that bootstrap dependencies have initialized."""
    checks = getattr(request.app.state, "readiness_checks", {"bootstrap": "not_ready"})
    if all(check == "ready" for check in checks.values()):
        return ReadinessReadyResponse()
    payload = ReadinessNotReadyResponse(checks=checks)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(by_alias=True),
    )
