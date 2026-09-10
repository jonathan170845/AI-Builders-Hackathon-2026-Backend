from fastapi import APIRouter, Request

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    """Confirm that the HTTP process is alive."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    """Confirm that bootstrap dependencies have initialized."""
    if not getattr(request.app.state, "ready", False):
        return {"status": "starting"}
    return {"status": "ready"}
