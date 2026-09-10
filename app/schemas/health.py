from __future__ import annotations

from typing import Literal

from app.schemas.analysis import ApiModel


class LivenessResponse(ApiModel):
    status: Literal["ok"] = "ok"


class ReadinessReadyResponse(ApiModel):
    status: Literal["ready"] = "ready"


class ReadinessNotReadyResponse(ApiModel):
    status: Literal["not_ready"] = "not_ready"
    checks: dict[str, Literal["ready", "not_ready"]]
