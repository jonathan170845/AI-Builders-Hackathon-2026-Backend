"""Single-process background execution boundary for persisted analysis jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol
from uuid import UUID

from app.db.analyses import AnalysisRepository, CapacityExceededError
from app.schemas.analysis import AnalysisAccepted, AnalysisStage, CreateAnalysisRequest
from app.services.pipeline import AnalysisPipeline, PipelineError, PipelineRun

logger = logging.getLogger(__name__)

StageReporter = Callable[[AnalysisStage], Awaitable[None]]


class PipelineFactory(Protocol):
    def __call__(self, report_stage: StageReporter) -> AnalysisPipeline: ...


class AnalysisJobService:
    """Schedules bounded in-process jobs; this is deliberately not a multi-worker queue."""

    def __init__(
        self,
        repository: AnalysisRepository,
        pipeline_factory: PipelineFactory,
        *,
        pipeline_version: str,
        max_concurrent: int,
        max_queued: int,
    ) -> None:
        self._repository = repository
        self._pipeline_factory = pipeline_factory
        self._pipeline_version = pipeline_version
        self._max_queued = max_queued
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def submit(self, payload: CreateAnalysisRequest) -> AnalysisAccepted:
        record = self._repository.create_queued(
            payload,
            pipeline_version=self._pipeline_version,
            max_queued=self._max_queued,
        )
        return AnalysisAccepted(id=record.id, created_at=record.created_at)

    async def run(self, analysis_id: UUID) -> None:
        """Run one job and turn every unexpected failure into a terminal public state."""
        async with self._semaphore:
            record = self._repository.start_if_queued(analysis_id)
            if record is None:
                return
            stage = AnalysisStage.EXTRACTING_ASSUMPTIONS

            async def report_stage(next_stage: AnalysisStage) -> None:
                nonlocal stage
                stage = next_stage
                self._repository.set_stage(analysis_id, next_stage)

            try:
                pipeline = self._pipeline_factory(report_stage)
                completed: PipelineRun = await pipeline.run(
                    analysis_id,
                    _payload_from_record(record.decision, record.financial_inputs_json),
                )
                self._repository.complete_if_processing(analysis_id, completed.result)
            except PipelineError as exc:
                self._repository.fail_if_nonterminal(
                    analysis_id,
                    code=exc.code,
                    message="Analysis could not be completed. Please try again.",
                    internal_error=exc.internal_message,
                    stage=_stage_or_default(exc.stage, stage),
                )
            except Exception as exc:  # noqa: BLE001 - this is the worker safety boundary
                logger.exception("analysis worker failed", extra={"analysis_id": str(analysis_id)})
                self._repository.fail_if_nonterminal(
                    analysis_id,
                    code="ANALYSIS_FAILED",
                    message="Analysis could not be completed. Please try again.",
                    internal_error=f"{type(exc).__name__}: {exc}",
                    stage=stage,
                )

    def reconcile_interrupted(self) -> int:
        return self._repository.reconcile_interrupted()

    @property
    def repository(self) -> AnalysisRepository:
        return self._repository


def _payload_from_record(decision: str, financial_inputs_json: str | None) -> CreateAnalysisRequest:
    return CreateAnalysisRequest.model_validate(
        {
            "decision": decision,
            "financialInputs": (
                None if financial_inputs_json is None else json.loads(financial_inputs_json)
            ),
        }
    )


def _stage_or_default(value: str, default: AnalysisStage) -> AnalysisStage:
    try:
        return AnalysisStage(value)
    except ValueError:
        return default


__all__ = ["AnalysisJobService", "CapacityExceededError", "PipelineFactory"]
