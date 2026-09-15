"""Repository for durable analysis jobs and stable history pagination."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Select, and_, func, or_, select, text, update
from sqlalchemy.orm import Session, load_only, sessionmaker

from app.db.models import AnalysisRecord
from app.schemas.analysis import (
    AnalysisCompleted,
    AnalysisDetail,
    AnalysisFailed,
    AnalysisHistory,
    AnalysisHistoryItem,
    AnalysisJobError,
    AnalysisPending,
    AnalysisResult,
    AnalysisStage,
    AnalysisStatus,
    CreateAnalysisRequest,
)

type Cursor = tuple[datetime, UUID]


class AnalysisRepositoryError(RuntimeError):
    """Base repository failure that must not be serialized directly to clients."""


class InvalidCursorError(AnalysisRepositoryError):
    pass


class CapacityExceededError(AnalysisRepositoryError):
    pass


class AnalysisRepository:
    """Creates a new database session for every operation, including worker operations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_queued(
        self,
        payload: CreateAnalysisRequest,
        *,
        pipeline_version: str,
        max_queued: int,
    ) -> AnalysisRecord:
        with self._session_factory() as session:
            # A deferred SQLite transaction would let two simultaneous submits both observe
            # spare capacity. BEGIN IMMEDIATE serializes this short count-and-insert section.
            if session.bind is not None and session.bind.dialect.name == "sqlite":
                session.execute(text("BEGIN IMMEDIATE"))
            else:
                session.begin()
            queued_count = session.scalar(
                select(func.count())
                .select_from(AnalysisRecord)
                .where(AnalysisRecord.status == AnalysisStatus.QUEUED)
            )
            if int(queued_count or 0) >= max_queued:
                raise CapacityExceededError("Analysis queue is full")
            now = datetime.now(UTC)
            record = AnalysisRecord(
                id=uuid4(),
                decision=payload.decision,
                financial_inputs_json=(
                    payload.financial_inputs.model_dump_json(by_alias=True)
                    if payload.financial_inputs is not None
                    else None
                ),
                status=AnalysisStatus.QUEUED,
                stage=AnalysisStage.QUEUED,
                pipeline_version=pipeline_version,
                assumption_count=0,
                financial_warning_count=0,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.commit()
        return record

    def get(self, analysis_id: UUID) -> AnalysisRecord | None:
        with self._session_factory() as session:
            return session.get(AnalysisRecord, analysis_id)

    def to_detail(self, record: AnalysisRecord) -> AnalysisDetail:
        created_at, updated_at = _utc(record.created_at), _utc(record.updated_at)
        if record.status in {AnalysisStatus.QUEUED, AnalysisStatus.PROCESSING}:
            return AnalysisPending(
                id=record.id,
                status=record.status,
                stage=record.stage or AnalysisStage.QUEUED,
                created_at=created_at,
                updated_at=updated_at,
            )
        if record.status == AnalysisStatus.COMPLETED:
            if record.result_json is None:
                raise AnalysisRepositoryError("Completed analysis has no result")
            return AnalysisCompleted(
                id=record.id,
                created_at=created_at,
                updated_at=updated_at,
                result=AnalysisResult.model_validate_json(record.result_json),
            )
        return AnalysisFailed(
            id=record.id,
            created_at=created_at,
            updated_at=updated_at,
            stage=record.stage or AnalysisStage.QUEUED,
            error=AnalysisJobError(
                code=record.public_error_code or "ANALYSIS_FAILED",
                message=record.public_error_message or "Analysis could not be completed",
            ),
        )

    def list_history(self, *, limit: int, cursor: str | None) -> AnalysisHistory:
        statement: Select[tuple[AnalysisRecord]] = select(AnalysisRecord).options(
            load_only(
                AnalysisRecord.id,
                AnalysisRecord.decision,
                AnalysisRecord.status,
                AnalysisRecord.created_at,
                AnalysisRecord.updated_at,
                AnalysisRecord.assumption_count,
                AnalysisRecord.financial_warning_count,
            )
        )
        if cursor is not None:
            created_at, analysis_id = decode_cursor(cursor)
            # SQLite stores DateTime without timezone metadata; comparing the same UTC instant
            # retains the `(created_at DESC, id DESC)` ordering on both SQLite and PostgreSQL.
            cursor_time = created_at.replace(tzinfo=None)
            statement = statement.where(
                or_(
                    AnalysisRecord.created_at < cursor_time,
                    and_(
                        AnalysisRecord.created_at == cursor_time,
                        AnalysisRecord.id < analysis_id,
                    ),
                )
            )
        statement = statement.order_by(
            AnalysisRecord.created_at.desc(), AnalysisRecord.id.desc()
        ).limit(limit + 1)
        with self._session_factory() as session:
            records = list(session.scalars(statement))
        has_next = len(records) > limit
        page = records[:limit]
        return AnalysisHistory(
            items=[
                AnalysisHistoryItem(
                    id=item.id,
                    decision=item.decision,
                    status=item.status,
                    created_at=_utc(item.created_at),
                    updated_at=_utc(item.updated_at),
                    assumption_count=item.assumption_count,
                    financial_warning_count=item.financial_warning_count,
                )
                for item in page
            ],
            next_cursor=encode_cursor(page[-1]) if has_next and page else None,
        )

    def start_if_queued(self, analysis_id: UUID) -> AnalysisRecord | None:
        now = datetime.now(UTC)
        with self._session_factory.begin() as session:
            result = session.execute(
                update(AnalysisRecord)
                .where(
                    AnalysisRecord.id == analysis_id,
                    AnalysisRecord.status == AnalysisStatus.QUEUED,
                )
                .values(
                    status=AnalysisStatus.PROCESSING,
                    stage=AnalysisStage.EXTRACTING_ASSUMPTIONS,
                    started_at=now,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                return None
        return self.get(analysis_id)

    def set_stage(self, analysis_id: UUID, stage: AnalysisStage) -> bool:
        with self._session_factory.begin() as session:
            result = session.execute(
                update(AnalysisRecord)
                .where(
                    AnalysisRecord.id == analysis_id,
                    AnalysisRecord.status == AnalysisStatus.PROCESSING,
                )
                .values(stage=stage, updated_at=datetime.now(UTC))
            )
            return result.rowcount == 1

    def complete_if_processing(
        self, analysis_id: UUID, result: AnalysisResult, metadata: dict | None = None
    ) -> bool:
        now = datetime.now(UTC)
        with self._session_factory.begin() as session:
            update_result = session.execute(
                update(AnalysisRecord)
                .where(
                    AnalysisRecord.id == analysis_id,
                    AnalysisRecord.status == AnalysisStatus.PROCESSING,
                )
                .values(
                    status=AnalysisStatus.COMPLETED,
                    stage=None,
                    result_json=result.model_dump_json(by_alias=True),
                    pipeline_metadata_json=json.dumps(metadata) if metadata is not None else None,
                    assumption_count=result.assumption_count,
                    financial_warning_count=result.financial_warning_count,
                    completed_at=now,
                    updated_at=now,
                )
            )
            return update_result.rowcount == 1

    def fail_if_nonterminal(
        self,
        analysis_id: UUID,
        *,
        code: str,
        message: str,
        internal_error: str,
        stage: AnalysisStage,
    ) -> bool:
        now = datetime.now(UTC)
        with self._session_factory.begin() as session:
            update_result = session.execute(
                update(AnalysisRecord)
                .where(
                    AnalysisRecord.id == analysis_id,
                    AnalysisRecord.status.in_([AnalysisStatus.QUEUED, AnalysisStatus.PROCESSING]),
                )
                .values(
                    status=AnalysisStatus.FAILED,
                    stage=stage,
                    public_error_code=code,
                    public_error_message=message,
                    internal_error=internal_error[:10_000],
                    completed_at=now,
                    updated_at=now,
                )
            )
            return update_result.rowcount == 1

    def reconcile_interrupted(self) -> int:
        """Ensure a process restart cannot leave a job permanently processing."""
        now = datetime.now(UTC)
        with self._session_factory.begin() as session:
            update_result = session.execute(
                update(AnalysisRecord)
                .where(
                    AnalysisRecord.status.in_([AnalysisStatus.QUEUED, AnalysisStatus.PROCESSING])
                )
                .values(
                    status=AnalysisStatus.FAILED,
                    public_error_code="WORKER_INTERRUPTED",
                    public_error_message="Analysis was interrupted by a backend restart",
                    internal_error=(
                        "Worker process ended before the analysis reached a terminal state"
                    ),
                    completed_at=now,
                    updated_at=now,
                )
            )
            return int(update_result.rowcount or 0)


def encode_cursor(record: AnalysisRecord) -> str:
    payload = {
        "createdAt": _utc(record.created_at).isoformat(),
        "id": str(record.id),
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
    return encoded.decode().rstrip("=")


def decode_cursor(value: str) -> Cursor:
    if not value or len(value) > 512:
        raise InvalidCursorError("Cursor is invalid")
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode(padded.encode())
        payload = json.loads(raw)
        if set(payload) != {"createdAt", "id"}:
            raise ValueError("unexpected cursor fields")
        created_at = datetime.fromisoformat(payload["createdAt"])
        if created_at.tzinfo is None:
            raise ValueError("cursor timestamp is timezone-naive")
        return created_at.astimezone(UTC), UUID(payload["id"])
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("Cursor is invalid") from exc


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
