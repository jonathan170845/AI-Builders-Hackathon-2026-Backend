from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.schemas.analysis import AnalysisStage, AnalysisStatus


def _enum_values(enum_type):
    return [member.value for member in enum_type]


class LLMCacheEntry(Base):
    __tablename__ = "llm_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    response_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    response_bytes: Mapped[int] = mapped_column(Integer, nullable=False)


class AnalysisRecord(Base):
    """Persisted job state. Internal error details never leave this persistence layer."""

    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_created_at_id", "created_at", "id"),
        Index("ix_analyses_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    financial_inputs_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(
            AnalysisStatus,
            name="analysis_status",
            native_enum=False,
            create_constraint=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    stage: Mapped[AnalysisStage | None] = mapped_column(
        Enum(
            AnalysisStage,
            name="analysis_stage",
            native_enum=False,
            create_constraint=True,
            values_callable=_enum_values,
        ),
        nullable=True,
    )
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    public_error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    public_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    internal_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    pipeline_version: Mapped[str] = mapped_column(String(100), nullable=False)
    pipeline_metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    assumption_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    financial_warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
