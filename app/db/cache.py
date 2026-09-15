from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import LLMCacheEntry


class LLMCache:
    """Exact-key SQLite cache. Values can contain sensitive model output; never log them."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        retention_days: int,
        max_rows: int,
    ) -> None:
        self._session_factory = session_factory
        self._retention = timedelta(days=retention_days)
        self._max_rows = max_rows

    def get(self, cache_key: str) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        with self._session_factory.begin() as session:
            entry = session.get(LLMCacheEntry, cache_key)
            if entry is None:
                return None
            if entry.expires_at.replace(tzinfo=UTC) <= now:
                session.delete(entry)
                return None
            entry.last_accessed_at = now
            return json.loads(entry.response_json)

    def put(
        self, cache_key: str, *, model: str, prompt_version: str, response: dict[str, Any]
    ) -> None:
        serialized = json.dumps(response, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        now = datetime.now(UTC)
        with self._session_factory.begin() as session:
            if session.bind is not None and session.bind.dialect.name == "sqlite":
                session.execute(text("BEGIN IMMEDIATE"))
            session.execute(delete(LLMCacheEntry).where(LLMCacheEntry.expires_at <= now))
            existing = session.get(LLMCacheEntry, cache_key)
            values = {
                "model": model,
                "prompt_version": prompt_version,
                "response_json": serialized,
                "created_at": now,
                "expires_at": now + self._retention,
                "last_accessed_at": now,
                "response_bytes": len(serialized.encode("utf-8")),
            }
            if existing is None:
                session.add(LLMCacheEntry(cache_key=cache_key, **values))
            else:
                for name, value in values.items():
                    setattr(existing, name, value)
            overflow = session.scalars(
                select(LLMCacheEntry.cache_key)
                .order_by(LLMCacheEntry.last_accessed_at.desc())
                .offset(self._max_rows)
            ).all()
            if overflow:
                session.execute(delete(LLMCacheEntry).where(LLMCacheEntry.cache_key.in_(overflow)))
