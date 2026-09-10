from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_engine_and_session_factory(database_url: str) -> tuple[Engine, sessionmaker[Session]]:
    """Create synchronous MVP persistence primitives without process-global sessions."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, future=True, pool_pre_ping=True, connect_args=connect_args)
    return engine, sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
