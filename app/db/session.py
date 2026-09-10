from __future__ import annotations

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def create_engine_and_session_factory(
    database_url: str, *, sqlite_busy_timeout_ms: int = 5_000
) -> tuple[Engine, sessionmaker[Session]]:
    """Create synchronous MVP persistence primitives without process-global sessions."""
    is_sqlite = database_url.startswith("sqlite")
    connect_args = (
        {"check_same_thread": False, "timeout": sqlite_busy_timeout_ms / 1_000} if is_sqlite else {}
    )
    engine = create_engine(database_url, future=True, pool_pre_ping=True, connect_args=connect_args)
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={sqlite_busy_timeout_ms}")
            cursor.close()

    return engine, sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
