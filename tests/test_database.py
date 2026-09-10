from __future__ import annotations

from alembic.config import Config
from sqlalchemy import inspect

from alembic import command


def test_llm_cache_migration_creates_empty_database(monkeypatch, tmp_path):
    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")

    command.upgrade(config, "head")

    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        assert "llm_cache" in inspect(engine).get_table_names()
    finally:
        engine.dispose()
