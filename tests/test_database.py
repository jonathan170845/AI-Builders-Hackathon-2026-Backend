from __future__ import annotations

from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from alembic import command
from app.db.analyses import AnalysisRepository
from app.schemas.analysis import CreateAnalysisRequest


def test_llm_cache_migration_creates_empty_database(monkeypatch, tmp_path):
    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")

    command.upgrade(config, "head")

    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        tables = inspect(engine).get_table_names()
        assert "llm_cache" in tables
        assert "analyses" in tables
        repository = AnalysisRepository(sessionmaker(bind=engine, expire_on_commit=False))
        record = repository.create_queued(
            CreateAnalysisRequest(decision="A valid decision for migration verification"),
            pipeline_version="test",
            max_queued=10,
        )
        assert repository.get(record.id) is not None
    finally:
        engine.dispose()
