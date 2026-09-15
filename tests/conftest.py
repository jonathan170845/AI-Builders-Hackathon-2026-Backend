from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.base import Base
from app.main import create_app


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    app = create_app(
        Settings(
            frontend_origins=["http://localhost:5173"],
            database_url=f"sqlite:///{tmp_path / 'test.db'}",
        )
    )
    with TestClient(app) as test_client:
        Base.metadata.create_all(app.state.db_engine)
        app.state.analysis_jobs.reconcile_interrupted()
        app.state.database_ready = True
        app.state.retrieval_service = object()
        yield test_client


@pytest.fixture(autouse=True)
def isolate_runtime_configuration(request, monkeypatch, tmp_path):
    # Offline tests must never use the developer's real provider, dataset, or database.
    if request.node.get_closest_marker("data") or request.node.get_closest_marker("external"):
        return
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "absent-data"))
    monkeypatch.setenv("EMBEDDING_MODEL_PATH_OR_ID", "")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'isolated.db'}")
