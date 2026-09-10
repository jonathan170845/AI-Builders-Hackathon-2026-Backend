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
        yield test_client
