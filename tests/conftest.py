from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(Settings(frontend_origins=["http://localhost:5173"]))
    with TestClient(app) as test_client:
        yield test_client
