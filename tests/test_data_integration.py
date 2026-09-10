from __future__ import annotations

import os

import pytest

from app.core.config import Settings
from app.services.artifacts import load_data_artifacts


@pytest.mark.data
def test_prepared_real_artifacts_are_loadable():
    """Opt-in check for a prepared, licensed artifact directory; never downloads a model."""
    if os.getenv("RUN_DATA_INTEGRATION") != "1":
        pytest.skip("Set RUN_DATA_INTEGRATION=1 after preparing licensed local artifacts")

    artifacts = load_data_artifacts(Settings().resolved_data_dir)

    assert len(artifacts.company_records) == artifacts.company_embeddings.shape[0]
    assert len(artifacts.failure_documents) == artifacts.failure_embeddings.shape[0]
