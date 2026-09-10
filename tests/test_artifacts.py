from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.artifacts import (
    MANIFEST_FILENAME,
    ArtifactValidationError,
    artifact_entry,
    load_data_artifacts,
)
from tests.artifact_factory import make_artifact_directory


def test_loader_loads_small_validated_fixture_once_with_memmap(tmp_path):
    artifacts = load_data_artifacts(make_artifact_directory(tmp_path))

    assert [record.company_id for record in artifacts.company_records] == [
        "company-a",
        "company-b",
        "company-c",
    ]
    assert isinstance(artifacts.company_embeddings, np.memmap)
    assert artifacts.embedding_dimension == 2


def test_loader_rejects_embedding_metadata_row_mismatch(tmp_path):
    artifact_dir = make_artifact_directory(tmp_path)
    np.save(
        artifact_dir / "software_it_candidate_embeddings.npy", np.array([[1, 0]], dtype=np.float32)
    )
    manifest_path = artifact_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["company_embeddings"] = artifact_entry(
        artifact_dir / "software_it_candidate_embeddings.npy", rows=1, dimension=2, dtype="float32"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="not aligned"):
        load_data_artifacts(artifact_dir)


def test_loader_rejects_same_count_metadata_with_different_stable_id_order(tmp_path):
    artifact_dir = make_artifact_directory(tmp_path)
    metadata = artifact_dir / "software_it_candidate_metadata.csv"
    with metadata.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    with metadata.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([rows[1], rows[0], rows[2]])
    manifest_path = artifact_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["company_metadata"] = artifact_entry(
        metadata, rows=3, stable_id_column="id"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="order"):
        load_data_artifacts(artifact_dir)


def test_missing_artifact_keeps_readiness_at_503_without_absolute_path(tmp_path):
    settings = Settings(data_dir=tmp_path, embedding_model_path_or_id="fixture")
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "database": "not_ready",
        "artifacts": "not_ready",
        "embedding_model": "not_ready",
    }
    assert str(tmp_path) not in response.text
