from __future__ import annotations

import json

import numpy as np
import pytest

from app.services.artifacts import load_data_artifacts
from app.services.retrieval import RetrievalError, RetrievalService
from tests.artifact_factory import make_artifact_directory


class FixtureEncoder:
    def dimension(self) -> int:
        return 2

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


@pytest.fixture
def retrieval_service(tmp_path) -> RetrievalService:
    return RetrievalService(
        load_data_artifacts(make_artifact_directory(tmp_path)), FixtureEncoder()
    )


def test_company_ranking_is_deterministic_for_similarity_ties(retrieval_service):
    results = retrieval_service.company_analogues("  alpha   software ", top_k=3)

    assert [result.company_id for result in results] == ["company-a", "company-c", "company-b"]
    assert [result.rank for result in results] == [1, 2, 3]


def test_failure_retrieval_and_assumption_dtos_are_json_compatible(retrieval_service):
    evidence = retrieval_service.retrieve_assumptions(
        [{"id": "A1", "assumption": "Demand exists", "retrieval_query": "alpha"}],
        top_company_analogues=2,
        top_failure_cases=2,
    )

    payload = evidence[0].as_dict()
    assert [failure["company"] for failure in payload["historical_failures"]] == [
        "Failed Alpha",
        "Failed Beta",
    ]
    json.dumps(payload)


def test_empty_query_is_rejected(retrieval_service):
    with pytest.raises(RetrievalError, match="must not be empty"):
        retrieval_service.company_analogues("   ")
