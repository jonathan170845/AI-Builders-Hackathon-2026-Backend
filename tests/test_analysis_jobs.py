from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.analyses import AnalysisRepository
from app.db.base import Base
from app.db.models import AnalysisRecord
from app.main import create_app
from app.schemas.analysis import (
    AnalysisResult,
    AnalysisStage,
    AnalysisStatus,
    CreateAnalysisRequest,
)
from app.services.jobs import AnalysisJobService
from app.services.pipeline import PipelineError, PipelineMetadata, PipelineRun

PAYLOAD = {"decision": "Scale on-demand grocery delivery to five new cities"}
StageReporter = Callable[[AnalysisStage], Awaitable[None]]


class SuccessfulPipeline:
    def __init__(self, report_stage: StageReporter) -> None:
        self._report_stage = report_stage

    async def run(self, analysis_id, payload) -> PipelineRun:
        await self._report_stage(AnalysisStage.RETRIEVING_EVIDENCE)
        await self._report_stage(AnalysisStage.BUILDING_REPORT)
        result = AnalysisResult(
            id=analysis_id,
            decision_title=payload.decision,
            summary="A deterministic fake result.",
            assumptions=[],
            direct_evidence_coverage="0 of 0 assumptions cite historical evidence",
            financial_warnings=[],
            financial_results=None,
            failure_mechanisms=[],
            company_analogues=[],
            validation_experiments=[],
            idx_benchmarks=[],
            sources=[],
            date="2026-09-10",
            assumption_count=0,
            financial_warning_count=0,
        )
        return PipelineRun(
            result=result,
            metadata=PipelineMetadata(
                model="fake", prompt_version="test", llm_calls=0, cached_calls=0, total_cost_usd=0
            ),
        )


class FailingPipeline:
    async def run(self, _analysis_id, _payload) -> PipelineRun:
        raise PipelineError(
            "LLM_PROVIDER_UNAVAILABLE",
            "extracting_assumptions",
            "Provider failure",
            internal_message="secret transport details must not leave the database",
        )


def _success_factory(report_stage: StageReporter) -> SuccessfulPipeline:
    return SuccessfulPipeline(report_stage)


def _failure_factory(_report_stage: StageReporter) -> FailingPipeline:
    return FailingPipeline()


@pytest.fixture
def persistent_app(tmp_path) -> Iterator[tuple[TestClient, str]]:
    database_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    app = create_app(Settings(database_url=database_url, max_queued_analyses=10))
    with TestClient(app) as client:
        Base.metadata.create_all(app.state.db_engine)
        repository = AnalysisRepository(app.state.db_session_factory)
        app.state.analysis_jobs = AnalysisJobService(
            repository,
            _success_factory,
            pipeline_version="test",
            max_concurrent=1,
            max_queued=10,
        )
        app.state.database_ready = True
        app.state.retrieval_service = object()
        yield client, database_url


def test_create_runs_fake_pipeline_and_persists_completed_result(persistent_app):
    client, _database_url = persistent_app

    created = client.post("/api/v1/analyses", json=PAYLOAD)

    assert created.status_code == 202
    analysis_id = created.json()["id"]
    detail = client.get(f"/api/v1/analyses/{analysis_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "completed"
    assert detail.json()["result"]["id"] == analysis_id

    history = client.get("/api/v1/analyses")
    assert history.status_code == 200
    assert history.json()["items"] == [
        {
            "id": analysis_id,
            "decision": PAYLOAD["decision"],
            "status": "completed",
            "createdAt": created.json()["createdAt"],
            "updatedAt": detail.json()["updatedAt"],
            "assumptionCount": 0,
            "financialWarningCount": 0,
        }
    ]
    assert "result" not in history.json()["items"][0]


def test_failed_pipeline_returns_safe_public_error_only(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'failure.db'}"
    app = create_app(Settings(database_url=database_url))
    with TestClient(app) as client:
        Base.metadata.create_all(app.state.db_engine)
        app.state.analysis_jobs = AnalysisJobService(
            AnalysisRepository(app.state.db_session_factory),
            _failure_factory,
            pipeline_version="test",
            max_concurrent=1,
            max_queued=10,
        )
        app.state.database_ready = True
        app.state.retrieval_service = object()
        created = client.post("/api/v1/analyses", json=PAYLOAD)
        body = client.get(f"/api/v1/analyses/{created.json()['id']}").json()

    assert body["status"] == "failed"
    assert body["error"]["code"] == "LLM_PROVIDER_UNAVAILABLE"
    assert "secret transport" not in str(body)


def test_cursor_is_stable_for_equal_timestamps_and_invalid_cursor_is_rejected(persistent_app):
    client, _database_url = persistent_app
    repository = client.app.state.analysis_jobs.repository
    shared_time = datetime(2026, 9, 10, 12, tzinfo=UTC)
    ids = sorted([uuid4(), uuid4(), uuid4()], reverse=True)
    with client.app.state.db_session_factory.begin() as session:
        for analysis_id in ids:
            session.add(
                AnalysisRecord(
                    id=analysis_id,
                    decision=f"Decision for {analysis_id}",
                    financial_inputs_json=None,
                    status=AnalysisStatus.QUEUED,
                    stage=AnalysisStage.QUEUED,
                    pipeline_version="test",
                    assumption_count=0,
                    financial_warning_count=0,
                    created_at=shared_time,
                    updated_at=shared_time,
                )
            )

    first = repository.list_history(limit=2, cursor=None)
    second = repository.list_history(limit=2, cursor=first.next_cursor)
    received = [item.id for item in first.items + second.items]
    assert set(ids).issubset(received)
    assert len(received) == len(set(received))

    invalid = client.get("/api/v1/analyses?cursor=not-a-cursor")
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "INVALID_CURSOR"


def test_capacity_rejection_does_not_insert_a_new_row(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'capacity.db'}"
    app = create_app(Settings(database_url=database_url, analysis_retry_after_seconds=7))
    with TestClient(app) as client:
        Base.metadata.create_all(app.state.db_engine)
        repository = AnalysisRepository(app.state.db_session_factory)
        app.state.analysis_jobs = AnalysisJobService(
            repository,
            _success_factory,
            pipeline_version="test",
            max_concurrent=1,
            max_queued=0,
        )
        app.state.database_ready = True
        app.state.retrieval_service = object()
        response = client.post("/api/v1/analyses", json=PAYLOAD)
        assert repository.list_history(limit=20, cursor=None).items == []

    assert response.status_code == 503
    assert response.headers["retry-after"] == "7"
    assert response.json()["error"]["code"] == "CAPACITY_EXCEEDED"


def test_terminal_compare_and_set_and_restart_reconciliation(persistent_app):
    client, _database_url = persistent_app
    repository = client.app.state.analysis_jobs.repository
    accepted = repository.create_queued(
        CreateAnalysisRequest(**PAYLOAD),
        pipeline_version="test",
        max_queued=10,
    )
    started = repository.start_if_queued(accepted.id)
    assert started is not None
    assert repository.start_if_queued(accepted.id) is None
    assert repository.reconcile_interrupted() == 1
    detail = repository.to_detail(repository.get(accepted.id))
    assert detail.status == AnalysisStatus.FAILED
    assert detail.error.code == "WORKER_INTERRUPTED"
    assert not repository.fail_if_nonterminal(
        accepted.id,
        code="OTHER",
        message="Other",
        internal_error="Other",
        stage=AnalysisStage.QUEUED,
    )
