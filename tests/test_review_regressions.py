import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import httpx
import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.db.base import Base
from app.db.cache import LLMCache
from app.db.session import create_engine_and_session_factory
from app.schemas.analysis import CreateAnalysisRequest, FinancialInputs
from app.services.financial import run_financial_stress_test
from app.services.llm import LLMError, OpenRouterLLMClient


class Output(BaseModel):
    required_value: str


def test_chunked_body_without_length_is_limited(client):
    response = client.post(
        "/api/v1/analyses",
        content=iter([b" " * 40000, b" " * 40000]),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_request_id_is_bounded_and_sanitized(client):
    response = client.get("/health/live", headers={"X-Request-ID": "x" * 1000})
    UUID(response.headers["x-request-id"])


def test_create_refuses_missing_artifacts_but_history_remains_available(client):
    client.app.state.retrieval_service = None
    response = client.post("/api/v1/analyses", json={"decision": "Expand to another city"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ARTIFACTS_NOT_READY"
    assert client.get("/api/v1/analyses").status_code == 200


def test_valid_maximum_financial_input_serializes():
    values = dict(
        monthlyOrders=10**12,
        revenuePerOrder=10**15,
        variableCostPerOrder=0,
        promoSubsidy=0,
        deliveryCost=0,
        fixedCost=0,
        driverCost=0,
        cashBalance=0,
    )
    result = run_financial_stress_test(FinancialInputs.model_validate(values)).to_api_result()
    assert result["contribution_margin"] == 1e27
    json.dumps(result, allow_nan=False)


def test_restart_reconciles_queued_jobs(client):
    repo = client.app.state.analysis_jobs.repository
    row = repo.create_queued(
        CreateAnalysisRequest(decision="Expand to another city"),
        pipeline_version="test",
        max_queued=10,
    )
    assert repo.reconcile_interrupted() == 1
    assert repo.to_detail(repo.get(row.id)).error.code == "WORKER_INTERRUPTED"


def test_provider_receives_schema():
    def handler(request):
        payload = json.loads(request.content)
        assert "required_value" in payload["messages"][0]["content"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"required_value":"ok"}'}}]}
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterLLMClient(
                Settings(openrouter_api_key="test", openrouter_model="test"), http_client=transport
            )
            result = await client.complete_json(
                system_prompt="JSON only", user_payload={}, response_model=Output
            )
            assert result.parse(Output).required_value == "ok"

    asyncio.run(scenario())


def test_streamed_provider_response_has_a_real_size_limit():
    class LargeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(10):
                yield b" " * 1024

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=LargeStream()))
        ) as transport:
            client = OpenRouterLLMClient(
                Settings(
                    openrouter_api_key="test", openrouter_model="test", llm_max_response_bytes=1024
                ),
                http_client=transport,
            )
            with pytest.raises(LLMError) as failure:
                await client.complete_json(
                    system_prompt="JSON only", user_payload={}, response_model=Output
                )
            assert failure.value.code == "LLM_RESPONSE_TOO_LARGE"

    asyncio.run(scenario())


def test_cache_concurrent_same_key_is_safe(tmp_path):
    engine, sessions = create_engine_and_session_factory(f"sqlite:///{tmp_path / 'concurrent.db'}")
    Base.metadata.create_all(engine)
    cache = LLMCache(sessions, retention_days=1, max_rows=10)
    try:

        def put(i):
            cache.put("same", model="test", prompt_version="1", response={"value": i})

        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(put, range(24)))
        assert cache.get("same") is not None
    finally:
        engine.dispose()


def test_model_identity_changes_with_tokenizer(tmp_path):
    from app.services.model_identity import local_model_fingerprint

    (tmp_path / "modules.json").write_text("[]")
    (tmp_path / "tokenizer.json").write_text("old")
    previous = local_model_fingerprint(tmp_path)
    (tmp_path / "tokenizer.json").write_text("new")
    assert local_model_fingerprint(tmp_path) != previous


def test_provider_wall_clock_timeout_is_enforced():
    async def handler(_request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterLLMClient(
                Settings(
                    openrouter_api_key="test",
                    openrouter_model="test",
                    llm_total_timeout_seconds=0.01,
                    llm_max_retries=0,
                ),
                http_client=transport,
            )
            with pytest.raises(LLMError) as failure:
                await client.complete_json(
                    system_prompt="JSON", user_payload={}, response_model=Output
                )
            assert failure.value.code == "LLM_PROVIDER_UNAVAILABLE"

    asyncio.run(scenario())


def test_job_timeout_reaches_terminal_state(client):
    from app.services.jobs import AnalysisJobService

    class SlowPipeline:
        async def run(self, *_args):
            await asyncio.sleep(1)

    repo = client.app.state.analysis_jobs.repository
    jobs = AnalysisJobService(
        repo,
        lambda _: SlowPipeline(),
        pipeline_version="test",
        max_concurrent=1,
        max_queued=10,
        timeout_seconds=0.01,
    )
    accepted = jobs.submit(CreateAnalysisRequest(decision="Expand to another city"))
    asyncio.run(jobs.run(accepted.id))
    detail = repo.to_detail(repo.get(accepted.id))
    assert detail.status == "failed"
    assert detail.error.code == "ANALYSIS_TIMEOUT"


def test_text_mode_omits_unsupported_provider_option_but_validates_schema():
    def handler(request):
        payload = json.loads(request.content)
        assert "response_format" not in payload
        assert "required_value" in payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '```json\n{"required_value":"ok"}\n```'}}]},
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterLLMClient(
                Settings(
                    openrouter_api_key="test", openrouter_model="test", llm_response_format="text"
                ),
                http_client=transport,
            )
            result = await client.complete_json(
                system_prompt="JSON", user_payload={}, response_model=Output
            )
            assert result.parse(Output).required_value == "ok"

    asyncio.run(scenario())


def test_compressed_provider_body_is_decoded_only_once():
    import gzip

    payload = json.dumps(
        {"choices": [{"message": {"content": '{"required_value":"ok"}'}}]}
    ).encode()

    def handler(_request):
        return httpx.Response(
            200, headers={"content-encoding": "gzip"}, content=gzip.compress(payload)
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterLLMClient(
                Settings(openrouter_api_key="test", openrouter_model="test"), http_client=transport
            )
            result = await client.complete_json(
                system_prompt="JSON", user_payload={}, response_model=Output
            )
            assert result.parse(Output).required_value == "ok"

    asyncio.run(scenario())


def test_truncated_output_is_a_typed_error_even_if_json_parses():
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "length", "message": {"content": '{"required_value":"ok"}'}}
                ]
            },
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = OpenRouterLLMClient(
                Settings(openrouter_api_key="test", openrouter_model="test"), http_client=transport
            )
            with pytest.raises(LLMError) as failure:
                await client.complete_json(
                    system_prompt="JSON", user_payload={}, response_model=Output
                )
            assert failure.value.code == "LLM_OUTPUT_TRUNCATED"

    asyncio.run(scenario())
