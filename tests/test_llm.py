from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.db.base import Base
from app.db.cache import LLMCache
from app.db.session import create_engine_and_session_factory
from app.services.llm import (
    CachedLLMClient,
    LLMCompletion,
    LLMError,
    OpenRouterLLMClient,
    cache_key,
)


class FakeLLM:
    model = "fake/test-model"

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls = 0

    async def complete_json(self, **_: Any) -> LLMCompletion:
        self.calls += 1
        return LLMCompletion(payload=self.outputs.pop(0), model=self.model)


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_cached_client_uses_only_exact_key(tmp_path: Path):
    engine, sessions = create_engine_and_session_factory(f"sqlite:///{tmp_path / 'cache.db'}")
    Base.metadata.create_all(engine)
    try:
        fake = FakeLLM([{"value": "ok"}])
        cache = LLMCache(sessions, retention_days=1, max_rows=10)
        client = CachedLLMClient(fake, cache, prompt_version="test-v1")

        first = run(
            client.complete_json(
                system_prompt="system",
                user_payload={"decision": "one"},
                response_model=ResponseModel,
            )
        )
        second = run(
            client.complete_json(
                system_prompt="system",
                user_payload={"decision": "one"},
                response_model=ResponseModel,
            )
        )

        assert fake.calls == 1
        assert first.cached is False
        assert second.cached is True
    finally:
        engine.dispose()


def test_cache_key_changes_when_temperature_changes():
    common = {
        "model": "provider/model",
        "prompt_version": "v1",
        "system_prompt": "system",
        "user_payload": {"decision": "one"},
    }

    assert cache_key(**common, sampling={"temperature": 0}) != cache_key(
        **common, sampling={"temperature": 0.5}
    )


def test_openrouter_retries_transient_response_and_validates_json():
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "busy"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"value":"ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.01},
            },
        )

    client = OpenRouterLLMClient(
        Settings(openrouter_api_key="secret", openrouter_model="provider/model", llm_max_retries=1),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    completion = run(
        client.complete_json(
            system_prompt="system", user_payload={"input": "test"}, response_model=ResponseModel
        )
    )

    assert attempts == 2
    assert completion.parse(ResponseModel).value == "ok"


def test_openrouter_does_not_retry_invalid_structured_response():
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    client = OpenRouterLLMClient(
        Settings(openrouter_api_key="secret", openrouter_model="provider/model", llm_max_retries=2),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(LLMError, match="invalid structured"):
        run(
            client.complete_json(
                system_prompt="system", user_payload={"input": "test"}, response_model=ResponseModel
            )
        )
    assert attempts == 1


class ResponseModel(BaseModel):
    value: str
