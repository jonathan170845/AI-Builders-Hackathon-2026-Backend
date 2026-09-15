"""Provider-neutral structured LLM client and OpenRouter implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.db.cache import LLMCache

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)
_FENCED_JSON = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """Typed provider or structured-output failure safe for the job boundary."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class LLMCompletion:
    payload: dict[str, Any]
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    cached: bool = False

    def parse(self, response_model: type[ModelT]) -> ModelT:
        try:
            return response_model.model_validate(self.payload)
        except ValidationError as exc:
            raise LLMError(
                "LLM_INVALID_RESPONSE", "Provider returned an invalid structured response"
            ) from exc


class LLMClient(Protocol):
    model: str
    cache_key_inputs: Mapping[str, Any]

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, Any],
        response_model: type[ModelT],
    ) -> LLMCompletion: ...


class OpenRouterLLMClient:
    """OpenAI-compatible OpenRouter adapter with bounded retries and concurrency."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        if not settings.openrouter_api_key or not settings.openrouter_model:
            raise LLMError("LLM_NOT_CONFIGURED", "LLM provider is not configured")
        self.model = settings.openrouter_model
        self._api_key = settings.openrouter_api_key
        self._base_url = settings.openrouter_base_url.rstrip("/")
        self._max_retries = settings.llm_max_retries
        self._max_tokens = settings.llm_max_tokens
        self._max_response_bytes = settings.llm_max_response_bytes
        self._temperature = settings.llm_temperature
        self._response_format = settings.llm_response_format
        self.cache_key_inputs: Mapping[str, Any] = {
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "response_format": self._response_format,
        }
        self._total_timeout = settings.llm_total_timeout_seconds
        self._timeout = httpx.Timeout(
            settings.llm_total_timeout_seconds,
            connect=settings.llm_connect_timeout_seconds,
            read=settings.llm_read_timeout_seconds,
        )
        self._client = http_client or httpx.AsyncClient(timeout=self._timeout)
        self._owns_client = http_client is None
        self._semaphore = semaphore or asyncio.Semaphore(settings.llm_global_concurrency)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, Any],
        response_model: type[ModelT],
    ) -> LLMCompletion:
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                    + "\nReturn exactly ONE populated JSON data object. Do not repeat the schema, "
                    + "include markdown fences, explanations, or multiple objects."
                    + "\n<response_schema>\n"
                    + json.dumps(response_model.model_json_schema(), sort_keys=True)
                    + "\n</response_schema>\nReturn the DATA only, never the schema.",
                },
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._response_format == "json_object":
            request["response_format"] = {"type": "json_object"}
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                async with asyncio.timeout(self._total_timeout), self._semaphore:
                    async with self._client.stream(
                        "POST",
                        f"{self._base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=request,
                        timeout=self._timeout,
                    ) as incoming:
                        content = bytearray()
                        async for chunk in incoming.aiter_bytes():
                            if len(content) + len(chunk) > self._max_response_bytes:
                                raise LLMError(
                                    "LLM_RESPONSE_TOO_LARGE", "Provider response is too large"
                                )
                            content.extend(chunk)
                        response = httpx.Response(
                            incoming.status_code,
                            headers={
                                key: value
                                for key, value in incoming.headers.items()
                                if key.lower() not in {"content-encoding", "content-length"}
                            },
                            content=bytes(content),
                        )
                completion = self._parse_response(response, response_model)
                logger.info(
                    "llm_completion model=%s attempt=%d status=%d latency_ms=%d "
                    "input_tokens=%s output_tokens=%s",
                    self.model,
                    attempt + 1,
                    response.status_code,
                    (time.perf_counter() - started) * 1000,
                    completion.input_tokens,
                    completion.output_tokens,
                )
                return completion
            except (TimeoutError, httpx.TimeoutException, httpx.NetworkError) as exc:
                error = LLMError(
                    "LLM_PROVIDER_UNAVAILABLE", "LLM provider is unavailable", retryable=True
                )
                error.__cause__ = exc
            except LLMError as exc:
                error = exc
            logger.warning(
                "llm_completion_failed model=%s attempt=%d code=%s retryable=%s latency_ms=%d",
                self.model,
                attempt + 1,
                error.code,
                error.retryable,
                (time.perf_counter() - started) * 1000,
            )
            if not error.retryable or attempt == self._max_retries:
                raise error
            await asyncio.sleep((0.25 * (2**attempt)) + random.uniform(0, 0.1))
        raise AssertionError("unreachable")

    def _parse_response(
        self, response: httpx.Response, response_model: type[ModelT]
    ) -> LLMCompletion:
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > self._max_response_bytes
            except ValueError:
                too_large = False
            if too_large:
                raise LLMError(
                    "LLM_RESPONSE_TOO_LARGE", "LLM response exceeds the configured size limit"
                )
        if response.status_code == 429 or response.status_code >= 500:
            raise LLMError(
                "LLM_PROVIDER_UNAVAILABLE", "LLM provider is unavailable", retryable=True
            )
        if response.status_code >= 400:
            raise LLMError("LLM_PROVIDER_REJECTED", "LLM provider rejected the request")
        if len(response.content) > self._max_response_bytes:
            raise LLMError(
                "LLM_RESPONSE_TOO_LARGE", "LLM response exceeds the configured size limit"
            )
        try:
            body = response.json()
            if body["choices"][0].get("finish_reason") == "length":
                raise LLMError("LLM_OUTPUT_TRUNCATED", "Provider exhausted its output token budget")
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("content is not text")
            payload = _parse_json_content(content)
            response_model.model_validate(payload)
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            raise LLMError(
                "LLM_INVALID_RESPONSE", "Provider returned an invalid structured response"
            ) from exc
        usage = body.get("usage") if isinstance(body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        return LLMCompletion(
            payload=payload,
            model=self.model,
            input_tokens=_optional_int(usage.get("prompt_tokens")),
            output_tokens=_optional_int(usage.get("completion_tokens")),
            cost_usd=_optional_float(usage.get("cost")),
        )


class CachedLLMClient:
    """Caches only exact structured calls; the key includes every behavior-affecting input."""

    def __init__(self, client: LLMClient, cache: LLMCache | None, *, prompt_version: str) -> None:
        self._client = client
        self._cache = cache
        self._prompt_version = prompt_version
        self.model = client.model

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, Any],
        response_model: type[ModelT],
    ) -> LLMCompletion:
        key = cache_key(
            model=self.model,
            prompt_version=self._prompt_version,
            system_prompt=system_prompt,
            user_payload=user_payload,
            sampling=getattr(self._client, "cache_key_inputs", {}),
            response_schema=json.dumps(response_model.model_json_schema(), sort_keys=True),
        )
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                logger.info("llm_cache_hit")
                completion = LLMCompletion(payload=cached, model=self.model, cached=True)
                completion.parse(response_model)
                return completion
        logger.info("llm_cache_miss")
        completion = await self._client.complete_json(
            system_prompt=system_prompt, user_payload=user_payload, response_model=response_model
        )
        if self._cache is not None:
            self._cache.put(
                key,
                model=self.model,
                prompt_version=self._prompt_version,
                response=completion.payload,
            )
        return completion


def cache_key(
    *,
    model: str,
    prompt_version: str,
    system_prompt: str,
    user_payload: Mapping[str, Any],
    sampling: Mapping[str, Any] | None = None,
    response_schema: str | None = None,
) -> str:
    canonical = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "system_prompt": system_prompt,
            "user_payload": user_payload,
            "sampling": sampling or {},
            "response_schema": response_schema,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _parse_json_content(content: str) -> dict[str, Any]:
    matched = _FENCED_JSON.match(content)
    if matched:
        content = matched.group(1)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("JSON response must be an object")
    return parsed


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )
