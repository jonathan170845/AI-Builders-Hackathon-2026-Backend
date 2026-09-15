"""Bound request bodies independently of client-provided Content-Length."""

import re
import time
from uuid import uuid4

from starlette.responses import JSONResponse

from app.core.logging import log_event, request_id_context


class RequestBoundary:
    def __init__(self, app, max_bytes=65536):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        candidate = headers.get(b"x-request-id", b"").decode("latin1")
        request_id = candidate if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", candidate) else str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_context.set(request_id)
        started = time.perf_counter()

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    (k, v) for k, v in message["headers"] if k.lower() != b"x-request-id"
                ]
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        async def too_large():
            response = JSONResponse(
                {
                    "error": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": "Request body exceeds the allowed size",
                        "requestId": request_id,
                    }
                },
                status_code=413,
            )
            await response(scope, receive, send_with_id)

        try:
            try:
                if int(headers.get(b"content-length", b"0")) > self.max_bytes:
                    return await too_large()
            except ValueError:
                pass
            chunks = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunks.extend(message.get("body", b""))
                if len(chunks) > self.max_bytes:
                    return await too_large()
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(chunks), "more_body": False}
                return await receive()

            await self.app(scope, replay, send_with_id)
        finally:
            log_event("http_request", duration_ms=round((time.perf_counter() - started) * 1000))
            request_id_context.reset(token)
