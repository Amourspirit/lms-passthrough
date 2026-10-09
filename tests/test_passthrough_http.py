"""Regression tests for passthrough request construction."""

from __future__ import annotations

import httpx
import pytest

from lms_passthrough.provider_api import ChatRequest
from lms_passthrough.providers.lm_studio import LMStudioProvider


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"models": []})


@pytest.mark.asyncio
async def test_get_requests_send_no_json_body() -> None:
    """Regression: GET passthrough must not send a JSON body (Express rejects it)."""
    transport = RecordingTransport()
    client = httpx.AsyncClient(transport=transport)
    provider = LMStudioProvider("lm-studio", "http://127.0.0.1:8999", client)
    await provider.list_models()
    request = transport.requests[0]
    assert request.method == "GET"
    assert request.content in {b"", None}


@pytest.mark.asyncio
async def test_post_chat_sends_raw_payload() -> None:
    transport = RecordingTransport()
    client = httpx.AsyncClient(transport=transport)
    provider = LMStudioProvider("lm-studio", "http://127.0.0.1:8999", client)
    request = ChatRequest(model="m", messages=[], raw={"model": "m", "input": "hi"})
    await provider.chat(request)
    sent = transport.requests[0]
    assert sent.method == "POST"
    assert b'"model":"m"' in sent.content or b'"model": "m"' in sent.content
