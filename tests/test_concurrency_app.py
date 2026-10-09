from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from lms_passthrough import create_app
from lms_passthrough.provider_api import CanonicalOutput, ChatResult, StreamEvent

from .conftest import StubHttp


def make_config(tmp_path: Path, *, max_concurrent: int | None, queue_timeout: float) -> Path:
    path = tmp_path / "config.yaml"
    concurrent = f"    max_concurrent: {max_concurrent}\n" if max_concurrent is not None else ""
    content = (
        "default_provider: android\n"
        f"persistence:\n  path: {tmp_path}/state.sqlite3\n"
        "providers:\n"
        "  - name: android\n"
        "    kind: openai_compatible\n"
        "    base_url: http://127.0.0.1:8080\n"
        f"{concurrent}    queue_timeout_seconds: {queue_timeout}\n"
        "    models:\n"
        "      - public: m\n"
        "        upstream: local-model\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


async def slow_chat_result(request) -> ChatResult:
    await asyncio.sleep(0.3)
    return ChatResult(model=request.model, outputs=[CanonicalOutput(content="answer")])


async def slow_stream_chat(request):
    await asyncio.sleep(0.3)
    yield StreamEvent(event="chat.start", data={})
    yield StreamEvent(event="message.delta", data={"content": "hello"})
    yield StreamEvent(event="chat.end", data={"result": {}})


@pytest.mark.asyncio
async def test_saturated_provider_overloads_with_503(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, max_concurrent=1, queue_timeout=0.05))
    provider = app.state.state.providers["android"]
    provider.chat = slow_chat_result  # type: ignore[method-assign]

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        async def call() -> httpx.Response:
            return await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-LMS-Provider": "android"},
            )

        results = await asyncio.gather(*[call(), call()])

    statuses = sorted(r.status_code for r in results)
    assert statuses == [200, 503]
    overloaded = next(r for r in results if r.status_code == 503)
    body = overloaded.json()
    assert body["error"]["message"] == "Provider overloaded"
    assert body["error"]["type"] == "overloaded"


@pytest.mark.asyncio
async def test_models_discovery_bypasses_limit(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, max_concurrent=1, queue_timeout=0.05))
    provider = app.state.state.providers["android"]
    provider.chat = slow_chat_result  # type: ignore[method-assign]

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        async def overload_chat() -> httpx.Response:
            return await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-LMS-Provider": "android"},
            )

        futures = [overload_chat(), overload_chat(), client.get("/v1/models")]
        results = await asyncio.gather(*futures)

    statuses = sorted(r.status_code for r in results)
    assert 200 in statuses and 503 in statuses
    models_response = next(r for r in results if r.request.url.path == "/v1/models")
    assert models_response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoints_bypass_limit(
    tmp_path: Path, stub_http: StubHttp
) -> None:
    app = create_app(make_config(tmp_path, max_concurrent=1, queue_timeout=0.05))
    provider = app.state.state.providers["android"]
    provider.chat = slow_chat_result  # type: ignore[method-assign]
    # Readiness probes upstream, so the 200 below needs an upstream to reach.
    stub_http(app, {"android": 200})

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        async def overload_chat() -> httpx.Response:
            return await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-LMS-Provider": "android"},
            )

        futures = [
            overload_chat(),
            overload_chat(),
            client.get("/health/live"),
            client.get("/health/ready"),
        ]
        results = await asyncio.gather(*futures)

    health_live = next(r for r in results if r.request.url.path == "/health/live")
    health_ready = next(r for r in results if r.request.url.path == "/health/ready")
    assert health_live.status_code == 200
    assert health_ready.status_code == 200


@pytest.mark.asyncio
async def test_streaming_overload_returns_503_not_mid_stream(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, max_concurrent=1, queue_timeout=0.05))
    provider = app.state.state.providers["android"]
    provider.chat = slow_chat_result  # type: ignore[method-assign]
    provider.stream_chat = slow_stream_chat  # type: ignore[method-assign]

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        async def stream() -> httpx.Response:
            return await client.post(
                "/api/v1/chat",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                headers={"X-LMS-Provider": "android"},
            )

        results = await asyncio.gather(*[stream(), stream(), stream()])

    overloaded = [r for r in results if r.status_code == 503]
    assert len(overloaded) >= 1
    assert overloaded[0].json()["error"]["type"] == "overloaded"
    ok = [r for r in results if r.status_code == 200]
    assert ok
    assert "chat.start" in ok[0].text
