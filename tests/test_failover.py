"""Failure-driven failover: down providers are skipped, healthy ones serve the model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lms_passthrough import create_app
from lms_passthrough.logging import JsonFormatter
from lms_passthrough.provider_api import (
    CanonicalOutput,
    ChatResult,
    ProviderHTTPError,
    ProviderUnavailableError,
)

GEMINI = "gemini-3.8-flash"


def make_config(tmp_path: Path, *, failover: bool, cooldown: float = 60.0) -> Path:
    """omlx first, then 1min-chat: the same shape as storage/config/docker-config.yaml."""
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
default_provider: 1min-chat
failover: {str(failover).lower()}
failover_cooldown_seconds: {cooldown}
persistence:
  path: {tmp_path}/state.sqlite3
providers:
  - name: omlx
    kind: openai_compatible
    base_url: http://127.0.0.1:8080
  - name: 1min-chat
    kind: openai_compatible
    base_url: http://127.0.0.1:8081
""".strip(),
        encoding="utf-8",
    )
    return path


async def ok(request: Any) -> ChatResult:
    return ChatResult(model=request.model, outputs=[CanonicalOutput(content="answer")])


def wire(app: Any, *, omlx_models: Any, omlx_chat: Any, chat_models: Any) -> list[dict[str, Any]]:
    """Stub both providers. Returns omlx's list_models call log."""
    calls: list[dict[str, Any]] = []

    async def omlx_list() -> list[dict[str, Any]]:
        calls.append({"provider": "omlx"})
        return await omlx_models()

    app.state.state.providers["omlx"].list_models = omlx_list  # type: ignore[method-assign]
    app.state.state.providers["omlx"].chat = omlx_chat  # type: ignore[method-assign]
    app.state.state.providers["1min-chat"].list_models = chat_models  # type: ignore[method-assign]
    app.state.state.providers["1min-chat"].chat = ok  # type: ignore[method-assign]
    return calls


def models_with(*keys: str) -> Any:
    async def list_models() -> list[dict[str, Any]]:
        return [{"id": key} for key in keys]

    return list_models


def offline() -> Any:
    async def list_models() -> list[dict[str, Any]]:
        raise ProviderUnavailableError("omlx is offline")

    return list_models


def body() -> dict[str, Any]:
    return {"model": GEMINI, "messages": [{"role": "user", "content": "hi"}]}


def test_down_provider_is_skipped_and_healthy_one_serves(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, failover=True))
    calls = wire(app, omlx_models=offline(), omlx_chat=ok, chat_models=models_with(GEMINI))

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body())

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "answer"
    assert response.headers["X-LMS-Provider"] == "1min-chat"
    assert app.state.state.is_down("omlx")
    assert len(calls) == 1


def test_failed_model_probe_is_not_repeated_during_cooldown(tmp_path: Path) -> None:
    """The regression this feature exists for: without the negative cache every
    request re-probes a dead host and pays timeout.total_seconds per call."""
    app = create_app(make_config(tmp_path, failover=True, cooldown=300.0))
    calls = wire(app, omlx_models=offline(), omlx_chat=ok, chat_models=models_with(GEMINI))

    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/v1/chat/completions", json=body()).status_code == 200

    assert len(calls) == 1, "omlx was re-probed while inside its cooldown window"


def test_failover_disabled_keeps_omlx_selected(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, failover=False))
    wire(app, omlx_models=models_with(GEMINI), omlx_chat=ok, chat_models=models_with(GEMINI))

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body())

    assert response.headers["X-LMS-Provider"] == "omlx"
    assert not app.state.state.is_down("omlx")


def test_header_is_absolute_and_never_failed_over(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, failover=True))
    wire(app, omlx_models=models_with(GEMINI), omlx_chat=ok, chat_models=models_with(GEMINI))

    with TestClient(app) as client:
        # omlx is healthy but marked down by an earlier transient failure.
        app.state.state.mark_down("omlx", 300.0)
        response = client.post(
            "/v1/chat/completions", json=body(), headers={"X-LMS-Provider": "omlx"}
        )

    assert response.status_code == 503
    assert "omlx" in response.json()["error"]["message"]


def test_5xx_fails_over_but_4xx_does_not(tmp_path: Path) -> None:
    async def upstream_500(request: Any) -> ChatResult:
        raise ProviderHTTPError(500, "boom")

    async def upstream_400(request: Any) -> ChatResult:
        raise ProviderHTTPError(400, "bad request")

    app = create_app(make_config(tmp_path, failover=True))
    wire(
        app,
        omlx_models=models_with(GEMINI),
        omlx_chat=upstream_500,
        chat_models=models_with(GEMINI),
    )
    with TestClient(app) as client:
        ok_response = client.post("/v1/chat/completions", json=body())
    assert ok_response.status_code == 200
    assert ok_response.headers["X-LMS-Provider"] == "1min-chat"

    (tmp_path / "b").mkdir()
    app = create_app(make_config(tmp_path / "b", failover=True))
    wire(
        app,
        omlx_models=models_with(GEMINI),
        omlx_chat=upstream_400,
        chat_models=models_with(GEMINI),
    )
    with TestClient(app) as client:
        bad_request = client.post("/v1/chat/completions", json=body())

    assert bad_request.status_code == 400
    assert bad_request.headers["X-LMS-Provider"] == "omlx"
    assert not app.state.state.is_down("omlx"), "a 4xx is the request's fault, not the provider's"


def test_no_available_provider_offers_the_model_is_503(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, failover=True))
    wire(app, omlx_models=models_with(GEMINI), omlx_chat=ok, chat_models=models_with("other"))

    with TestClient(app) as client:
        app.state.state.mark_down("omlx", 300.0)
        response = client.post("/v1/chat/completions", json=body())

    assert response.status_code == 503
    assert "omlx" in response.json()["error"]["message"]


def test_catalog_hides_down_providers(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, failover=True))
    wire(app, omlx_models=models_with(GEMINI), omlx_chat=ok, chat_models=models_with("other"))

    with TestClient(app) as client:
        visible = [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert "other" in visible

        app.state.state.mark_down("1min-chat", 300.0)
        hidden = [m["id"] for m in client.get("/v1/models").json()["data"]]
        # A client that explicitly named the down provider still sees it.
        pinned = [
            m["id"]
            for m in client.get(
                "/v1/models", headers={"X-LMS-Provider": "1min-chat"}
            ).json()["data"]
        ]

    assert "other" not in hidden
    assert pinned == ["other"]


@pytest.mark.parametrize("path", ["/api/v1/chat", "/v1/chat/completions", "/v1/responses"])
def test_every_translated_chat_route_fails_over(tmp_path: Path, path: str) -> None:
    app = create_app(make_config(tmp_path, failover=True))
    wire(app, omlx_models=offline(), omlx_chat=ok, chat_models=models_with(GEMINI))
    if path == "/api/v1/chat":
        app.state.state.providers["1min-chat"].responses = ok  # type: ignore[method-assign]

    payload = {"model": GEMINI, "messages": [{"role": "user", "content": "hi"}]}
    if path == "/v1/responses":
        payload = {"model": GEMINI, "input": "hi"}

    with TestClient(app) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 200
    assert response.headers["X-LMS-Provider"] == "1min-chat"


def test_failover_log_field_names_both_providers(tmp_path: Path) -> None:
    """A dispatch-time failover: omlx was selected, tried, and failed mid-request.

    (A provider skipped during routing never reaches `dispatch`, so there is no
    failover to report — that case shows up as the down-state plus the
    `X-LMS-Provider` header naming whoever actually served.)
    """
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    async def upstream_500(request: Any) -> ChatResult:
        raise ProviderHTTPError(503, "boom")

    app = create_app(make_config(tmp_path, failover=True))
    wire(
        app,
        omlx_models=models_with(GEMINI),
        omlx_chat=upstream_500,
        chat_models=models_with(GEMINI),
    )
    logger = logging.getLogger("lms_passthrough.request")
    handler = Capture()
    logger.addHandler(handler)

    try:
        with TestClient(app) as client:
            assert client.post("/v1/chat/completions", json=body()).status_code == 200
    finally:
        logger.removeHandler(handler)

    spans = [json.loads(JsonFormatter().format(r)) for r in records]
    spans = [s for s in spans if s.get("path") == "/v1/chat/completions"]
    assert [s["failover"] for s in spans] == ["omlx->1min-chat"]
    assert [s["provider"] for s in spans] == ["1min-chat"]
