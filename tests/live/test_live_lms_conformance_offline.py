"""Offline LM Studio conformance tests.

These run on every `pytest` invocation with no external server: the proxy
(transparent lm-studio passthrough) is exercised against a respx-mocked
upstream. They pin the protocol shapes the live suite asserts against a real
server: model catalog byte-equivalence, native error envelope shape, and
native SSE lifecycle ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ._mock import (
    FIXTURES_DIR,
    MODEL_CATALOG,
    build_proxy_app,
    model_keys,
    parse_sse,
    register_mock_upstream,
)

MOCK_BASE = "http://mock-upstream:8999"


def test_models_byte_equivalent_through_proxy(tmp_path: Path) -> None:
    """Byte-equivalence: proxy output for /api/v1/models must equal the upstream body."""
    app = build_proxy_app(tmp_path, MOCK_BASE)
    body = json.dumps(MODEL_CATALOG).encode("utf-8")
    with register_mock_upstream(base_url=MOCK_BASE) as router:
        direct = httpx.get(f"{MOCK_BASE}/api/v1/models")
        assert direct.content == body

        client = TestClient(app)
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        assert response.content == body
        assert router.get("/api/v1/models").call_count == 2


def test_chat_byte_equivalent_through_proxy(tmp_path: Path) -> None:
    """Non-streaming /api/v1/chat is forwarded byte-for-byte (no re-serialization)."""
    app = build_proxy_app(tmp_path, MOCK_BASE)
    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    with register_mock_upstream(base_url=MOCK_BASE):
        direct = httpx.post(f"{MOCK_BASE}/api/v1/chat", json=payload)
        assert direct.status_code == 200

        client = TestClient(app)
        response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    assert response.content == direct.content


def test_unknown_model_error_envelope(tmp_path: Path) -> None:
    """Native error envelope shape is forwarded for an unknown model."""
    app = build_proxy_app(tmp_path, MOCK_BASE)
    with register_mock_upstream(base_url=MOCK_BASE):
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400
    payload = response.json()
    assert set(payload) == {"error"}
    error = payload["error"]
    assert "message" in error
    assert error["type"] == "invalid_request"


def test_invalid_json_error_envelope(tmp_path: Path) -> None:
    """Invalid JSON yields a native error envelope without reaching upstream."""
    app = build_proxy_app(tmp_path, MOCK_BASE)
    with register_mock_upstream(base_url=MOCK_BASE) as router:
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            content=b"{not-valid-json",
            headers={"content-type": "application/json"},
        )
        chat_calls = router.post("/api/v1/chat").call_count
    assert response.status_code == 400
    assert chat_calls == 0
    payload = response.json()
    assert "error" in payload
    assert "message" in payload["error"]
    assert payload["error"]["type"] == "invalid_request"


def test_native_sse_lifecycle_ordering(tmp_path: Path) -> None:
    """Streamed chat through the proxy preserves native SSE lifecycle ordering."""
    app = build_proxy_app(tmp_path, MOCK_BASE)
    with register_mock_upstream(base_url=MOCK_BASE):
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = parse_sse(response.text)
    lifecycle = [f[0] for f in frames]
    assert "chat.start" in lifecycle
    assert "chat.end" in lifecycle
    assert lifecycle.count("chat.end") == 1
    assert lifecycle[0] == "chat.start"
    assert lifecycle[-1] == "chat.end"
    assert "message.start" in lifecycle
    assert "message.end" in lifecycle
    assert lifecycle.count("message.delta") >= 1
    assert lifecycle.index("chat.start") < lifecycle.index("message.start")
    assert lifecycle.index("message.start") < lifecycle.index("message.delta")
    assert lifecycle.index("message.end") < lifecycle.index("chat.end")


def test_captured_fixture_models_byte_equivalent(tmp_path: Path) -> None:
    """Replay a captured live fixture (if any) through the proxy byte-for-byte."""
    fixture = FIXTURES_DIR / "models.json"
    if not fixture.exists():
        pytest.skip("no captured live fixture; run pytest --live-lms <url> to generate one")

    body = fixture.read_bytes()
    catalog = json.loads(body)
    assert "models" in catalog

    app = build_proxy_app(tmp_path, MOCK_BASE)
    with register_mock_upstream(base_url=MOCK_BASE, catalog=catalog):
        client = TestClient(app)
        response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert response.content == body
    assert model_keys(catalog)
