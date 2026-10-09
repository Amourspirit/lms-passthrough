"""Live LM Studio conformance suite (opt-in).

Only runs when the `--live-lms <url>` pytest option is provided. It exercises
the proxy's transparent passthrough against a real LM Studio server and
asserts protocol shape by presence, never exact wording. It also captures
sanitized fixtures that the offline suite replays (see
tests/live/test_live_lms_conformance_offline.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ._mock import (
    FIXTURES_DIR,
    build_proxy_app,
    live_chat_payload,
    model_keys,
    parse_sse,
)

pytestmark = pytest.mark.live


def test_live_error_envelope_unknown_model(
    tmp_path: Path,
    require_live_lms: str,
) -> None:
    """Unknown model yields the native error envelope shape through the proxy."""
    app = build_proxy_app(tmp_path, require_live_lms)
    content = (
        '{"model": "definitely-not-a-real-model", '
        '"messages": [{"role": "user", "content": "hi"}]}'
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        content=content,
        headers={"content-type": "application/json"},
    )
    assert response.status_code >= 400
    payload = response.json()
    assert "error" in payload
    error = payload["error"]
    assert "message" in error
    assert isinstance(error["message"], str)
    assert "type" in error
    assert isinstance(error["type"], str)


def test_live_error_envelope_invalid_json(tmp_path: Path, require_live_lms: str) -> None:
    """Invalid JSON yields the native error envelope shape through the proxy."""
    app = build_proxy_app(tmp_path, require_live_lms)
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        content=b"{not-valid-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    payload = response.json()
    assert "error" in payload
    assert "message" in payload["error"]
    assert payload["error"]["type"] == "invalid_request"


@pytest.fixture
def live_catalog(tmp_path: Path, require_live_lms: str) -> dict[str, object]:
    """The live server's /api/v1/models body via the proxy (and captured as a fixture)."""
    app = build_proxy_app(tmp_path, require_live_lms)
    client = TestClient(app)
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    catalog = json.loads(response.content)
    assert "models" in catalog
    # Store the catalog in canonical json.dumps form so the offline replay test
    # can compare byte-for-byte regardless of the upstream's original formatting.
    body = json.dumps(catalog, ensure_ascii=False).encode("utf-8")
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "models.json").write_bytes(body)
    return catalog


def _stable_catalog(catalog: dict[str, object]) -> dict[str, object]:
    """Catalog minus server-owned volatile fields (loaded-instance TTL ticks each second)."""

    def scrub(node: object) -> object:
        if isinstance(node, dict):
            return {k: scrub(v) for k, v in node.items() if k != "remaining_ttl_seconds"}
        if isinstance(node, list):
            return [scrub(v) for v in node]
        return node

    result = scrub(catalog)
    assert isinstance(result, dict)
    return result


def test_live_models_byte_equivalent(
    tmp_path: Path,
    require_live_lms: str,
    live_catalog: dict[str, object],
) -> None:
    """Model catalog is identical through the proxy vs direct.

    The server itself changes a loaded-instance TTL on every response, so the
    two live snapshots are compared with that ticking field stripped.
    """
    app = build_proxy_app(tmp_path, require_live_lms)
    direct = httpx.get(f"{require_live_lms}/api/v1/models")
    assert direct.status_code == 200
    client = TestClient(app)
    proxy = client.get("/api/v1/models")
    assert proxy.status_code == 200
    assert _stable_catalog(direct.json()) == _stable_catalog(proxy.json())


def test_live_native_sse_lifecycle_ordering(
    tmp_path: Path,
    require_live_lms: str,
    live_catalog: dict[str, object],
) -> None:
    """Native SSE lifecycle ordering for a real streamed chat."""
    keys = model_keys(live_catalog)
    if not keys:
        pytest.skip("live server reported no model ids")

    app = build_proxy_app(tmp_path, require_live_lms)
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json=live_chat_payload(require_live_lms, keys[0]),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = parse_sse(response.text)
    lifecycle = [f[0] for f in frames]
    assert "chat.start" in lifecycle
    assert "chat.end" in lifecycle
    assert lifecycle[0] == "chat.start"
    assert lifecycle[-1] == "chat.end"
    assert any(name == "message.delta" for name in lifecycle)
