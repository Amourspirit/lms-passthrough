"""`/health/ready` as a real signal: probe-driven status, not a hardcoded "ok"."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi.testclient import TestClient

from lms_passthrough import create_app

from .conftest import StubHttp


def make_config(
    tmp_path: Path, *, local: dict[str, Any] | None = None, probe: float = 2.0
) -> Path:
    """Two `openai_compatible` providers: `local` (8080) and `remote` (8081).

    `local` overrides keys on the `local` provider; `probe` is the probe budget.
    """
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "default_provider": "local",
                "readiness_timeout_seconds": probe,
                "persistence": {"path": str(tmp_path / "state.sqlite3")},
                "providers": [
                    {
                        "name": "local",
                        "kind": "openai_compatible",
                        "base_url": "http://127.0.0.1:8080",
                        **(local or {}),
                    },
                    {
                        "name": "remote",
                        "kind": "openai_compatible",
                        "base_url": "http://127.0.0.1:8081",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def ready(client: TestClient) -> Any:
    return client.get("/health/ready")


def test_no_reachable_provider_is_not_ready(tmp_path: Path, stub_http: StubHttp) -> None:
    app = create_app(make_config(tmp_path))
    stub_http(app, {"local": 500, "remote": 500})

    with TestClient(app) as client:
        response = ready(client)

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "providers": {"local": False, "remote": False},
    }


def test_one_reachable_provider_is_ready_and_reports_each(
    tmp_path: Path, stub_http: StubHttp
) -> None:
    app = create_app(make_config(tmp_path))
    stub_http(app, {"local": 404, "remote": 500})

    with TestClient(app) as client:
        response = ready(client)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "providers": {"local": True, "remote": False},
    }


def test_failing_required_provider_is_not_ready(tmp_path: Path, stub_http: StubHttp) -> None:
    app = create_app(make_config(tmp_path, local={"required": True}))
    stub_http(app, {"local": 500, "remote": 200})

    with TestClient(app) as client:
        response = ready(client)

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "providers": {"local": False, "remote": True},
    }


def test_hung_providers_are_probed_concurrently_on_a_short_budget(tmp_path: Path) -> None:
    """A hung upstream costs one `readiness_timeout_seconds` budget, not one per provider.

    Neither `timeout.total_seconds` (600s by default) nor sequential probing is
    acceptable: the first would hang the endpoint for ten minutes, the second
    would multiply that by the number of providers.
    """
    app = create_app(make_config(tmp_path, probe=0.5))
    entered: list[float] = []
    started = time.monotonic()

    async def hang(request: httpx.Request) -> httpx.Response:
        entered.append(time.monotonic() - started)
        await asyncio.sleep(30)
        return httpx.Response(200)

    for name in ("local", "remote"):
        app.state.state.http_clients[name] = httpx.AsyncClient(
            transport=httpx.MockTransport(hang)
        )

    with TestClient(app) as client:
        response = ready(client)
    elapsed = time.monotonic() - started

    assert response.status_code == 503
    # Probes overlap, so both start long before the first one would time out;
    # sequentially the second could not start before 0.5s.
    assert len(entered) == 2, "probes did not run concurrently"
    assert max(entered) < 0.25, "the second probe only started after the first finished"
    assert elapsed < 2.0, "the probe budget is not enforced"


def test_probe_targets_health_path_with_configured_headers(tmp_path: Path) -> None:
    """`health_path` and `extra_headers` make the probe target a protected endpoint.

    The api key is deliberately *not* sent: the header name differs per kind
    (`Bearer` for openai_compatible, `API-KEY` for one_min), so an operator
    needing auth at the probe path puts it in `extra_headers`.
    """
    app = create_app(
        make_config(
            tmp_path,
            local={"health_path": "/v1/models", "extra_headers": {"x-token": "abc"}},
        )
    )
    seen: dict[int, httpx.Request] = {}

    def record(request: httpx.Request) -> httpx.Response:
        seen[request.url.port] = request
        return httpx.Response(404)

    for name in ("local", "remote"):
        app.state.state.http_clients[name] = httpx.AsyncClient(
            transport=httpx.MockTransport(record)
        )

    with TestClient(app) as client:
        assert ready(client).status_code == 200

    probe = seen[8080]
    assert probe.url.path == "/v1/models"
    assert probe.headers["x-token"] == "abc"
    assert "authorization" not in probe.headers


def test_unbuildable_probe_url_is_unready_not_a_500(tmp_path: Path) -> None:
    """A health endpoint that 500s on a bad URL is useless to whatever is deciding
    whether to send it traffic. httpx raises `InvalidURL` — not an `httpx.HTTPError` —
    for a URL it cannot even build, so the probe has to treat any exception as
    "not answered"."""
    app = create_app(make_config(tmp_path, local={"health_path": "/\x00x"}))

    with TestClient(app) as client:
        response = ready(client)

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "providers": {"local": False, "remote": False},
    }
