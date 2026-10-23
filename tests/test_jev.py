"""`/api/v1/systemone` and `/v1/systemone`: System 1 JEV passthrough."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from lms_passthrough import create_app
from lms_passthrough.provider_api import Capability

JEV = "http://mock-jev:9000"
JEV2 = "http://mock-jev2:9000"
LMS = "http://mock-lms:8999"
JEV_KEY_ENV = "LMS_PT_TEST_JEV_KEY"
PATHS = ["/api/v1/systemone", "/v1/systemone"]

EXAMPLE: dict[str, Any] = {
    "model": "jev-latest",
    "state": "I was charged twice for my subscription this month. Please refund the extra charge.",
    "questions": {
        "billing": {
            "type": "noul",
            "instructions": "Is this message about billing?",
        },
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"true": "billing", "false": "other"},
        },
    },
    "assets": {"image_1": {"url": "https://example.com/receipt.png"}},
}

RESPONSE: dict[str, Any] = {
    "model": "jev-latest",
    "answers": {
        "billing": {"type": "noul", "noul": 0.98},
        "route": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.91,
            "probabilities": {"billing": 0.91, "other": 0.09},
        },
    },
    "usage": {"input_tokens": 120, "output_tokens": 12},
}


def make_app(tmp_path: Path, providers: str, *, failover: bool = False) -> Any:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"default_provider: lm-studio\nfailover: {str(failover).lower()}\n"
        f"persistence:\n  path: {tmp_path}/state.sqlite3\n"
        f"providers:\n  - name: lm-studio\n    kind: lm_studio\n    base_url: {LMS}\n"
        + providers,
        encoding="utf-8",
    )
    return create_app(config)


SYSTEM1_JEV = f"""  - name: system1-jev
    kind: system1_jev
    base_url: {JEV}
    extra_headers:
      CF-Access-Client-Id: cf-id
      CF-Access-Client-Secret: cf-secret
    models:
      - public: jev-latest
        upstream: jev-upstream
"""


def capture(seen: list[httpx.Request], response: httpx.Response) -> Any:
    def side_effect(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return side_effect


@pytest.mark.parametrize("path", PATHS)
def test_example_round_trip(tmp_path: Path, path: str) -> None:
    seen: list[httpx.Request] = []
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{JEV}/v1/systemone").mock(
            side_effect=capture(seen, httpx.Response(200, json=RESPONSE))
        )
        response = TestClient(make_app(tmp_path, SYSTEM1_JEV)).post(path, json=EXAMPLE)
    assert response.status_code == 200
    assert response.json() == RESPONSE
    assert response.headers["x-lms-provider"] == "system1-jev"
    sent = json.loads(seen[0].content)
    # The public model id is mapped to the upstream id; assets survive.
    assert sent == {**EXAMPLE, "model": "jev-upstream"}
    assert seen[0].headers["cf-access-client-id"] == "cf-id"
    assert seen[0].headers["cf-access-client-secret"] == "cf-secret"
    assert "authorization" not in seen[0].headers


def test_api_key_sent_as_bearer(tmp_path: Path) -> None:
    os.environ[JEV_KEY_ENV] = "jev-secret"
    seen: list[httpx.Request] = []
    try:
        with respx.mock() as router:
            router.post(f"{JEV}/v1/systemone").mock(
                side_effect=capture(seen, httpx.Response(200, json=RESPONSE))
            )
            app = make_app(tmp_path, SYSTEM1_JEV + f"    api_key_env: {JEV_KEY_ENV}\n")
            response = TestClient(app).post(PATHS[0], json=EXAMPLE)
    finally:
        del os.environ[JEV_KEY_ENV]
    assert response.status_code == 200
    assert seen[0].headers["authorization"] == "Bearer jev-secret"


@pytest.mark.parametrize(
    "body",
    [
        {**EXAMPLE, "model": None},
        {k: v for k, v in EXAMPLE.items() if k != "state"},
        {k: v for k, v in EXAMPLE.items() if k != "questions"},
        {**EXAMPLE, "stream": True},
        {
            **EXAMPLE,
            "questions": {"q": {"type": "maybe", "instructions": "?"}},
        },
        {
            **EXAMPLE,
            "questions": {"q": {"type": "noul"}},
        },
        {
            **EXAMPLE,
            "questions": {"q": {"type": "noul", "instructions": "?", "extra": 1}},
        },
    ],
)
def test_invalid_request_is_422(tmp_path: Path, body: dict[str, Any]) -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.post(f"{JEV}/v1/systemone")
        response = TestClient(make_app(tmp_path, SYSTEM1_JEV)).post(PATHS[0], json=body)
    assert response.status_code == 422
    assert not route.called


def test_header_pins_provider(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path, SYSTEM1_JEV))
    lms = client.post(PATHS[0], json=EXAMPLE, headers={"X-LMS-Provider": "lm-studio"})
    assert lms.status_code == 400
    assert "does not support JEV" in lms.json()["error"]["message"]
    unknown = client.post(PATHS[0], json=EXAMPLE, headers={"X-LMS-Provider": "nope"})
    assert unknown.status_code == 400


def test_no_jev_provider_is_400(tmp_path: Path) -> None:
    response = TestClient(make_app(tmp_path, "")).post(PATHS[1], json=EXAMPLE)
    assert response.status_code == 400
    assert "JEV" in response.json()["error"]["message"]


def test_upstream_unreachable_is_503(tmp_path: Path) -> None:
    with respx.mock() as router:
        router.post(f"{JEV}/v1/systemone").mock(side_effect=httpx.ConnectError("down"))
        response = TestClient(make_app(tmp_path, SYSTEM1_JEV)).post(PATHS[0], json=EXAMPLE)
    assert response.status_code == 503


def test_upstream_4xx_is_relayed(tmp_path: Path) -> None:
    with respx.mock() as router:
        router.post(f"{JEV}/v1/systemone").mock(
            return_value=httpx.Response(400, json={"message": "bad"})
        )
        response = TestClient(make_app(tmp_path, SYSTEM1_JEV)).post(PATHS[0], json=EXAMPLE)
    assert response.status_code == 400
    assert response.json()["error"]["provider_error"] == {"message": "bad"}


def test_failover_to_second_jev_provider(tmp_path: Path) -> None:
    second = f"  - name: system1-jev-b\n    kind: system1_jev\n    base_url: {JEV2}\n"
    with respx.mock() as router:
        router.post(f"{JEV}/v1/systemone").mock(return_value=httpx.Response(502))
        router.post(f"{JEV2}/v1/systemone").mock(
            return_value=httpx.Response(200, json=RESPONSE)
        )
        app = make_app(tmp_path, SYSTEM1_JEV + second, failover=True)
        response = TestClient(app).post(PATHS[0], json=EXAMPLE)
    assert response.status_code == 200
    assert response.headers["x-lms-provider"] == "system1-jev-b"


def test_chat_against_system1_jev_is_400(tmp_path: Path) -> None:
    response = TestClient(make_app(tmp_path, SYSTEM1_JEV)).post(
        "/v1/chat/completions",
        json={"model": "jev-latest", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-LMS-Provider": "system1-jev"},
    )
    assert response.status_code == 400


def test_provider_declares_only_jev(tmp_path: Path) -> None:
    app = make_app(tmp_path, SYSTEM1_JEV)
    provider = app.state.state.providers["system1-jev"]
    assert provider.info.capabilities == {Capability.JEV}
