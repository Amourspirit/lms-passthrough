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

# The exact payload shape from the TypeSafe System One docs: every question
# type has its own ``criteria`` shape (choice: mapping, score: ordered list).
EXAMPLE: dict[str, Any] = {
    "model": "jev-latest",
    "state": "I was charged twice for my subscription this month. Please refund the extra charge.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": None, "shipping": None, "technical": None},
        },
        "angry": {
            "type": "noul",
            "instructions": "Is the customer angry?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "this week", "today", "right now"],
        },
    },
}

RESPONSE: dict[str, Any] = {
    "id": "shim-1792786975876",
    "model": "jev-latest",
    "answers": {
        "route": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9998, "shipping": 0.0001, "technical": 0.0001},
            "confidence": 0.9996,
        },
        "angry": {"type": "noul", "noul": 0.6371},
        "urgency": {
            "type": "score",
            "score": 2.0878,
            "legend": {"0": "can wait", "1": "this week", "2": "today", "3": "right now"},
            "probabilities": {"0": 0.0023, "1": 0.1155, "2": 0.6743, "3": 0.2079},
            "confidence": 0.6719,
        },
    },
    "usage": {"input_tokens": 239, "output_tokens": 0},
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
    # The public model id is mapped to the upstream id; per-type criteria survive.
    assert sent == {**EXAMPLE, "model": "jev-upstream"}
    assert seen[0].headers["cf-access-client-id"] == "cf-id"
    assert seen[0].headers["cf-access-client-secret"] == "cf-secret"
    assert "authorization" not in seen[0].headers


def test_noul_instructions_are_optional(tmp_path: Path) -> None:
    body = {"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul"}}}
    seen: list[httpx.Request] = []
    with respx.mock() as router:
        router.post(f"{JEV}/v1/systemone").mock(
            side_effect=capture(seen, httpx.Response(200, json=RESPONSE))
        )
        response = TestClient(make_app(tmp_path, SYSTEM1_JEV)).post(PATHS[0], json=body)
    assert response.status_code == 200
    assert json.loads(seen[0].content) == {**body, "model": "jev-upstream"}


def test_assets_and_box_question(tmp_path: Path) -> None:
    body = {
        "model": "jev-latest",
        "state": "photo",
        "questions": {
            "red_rectangle": {"type": "box", "instructions": "Locate the red rectangle."}
        },
        "assets": {
            "photo": {
                "type": "image",
                "format": "png",
                "data_base64": "iVBORw0KGgo=",
            }
        },
    }
    seen: list[httpx.Request] = []
    with respx.mock() as router:
        router.post(f"{JEV}/v1/systemone").mock(
            side_effect=capture(seen, httpx.Response(200, json=RESPONSE))
        )
        response = TestClient(make_app(tmp_path, SYSTEM1_JEV)).post(PATHS[0], json=body)
    assert response.status_code == 200
    assert json.loads(seen[0].content) == {**body, "model": "jev-upstream"}


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
            "questions": {"q": {"type": "choice", "instructions": "?"}},
        },
        {
            **EXAMPLE,
            "questions": {"q": {"type": "score", "instructions": "?", "criteria": "one"}},
        },
        {
            **EXAMPLE,
            "questions": {"q": {"type": "score", "instructions": "?", "criteria": []}},
        },
        {
            **EXAMPLE,
            "questions": {"q": {"type": "noul", "instructions": "?", "extra": 1}},
        },
        {**EXAMPLE, "questions": {}},
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
