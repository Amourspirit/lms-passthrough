"""`/api/v1/decisions` and `/v1/decisions`: System 1 (JEV `jev_decide`) passthrough."""

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

JEV = "http://mock-jev:9000"
JEV2 = "http://mock-jev2:9000"
LMS = "http://mock-lms:8999"
JEV_KEY_ENV = "LMS_PT_TEST_JEV_KEY"
PATHS = ["/api/v1/decisions", "/v1/decisions"]

EXAMPLE_1: dict[str, Any] = {
    "model": "openjev",
    "state": "Customer message: I was charged twice for my order last week.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": None, "shipping": None, "technical": None},
        },
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "this week", "today", "right now"],
        },
    },
}

EXAMPLE_2: dict[str, Any] = {
    "model": "openjev",
    "state": {
        "customer_message": "I was charged twice for order ord_7429.",
        "duplicate_charge_usd": 680,
        "customer_identity_verified": True,
        "policy": "Refunds above USD 500 require human approval.",
    },
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "Choose the safest next action.",
            "criteria": {
                "allow": "Issue the refund immediately.",
                "review": "Require human approval before issuing the refund.",
                "deny": "Reject the refund request.",
            },
        },
        "needs_human_review": {
            "type": "noul",
            "instructions": "Does this refund require human review under the stated policy?",
        },
        "risk": {
            "type": "score",
            "instructions": "Score the financial and policy risk.",
            "criteria": ["Low", "Moderate", "High", "Critical"],
        },
    },
}

ENVELOPE: dict[str, Any] = {
    "code": 0,
    "message": "ok",
    "data": {
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
        }
    },
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


SYSTEM1 = f"""  - name: system1
    kind: system1
    base_url: {JEV}
    extra_headers:
      CF-Access-Client-Id: cf-id
      CF-Access-Client-Secret: cf-secret
    models:
      - public: openjev
        upstream: jev-upstream
"""


def capture(seen: list[httpx.Request], response: httpx.Response) -> Any:
    def side_effect(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return side_effect


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("example", [EXAMPLE_1, EXAMPLE_2])
def test_examples_round_trip(tmp_path: Path, path: str, example: dict[str, Any]) -> None:
    seen: list[httpx.Request] = []
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{JEV}/api/v1/decisions").mock(
            side_effect=capture(seen, httpx.Response(200, json=ENVELOPE))
        )
        response = TestClient(make_app(tmp_path, SYSTEM1)).post(path, json=example)
    assert response.status_code == 200
    assert response.json() == ENVELOPE
    assert response.headers["x-lms-provider"] == "system1"
    sent = json.loads(seen[0].content)
    # Null choice criteria survive; the public model id is mapped to upstream.
    assert sent == {**example, "model": "jev-upstream"}
    assert seen[0].headers["cf-access-client-id"] == "cf-id"
    assert seen[0].headers["cf-access-client-secret"] == "cf-secret"
    assert "authorization" not in seen[0].headers


def test_model_is_optional_and_not_invented(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    body = {k: v for k, v in EXAMPLE_1.items() if k != "model"}
    with respx.mock() as router:
        router.post(f"{JEV}/api/v1/decisions").mock(
            side_effect=capture(seen, httpx.Response(200, json=ENVELOPE))
        )
        response = TestClient(make_app(tmp_path, SYSTEM1)).post(PATHS[0], json=body)
    assert response.status_code == 200
    assert "model" not in json.loads(seen[0].content)


def test_api_key_sent_as_bearer(tmp_path: Path) -> None:
    os.environ[JEV_KEY_ENV] = "jev-secret"
    seen: list[httpx.Request] = []
    try:
        with respx.mock() as router:
            router.post(f"{JEV}/api/v1/decisions").mock(
                side_effect=capture(seen, httpx.Response(200, json=ENVELOPE))
            )
            app = make_app(tmp_path, SYSTEM1 + f"    api_key_env: {JEV_KEY_ENV}\n")
            response = TestClient(app).post(PATHS[0], json=EXAMPLE_1)
    finally:
        del os.environ[JEV_KEY_ENV]
    assert response.status_code == 200
    assert seen[0].headers["authorization"] == "Bearer jev-secret"


def test_nonzero_code_passes_through(tmp_path: Path) -> None:
    envelope = {"code": 7, "message": "model busy", "data": {"answers": {}}}
    with respx.mock() as router:
        router.post(f"{JEV}/api/v1/decisions").mock(
            return_value=httpx.Response(200, json=envelope)
        )
        response = TestClient(make_app(tmp_path, SYSTEM1)).post(PATHS[0], json=EXAMPLE_1)
    assert response.status_code == 200
    assert response.json() == envelope


@pytest.mark.parametrize(
    "question",
    [
        {"type": "maybe", "instructions": "?"},
        {"type": "score", "instructions": "?", "criteria": ["only one"]},
        {"type": "choice", "instructions": "?"},
        {"type": "noul"},
        {"type": "noul", "instructions": "?", "extra": 1},
        {"type": "noul", "instructions": "?", "criteria": {"maybe": "x"}},
    ],
)
def test_invalid_question_is_422(tmp_path: Path, question: dict[str, Any]) -> None:
    body = {"state": "s", "questions": {"q": question}}
    with respx.mock(assert_all_called=False) as router:
        route = router.post(f"{JEV}/api/v1/decisions")
        response = TestClient(make_app(tmp_path, SYSTEM1)).post(PATHS[0], json=body)
    assert response.status_code == 422
    assert not route.called


def test_missing_state_and_unknown_field_are_422(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path, SYSTEM1))
    assert client.post(PATHS[0], json={"questions": {}}).status_code == 422
    extra = {**EXAMPLE_1, "stream": True}
    assert client.post(PATHS[0], json=extra).status_code == 422


def test_header_pins_provider(tmp_path: Path) -> None:
    client = TestClient(make_app(tmp_path, SYSTEM1))
    lms = client.post(PATHS[0], json=EXAMPLE_1, headers={"X-LMS-Provider": "lm-studio"})
    assert lms.status_code == 400
    assert "does not support decisions" in lms.json()["error"]["message"]
    unknown = client.post(PATHS[0], json=EXAMPLE_1, headers={"X-LMS-Provider": "nope"})
    assert unknown.status_code == 400


def test_no_decisions_provider_is_400(tmp_path: Path) -> None:
    response = TestClient(make_app(tmp_path, "")).post(PATHS[1], json=EXAMPLE_1)
    assert response.status_code == 400
    assert "decisions" in response.json()["error"]["message"]


def test_upstream_unreachable_is_503(tmp_path: Path) -> None:
    with respx.mock() as router:
        router.post(f"{JEV}/api/v1/decisions").mock(side_effect=httpx.ConnectError("down"))
        response = TestClient(make_app(tmp_path, SYSTEM1)).post(PATHS[0], json=EXAMPLE_1)
    assert response.status_code == 503


def test_upstream_4xx_is_relayed(tmp_path: Path) -> None:
    with respx.mock() as router:
        router.post(f"{JEV}/api/v1/decisions").mock(
            return_value=httpx.Response(400, json={"code": 1, "message": "bad"})
        )
        response = TestClient(make_app(tmp_path, SYSTEM1)).post(PATHS[0], json=EXAMPLE_1)
    assert response.status_code == 400
    assert response.json()["error"]["provider_error"] == {"code": 1, "message": "bad"}


def test_failover_to_second_decisions_provider(tmp_path: Path) -> None:
    second = f"  - name: system1-b\n    kind: system1\n    base_url: {JEV2}\n"
    with respx.mock() as router:
        router.post(f"{JEV}/api/v1/decisions").mock(return_value=httpx.Response(502))
        router.post(f"{JEV2}/api/v1/decisions").mock(
            return_value=httpx.Response(200, json=ENVELOPE)
        )
        app = make_app(tmp_path, SYSTEM1 + second, failover=True)
        response = TestClient(app).post(PATHS[0], json=EXAMPLE_1)
    assert response.status_code == 200
    assert response.headers["x-lms-provider"] == "system1-b"


def test_chat_against_system1_is_400(tmp_path: Path) -> None:
    response = TestClient(make_app(tmp_path, SYSTEM1)).post(
        "/v1/chat/completions",
        json={"model": "openjev", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-LMS-Provider": "system1"},
    )
    assert response.status_code == 400
