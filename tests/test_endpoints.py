from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lms_passthrough import create_app
from lms_passthrough.provider_api import CanonicalOutput, ChatResult, ProviderInfo


def make_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
default_provider: android
persistence:
  path: {tmp_path}/state.sqlite3
providers:
  - name: android
    kind: openai_compatible
    base_url: http://127.0.0.1:8080
    models:
      - public: m
        upstream: local-model
""".strip(),
        encoding="utf-8",
    )
    return path


async def fake_chat_result(request) -> ChatResult:
    return ChatResult(model=request.model, outputs=[CanonicalOutput(content="answer")])


def test_responses_shape(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]
    provider.chat = fake_chat_result  # type: ignore[method-assign]
    provider.responses = fake_chat_result  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        json={"model": "m", "input": "hi"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["id"].startswith("resp_")
    assert body["status"] == "completed"
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert body["output"][0]["content"][0]["text"] == "answer"
    assert body["output_text"] == "answer"


def test_stateful_continuation(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]
    seen: list[list[dict[str, object]]] = []

    async def capture(request) -> ChatResult:
        seen.append(list(request.messages))
        return await fake_chat_result(request)

    provider.chat = capture  # type: ignore[method-assign]
    provider.responses = capture  # type: ignore[method-assign]
    client = TestClient(app)
    first = client.post("/v1/responses", json={"model": "m", "input": "hello"})
    assert first.status_code == 200
    response_id = first.json()["id"]
    second = client.post(
        "/v1/responses",
        json={"model": "m", "input": "follow up", "previous_response_id": response_id},
    )
    assert second.status_code == 200
    assert len(seen) == 2
    assert seen[1][0] == {"role": "assistant", "content": "answer"}
    assert seen[1][1] == {"role": "user", "content": "follow up"}


def test_unknown_previous_response_id(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]
    provider.chat = fake_chat_result  # type: ignore[method-assign]
    provider.responses = fake_chat_result  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        json={"model": "m", "input": "hi", "previous_response_id": "resp_missing"},
    )
    assert response.status_code == 404


def test_unsupported_model_rejected(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "unknown-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400


def test_embeddings_unsupported_provider(tmp_path: Path, monkeypatch) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]

    @property
    def limited_info(self) -> ProviderInfo:
        return ProviderInfo(name="android", capabilities=set())

    monkeypatch.setattr(type(provider), "info", limited_info)
    client = TestClient(app)
    response = client.post("/v1/embeddings", json={"model": "m", "input": "hi"})
    assert response.status_code == 400
