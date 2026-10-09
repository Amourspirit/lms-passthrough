from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from lms_passthrough import create_app
from lms_passthrough.provider_api import CanonicalOutput, ChatResult, StreamEvent


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


def stream_events() -> object:
    async def fake_stream(request) -> object:  # type: ignore[misc]
        yield StreamEvent(event="chat.start", data={})
        yield StreamEvent(event="message.delta", data={"content": "hello"})
        yield StreamEvent(event="message.delta", data={"content": " world"})
        yield StreamEvent(event="chat.end", data={"result": {}})

    return fake_stream


def parse_frames(text: str) -> list[tuple[str | None, str]]:
    frames = []
    for block in text.strip().split("\n\n"):
        event = None
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        frames.append((event, data))
    return frames


def test_streamed_native_chat_stores_aggregated_result(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]
    provider.stream_chat = stream_events()  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"model": "m", "input": "hi", "stream": True},
        headers={"X-LMS-Provider": "android"},
    )
    assert response.status_code == 200
    frames = parse_frames(response.text)
    chat_end_data = json.loads(next(data for name, data in frames if name == "chat.end"))
    assert chat_end_data["type"] == "chat.end"
    response_id = chat_end_data.get("response_id")
    assert response_id and response_id.startswith("resp_")

    stored = app.state.state.state_store.get_response(response_id)
    assert stored is not None
    assert stored.payload["outputs"][0]["content"] == "hello world"


def test_streamed_native_chat_continuation(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]
    seen: list[list[dict[str, str]]] = []
    provider.stream_chat = stream_events()  # type: ignore[method-assign]

    async def fake_chat(request) -> ChatResult:
        seen.append(list(request.messages))
        return ChatResult(model=request.model, outputs=[CanonicalOutput(content="answered")])

    provider.chat = fake_chat  # type: ignore[method-assign]
    client = TestClient(app)
    first = client.post(
        "/api/v1/chat",
        json={"model": "m", "input": "hi", "stream": True},
        headers={"X-LMS-Provider": "android"},
    )
    assert first.status_code == 200
    first_frames = parse_frames(first.text)
    chat_end = json.loads(next(data for name, data in first_frames if name == "chat.end"))
    response_id = chat_end["response_id"]

    second = client.post(
        "/api/v1/chat",
        json={
            "model": "m",
            "input": "follow up",
            "previous_response_id": response_id,
        },
        headers={"X-LMS-Provider": "android"},
    )
    assert second.status_code == 200
    assert len(seen) == 1
    assert seen[0][0] == {"role": "assistant", "content": "hello world"}
    assert seen[0][1] == {"role": "user", "content": "follow up"}


def test_streamed_native_chat_store_false_stores_nothing(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]
    provider.stream_chat = stream_events()  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"model": "m", "input": "hi", "stream": True, "store": False},
        headers={"X-LMS-Provider": "android"},
    )
    assert response.status_code == 200
    frames = parse_frames(response.text)
    chat_end = json.loads(next(data for name, data in frames if name == "chat.end"))
    assert "response_id" not in chat_end


def test_streamed_native_chat_store_false_no_persisted_chain(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]
    provider.stream_chat = stream_events()  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"model": "m", "input": "hi", "stream": True, "store": False},
        headers={"X-LMS-Provider": "android"},
    )
    assert response.status_code == 200
    assert response.content.count(b"response_id") == 0
