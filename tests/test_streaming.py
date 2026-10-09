from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lms_passthrough import create_app
from lms_passthrough.provider_api import StreamEvent


def test_streaming_native_shape(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
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
    app = create_app(config)
    provider = app.state.state.providers["android"]
    async def fake_stream(request):
        yield StreamEvent(event="chat.start", data={})
        yield StreamEvent(event="message.delta", data={"content": "hello"})
        yield StreamEvent(event="chat.end", data={"result": {}})

    provider.stream_chat = fake_stream  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"model": "m", "input": "hi", "stream": True},
        headers={"X-LMS-Provider": "android"},
    )
    assert response.status_code == 200
    assert "chat.start" in response.text
