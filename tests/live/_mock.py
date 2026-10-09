"""Shared helpers for LM Studio conformance tests.

Provides a mock upstream LM Studio server (routed via respx) that is used
offline to exercise the proxy's transparent passthrough without a real
server, plus helpers to build a proxy app pointed at it.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from lms_passthrough import create_app

# Known location where the live suite captures sanitized fixtures.
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "live_lms"

# Independent source of truth for the expected native model catalog shape.
MODEL_CATALOG: dict[str, object] = {
    "models": [
        {
            "type": "llm",
            "publisher": "mock",
            "key": "mock-model",
            "display_name": "Mock Model",
            "quantization": "Q8_0",
            "size_bytes": 123456789,
            "params_string": "7B",
            "max_context_length": 4096,
            "format": "gguf",
            "loaded_instances": [],
        }
    ]
}


def native_error_body(status: str, message: str) -> str:
    """The LM Studio native error envelope."""
    return json.dumps({"error": {"message": message, "type": status}})


def parse_sse(text: str) -> list[tuple[str | None, dict[str, object]]]:
    """Parse an SSE body into (event_name, data) frames."""
    frames: list[tuple[str | None, dict[str, object]]] = []
    event: str | None = None
    for block in text.split("\n\n"):
        lines = [line for line in block.splitlines() if line]
        if not lines:
            continue
        data_lines: list[str] = []
        for line in lines:
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
        if not data_lines:
            continue
        frames.append((event, json.loads("\n".join(data_lines))))
        event = None
    return frames


def live_chat_payload(server: str, model: str) -> dict[str, object]:
    """Streaming /api/v1/chat body accepted by a given live LM Studio server.

    Modern LM Studio requires `input` (a plain string); the older generation
    accepted `messages`. Probe the direct server once and return the shape it
    accepts, so the conformance suite works against both.
    """
    prompt = "Say the word conformance."
    input_body: dict[str, object] = {"model": model, "input": prompt, "stream": True}
    probe = httpx.post(f"{server}/api/v1/chat", json=input_body)
    if probe.is_server_error:
        probe.raise_for_status()
    if probe.is_client_error:
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
    return input_body


def model_keys(catalog: dict[str, object]) -> list[str]:
    """Extract native model ids from a /api/v1/models body."""
    models = catalog.get("models", [])
    keys: list[str] = []
    for entry in models if isinstance(models, list) else []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key") or entry.get("id")
        if isinstance(key, str):
            keys.append(key)
    return keys


def native_sse(*events: tuple[str | None, dict[str, object]]) -> bytes:
    """Render a native LM Studio SSE body from (event, data) frames."""
    frames: list[str] = []
    for event_name, data in events:
        if event_name is not None:
            frames.append(f"event: {event_name}")
        frames.append(f"data: {json.dumps(data)}")
        frames.append("")
    return ("\n".join(frames) + "\n").encode("utf-8")


def chat_stream_events(text: str) -> list[tuple[str | None, dict[str, object]]]:
    """A canonical native streamed chat lifecycle for the mock upstream."""
    return [
        ("chat.start", {"type": "chat.start"}),
        ("message.start", {"type": "message.start", "role": "assistant"}),
        ("message.delta", {"type": "message.delta", "content": ""}),
        (None, {"type": "message.delta", "content": text}),
        ("message.end", {"type": "message.end"}),
        ("chat.end", {"type": "chat.end", "response_id": "mock-resp-1"}),
    ]


def register_mock_upstream(
    *,
    base_url: str = "http://mock-upstream:8999",
    catalog: dict[str, object] | None = None,
    unknown_model_status: int = 400,
) -> respx.MockRouter:
    """Register respx routes emulating an LM Studio server."""
    catalog_body = json.dumps(catalog if catalog is not None else MODEL_CATALOG).encode("utf-8")

    def handle_chat(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        model = payload.get("model")
        if model != "mock-model":
            return httpx.Response(
                unknown_model_status,
                content=native_error_body(
                    "invalid_request",
                    f"Model '{model}' not found",
                ).encode("utf-8"),
                headers={"content-type": "application/json"},
            )
        if payload.get("stream"):
            return httpx.Response(
                200,
                content=native_sse(*chat_stream_events("live conformance text")),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            content=json.dumps({"id": "mock-chat", "choices": [{"text": "hi"}]}).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    router = respx.mock(
        base_url=base_url,
        assert_all_called=False,
    )
    router.get("/api/v1/models").mock(
        return_value=httpx.Response(
            200,
            content=catalog_body,
            headers={"content-type": "application/json"},
        )
    )
    router.post("/api/v1/chat").mock(side_effect=handle_chat)
    return router


def build_proxy_app(tmp_path: Path, base_url: str = "http://mock-upstream:8999"):
    """Build an in-process proxy app configured to passthrough to `base_url`."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
default_provider: lm-studio
persistence:
  path: {tmp_path}/state.sqlite3
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: {base_url}
""".strip(),
        encoding="utf-8",
    )
    return create_app(config)
