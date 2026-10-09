"""Tests for the typed OpenAI chat completions HTTP boundary (issue 03)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from lms_passthrough import create_app
from lms_passthrough.api.schemas.openai_chat import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from lms_passthrough.api.schemas.translate import (
    openai_chat_request_to_canonical,
    to_chat_request,
)
from lms_passthrough.provider_api import CanonicalOutput, ChatResult


def make_android_config(tmp_path: Path) -> Path:
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


def make_lm_studio_config(tmp_path: Path, base_url: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
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
    return path


async def fake_chat_result(request) -> ChatResult:
    return ChatResult(model=request.model, outputs=[CanonicalOutput(content="answer")])


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def openai_payload() -> dict[str, Any]:
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
        ],
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 64,
        "stop": ["\n"],
        "seed": 42,
        "parallel_tool_calls": True,
        "response_format": {"type": "json_object"},
        "logit_bias": {"1": 2},
        "new_openai_field": {"nested": [1, 2]},
    }


def test_translation_matches_to_chat_request_for_openai_payload() -> None:
    payload = openai_payload()
    model = OpenAIChatCompletionRequest.model_validate(payload)
    actual = asdict(openai_chat_request_to_canonical(model))
    expected = asdict(to_chat_request(payload))
    for field in set(actual) - {"raw"}:
        assert _drop_none_values(actual[field]) == _drop_none_values(expected[field]), field
    assert _drop_none_values(actual["raw"]) == _drop_none_values(expected["raw"])


def test_max_completion_tokens_maps_to_max_output_tokens() -> None:
    model = OpenAIChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 128,
        }
    )
    assert openai_chat_request_to_canonical(model).max_output_tokens == 128


def test_openai_chat_request_parses_message_content_parts() -> None:
    payload = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
    }
    model = OpenAIChatCompletionRequest.model_validate(payload)
    canonical = openai_chat_request_to_canonical(model)
    assert canonical.messages[0]["content"][0]["type"] == "text"
    assert canonical.messages[0]["content"][1]["type"] == "image_url"


def test_single_string_stop_is_wrapped_not_dropped() -> None:
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "END",
    }
    model = OpenAIChatCompletionRequest.model_validate(payload)
    canonical = openai_chat_request_to_canonical(model)
    assert canonical.stop == ["END"]


def test_valid_chat_completion_returns_typed_response(tmp_path: Path) -> None:
    app = create_app(make_android_config(tmp_path))
    provider = app.state.state.providers["android"]
    provider.chat = fake_chat_result  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "local-model"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["content"] == "answer"
    assert "usage" in body
    assert OpenAIChatCompletionResponse.model_validate(body)


def test_malformed_stream_returns_422_with_field_path(tmp_path: Path) -> None:
    app = create_app(make_android_config(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": "yes"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(tuple(e["loc"]) == ("body", "stream") for e in errors)


def test_malformed_content_part_discriminator_returns_422(tmp_path: Path) -> None:
    app = create_app(make_android_config(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": {"format": "mp3"}}],
                }
            ],
        },
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(tuple(e["loc"])[:4] == ("body", "messages", 0, "content") for e in errors)


def test_unknown_top_level_field_reaches_provider(tmp_path: Path) -> None:
    app = create_app(make_android_config(tmp_path))
    provider = app.state.state.providers["android"]
    seen: list[dict[str, object]] = []

    async def capture(request) -> ChatResult:
        seen.append(dict(request.raw))
        return await fake_chat_result(request)

    provider.chat = capture  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "future_openai_field": {"a": [1, 2]},
        },
    )
    assert response.status_code == 200
    assert seen[0]["future_openai_field"] == {"a": [1, 2]}


def test_openai_chat_stream_path_returns_sse(tmp_path: Path) -> None:
    app = create_app(make_android_config(tmp_path))
    provider = app.state.state.providers["android"]

    async def fake_stream(request):
        yield {"event": "chat.start", "data": {}}
        yield {"event": "message.delta", "data": {"content": "hello"}}
        yield {"event": "chat.end", "data": {"result": {}}}

    provider.stream_chat = fake_stream  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "chat.completion.chunk" in response.text
    assert "[DONE]" in response.text


def test_transparent_lm_studio_forwards_raw_body_bytes(tmp_path: Path) -> None:
    app = create_app(make_lm_studio_config(tmp_path, "http://mock-upstream:8999"))
    raw = (
        b'{"model":"mock-model",\n'
        b' "messages":[{"role":"user","content":"caf\xc3\xa9"}],\n'
        b' "stream": false}\n'
    )
    captured: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return httpx.Response(
            200,
            json={"id": "x", "object": "chat.completion", "choices": []},
            headers={"content-type": "application/json"},
        )

    with respx.mock(base_url="http://mock-upstream:8999") as router:
        router.post("/v1/chat/completions").mock(side_effect=handle)
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            content=raw,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 200
    assert captured[0] == raw


def test_openapi_describes_chat_completions_schema(tmp_path: Path) -> None:
    app = create_app(make_android_config(tmp_path))
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    path_item = schema["paths"]["/v1/chat/completions"]["post"]
    request_ref = path_item["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request_name = request_ref.rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][request_name]["properties"]
    for field in [
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_completion_tokens",
        "max_tokens",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    ]:
        assert field in properties
    assert "OpenAIChatCompletionResponse" in schema["components"]["schemas"]
    content = path_item["responses"]["200"]["content"]
    assert "application/json" in content
    assert "text/event-stream" in content


def test_chat_succeeds_on_fresh_start_with_empty_model_cache(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
default_provider: android
persistence:
  path: {tmp_path}/state.sqlite3
providers:
  - name: android
    kind: openai_compatible
    base_url: http://mock-upstream:8080
""".strip(),
        encoding="utf-8",
    )
    app = create_app(config)
    with respx.mock(base_url="http://mock-upstream:8080") as router:
        models_route = router.get("/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "Qwen3.8-27B-MTP-8bit", "object": "model"}],
                },
            )
        )
        router.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                },
            )
        )
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            headers={"X-LMS-Provider": "android"},
            json={
                "model": "Qwen3.8-27B-MTP-8bit",
                "messages": [{"role": "user", "content": "Say hi"}],
            },
        )
    assert models_route.called
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi"
