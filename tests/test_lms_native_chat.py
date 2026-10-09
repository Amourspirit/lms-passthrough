"""Typed HTTP boundary for `/api/v1/chat` (LM Studio native chat)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from lms_passthrough import create_app
from lms_passthrough.api.compat import result_to_native
from lms_passthrough.api.schemas.lm_studio import LMSChatRequest, LMSChatResponse
from lms_passthrough.api.schemas.translate import (
    lms_chat_request_to_canonical,
    to_chat_request,
)
from lms_passthrough.provider_api import CanonicalOutput, CanonicalUsage, ChatResult

MOCK_BASE = "http://mock-upstream:8999"

NATIVE_PAYLOAD = {
    "model": "m",
    "input": "hi",
    "system_prompt": "be brief",
    "previous_response_id": "resp_abc",
    "store": True,
    "stream": False,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "max_output_tokens": 512,
    "max_tokens": 256,
    "stop": ["END"],
    "seed": 7,
    "reasoning": "high",
    "context_length": 8192,
    "future_lms_field": {"a": 1},
}


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


def build_lms_proxy(tmp_path: Path, base_url: str = MOCK_BASE):
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


def schema_for(app, path: str, part: str) -> dict[str, object]:
    schema = app.openapi()
    operation = schema["paths"][path]["post"]
    if part == "responses":
        content = operation["responses"]["200"]["content"]["application/json"]
    else:
        content = operation[part]["content"]["application/json"]
    ref = content["schema"]["$ref"]
    name = ref.rsplit("/", 1)[-1]
    return schema["components"]["schemas"][name]


class TestLMSChatRequest:
    def test_parses_native_fields(self) -> None:
        request = LMSChatRequest.model_validate(NATIVE_PAYLOAD)
        assert request.model == "m"
        assert request.input == "hi"
        assert request.system_prompt == "be brief"
        assert request.top_k == 40
        assert request.min_p == 0.05
        assert request.repeat_penalty == 1.1
        assert request.context_length == 8192
        assert request.reasoning == "high"

    def test_accepts_unknown_top_level_fields(self) -> None:
        payload = {"model": "m", "input": "hi", "brand_new_lms_field": {"enabled": True}}
        request = LMSChatRequest.model_validate(payload)
        assert request.model_dump()["brand_new_lms_field"] == {"enabled": True}

    def test_accepts_messages_form_with_string_content(self) -> None:
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        request = LMSChatRequest.model_validate(payload)
        assert len(request.messages or []) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "hi"

    def test_rejects_bad_type_with_field_path(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            LMSChatRequest.model_validate({"model": "m", "top_k": "many"})
        assert any("top_k" in str(item["loc"]) for item in excinfo.value.errors())

    def test_rejects_missing_model(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            LMSChatRequest.model_validate({})
        assert any("model" in str(item["loc"]) for item in excinfo.value.errors())


class TestLMSChatRequestToCanonical:
    def test_equivalent_to_to_chat_request_for_native_payload(self) -> None:
        canonical = lms_chat_request_to_canonical(LMSChatRequest.model_validate(NATIVE_PAYLOAD))
        expected = to_chat_request(NATIVE_PAYLOAD)
        for field in (
            "model",
            "messages",
            "system_prompt",
            "stream",
            "previous_response_id",
            "store",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "max_output_tokens",
            "stop",
            "seed",
            "reasoning",
            "context_length",
        ):
            assert getattr(canonical, field) == getattr(expected, field), field
        assert canonical.raw["future_lms_field"] == {"a": 1}
        assert canonical.raw["system_prompt"] == "be brief"

    def test_messages_form_preserves_message_dicts(self) -> None:
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi", "style": "loud"}]}
        canonical = lms_chat_request_to_canonical(LMSChatRequest.model_validate(payload))
        expected = to_chat_request(payload)
        assert canonical.messages == [{"role": "user", "content": "hi", "style": "loud"}]
        assert canonical.messages == expected.messages

    def test_system_prompt_flows_through(self) -> None:
        canonical = lms_chat_request_to_canonical(
            LMSChatRequest.model_validate({**NATIVE_PAYLOAD, "input": "hello"})
        )
        assert canonical.system_prompt == "be brief"


class TestLMSChatResponse:
    def test_matches_result_to_native_shape(self) -> None:
        result = ChatResult(
            model="m",
            outputs=[
                CanonicalOutput(content="hi"),
                CanonicalOutput(type="reasoning", content="think"),
                CanonicalOutput(type="tool_call", tool="search", arguments={"q": "x"}),
            ],
            usage=CanonicalUsage(
                input_tokens=10,
                output_tokens=5,
                reasoning_output_tokens=3,
                tokens_per_second=12.5,
            ),
            response_id="resp_123",
        )
        native = result_to_native(result, "resp_123")
        response = LMSChatResponse.model_validate(native)
        assert response.model_instance_id == "m"
        assert [output.type for output in response.output] == ["message", "reasoning", "tool_call"]
        assert response.output[0].content == "hi"
        assert response.output[2].tool == "search"
        assert response.stats.input_tokens == 10
        assert response.stats.output_tokens == 5
        assert response.stats.tokens_per_second == 12.5
        assert response.response_id == "resp_123"


class TestNativeChatEndpoint:
    def test_malformed_payload_returns_422_with_field_path(self, tmp_path: Path) -> None:
        client = TestClient(create_app(make_android_config(tmp_path)))
        response = client.post(
            "/api/v1/chat",
            json={"model": "m", "input": "hi", "top_k": "many"},
            headers={"X-LMS-Provider": "android"},
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "top_k"]

    def test_unknown_fields_reach_provider(self, tmp_path: Path) -> None:
        app = create_app(make_android_config(tmp_path))
        seen: dict[str, object] = {}
        provider = app.state.state.providers["android"]

        async def fake_chat(request) -> ChatResult:
            seen["raw"] = request.raw
            return ChatResult(model=request.model, outputs=[CanonicalOutput(content="ok")])

        provider.chat = fake_chat  # type: ignore[method-assign]
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={
                "model": "m",
                "input": "hi",
                "brand_new_lms_field": {"enabled": True},
            },
            headers={"X-LMS-Provider": "android"},
        )
        assert response.status_code == 200
        assert seen["raw"].get("brand_new_lms_field") == {"enabled": True}
        assert response.json()["output"][0]["content"] == "ok"

    def test_openapi_native_shape_distinct_from_chat_completions(self, tmp_path: Path) -> None:
        app = create_app(make_android_config(tmp_path))
        schema = app.openapi()
        native = schema["paths"]["/api/v1/chat"]["post"]
        request_schema = schema_for(app, "/api/v1/chat", "requestBody")
        props = request_schema["properties"]
        for field in (
            "system_prompt",
            "top_k",
            "min_p",
            "repeat_penalty",
            "context_length",
            "reasoning",
            "input",
        ):
            assert field in props, field
        response_schema = schema_for(app, "/api/v1/chat", "responses")
        resp_props = response_schema["properties"]
        for field in ("model_instance_id", "output", "stats", "response_id"):
            assert field in resp_props, field
        assert "text/event-stream" in native["responses"]["200"]["content"]
        openai_schema = schema_for(app, "/v1/chat/completions", "requestBody")
        openai_name = openai_schema.get("title")
        assert openai_name == "OpenAIChatCompletionRequest"


class TestTransparentNativeChatPath:
    def test_raw_body_bytes_forwarded_unchanged(self, tmp_path: Path) -> None:
        captured: list[bytes] = []

        def handle_chat(request: httpx.Request) -> httpx.Response:
            captured.append(request.content)
            return httpx.Response(
                200,
                content=json.dumps({"id": "ok"}).encode("utf-8"),
                headers={"content-type": "application/json"},
            )

        app = build_lms_proxy(tmp_path)
        with respx.mock(base_url=MOCK_BASE, assert_all_called=False) as router:
            router.post("/api/v1/chat").mock(side_effect=handle_chat)
            client = TestClient(app)
            body = b'{\n  "model": "m",\n  "input": "hi"\n}'
            response = client.post(
                "/api/v1/chat",
                content=body,
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 200
        assert captured == [body]

    def test_invalid_json_returns_native_400_before_upstream(self, tmp_path: Path) -> None:
        app = build_lms_proxy(tmp_path)
        with respx.mock(base_url=MOCK_BASE, assert_all_called=False) as router:
            router.post("/api/v1/chat").mock(
                return_value=httpx.Response(
                    200,
                    content=b"unused",
                    headers={"content-type": "application/json"},
                )
            )
            client = TestClient(app)
            response = client.post(
                "/api/v1/chat",
                content=b"{not-valid-json",
                headers={"content-type": "application/json"},
            )
            assert router.post("/api/v1/chat").call_count == 0
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request"


class TestTransparentModelsPath:
    def test_transparent_models_still_forward(self, tmp_path: Path) -> None:
        app = build_lms_proxy(tmp_path)
        with respx.mock(base_url=MOCK_BASE, assert_all_called=False) as router:
            router.get("/api/v1/models").mock(
                return_value=httpx.Response(
                    200,
                    content=json.dumps({"models": []}).encode("utf-8"),
                    headers={"content-type": "application/json"},
                )
            )
            client = TestClient(app)
            response = client.get("/api/v1/models")
        assert response.status_code == 200
        assert response.json() == {"models": []}
