from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from lms_passthrough import create_app
from lms_passthrough.api.schemas.openai_responses import (
    ImageInputItem,
    InputItem,
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
)
from lms_passthrough.api.schemas.translate import openai_responses_request_to_canonical
from lms_passthrough.provider_api import CanonicalOutput, ChatResult


def test_input_string_request() -> None:
    req = OpenAIResponsesRequest.model_validate({"model": "gpt-4o", "input": "hello"})
    assert req.input == "hello"


def test_input_message_list_request() -> None:
    req = OpenAIResponsesRequest.model_validate(
        {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "message", "role": "assistant", "content": "hi there"},
            ],
        }
    )
    assert isinstance(req.input, list)
    assert isinstance(req.input[0], InputItem)
    assert req.input[0].role == "user"
    assert req.input[1].role == "assistant"


def test_input_mixed_message_and_image() -> None:
    req = OpenAIResponsesRequest.model_validate(
        {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "what is this?"},
                {"type": "image", "image_url": {"url": "http://example.com/img.png"}},
            ],
        }
    )
    assert isinstance(req.input[1], ImageInputItem)


def test_unknown_top_level_field_preserved() -> None:
    payload = {
        "model": "gpt-4o",
        "input": "hello",
        "truncation": "auto",
        "include": ["output_text"],
    }
    req = OpenAIResponsesRequest.model_validate(payload)
    raw = req.model_dump()
    assert raw["truncation"] == "auto"
    assert raw["include"] == ["output_text"]


def test_invalid_stream_type_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        OpenAIResponsesRequest.model_validate(
            {"model": "gpt-4o", "input": "hello", "stream": ["not", "a", "bool"]}
        )
    errors = excinfo.value.errors()
    assert errors
    assert any("stream" in str(e["loc"]) for e in errors)


def test_unknown_input_item_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        OpenAIResponsesRequest.model_validate(
            {
                "model": "gpt-4o",
                "input": [{"type": "input_item_reference"}],
            }
        )
    errors = excinfo.value.errors()
    assert errors


def test_openai_responses_request_to_canonical_string_input() -> None:
    req = OpenAIResponsesRequest.model_validate({"model": "gpt-4o", "input": "hello"})
    canonical = openai_responses_request_to_canonical(req)
    assert canonical.model == "gpt-4o"
    assert canonical.messages == [{"role": "user", "content": "hello"}]


def test_openai_responses_request_to_canonical_message_list() -> None:
    req = OpenAIResponsesRequest.model_validate(
        {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "message", "role": "assistant", "content": "hi"},
            ],
        }
    )
    canonical = openai_responses_request_to_canonical(req)
    assert canonical.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_openai_responses_request_to_canonical_mixed_image() -> None:
    req = OpenAIResponsesRequest.model_validate(
        {
            "model": "gpt-4o",
            "input": [
                {"type": "image", "image_url": {"url": "http://example.com/img.png"}},
                {"type": "message", "role": "user", "content": "describe"},
            ],
        }
    )
    canonical = openai_responses_request_to_canonical(req)
    image_item = canonical.messages[0]["content"][0]
    assert image_item["type"] == "image"
    assert image_item["image_url"]["url"] == "http://example.com/img.png"
    assert canonical.messages[1] == {"role": "user", "content": "describe"}


def test_openai_responses_response_shape() -> None:
    response = OpenAIResponsesResponse.model_validate(
        {
            "id": "resp_abc",
            "object": "response",
            "created_at": 1234567890,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "id": "msg_abc",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                }
            ],
            "output_text": "answer",
        }
    )
    dumped = response.model_dump(mode="json", exclude_none=True)
    assert dumped["object"] == "response"
    assert dumped["output_text"] == "answer"


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


async def fake_responses_result(request) -> ChatResult:
    return ChatResult(model=request.model, outputs=[CanonicalOutput(content="answer")])


def test_endpoint_responses_422(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        json={"model": "m", "input": "hello", "stream": ["not", "a", "bool"]},
    )
    assert response.status_code == 422


def test_openapi_schema_does_not_advertise_unsupported_fields(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    request_schema = schema["components"]["schemas"]["OpenAIResponsesRequest"]
    properties = request_schema["properties"]
    assert "input_items" not in properties
    assert "include" not in properties
    assert "truncation" not in properties
    assert "model" in properties
    assert "input" in properties


def test_unknown_top_level_field_accepted_at_runtime(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    provider = app.state.state.providers["android"]
    provider.chat = fake_responses_result  # type: ignore[method-assign]
    provider.responses = fake_responses_result  # type: ignore[method-assign]
    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": "hello",
            "truncation": "auto",
            "include": ["output_text"],
        },
    )
    assert response.status_code == 200
