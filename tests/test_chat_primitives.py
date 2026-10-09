from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from lms_passthrough.api.schemas.chat_primitives import (
    ChatMessage,
    ImageUrlPart,
    TextPart,
    Tool,
    ToolChoice,
)


def test_string_content() -> None:
    message = ChatMessage.model_validate({"role": "user", "content": "hello"})
    assert message.role == "user"
    assert message.content == "hello"
    assert message.name is None
    assert message.tool_calls is None


def test_string_content_round_trip() -> None:
    payload = {"role": "user", "content": "hello"}
    assert ChatMessage.model_validate(payload).model_dump(mode="json", exclude_none=True) == payload


def test_assistant_content_none() -> None:
    message = ChatMessage.model_validate({"role": "assistant", "content": None})
    assert message.content is None


def test_mixed_text_and_image_content() -> None:
    payload = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"},
            },
        ],
    }
    message = ChatMessage.model_validate(payload)
    parts = message.content
    assert isinstance(parts, list)
    assert isinstance(parts[0], TextPart)
    assert isinstance(parts[1], ImageUrlPart)
    assert parts[1].image_url.url == "data:image/png;base64,AAAA"
    assert parts[1].image_url.detail == "high"
    assert message.model_dump(mode="json", exclude_none=True) == payload


def test_invalid_content_part_discriminator() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ChatMessage.model_validate(
            {"role": "user", "content": [{"type": "audio", "audio": {"format": "mp3"}}]}
        )
    part_errors = _content_part_errors(excinfo.value.errors())
    assert part_errors
    assert any("audio" in e["msg"] for e in part_errors)


def test_content_part_requires_discriminator() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ChatMessage.model_validate({"role": "user", "content": [{"text": "hello"}]})
    part_errors = _content_part_errors(excinfo.value.errors())
    assert part_errors


def _content_part_errors(errors: list[dict[str, object]]) -> list[dict[str, object]]:
    return [e for e in errors if tuple(e["loc"])[:1] == ("content",) and tuple(e["loc"])[-1] == 0]


def test_invalid_message_missing_role() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ChatMessage.model_validate({"content": "no role"})
    errors = excinfo.value.errors()
    assert errors[0]["loc"] == ("role",)


def test_tool_call_round_trip() -> None:
    payload = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            }
        ],
    }
    message = ChatMessage.model_validate(payload)
    assert message.tool_calls is not None
    assert message.tool_calls[0].id == "call_abc"
    assert message.tool_calls[0].type == "function"
    assert message.tool_calls[0].function.name == "get_weather"
    assert message.model_dump(mode="json", exclude_none=True) == payload


def test_tool_message_with_tool_call_id() -> None:
    message = ChatMessage.model_validate(
        {"role": "tool", "tool_call_id": "call_abc", "content": '{"city": "London"}'}
    )
    assert message.tool_call_id == "call_abc"
    assert message.content == '{"city": "London"}'


def test_unknown_fields_preserved() -> None:
    payload = {"role": "assistant", "content": "hi", "custom_field": {"a": 1}}
    assert ChatMessage.model_validate(payload).model_dump(mode="json", exclude_none=True) == payload


def test_tool_function_shape() -> None:
    tool = Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    )
    assert tool.type == "function"
    assert tool.function.name == "get_weather"
    assert tool.function.description == "Get the weather"
    assert tool.function.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    assert tool.function.strict is None


@pytest.mark.parametrize("value", ["auto", "none", "required"])
def test_tool_choice_string_forms(value: str) -> None:
    assert TypeAdapter(ToolChoice).validate_python(value) == value


def test_tool_choice_function_form() -> None:
    payload = {"type": "function", "function": {"name": "get_weather"}}
    adapter = TypeAdapter(ToolChoice)
    assert adapter.dump_python(adapter.validate_python(payload)) == payload
