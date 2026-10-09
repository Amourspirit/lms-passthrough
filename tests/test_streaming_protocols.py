from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from lms_passthrough.api.compat import (
    native_sse_from_canonical,
    openai_chat_sse_from_canonical,
    responses_sse_from_canonical,
)
from lms_passthrough.provider_api import CanonicalOutput, ChatResult, StreamEvent


def parse_sse(text: str) -> list[tuple[str | None, str]]:
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


async def events(*items: StreamEvent) -> AsyncIterator[StreamEvent]:
    for item in items:
        yield item


async def failing_events() -> AsyncIterator[StreamEvent]:
    yield StreamEvent(event="message.delta", data={"content": "partial"})
    raise RuntimeError("upstream died")


def result() -> ChatResult:
    return ChatResult(model="m", outputs=[CanonicalOutput(content="hello")])


@pytest.mark.asyncio
async def test_native_sse_lifecycle() -> None:
    text = ""
    async for frame in native_sse_from_canonical(
        events(
            StreamEvent(event="chat.start", data={"type": "chat.start"}),
            StreamEvent(event="message.delta", data={"content": "hi"}),
            StreamEvent(event="chat.end", data={"result": {}}),
        )
    ):
        text += frame
    frames = parse_sse(text)
    assert frames[0][0] == "chat.start"
    assert frames[-1][0] == "chat.end"
    assert frames.count(("chat.start", frames[0][1])) == 1


@pytest.mark.asyncio
async def test_native_sse_error_still_ends_with_chat_end() -> None:
    text = ""
    async for frame in native_sse_from_canonical(failing_events()):
        text += frame
    frames = parse_sse(text)
    names = [name for name, _ in frames]
    assert "error" in names
    assert names[-1] == "chat.end"


@pytest.mark.asyncio
async def test_openai_chat_sse_shape() -> None:
    text = ""
    async for frame in openai_chat_sse_from_canonical(
        events(StreamEvent(event="message.delta", data={"content": "hello"})),
        "chatcmpl-1",
        "m",
    ):
        text += frame
    frames = parse_sse(text)
    first = json.loads(frames[0][1])
    assert first["choices"][0]["delta"] == {"role": "assistant"}
    assert frames[-1][1] == "[DONE]"
    finish = json.loads(frames[-2][1])
    assert finish["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_responses_sse_shape() -> None:
    text = ""
    async for frame in responses_sse_from_canonical(
        events(StreamEvent(event="message.delta", data={"content": "hello"})),
        "resp_1",
        "m",
    ):
        text += frame
    frames = parse_sse(text)
    names = [name for name, _ in frames]
    assert names[0] == "response.created"
    assert "response.output_text.delta" in names
    assert names[-1] == "response.completed"
    completed = json.loads(frames[-1][1])
    assert completed["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_responses_sse_error() -> None:
    text = ""
    async for frame in responses_sse_from_canonical(failing_events(), "resp_1", "m"):
        text += frame
    frames = parse_sse(text)
    names = [name for name, _ in frames]
    assert names[-1] == "error"
