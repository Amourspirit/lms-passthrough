"""Helpers for synthesizing canonical stream lifecycles."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from lms_passthrough.provider_api.base import ChatRequest, ChatResult, StreamEvent


async def synthesize_stream(
    request: ChatRequest,
    invoke: Any,
) -> AsyncIterator[StreamEvent]:
    """Synthesize a full lifecycle stream from a non-streaming result.

    Emits the canonical lifecycle: chat.start, message.start, message.delta
    per output, message.end, and chat.end with the aggregated result. On
    failure it emits a terminal error event; nothing follows it.
    """
    yield StreamEvent(event="chat.start", data={"type": "chat.start"})
    try:
        result: ChatResult = await invoke(request)
    except Exception as exc:
        yield StreamEvent(event="error", data={"type": "unknown", "message": str(exc)})
        return
    yield StreamEvent(event="message.start", data={"type": "message.start"})
    for output in result.outputs:
        yield StreamEvent(
            event="message.delta",
            data={"content": output.content},
        )
    yield StreamEvent(event="message.end", data={"type": "message.end"})
    yield StreamEvent(event="chat.end", data={"result": result.raw})
