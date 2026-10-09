"""Protocol compatibility helpers."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from lms_passthrough.provider_api import ChatResult, StreamEvent


def json_dumps(data: Any) -> str:
    """Serialize JSON with stable separators."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def sse_frame(event: str | None, data: Any) -> str:
    """Render one SSE frame; event name omitted when None."""
    frame = ""
    if event is not None:
        frame += f"event: {event}\n"
    frame += f"data: {json_dumps(data)}\n\n"
    return frame


async def native_sse_from_canonical(
    events: AsyncIterator[StreamEvent],
    response_id: str | None = None,
) -> AsyncIterator[str]:
    """Render canonical events as LM Studio native SSE.

    Guarantees chat.start first and chat.end last, even if the provider's
    stream fails partway through. When ``response_id`` is given, it is
    included in the chat.end frame data.
    """
    yield sse_frame("chat.start", {"type": "chat.start"})
    ended = False
    try:
        async for event in events:
            if event.event == "chat.start":
                continue
            if event.event == "chat.end":
                ended = True
                yield sse_frame("chat.end", _chat_end_data(event.data, response_id))
                continue
            yield sse_frame(event.event, event.data)
    except Exception as exc:
        yield sse_frame("error", {"type": "unknown", "message": str(exc)})
    if not ended:
        yield sse_frame("chat.end", _chat_end_data({}, response_id))


def _chat_end_data(data: dict[str, Any], response_id: str | None) -> dict[str, Any]:
    data = dict(data)
    if "type" not in data:
        data["type"] = "chat.end"
    if response_id:
        data["response_id"] = response_id
    return data


async def openai_chat_sse_from_canonical(
    events: AsyncIterator[StreamEvent],
    completion_id: str,
    model: str,
) -> AsyncIterator[str]:
    """Render canonical events as OpenAI chat-completions SSE."""
    created = int(time.time())
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    yield sse_frame(None, {**base, "choices": [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
    ]})
    try:
        async for event in events:
            if event.event != "message.delta":
                continue
            yield sse_frame(None, {**base, "choices": [
                {
                    "index": 0,
                    "delta": {"content": event.data.get("content", "")},
                    "finish_reason": None,
                }
            ]})
    except Exception as exc:
        yield sse_frame(None, {
            "error": {"message": str(exc), "type": "server_error"},
        })
        yield "data: [DONE]\n\n"
        return
    yield sse_frame(None, {**base, "choices": [
        {"index": 0, "delta": {}, "finish_reason": "stop"}
    ]})
    yield "data: [DONE]\n\n"


async def responses_sse_from_canonical(
    events: AsyncIterator[StreamEvent],
    response_id: str,
    model: str,
) -> AsyncIterator[str]:
    """Render canonical events as OpenAI Responses SSE."""
    created = int(time.time())
    response_base: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "model": model,
        "status": "in_progress",
    }
    yield sse_frame("response.created", {"response": response_base})
    item_id = f"msg_{uuid.uuid4().hex}"
    yield sse_frame("response.output_item.added", {
        "output_index": 0,
        "item": {"id": item_id, "type": "message", "status": "in_progress"},
    })
    yield sse_frame("response.content_part.added", {
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    })
    accumulated = ""
    try:
        async for event in events:
            if event.event != "message.delta":
                continue
            delta = str(event.data.get("content", ""))
            accumulated += delta
            yield sse_frame("response.output_text.delta", {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
            })
    except Exception as exc:
        yield sse_frame("error", {
            "type": "server_error",
            "message": str(exc),
        })
        return
    yield sse_frame("response.output_text.done", {
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": accumulated,
    })
    yield sse_frame("response.content_part.done", {
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": accumulated},
    })
    yield sse_frame("response.output_item.done", {
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": accumulated}],
        },
    })
    completed = {
        **response_base,
        "status": "completed",
        "output": [
            {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": accumulated}],
            }
        ],
    }
    yield sse_frame("response.completed", {"response": completed})


def result_to_native(result: ChatResult, response_id: str | None) -> dict[str, Any]:
    """Render canonical result in native LM Studio shape."""
    payload: dict[str, Any] = {
        "model_instance_id": result.raw.get("model_instance_id", result.model),
        "output": [asdict(output) for output in result.outputs],
        "stats": asdict(result.usage),
    }
    if response_id:
        payload["response_id"] = response_id
    return payload


def result_to_chat_completion(result: ChatResult) -> dict[str, Any]:
    """Render canonical result in OpenAI chat completion shape."""
    content = "\n".join(output.content for output in result.outputs)
    return {
        "id": result.response_id,
        "object": "chat.completion",
        "created": result.raw.get("created", 0),
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.usage.input_tokens or 0,
            "completion_tokens": result.usage.output_tokens or 0,
            "total_tokens": result.usage.total_tokens or 0,
        },
    }


def result_to_responses(result: ChatResult, response_id: str) -> dict[str, Any]:
    """Render canonical result in OpenAI Responses API shape."""
    output_items: list[dict[str, Any]] = []
    for output in result.outputs:
        output_items.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output.content,
                        "annotations": [],
                    }
                ],
            }
        )
    usage: dict[str, Any] = {}
    if result.usage.input_tokens is not None or result.usage.output_tokens is not None:
        usage = {
            "input_tokens": result.usage.input_tokens or 0,
            "output_tokens": result.usage.output_tokens or 0,
            "total_tokens": result.usage.total_tokens
            or (result.usage.input_tokens or 0) + (result.usage.output_tokens or 0),
        }
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": result.model,
        "status": "completed",
        "output": output_items,
        "output_text": "\n".join(output.content for output in result.outputs),
    }
    if usage:
        payload["usage"] = usage
    return payload
