"""HTTP-model to canonical-type translation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from lms_passthrough.api.schemas.lm_studio import LMSChatRequest
from lms_passthrough.api.schemas.openai_chat import OpenAIChatCompletionRequest
from lms_passthrough.api.schemas.openai_responses import OpenAIResponsesRequest
from lms_passthrough.provider_api import ChatRequest


def _normalize_messages_from_input(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, list):
        messages: list[dict[str, Any]] = []
        for item in input_value:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                messages.append(
                    {
                        "role": item.get("role", "user"),
                        "content": item.get("content", ""),
                    }
                )
            elif item.get("type") == "image":
                messages.append({"role": "user", "content": [item]})
            else:
                messages.append(item)
        return messages
    return []


def openai_responses_request_to_canonical(request: OpenAIResponsesRequest) -> ChatRequest:
    """Translate an OpenAI Responses request to the canonical ChatRequest.

    This delegates input normalization to the same path used by the legacy
    dict-based translator so that string, message, and image item handling stay
    identical.
    """
    raw = request.model_dump(by_alias=True, exclude_unset=False)
    if isinstance(request.input, str):
        messages = [{"role": "user", "content": request.input}]
    else:
        messages = _normalize_messages_from_input(
            [item.model_dump(by_alias=True, exclude_unset=False) for item in request.input]
        )
    return ChatRequest(
        model=request.model,
        messages=messages,
        system_prompt=request.instructions,
        stream=request.stream,
        previous_response_id=request.previous_response_id,
        store=request.store,
        response_format=request.response_format,
        tools=[tool.model_dump(by_alias=True) for tool in request.tools] if request.tools else None,
        raw=raw,
    )


def to_chat_request(payload: dict[str, Any]) -> ChatRequest:
    """Build a canonical request from supported API payloads."""
    return _chat_request_from_payload(payload)


def _chat_from_typed_model(model: BaseModel) -> ChatRequest:
    """Build a canonical request from a typed chat request model.

    ``model_dump`` normalizes the typed model back to a wire-ready payload
    (dropping ``None`` values, which are wire-equivalent to absent keys) that
    is then fed through the same dict-based extraction as ``to_chat_request``,
    so each API shape's mapping stays identical to the untyped path while
    unmodelled fields keep flowing through ``ChatRequest.raw``.
    """
    return _chat_request_from_payload(model.model_dump(mode="json", exclude_none=True))


def openai_chat_request_to_canonical(model: OpenAIChatCompletionRequest) -> ChatRequest:
    """Build a canonical request from a typed OpenAI chat completion request."""
    return _chat_from_typed_model(model)


def lms_chat_request_to_canonical(request: LMSChatRequest) -> ChatRequest:
    """Build a canonical request from a typed LM Studio native chat request.

    The LM Studio-shape mapping (including ``system_prompt`` handling) stays
    identical to ``to_chat_request``.
    """
    return _chat_from_typed_model(request)


def _normalize_stop(stop: Any) -> list[str] | None:
    """Normalize ``stop`` to the canonical ``list[str] | None`` shape.

    OpenAI accepts a single string or a list, so a lone string is wrapped in a
    list rather than silently dropped; anything else is treated as absent.
    """
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return stop
    return None


def _chat_request_from_payload(payload: dict[str, Any]) -> ChatRequest:
    messages: list[dict[str, Any]] = []
    system_prompt = payload.get("system_prompt")
    if "messages" in payload:
        messages = list(payload.get("messages") or [])
    elif "input" in payload:
        messages = _normalize_messages_from_input(payload.get("input"))
    return ChatRequest(
        model=str(payload.get("model", "")),
        messages=messages,
        system_prompt=system_prompt,
        stream=bool(payload.get("stream", False)),
        previous_response_id=payload.get("previous_response_id"),
        store=bool(payload.get("store", True)),
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        top_k=payload.get("top_k"),
        min_p=payload.get("min_p"),
        repeat_penalty=payload.get("repeat_penalty"),
        max_output_tokens=(
            payload.get("max_output_tokens")
            or payload.get("max_completion_tokens")
            or payload.get("max_tokens")
        ),
        stop=_normalize_stop(payload.get("stop")),
        seed=payload.get("seed"),
        reasoning=payload.get("reasoning"),
        context_length=payload.get("context_length"),
        response_format=payload.get("response_format"),
        tools=payload.get("tools"),
        integrations=payload.get("integrations"),
        raw=dict(payload),
    )
