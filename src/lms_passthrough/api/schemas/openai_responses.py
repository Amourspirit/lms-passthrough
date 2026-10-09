"""OpenAI Responses API request/response models (alias-scoped)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lms_passthrough.api.schemas.chat_primitives import (
    ContentPart,
    ImageUrl,
    Tool,
    ToolChoice,
)


class InputItem(BaseModel):
    """A single user/assistant message-style item in an `input` list."""

    model_config = ConfigDict(extra="allow")

    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"] = "user"
    content: str | list[Annotated[ContentPart, Field(discriminator="type")]] | None = None


class ImageInputItem(BaseModel):
    """An image item in an `input` list, forwarded as a user message."""

    model_config = ConfigDict(extra="allow")

    type: Literal["image"]
    # Re-use the image URL shape from chat primitives; keep ``source`` loose
    # for other image-encoding conventions that may appear in raw payloads.
    image_url: ImageUrl | None = None
    source: Any | None = None


InputItemUnion = Annotated[InputItem | ImageInputItem, Field(discriminator="type")]


class OpenAIResponsesRequest(BaseModel):
    """OpenAI Responses API request covering only the fields we translate.

    The Responses spec defines additional fields (``input_items``, ``include``,
    ``truncation``, async/background responses, real tool-call chain replay). We
    deliberately do not model those so that ``/docs`` honestly reflects our
    alias-to-chat implementation, but ``extra="allow"`` keeps them passing
    through at runtime.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[Annotated[InputItemUnion, Field(discriminator="type")]]
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool = True
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    response_format: dict[str, Any] | None = None


class OpenAIResponsesOutputText(BaseModel):
    """An output_text content part in a Responses output message."""

    model_config = ConfigDict(extra="allow")

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = []


class OpenAIResponsesOutputMessage(BaseModel):
    """A message-shaped item in the Responses ``output`` array."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["message"] = "message"
    status: Literal["in_progress", "completed", "incomplete"]
    role: Literal["assistant"] = "assistant"
    content: list[OpenAIResponsesOutputText]


class OpenAIResponsesUsage(BaseModel):
    """Token usage for a Responses response."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int
    total_tokens: int | None = None


class OpenAIResponsesResponse(BaseModel):
    """OpenAI Responses API non-streaming response."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: Literal["completed", "in_progress", "incomplete"]
    output: list[OpenAIResponsesOutputMessage]
    output_text: str
    usage: OpenAIResponsesUsage | None = None
