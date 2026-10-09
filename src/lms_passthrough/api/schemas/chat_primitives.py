"""Chat primitives shared between the LM Studio native and OpenAI chat HTTP schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextPart(BaseModel):
    """A text content part within a message."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    """An image URL reference with an optional rendering hint."""

    model_config = ConfigDict(extra="allow")

    url: str
    detail: str | None = None


class ImageUrlPart(BaseModel):
    """An image content part within a message."""

    model_config = ConfigDict(extra="allow")

    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Annotated[TextPart | ImageUrlPart, Field(discriminator="type")]


class ToolCallFunction(BaseModel):
    """The function payload of a tool call."""

    model_config = ConfigDict(extra="allow")

    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    """A tool invocation emitted on an assistant message."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ToolFunction(BaseModel):
    """The function a tool exposes to the model."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class Tool(BaseModel):
    """A tool offered to the model."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: ToolFunction


class ToolChoiceFunction(BaseModel):
    """The function a tool_choice object pins the model to."""

    model_config = ConfigDict(extra="allow")

    name: str


class ToolChoiceObject(BaseModel):
    """The object form of tool_choice."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: ToolChoiceFunction


ToolChoice = Literal["auto", "none", "required"] | ToolChoiceObject


class ChatMessage(BaseModel):
    """A single chat message shared across the chat HTTP schemas."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
