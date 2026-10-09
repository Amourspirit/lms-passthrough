"""OpenAI chat completions HTTP request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from lms_passthrough.api.schemas.chat_primitives import (
    ChatMessage,
    Tool,
    ToolChoice,
)


class StreamOptions(BaseModel):
    """OpenAI ``stream_options`` parameter."""

    model_config = ConfigDict(extra="allow")

    include_usage: bool | None = None
    continuous_usage_stats: bool | None = None


class ResponseFormat(BaseModel):
    """OpenAI ``response_format`` parameter."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class OpenAIChatCompletionRequest(BaseModel):
    """The ``/v1/chat/completions`` request body.

    Field commitments follow issue 03 (typed HTTP boundary): core fields and
    sampling knobs typed (including OpenAI's ``max_completion_tokens`` alias),
    tools / structured output typed as first-class, and everything else
    declared with loose types plus ``extra="allow"`` so unmodelled fields pass
    through unchanged.
    """

    model_config = ConfigDict(extra="allow")

    # Tier A: core.
    model: str
    messages: list[ChatMessage]
    stream: StrictBool = False
    stream_options: StreamOptions | None = None

    # Tier B: sampling knobs.
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    # Tier C: tools / structured output, typed as first-class.
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: ResponseFormat | None = None

    # State continuation used by this proxy.
    previous_response_id: str | None = None

    # Tier D: passed through, not validated.
    user: Any = Field(default=None, description="Passed through; not validated.")
    logit_bias: Any = Field(default=None, description="Passed through; not validated.")
    n: Any = Field(default=None, description="Passed through; not validated.")
    logprobs: Any = Field(default=None, description="Passed through; not validated.")
    top_logprobs: Any = Field(default=None, description="Passed through; not validated.")
    service_tier: Any = Field(default=None, description="Passed through; not validated.")
    metadata: Any = Field(default=None, description="Passed through; not validated.")
    store: Any = Field(default=None, description="Passed through; not validated.")


class OpenAIChatCompletionUsage(BaseModel):
    """Usage numbers of a chat completion."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionMessage(BaseModel):
    """The assistant message inside a chat completion choice."""

    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[Any] | None = None


class OpenAIChatCompletionChoice(BaseModel):
    """One completion choice."""

    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: OpenAIChatCompletionMessage = Field(default_factory=OpenAIChatCompletionMessage)
    finish_reason: str | None = None


class OpenAIChatCompletionResponse(BaseModel):
    """The non-streaming ``/v1/chat/completions`` response body."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChoice]
    usage: OpenAIChatCompletionUsage | None = None
