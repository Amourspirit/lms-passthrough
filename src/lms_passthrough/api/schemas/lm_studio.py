"""LM Studio native chat HTTP request/response schemas.

Mirrors the LM Studio 0.4.x native ``POST /api/v1/chat`` shape. Deliberately
separate from ``openai_chat.OpenAIChatCompletionRequest`` per ADR-0001: per-spec
models, no inheritance or shared base.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from lms_passthrough.api.schemas.chat_primitives import ChatMessage


class LMSChatRequest(BaseModel):
    """The ``/api/v1/chat`` request body.

    Field commitments follow the typed HTTP boundary ADR: core fields and the
    native-only sampling knobs (``system_prompt``, ``top_k``, ``min_p``,
    ``repeat_penalty``, ``context_length``, ``reasoning``) are typed, and
    everything else is declared with loose types plus ``extra="allow"`` so
    unmodelled LM Studio fields pass through into ``ChatRequest.raw``.
    """

    model_config = ConfigDict(extra="allow")

    # Core.
    model: str
    input: str | list[dict[str, Any]] | None = Field(
        default=None,
        description="Passed through; not validated.",
    )
    messages: list[ChatMessage] | None = None
    system_prompt: str | None = None
    previous_response_id: str | None = None
    store: StrictBool = True
    stream: StrictBool = False

    # Sampling knobs (native-only ones included).
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    reasoning: str | None = Field(
        default=None,
        description='LM Studio reasoning setting: "off" | "low" | "medium" | "high" | "on".',
    )
    context_length: int | None = None

    # Tools / structured output are not part of the native chat spec but flow
    # through to translated providers; kept loose so unknown shapes pass.
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Passed through; not validated.",
    )
    response_format: dict[str, Any] | None = Field(
        default=None,
        description="Passed through; not validated.",
    )

    # LM Studio MCP integrations (plugins, ephemeral servers).
    integrations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Passed through; not validated.",
    )


class LMSChatOutput(BaseModel):
    """One output item of a native chat response (canonical output shape)."""

    model_config = ConfigDict(extra="allow")

    type: str = "message"
    content: str = ""
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    output: str | None = None
    provider_info: dict[str, Any] | None = None


class LMSChatStats(BaseModel):
    """Token usage and performance metrics of a native chat response."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    tokens_per_second: float | None = None
    time_to_first_token_seconds: float | None = None
    model_load_time_seconds: float | None = None


class LMSChatResponse(BaseModel):
    """The non-streaming ``/api/v1/chat`` response body."""

    model_config = ConfigDict(extra="allow")

    model_instance_id: str
    output: list[LMSChatOutput]
    stats: LMSChatStats
    response_id: str | None = None
