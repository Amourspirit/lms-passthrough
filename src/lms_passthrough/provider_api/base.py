"""Canonical provider contracts for translated providers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol


class Capability(StrEnum):
    """Capabilities a provider may support."""

    CHAT = "chat"
    RESPONSES = "responses"
    EMBEDDINGS = "embeddings"
    MODELS = "models"
    TOOLS = "tools"
    IMAGES = "images"
    STRUCTURED_OUTPUT = "structured_output"
    STREAMING = "streaming"
    STATE = "state"
    TRANSCRIPTIONS = "transcriptions"
    DECISIONS = "decisions"


class CanonicalMessage(dict[str, Any]):
    """Placeholder for canonical message."""


@dataclass(slots=True)
class CanonicalOutput:
    """Canonical model output."""

    type: Literal["message", "reasoning", "tool_call", "invalid_tool_call"] = "message"
    content: str = ""
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    output: str | None = None
    provider_info: dict[str, Any] | None = None


@dataclass(slots=True)
class CanonicalUsage:
    """Usage statistics."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    tokens_per_second: float | None = None
    time_to_first_token_seconds: float | None = None
    model_load_time_seconds: float | None = None


@dataclass(slots=True)
class ChatRequest:
    """Canonical inference request."""

    model: str
    messages: list[dict[str, Any]]
    system_prompt: str | None = None
    stream: bool = False
    previous_response_id: str | None = None
    store: bool = True
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    max_output_tokens: int | None = None
    stop: list[str] | None = None
    seed: int | None = None
    reasoning: str | None = None
    context_length: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    integrations: list[Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResult:
    """Canonical inference result."""

    model: str
    outputs: list[CanonicalOutput]
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)
    response_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamEvent:
    """Canonical stream event."""

    event: str
    data: dict[str, Any]


@dataclass(slots=True)
class TranscriptionRequest:
    """Canonical speech-to-text request.

    ``audio`` is the decoded file bytes. The HTTP boundary is responsible for
    parsing the multipart envelope; providers receive the audio, never the
    envelope.
    """

    model: str
    audio: bytes
    filename: str
    content_type: str | None = None
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"


@dataclass(slots=True)
class TranscriptionResult:
    """Canonical speech-to-text result.

    Only ``text`` is a commitment. The remaining fields are whatever the
    upstream volunteered; providers that do not report them leave them unset.
    """

    model: str
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict[str, Any]] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderInfo:
    """Provider metadata."""

    name: str
    capabilities: set[Capability]


class ProviderError(Exception):
    """Base provider error."""


class ProviderHTTPError(ProviderError):
    """Provider HTTP error with response metadata."""

    def __init__(self, status_code: int, body: dict[str, Any] | str) -> None:
        super().__init__(f"Provider HTTP error: {status_code}")
        self.status_code = status_code
        self.body = body


class ProviderUnavailableError(ProviderError):
    """Provider is unavailable."""


class ProviderOverloadedError(ProviderError):
    """Provider is at capacity and cannot accept more inference calls."""


class ProviderUnsupportedError(ProviderError):
    """Provider does not support the requested capability."""


class Provider(Protocol):
    """Async provider protocol."""

    @property
    def info(self) -> ProviderInfo:
        """Return provider metadata."""
        ...

    async def chat(self, request: ChatRequest) -> ChatResult:
        """Perform a chat-style inference call."""
        ...

    async def responses(self, request: ChatRequest) -> ChatResult:
        """Perform an OpenAI Responses-style inference call."""
        ...

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Perform embeddings call."""
        ...

    async def transcriptions(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Perform a speech-to-text call."""
        ...

    async def list_models(self) -> list[dict[str, Any]]:
        """List models exposed by the provider."""
        ...

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Stream canonical chat events."""
        ...
