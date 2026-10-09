"""A minimal example translated provider.

This provider mirrors the structure third-party authors should follow when
implementing against ``lms_passthrough.provider_api``. It is used in the docs
and by the provider contract kit to prove the suite passes for a well-formed
implementation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from lms_passthrough.provider_api import (
    CanonicalOutput,
    Capability,
    ChatRequest,
    ChatResult,
    ProviderHTTPError,
    ProviderInfo,
    ProviderUnavailableError,
    StreamEvent,
    TranscriptionRequest,
    TranscriptionResult,
    synthesize_stream,
)

HTTP_ERROR_THRESHOLD = 400


class MinimalProvider:
    """Trivial OpenAI-compatible provider illustrating the provider contract."""

    def __init__(self, name: str, base_url: str, client: httpx.AsyncClient) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._client = client

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self._name,
            capabilities={
                Capability.CHAT,
                Capability.RESPONSES,
                Capability.MODELS,
                Capability.STREAMING,
            },
        )

    async def chat(self, request: ChatRequest) -> ChatResult:
        response = await self._post(
            "/v1/chat/completions",
            {"model": request.model, "messages": request.messages},
        )
        data = response.json()
        return ChatResult(
            model=request.model,
            outputs=[CanonicalOutput(content=str(data.get("output", "")))],
            raw=data,
        )

    async def responses(self, request: ChatRequest) -> ChatResult:
        return await self.chat(request)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def transcriptions(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise NotImplementedError

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._get("/v1/models")
        return list(response.json().get("data", []))

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        async for event in synthesize_stream(request, self.chat):
            yield event

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            response = await self._client.post(f"{self._base_url}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            raise ProviderHTTPError(response.status_code, response.text)
        return response

    async def _get(self, path: str) -> httpx.Response:
        try:
            response = await self._client.get(f"{self._base_url}{path}")
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            raise ProviderHTTPError(response.status_code, response.text)
        return response
