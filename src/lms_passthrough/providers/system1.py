"""System 1 decision-model provider (JEV ``jev_decide``-style ``/api/v1/decisions``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from lms_passthrough.provider_api import (
    Capability,
    ChatRequest,
    ChatResult,
    ProviderHTTPError,
    ProviderInfo,
    ProviderUnavailableError,
    ProviderUnsupportedError,
    StreamEvent,
    TranscriptionRequest,
    TranscriptionResult,
)

HTTP_ERROR_THRESHOLD = 400


class System1Provider:
    """Forwards decision requests; the upstream envelope is returned unchanged."""

    def __init__(
        self,
        name: str,
        base_url: str,
        client: httpx.AsyncClient,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._headers = dict(extra_headers or {})
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self._name, capabilities={Capability.DECISIONS})

    async def decisions(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                f"{self._base_url}/api/v1/decisions", json=payload, headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            try:
                body: dict[str, Any] | str = response.json()
            except ValueError:
                body = response.text
            raise ProviderHTTPError(response.status_code, body)
        try:
            return dict(response.json())
        except ValueError as exc:
            raise ProviderUnavailableError(f"Malformed decision from {self._name}") from exc

    async def chat(self, request: ChatRequest) -> ChatResult:
        raise ProviderUnsupportedError(f"Provider {self._name} only serves decisions")

    async def responses(self, request: ChatRequest) -> ChatResult:
        return await self.chat(request)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        raise ProviderUnsupportedError(f"Provider {self._name} only serves decisions")

    async def transcriptions(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise ProviderUnsupportedError(f"Provider {self._name} only serves decisions")

    async def list_models(self) -> list[dict[str, Any]]:
        # No upstream catalog; models come from config `models:` mappings only.
        return []

    def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        raise ProviderUnsupportedError(f"Provider {self._name} only serves decisions")
