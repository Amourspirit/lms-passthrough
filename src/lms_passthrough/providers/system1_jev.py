"""System 1 JEV provider (``/v1/systemone``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from lms_passthrough.logging import get_logger, provider_request_extra
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

logger = get_logger("system1_jev")


class System1JevProvider:
    """Forwards JEV system-one requests; the upstream response is returned unchanged."""

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
        return ProviderInfo(name=self._name, capabilities={Capability.JEV})

    async def jev(self, payload: dict[str, Any]) -> dict[str, Any]:
        logger.debug(
            "provider request",
            extra=provider_request_extra(
                self._name,
                "POST",
                f"{self._base_url}/v1/systemone",
                headers=self._headers,
                body=payload,
            ),
        )
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/systemone", json=payload, headers=self._headers
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
            raise ProviderUnavailableError(f"Malformed JEV response from {self._name}") from exc

    def _unsupported(self) -> ProviderUnsupportedError:
        return ProviderUnsupportedError(
            f"Provider {self._name} only serves JEV system-one requests"
        )

    async def chat(self, request: ChatRequest) -> ChatResult:
        raise self._unsupported()

    async def responses(self, request: ChatRequest) -> ChatResult:
        return await self.chat(request)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        raise self._unsupported()

    async def transcriptions(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise self._unsupported()

    async def list_models(self) -> list[dict[str, Any]]:
        # No upstream catalog; models come from config `models:` mappings only.
        return []

    def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        raise self._unsupported()
