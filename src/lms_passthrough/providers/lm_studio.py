"""Transparent LM Studio provider."""

from __future__ import annotations

import json
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
    StreamEvent,
    TranscriptionRequest,
    TranscriptionResult,
)

HTTP_ERROR_THRESHOLD = 400


class LMStudioProvider:
    """Transparent provider that forwards LM Studio requests."""

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
                Capability.EMBEDDINGS,
                Capability.MODELS,
                Capability.TOOLS,
                Capability.IMAGES,
                Capability.STRUCTURED_OUTPUT,
                Capability.STREAMING,
                Capability.STATE,
            },
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}{path}",
                json=json_body,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            try:
                body = response.json()
            except ValueError:
                body = response.text
            raise ProviderHTTPError(response.status_code, body)
        return response

    async def chat(self, request: ChatRequest) -> ChatResult:
        response = await self._request("POST", "/api/v1/chat", json_body=request.raw)
        payload = response.json()
        return ChatResult(model=request.model, outputs=[], raw=payload)

    async def responses(self, request: ChatRequest) -> ChatResult:
        return await self.chat(request)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        response = await self._request("POST", "/v1/embeddings", json_body=request)
        return dict(response.json())

    async def transcriptions(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise NotImplementedError

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._request("GET", "/api/v1/models")
        return list(response.json().get("models", []))

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Incrementally parse the upstream native SSE stream."""
        body = {**request.raw, "stream": True}
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/v1/chat",
                json=body,
            ) as response:
                if response.status_code >= HTTP_ERROR_THRESHOLD:
                    await response.aread()
                    raise ProviderHTTPError(response.status_code, response.text)
                event_name: str | None = None
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event_name = line.removeprefix("event:").strip()
                    elif line.startswith("data:"):
                        data_lines.append(line.removeprefix("data:").lstrip())
                    elif not line.strip() and event_name is not None:
                        raw_data = "\n".join(data_lines) if data_lines else "{}"
                        try:
                            data = json.loads(raw_data)
                        except ValueError:
                            data = {"raw": raw_data}
                        yield StreamEvent(event=event_name, data=data)
                        event_name = None
                        data_lines = []
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
