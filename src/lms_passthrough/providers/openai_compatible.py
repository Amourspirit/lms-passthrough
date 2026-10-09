"""OpenAI-compatible provider."""

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
)
from lms_passthrough.provider_api.streaming import synthesize_stream

HTTP_ERROR_THRESHOLD = 400


class OpenAICompatibleProvider:
    """Configurable OpenAI-compatible provider."""

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
        self._api_key = api_key
        self._extra_headers = extra_headers or {}

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self._name,
            capabilities={
                Capability.CHAT,
                Capability.RESPONSES,
                Capability.EMBEDDINGS,
                Capability.MODELS,
                Capability.STREAMING,
                Capability.TRANSCRIPTIONS,
            },
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        headers.update(self._extra_headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = await self._client.post(
                f"{self._base_url}{path}",
                json=payload,
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
        messages = request.messages
        if messages and all(m.get("role") == "system" for m in messages):
            # z.ai-backed upstreams reject system-only payloads (code 1214); fabric
            # sends pattern+input as a single system message. Demote to user.
            messages = [{**m, "role": "user"} for m in messages]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        response = await self._post("/v1/chat/completions", payload)
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return ChatResult(
            model=request.model,
            outputs=[CanonicalOutput(content=str(message.get("content", "")))],
            raw=data,
        )

    async def responses(self, request: ChatRequest) -> ChatResult:
        return await self.chat(request)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        response = await self._post("/v1/embeddings", request)
        return dict(response.json())

    async def transcriptions(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Forward a transcription to the upstream OpenAI-compatible route.

        httpx builds the multipart envelope and its own boundary, so the
        forwarded `Content-Type` is never the inbound one — unlike the
        transparent lm-studio path, which is deliberately byte-for-byte.
        """
        fields = {"model": request.model, "response_format": request.response_format}
        if request.language is not None:
            fields["language"] = request.language
        if request.prompt is not None:
            fields["prompt"] = request.prompt
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/audio/transcriptions",
                headers=self._headers(),
                files={
                    "file": (
                        request.filename,
                        request.audio,
                        request.content_type or "application/octet-stream",
                    )
                },
                data=fields,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            try:
                body: dict[str, Any] | str = response.json()
            except ValueError:
                body = response.text
            raise ProviderHTTPError(response.status_code, body)
        data = response.json()
        return TranscriptionResult(
            model=request.model,
            text=str(data.get("text", "")),
            language=data.get("language"),
            duration=data.get("duration"),
            segments=data.get("segments"),
            raw=data,
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        headers.update(self._extra_headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def list_models(self) -> list[dict[str, Any]]:
        # Routed through the same error mapping as `_post`: the proxy marks a
        # provider down from a failed model probe, so a raw ConnectError or a
        # JSON decode error here would be silently misclassified.
        try:
            response = await self._client.get(
                f"{self._base_url}/v1/models", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            raise ProviderHTTPError(response.status_code, response.text)
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(
                f"Malformed model list from {self._name}"
            ) from exc
        return list(data.get("data", []))

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        async for event in synthesize_stream(request, self.chat):
            yield event
