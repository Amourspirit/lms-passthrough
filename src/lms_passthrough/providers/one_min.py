"""1min.AI providers."""

from __future__ import annotations

import json
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
from lms_passthrough.providers.base import serialize_transcript

HTTP_ERROR_THRESHOLD = 400


class OneMinBaseProvider:
    """Shared 1min.AI functionality."""

    def __init__(self, name: str, base_url: str, api_key: str, client: httpx.AsyncClient) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client

    @property
    def info(self) -> ProviderInfo:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "API-KEY": self._api_key}

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        stream: bool = False,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        if stream:
            url += "?isStreaming=true"
        try:
            response = await self._client.post(url, headers=self._headers(), json=payload)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            try:
                body: dict[str, Any] | str = response.json()
            except ValueError:
                body = response.text
            raise ProviderHTTPError(response.status_code, body)
        return response

    async def _get(self, path: str) -> httpx.Response:
        try:
            response = await self._client.get(f"{self._base_url}{path}", headers=self._headers())
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            raise ProviderHTTPError(response.status_code, response.text)
        return response

    async def _post_multipart(
        self,
        path: str,
        *,
        field: str,
        filename: str,
        audio: bytes,
        content_type: str | None,
    ) -> httpx.Response:
        """POST raw audio as a single multipart file field.

        `_headers` is deliberately not reused: it pins `Content-Type:
        application/json`, which would override the multipart boundary httpx
        generates and produce an unparseable body upstream.
        """
        try:
            response = await self._client.post(
                f"{self._base_url}{path}",
                headers={"API-KEY": self._api_key},
                files={field: (filename, audio, content_type or "application/octet-stream")},
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            try:
                body: dict[str, Any] | str = response.json()
            except ValueError:
                body = response.text
            raise ProviderHTTPError(response.status_code, body)
        return response


def _outputs_from_ai_record(data: dict[str, Any]) -> list[CanonicalOutput]:
    result = data.get("aiRecord", {}).get("aiRecordDetail", {}).get("resultObject", [])
    return [CanonicalOutput(content=str(item)) for item in result]


class OneMinChatProvider(OneMinBaseProvider):
    """1min chat provider."""

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self._name,
            capabilities={
                Capability.CHAT,
                Capability.RESPONSES,
                Capability.MODELS,
                Capability.STREAMING,
                Capability.TRANSCRIPTIONS,
            },
        )

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        return {
            "type": "UNIFY_CHAT_WITH_AI",
            "model": request.model,
            "promptObject": {
                "prompt": serialize_transcript(request.messages, request.system_prompt),
                "settings": {
                    "historySettings": {"isMixed": False, "historyMessageLimit": 10},
                    "withMemories": False,
                },
            },
        }

    async def chat(self, request: ChatRequest) -> ChatResult:
        response = await self._post("/api/chat-with-ai", self._payload(request))
        data = response.json()
        return ChatResult(
            model=request.model,
            outputs=_outputs_from_ai_record(data),
            raw=data,
        )

    async def responses(self, request: ChatRequest) -> ChatResult:
        return await self.chat(request)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def transcriptions(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe via the 1min `SPEECH_TO_TEXT` feature.

        Two hops, because the feature API takes no bytes: audio is uploaded
        through the Asset API first and the returned `asset.key` is passed back
        as `promptObject.audioUrl`.

        `response_format` is always sent as `text` to 1min. The proxy builds
        the OpenAI-shaped envelope itself, and 1min's own `json` format would
        hand back a JSON document nested inside a string in `resultObject`.
        """
        upload = await self._post_multipart(
            "/api/assets",
            field="asset",
            filename=request.filename,
            audio=request.audio,
            content_type=request.content_type,
        )
        asset_key = upload.json().get("asset", {}).get("key")
        if not asset_key:
            raise ProviderUnavailableError("1min asset upload returned no key")

        prompt_object: dict[str, Any] = {
            "audioUrl": str(asset_key),
            "response_format": "text",
        }
        if request.language is not None:
            prompt_object["language"] = request.language
        if request.prompt is not None:
            prompt_object["prompt"] = request.prompt

        response = await self._post(
            "/api/features",
            {
                "type": "SPEECH_TO_TEXT",
                "model": request.model,
                "promptObject": prompt_object,
            },
        )
        data = response.json()
        outputs = _outputs_from_ai_record(data)
        return TranscriptionResult(
            model=request.model,
            text=outputs[0].content if outputs else "",
            raw=data,
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """Chat and speech-to-text models, as one catalog.

        The two listings are fetched independently on purpose. `is_model_supported`
        marks a provider Down when `list_models` raises and `failover` is on, so
        letting a failing STT listing propagate would take the provider out of
        chat service too. A failed STT listing degrades to an empty STT half.
        """
        response = await self._get("/models?feature=UNIFY_CHAT_WITH_AI")
        models = list(response.json().get("models", []))
        try:
            stt = await self._get("/models?feature=SPEECH_TO_TEXT")
        except (ProviderHTTPError, ProviderUnavailableError):
            return models
        return models + list(stt.json().get("models", []))

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Stream real 1min chat SSE, translated to canonical events."""
        payload = self._payload(request)
        url = f"{self._base_url}/api/chat-with-ai?isStreaming=true"
        yield StreamEvent(event="chat.start", data={"type": "chat.start"})
        try:
            async with self._client.stream(
                "POST", url, headers=self._headers(), json=payload
            ) as response:
                if response.status_code >= HTTP_ERROR_THRESHOLD:
                    await response.aread()
                    raise ProviderHTTPError(response.status_code, response.text)
                message_open = False
                async for event_name, data in _sse_frames(response.aiter_lines()):
                    if event_name == "content":
                        # 1min brackets the stream with empty content frames; drop them.
                        content = data.get("content", "")
                        if not content:
                            continue
                        if not message_open:
                            message_open = True
                            yield StreamEvent(event="message.start", data={"type": "message.start"})
                        yield StreamEvent(event="message.delta", data={"content": content})
                    elif event_name == "result":
                        if message_open:
                            yield StreamEvent(event="message.end", data={"type": "message.end"})
                        yield StreamEvent(event="chat.end", data={"result": data})
                    elif event_name == "error":
                        # Terminal: nothing follows an error event.
                        yield StreamEvent(event="error", data=data)
                        return
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc


async def _sse_frames(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Group raw SSE lines into (event, parsed JSON data) frames."""
    event_name: str | None = None
    data_lines: list[str] = []
    async for line in lines:
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())
        elif not line.strip() and event_name is not None:
            raw = "\n".join(data_lines) if data_lines else "{}"
            try:
                data: dict[str, Any] = json.loads(raw)
            except ValueError:
                data = {"raw": raw}
            yield event_name, data
            event_name = None
            data_lines = []


class OneMinCodeGeneratorProvider(OneMinBaseProvider):
    """1min code generator provider."""

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self._name,
            capabilities={Capability.CHAT, Capability.RESPONSES, Capability.MODELS},
        )

    async def chat(self, request: ChatRequest) -> ChatResult:
        payload = {
            "type": "CODE_GENERATOR",
            "model": request.model,
            "promptObject": {
                "prompt": serialize_transcript(request.messages, request.system_prompt),
                "webSearch": False,
            },
        }
        response = await self._post("/api/features", payload)
        data = response.json()
        return ChatResult(
            model=request.model,
            outputs=_outputs_from_ai_record(data),
            raw=data,
        )

    async def responses(self, request: ChatRequest) -> ChatResult:
        return await self.chat(request)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def transcriptions(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise NotImplementedError

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._get("/models?feature=CODE_GENERATOR")
        return list(response.json().get("models", []))

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Code Generator has no documented stream; synthesize one."""
        async for event in synthesize_stream(request, self.chat):
            yield event
