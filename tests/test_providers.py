from __future__ import annotations

import pytest

from lms_passthrough.provider_api import ChatRequest, ProviderHTTPError
from lms_passthrough.provider_api.contracts import FakeClient as ContractFakeClient
from lms_passthrough.providers.lm_studio import LMStudioProvider
from lms_passthrough.providers.one_min import OneMinChatProvider, OneMinCodeGeneratorProvider
from lms_passthrough.providers.openai_compatible import OpenAICompatibleProvider


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    async def request(self, *args, **kwargs):
        return self._response

    async def post(self, *args, **kwargs):
        return self._response

    async def get(self, *args, **kwargs):
        return self._response

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_lm_studio_chat_returns_raw_payload() -> None:
    client = FakeClient(FakeResponse(200, {"model_instance_id": "inst-1", "output": []}))
    provider = LMStudioProvider("lm-studio", "http://127.0.0.1:1234", client)  # type: ignore[arg-type]
    result = await provider.chat(ChatRequest(model="m", messages=[]))
    assert result.raw["model_instance_id"] == "inst-1"


@pytest.mark.asyncio
async def test_lm_studio_http_error_raises() -> None:
    client = FakeClient(FakeResponse(500, {"error": "boom"}))
    provider = LMStudioProvider("lm-studio", "http://127.0.0.1:1234", client)  # type: ignore[arg-type]
    with pytest.raises(ProviderHTTPError):
        await provider.chat(ChatRequest(model="m", messages=[]))


@pytest.mark.asyncio
async def test_one_min_chat_maps_result_object() -> None:
    payload = {"aiRecord": {"aiRecordDetail": {"resultObject": ["a", "b"]}}}
    client = FakeClient(FakeResponse(200, payload))
    provider = OneMinChatProvider("1min-chat", "https://api.1min.ai", "k", client)  # type: ignore[arg-type]
    result = await provider.chat(
        ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    )
    assert [o.content for o in result.outputs] == ["a", "b"]


@pytest.mark.asyncio
async def test_code_generator_maps_result_object() -> None:
    payload = {"aiRecord": {"aiRecordDetail": {"resultObject": ["code"]}}}
    client = FakeClient(FakeResponse(200, payload))
    provider = OneMinCodeGeneratorProvider(
        "1min-code-generator",
        "https://api.1min.ai",
        "k",
        client,
    )  # type: ignore[arg-type]
    result = await provider.chat(
        ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    )
    assert [o.content for o in result.outputs] == ["code"]


@pytest.mark.asyncio
async def test_openai_compatible_chat_maps_choice() -> None:
    payload = {"choices": [{"message": {"content": "hello"}}]}
    client = FakeClient(FakeResponse(200, payload))
    provider = OpenAICompatibleProvider("android", "http://android", client)  # type: ignore[arg-type]
    result = await provider.chat(
        ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    )
    assert result.outputs[0].content == "hello"


class RecordingPostClient(FakeClient):
    def __init__(self, response: FakeResponse) -> None:
        super().__init__(response)
        self.last_json: object = None

    async def post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.last_json = kwargs.get("json")
        return self._response


@pytest.mark.asyncio
async def test_openai_compatible_demotes_system_only_to_user() -> None:
    payload = {"choices": [{"message": {"content": "hello"}}]}
    client = RecordingPostClient(FakeResponse(200, payload))
    provider = OpenAICompatibleProvider("tokenra", "https://tokenra.io", client)  # type: ignore[arg-type]

    await provider.chat(
        ChatRequest(model="m", messages=[{"role": "system", "content": "Say hi"}])
    )
    assert client.last_json == {
        "model": "m",
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": False,
    }

    await provider.chat(
        ChatRequest(
            model="m",
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        )
    )
    assert client.last_json == {  # type: ignore[comparison-overlap]
        "model": "m",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
    }


class RecordingClient(ContractFakeClient):
    """FakeClient that remembers the last ``stream`` call."""

    def stream(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        self.last = (method, url, kwargs)
        return super().stream(method, url, **kwargs)


# Shape captured live from api.1min.ai on 2026-09-15 (empty content frames bracket the stream).
ONE_MIN_STREAM = (
    "event: content",
    'data: {"content":""}',
    "",
    "event: content",
    'data: {"content":"Hello"}',
    "",
    "event: content",
    'data: {"content":" there"}',
    "",
    "event: content",
    'data: {"content":""}',
    "",
    "event: result",
    'data: {"aiRecord":{"aiRecordDetail":{"resultObject":["Hello there"]}}}',
    "",
    "event: done",
    'data: {"message":"Stream completed"}',
    "",
)


@pytest.mark.asyncio
async def test_one_min_stream_wire_format_and_canonical_grammar() -> None:
    client = RecordingClient(stream_lines=ONE_MIN_STREAM)
    provider = OneMinChatProvider("1min-chat", "https://api.1min.ai", "k", client)  # type: ignore[arg-type]
    events = [
        e
        async for e in provider.stream_chat(
            ChatRequest(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        )
    ]

    method, url, kwargs = client.last
    assert (method, url) == ("POST", "https://api.1min.ai/api/chat-with-ai?isStreaming=true")
    assert kwargs["headers"]["API-KEY"] == "k"
    assert kwargs["json"]["type"] == "UNIFY_CHAT_WITH_AI"
    assert kwargs["json"]["model"] == "gpt-4o-mini"
    assert kwargs["json"]["promptObject"]["prompt"] == "User: hi"

    assert [e.event for e in events] == [
        "chat.start",
        "message.start",
        "message.delta",
        "message.delta",
        "message.end",
        "chat.end",
    ]
    assert "".join(e.data["content"] for e in events if e.event == "message.delta") == "Hello there"
    result = events[-1].data["result"]
    assert result["aiRecord"]["aiRecordDetail"]["resultObject"] == ["Hello there"]
