"""Contract suite coverage for built-in providers and the example provider."""

from __future__ import annotations

import pytest

from lms_passthrough.provider_api import (
    Capability,
    ChatRequest,
    ChatResult,
    ContractViolation,
    ProviderInfo,
    StreamEvent,
    run_contract_tests,
)
from lms_passthrough.provider_api import contracts as provider_contracts
from lms_passthrough.provider_api.contracts import FakeClient
from lms_passthrough.provider_api.example import MinimalProvider
from lms_passthrough.providers.lm_studio import LMStudioProvider
from lms_passthrough.providers.one_min import OneMinChatProvider, OneMinCodeGeneratorProvider
from lms_passthrough.providers.openai_compatible import OpenAICompatibleProvider


def make_lm_studio() -> LMStudioProvider:
    client = FakeClient(
        stream_lines=(
            "event: chat.start",
            "data: {}",
            "",
            "event: message.delta",
            'data: {"content": "hi"}',
            "",
            "event: chat.end",
            "data: {}",
            "",
        )
    )
    return LMStudioProvider("lm-studio", "http://127.0.0.1:1234", client)  # type: ignore[arg-type]


def make_openai_compatible() -> OpenAICompatibleProvider:
    client = FakeClient(payload={"choices": [{"message": {"content": "hello"}}]})
    return OpenAICompatibleProvider("android", "http://android", client)  # type: ignore[arg-type]


def make_one_min_chat() -> OneMinChatProvider:
    client = FakeClient(
        payload={"aiRecord": {"aiRecordDetail": {"resultObject": ["a"]}}},
        stream_lines=(
            'event: content',
            'data: {"content": "hi"}',
            "",
            "event: result",
            "data: {}",
            "",
        ),
    )
    return OneMinChatProvider("1min-chat", "https://api.1min.ai", "k", client)  # type: ignore[arg-type]


def make_one_min_codegen() -> OneMinCodeGeneratorProvider:
    client = FakeClient(payload={"aiRecord": {"aiRecordDetail": {"resultObject": ["code"]}}})
    return OneMinCodeGeneratorProvider(  # type: ignore[arg-type]
        "1min-code-generator", "https://api.1min.ai", "k", client
    )


def make_minimal() -> MinimalProvider:
    client = FakeClient(payload={"output": "hello"})
    return MinimalProvider("minimal", "http://example", client)  # type: ignore[arg-type]


def test_lm_studio_passes_contract() -> None:
    run_contract_tests(make_lm_studio())


def test_openai_compatible_passes_contract() -> None:
    run_contract_tests(make_openai_compatible())


def test_one_min_chat_passes_contract() -> None:
    run_contract_tests(make_one_min_chat())


def test_one_min_codegen_passes_contract() -> None:
    run_contract_tests(make_one_min_codegen())


def test_minimal_example_passes_contract() -> None:
    run_contract_tests(make_minimal())


def test_contract_suite_importable_from_provider_api() -> None:
    assert run_contract_tests is provider_contracts.run_contract_tests
    assert ContractViolation is provider_contracts.ContractViolation


async def test_run_contract_tests_works_from_async_test() -> None:
    run_contract_tests(make_minimal())


async def test_synthesized_stream_closes_cleanly() -> None:
    provider = make_minimal()
    stream = provider.stream_chat(ChatRequest(model="m", messages=[]))
    await stream.__anext__()
    await stream.aclose()
    await stream.aclose()


class DeclaresEmbeddingsButBroken:
    """A provider that declares EMBEDDINGS but does not implement it."""

    def __init__(self) -> None:
        self._client = FakeClient()

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="broken",
            capabilities={Capability.CHAT, Capability.EMBEDDINGS},
        )

    async def chat(self, request: ChatRequest) -> ChatResult:
        return ChatResult(model=request.model, outputs=[])

    async def responses(self, request: ChatRequest) -> ChatResult:
        return await self.chat(request)

    async def embeddings(self, request: dict) -> dict:
        raise NotImplementedError

    async def list_models(self) -> list[dict]:
        return []

    async def stream_chat(self, request: ChatRequest):
        yield StreamEvent(event="chat.start", data={})
        yield StreamEvent(event="chat.end", data={})


def test_provider_missing_declared_capability_fails() -> None:
    with pytest.raises(ContractViolation):
        run_contract_tests(DeclaresEmbeddingsButBroken())


async def test_lm_studio_stream_error_still_terminates() -> None:
    provider = LMStudioProvider(  # type: ignore[arg-type]
        "lm-studio",
        "http://127.0.0.1:1234",
        FakeClient(
            stream_lines=(
                "event: chat.start",
                "data: {}",
                "",
                "event: error",
                'data: {"message": "boom"}',
                "",
            )
        ),
    )
    events = [event async for event in provider.stream_chat(ChatRequest(model="m", messages=[]))]
    assert events[0].event == "chat.start"
    assert events[-1].event == "error"


async def test_one_min_chat_stream_error_still_terminates() -> None:
    provider = OneMinChatProvider(  # type: ignore[arg-type]
        "1min-chat",
        "https://api.1min.ai",
        "k",
        FakeClient(
            stream_lines=(
                'event: content',
                'data: {"content": "partial"}',
                "",
                'event: error',
                'data: {"message": "boom"}',
                "",
            )
        ),
    )
    events = [event async for event in provider.stream_chat(ChatRequest(model="m", messages=[]))]
    assert [e.event for e in events] == ["chat.start", "message.start", "message.delta", "error"]


async def test_synthesized_stream_error_is_terminal() -> None:
    provider = OpenAICompatibleProvider(  # type: ignore[arg-type]
        "android",
        "http://android",
        FakeClient(status_code=500),
    )
    events = [event async for event in provider.stream_chat(ChatRequest(model="m", messages=[]))]
    names = [e.event for e in events]
    assert names == ["chat.start", "error"]
