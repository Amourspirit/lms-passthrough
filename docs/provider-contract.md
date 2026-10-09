# Provider authoring guide

Implement a third-party provider by conforming to the `Provider` protocol in
`lms_passthrough.provider_api`. This guide walks through the contract, the
canonical types, the capability system, error handling, and the reusable
contract test suite that verifies conformance.

## The Provider protocol

A provider is an async object that exposes one property and five methods:

```python
from lms_passthrough.provider_api import Provider

class MyProvider:
    @property
    def info(self) -> ProviderInfo: ...

    async def chat(self, request: ChatRequest) -> ChatResult: ...
    async def responses(self, request: ChatRequest) -> ChatResult: ...
    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]: ...
    async def list_models(self) -> list[dict[str, Any]]: ...
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]: ...
```

The protocol is structural — Python's `Protocol` is duck-typed. You do not need
to inherit from it; your class just needs to match the interface.

### The `info` property

`ProviderInfo` carries metadata the proxy uses for capability validation and
routing:

```python
@dataclass(slots=True)
class ProviderInfo:
    name: str
    capabilities: set[Capability]
```

`name` is the provider's user-facing identifier. `capabilities` is a set of
`Capability` members the provider actually implements. Declare only what you
implement — the proxy rejects requests whose features exceed your declared set
before they reach your code.

Available capabilities:

| Capability | Endpoint / feature gated |
| --- | --- |
| `CHAT` | `POST /api/v1/chat`, `POST /v1/chat/completions` |
| `RESPONSES` | `POST /v1/responses` |
| `EMBEDDINGS` | `POST /v1/embeddings` |
| `MODELS` | `GET /api/v1/models`, `GET /v1/models` |
| `TOOLS` | `tools` parameter on chat requests |
| `IMAGES` | Image content in message payloads |
| `STRUCTURED_OUTPUT` | `response_format` parameter on chat requests |
| `STREAMING` | `stream: true` on chat requests |
| `STATE` | Conversation state (not currently used by the proxy's validation) |

### Required methods

| Method | Parameter type | Return type | Notes |
| --- | --- | --- | --- |
| `chat` | `ChatRequest` | `ChatResult` | Primary inference path. The canonical type is the same shape the proxy uses internally. |
| `responses` | `ChatRequest` | `ChatResult` | OpenAI Responses API entry point. Often implemented by delegating to `chat`. |
| `embeddings` | `dict[str, Any]` | `dict[str, Any]` | Plain dict I/O — no canonical wrapper. Raise `NotImplementedError` if not supported. |
| `list_models` | _(no args)_ | `list[dict[str, Any]]` | Return a list of model catalog entries. |
| `stream_chat` | `ChatRequest` | `AsyncIterator[StreamEvent]` | Streaming entry point. Must emit the canonical SSE lifecycle in order. |

## Canonical types

The proxy parses external request payloads (LM Studio native, OpenAI chat
completions, OpenAI responses) into canonical types before calling provider
methods. Providers never see raw HTTP payloads — only `ChatRequest`,
`ChatResult`, `StreamEvent`, and `dict[str, Any]`.

### `ChatRequest`

```python
@dataclass(slots=True)
class ChatRequest:
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
```

Only `model` and `messages` are guaranteed to be set. All other fields come from
the translated client request and may be `None`. Use `raw` to pass through
provider-specific fields the proxy cannot model as canonical types.

### `ChatResult`

```python
@dataclass(slots=True)
class ChatResult:
    model: str
    outputs: list[CanonicalOutput]
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)
    response_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
```

`model` should be the model the provider used (often `request.model`, which
may have been remapped by the proxy). `outputs` carries the provider's response
data; `response_id` is optional and used by the conversation-state store.
`raw` is the provider's full response payload, preserved for the proxy's
compatibility layer.

### `CanonicalOutput`

```python
@dataclass(slots=True)
class CanonicalOutput:
    type: Literal["message", "reasoning", "tool_call", "invalid_tool_call"] = "message"
    content: str = ""
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    output: str | None = None
    provider_info: dict[str, Any] | None = None
```

`type` classifies the output. `content` is the text payload for `message` and
`reasoning` types. `tool` / `arguments` carry function-calling data for tool
call types. `provider_info` passes through opaque provider-specific metadata.

### `CanonicalUsage`

```python
@dataclass(slots=True)
class CanonicalUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    tokens_per_second: float | None = None
    time_to_first_token_seconds: float | None = None
    model_load_time_seconds: float | None = None
```

All fields are optional. Set what the provider reports; unset fields are
omitted from the response.

### `StreamEvent`

```python
@dataclass(slots=True)
class StreamEvent:
    event: str
    data: dict[str, Any]
```

Events are emitted by `stream_chat` in a specific lifecycle order (see
[Streaming](#streaming) below).

## Error handling

Providers must raise typed errors — never raw `httpx.HTTPError` or other
library exceptions. The proxy translates them into consistent JSON error
envelopes:

| Error class | HTTP status | Use case |
| --- | --- | --- |
| `ProviderHTTPError(status_code, body)` | Forwarded from upstream | Non-2xx response from the upstream provider |
| `ProviderUnavailableError` | `503` | Network failure, DNS error, connection refused |
| `ProviderOverloadedError` | `503` | Concurrency limit reached (see [Configuration](configuration.md)) |
| `ProviderUnsupportedError` | `400` | Provider rejects a request feature it does not support |

### Typed-error pattern

All built-in providers wrap `httpx` calls in a helper that raises typed errors:

```python
HTTP_ERROR_THRESHOLD = 400

async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
    try:
        response = await self._client.post(f"{self._base_url}{path}", json=payload)
    except httpx.HTTPError as exc:
        raise ProviderUnavailableError(str(exc)) from exc
    if response.status_code >= HTTP_ERROR_THRESHOLD:
        raise ProviderHTTPError(response.status_code, response.text)
    return response
```

The contract suite swaps the provider's `_client` to verify this pattern works
for both 5xx and network-error paths.

## Streaming

### The canonical lifecycle

`stream_chat` must yield `StreamEvent` objects in a specific order:

```
chat.start → message.start → message.delta (×N) → message.end → chat.end
```

On failure:

```
chat.start → (any events) → error → chat.end
```

Events:

| Event | Data shape | Position |
| --- | --- | --- |
| `chat.start` | `{"type": "chat.start"}` | First |
| `message.start` | `{"type": "message.start"}` | After `chat.start` |
| `message.delta` | `{"content": "..."}` | Per chunk of output text |
| `message.end` | `{"type": "message.end"}` | After all deltas |
| `chat.end` | `{"result": {...}}` | Last — aggregated result or empty on error |
| `error` | `{"type": "...", "message": "..."}` | Before `chat.end` on failure |

### Native vs synthesized streaming

If your provider natively supports SSE, yield real `StreamEvent` objects as
chunks arrive. If it does not, delegate to
`provider_api.synthesize_stream`, which emits the canonical lifecycle from a
completed non-streaming `chat` call:

```python
from lms_passthrough.provider_api import synthesize_stream

async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
    async for event in synthesize_stream(request, self.chat):
        yield event
```

`synthesize_stream` handles error paths: on failure it emits an `error` event
followed by `chat.end` so the stream always terminates cleanly.

### Streaming lifecycle requirements

The contract suite verifies:

1. The first event yielded is `chat.start`.
2. The last event yielded is `chat.end` or `error`.
3. The stream yields at least one event.

Real-SSE providers should wrap the upstream request in an `async with` context
manager that closes (and releases the response) when the generator is closed
or cancelled.

## Running the contract suite

The suite is in `lms_passthrough.provider_api.contracts` and runs as a plain
synchronous function callable from any pytest test:

```python
from lms_passthrough.provider_api.contracts import run_contract_tests

def test_my_provider_contract() -> None:
    run_contract_tests(MyProvider(...))
```

`run_contract_tests` raises `ContractViolation` (an `AssertionError`) on the
first failing assertion. It is safe to call from both sync and async tests.

### What the suite checks

| Check | What it verifies |
| --- | --- |
| Capability set | `info.capabilities` is a non-empty `set[Capability]` |
| Declared capabilities work | Every capability declared is invocable and returns the correct type |
| Undeclared capabilities fail | Methods for undeclared capabilities raise `NotImplementedError` or a `ProviderError` |
| HTTP errors typed | 5xx responses raise `ProviderHTTPError`; network errors raise `ProviderUnavailableError` |
| Streaming lifecycle | First event is `chat.start`, last is `chat.end` or `error` |

### Prerequisites for full coverage

The HTTP-error checks temporarily swap the provider's `_client` attribute, so
providers must expose their `httpx.AsyncClient` as `_client`. All built-in
providers do this.

## Complete minimal provider example

This is the smallest shape that passes `run_contract_tests`. It declares
`CHAT`, `RESPONSES`, `MODELS`, and `STREAMING` (synthesized):

```python
from collections.abc import AsyncIterator
from typing import Any

import httpx

from lms_passthrough.provider_api import (
    Capability,
    CanonicalOutput,
    ChatRequest,
    ChatResult,
    ProviderHTTPError,
    ProviderInfo,
    ProviderUnavailableError,
    StreamEvent,
    synthesize_stream,
)

HTTP_ERROR_THRESHOLD = 400


class MinimalProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        client: httpx.AsyncClient,
    ) -> None:
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

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._get("/v1/models")
        return list(response.json().get("data", []))

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        async for event in synthesize_stream(request, self.chat):
            yield event

    async def _post(
        self, path: str, payload: dict[str, Any]
    ) -> httpx.Response:
        try:
            response = await self._client.post(
                f"{self._base_url}{path}", json=payload
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        if response.status_code >= HTTP_ERROR_THRESHOLD:
            raise ProviderHTTPError(response.status_code, response.text)
        return response

    async def _get(self, path: str) -> httpx.Response:
        try:
            return await self._client.get(f"{self._base_url}{path}")
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
```

### Notes on the example

- It declares only the capabilities it implements. `EMBEDDINGS` is deliberately
  absent, so `embeddings` raises `NotImplementedError`.
- Streaming delegates to `synthesize_stream`, which produces the canonical
  lifecycle from a non-streaming `chat` call.
- The `_client` attribute is exposed for the contract suite's HTTP-error
  checks.
- All upstream errors are wrapped as `ProviderHTTPError` (for `>= 400`) or
  `ProviderUnavailableError` (for network failures).

## Wiring your provider into the proxy

To register a custom provider, add it to `providers:` in your config and use
a `kind` value that your application recognizes. Built-in providers are
constructed in `src/lms_passthrough/app.py` `AppState.__init__` by matching
`provider.kind` against the four literal strings:
`lm_studio`, `openai_compatible`, `one_min_chat`, `one_min_code_generator`.

For a third-party kind you would:

1. Implement the `Provider` protocol.
2. Add a constructor path in `AppState.__init__` for your new `kind`.
3. Declare the new kind in the `Literal` type of `ProviderConfig.kind` in
   `src/lms_passthrough/config.py`.
4. Add a row to the compatibility matrix in [Providers](providers.md).
