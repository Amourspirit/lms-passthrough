"""Reusable contract suite for translated providers.

Third-party provider authors can run this suite against their own provider
implementation to verify it honours the canonical ``provider_api`` contract:
capability declarations match behaviour, upstream HTTP failures surface as the
typed errors, unsupported operations fail predictably, and streaming follows
the canonical lifecycle.

The primary entry point is :func:`run_contract_tests`, a plain synchronous
function that can be invoked from any pytest test (or script)::

    from lms_passthrough.provider_api.contracts import run_contract_tests

    def test_my_provider_contract() -> None:
        run_contract_tests(MyProvider(...))

The suite temporarily swaps the provider's HTTP client (``provider._client``)
to exercise error paths, so providers must expose their ``httpx.AsyncClient``
as ``_client`` for full error-path coverage. All built-in providers do.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

import httpx

from lms_passthrough.provider_api.base import (
    Capability,
    ChatRequest,
    ChatResult,
    Provider,
    ProviderError,
    ProviderHTTPError,
    ProviderUnavailableError,
    ProviderUnsupportedError,
    StreamEvent,
)


class ContractViolation(AssertionError):
    """Raised when a provider fails a contract assertion."""


HTTP_ERROR_THRESHOLD = 400


def _req() -> ChatRequest:
    return ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])


def _fail(msg: str) -> None:
    raise ContractViolation(msg)


class FakeResponse:
    """Minimal stand-in for the parts of ``httpx.Response`` providers touch."""

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = {} if payload is None else payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeStream:
    """Async context manager yielding configured SSE lines."""

    def __init__(self, response: FakeResponse, lines: tuple[str, ...]) -> None:
        self._response = response
        self._lines = lines

    @property
    def status_code(self) -> int:
        return self._response.status_code

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class FakeClient:
    """Minimal ``httpx.AsyncClient`` stand-in with selectable responses.

    With ``status_code >= 400`` requests return a non-2xx response so providers
    surface ``ProviderHTTPError``; with ``network_error=True`` they raise an
    ``httpx`` network error so providers surface ``ProviderUnavailableError``.
    """

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stream_lines: tuple[str, ...] = (),
        status_code: int = 200,
        network_error: bool = False,
    ) -> None:
        self._payload = {} if payload is None else payload
        self._stream_lines = stream_lines
        self._status_code = status_code
        self._network_error = network_error

    def _response(self) -> FakeResponse:
        if self._network_error:
            raise httpx.ConnectError("network unavailable")
        if self._status_code >= HTTP_ERROR_THRESHOLD:
            return FakeResponse(status_code=self._status_code, text="upstream error")
        return FakeResponse(self._payload, status_code=self._status_code)

    async def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        return self._response()

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._response()

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._response()

    def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStream:
        return _FakeStream(self._response(), self._stream_lines)

    async def aclose(self) -> None:
        return None


async def _collect_stream(provider: Any, request: ChatRequest) -> list[StreamEvent]:
    collected: list[StreamEvent] = []
    async for event in provider.stream_chat(request):
        collected.append(event)
    return collected


def _check_capability_set(provider: Any, caps: Any) -> None:
    if not isinstance(caps, set) or not caps:
        _fail(f"{provider.info.name}: capabilities must be a non-empty set of Capability")
    for capability in caps:
        if not isinstance(capability, Capability):
            _fail(f"{provider.info.name}: {capability!r} is not a Capability")


def _require_failure(provider: Any, label: str) -> Exception:
    raise ContractViolation(
        f"{provider.info.name}: declared {label} capability but invoking it failed"
    )


async def _check_declared_behaviors(provider: Any, caps: set[Capability]) -> None:
    if Capability.CHAT in caps:
        try:
            result = await provider.chat(_req())
        except Exception as exc:  # noqa: BLE001 - convert any failure into a contract violation
            raise _require_failure(provider, "chat") from exc
        if not isinstance(result, ChatResult):
            _fail(f"{provider.info.name}: chat() must return ChatResult")
    if Capability.RESPONSES in caps:
        try:
            result = await provider.responses(_req())
        except Exception as exc:  # noqa: BLE001 - convert any failure into a contract violation
            raise _require_failure(provider, "responses") from exc
        if not isinstance(result, ChatResult):
            _fail(f"{provider.info.name}: responses() must return ChatResult")
    if Capability.EMBEDDINGS in caps:
        try:
            result = await provider.embeddings({})
        except Exception as exc:  # noqa: BLE001 - convert any failure into a contract violation
            raise _require_failure(provider, "embeddings") from exc
        if not isinstance(result, dict):
            _fail(f"{provider.info.name}: embeddings() must return dict")
    if Capability.MODELS in caps:
        try:
            models = await provider.list_models()
        except Exception as exc:  # noqa: BLE001 - convert any failure into a contract violation
            raise _require_failure(provider, "list_models") from exc
        if not isinstance(models, list):
            _fail(f"{provider.info.name}: list_models() must return list")


async def _check_undeclared_predictable_failure(provider: Any, caps: set[Capability]) -> None:
    if Capability.EMBEDDINGS in caps:
        return
    try:
        await provider.embeddings({})
    except (NotImplementedError, ProviderUnsupportedError, ProviderError):
        return
    _fail(
        f"{provider.info.name}: embeddings() is callable for an undeclared EMBEDDINGS "
        "capability; raise NotImplementedError or a Provider error instead"
    )


async def _check_stream_lifecycle(provider: Any, caps: set[Capability]) -> None:
    if Capability.STREAMING not in caps:
        return
    events = await _collect_stream(provider, _req())
    if not events:
        _fail(f"{provider.info.name}: stream_chat() yielded no events")
    if events[0].event != "chat.start":
        _fail(
            f"{provider.info.name}: first stream event must be chat.start, "
            f"got {events[0].event!r}"
        )
    if events[-1].event not in {"chat.end", "error"}:
        _fail(
            f"{provider.info.name}: last stream event must be chat.end or error, "
            f"got {events[-1].event!r}"
        )


async def _check_http_errors(provider: Any, caps: set[Capability]) -> None:
    if Capability.CHAT not in caps or not hasattr(provider, "_client"):
        return
    original = provider._client
    try:
        provider._client = FakeClient(status_code=500)
        try:
            await provider.chat(_req())
        except ProviderHTTPError:
            pass
        else:
            _fail(f"{provider.info.name}: chat() must raise ProviderHTTPError on an upstream 5xx")

        provider._client = FakeClient(network_error=True)
        try:
            await provider.chat(_req())
        except ProviderUnavailableError:
            pass
        else:
            _fail(
                f"{provider.info.name}: chat() must raise ProviderUnavailableError "
                "on an upstream network error"
            )
    finally:
        provider._client = original


async def _run_all_checks(provider: Any, caps: set[Capability]) -> None:
    await _check_declared_behaviors(provider, caps)
    await _check_undeclared_predictable_failure(provider, caps)
    await _check_stream_lifecycle(provider, caps)
    await _check_http_errors(provider, caps)


def _run_async(coro: Any) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
    else:
        failed: list[BaseException] = []

        def target() -> None:
            try:
                asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - re-raise on the caller's thread
                failed.append(exc)

        thread = threading.Thread(target=target)
        thread.start()
        thread.join()
        if failed:
            raise failed[0]


def run_contract_tests(provider: Provider, *, capabilities: set[Capability] | None = None) -> None:
    """Run the full contract suite against ``provider``.

    Raises :class:`ContractViolation` (an ``AssertionError``) on the first
    failing assertion. Safe to call from both sync and async pytest tests.
    """
    caps = capabilities if capabilities is not None else set(provider.info.capabilities)
    _check_capability_set(provider, caps)
    _run_async(_run_all_checks(provider, caps))
