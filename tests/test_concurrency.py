from __future__ import annotations

import asyncio

import pytest

from lms_passthrough.concurrency import ProviderLimiter
from lms_passthrough.provider_api import ProviderOverloadedError


@pytest.mark.asyncio
async def test_unlimited_limiter_is_passthrough() -> None:
    limiter = ProviderLimiter()
    entered = 0

    async def call() -> None:
        nonlocal entered
        async with limiter.slot():
            entered += 1

    await asyncio.gather(*[call() for _ in range(20)])
    assert entered == 20


@pytest.mark.asyncio
async def test_limits_concurrent_calls_to_max_concurrent() -> None:
    limiter = ProviderLimiter(max_concurrent=2)
    active = 0
    peak = 0

    async def call() -> None:
        nonlocal active, peak
        async with limiter.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(*[call() for _ in range(10)])
    assert peak <= 2


@pytest.mark.asyncio
async def test_nth_plus_one_concurrent_call_overloads() -> None:
    limiter = ProviderLimiter(max_concurrent=1, queue_timeout_seconds=0.05)

    async def hold_forever() -> None:
        async with limiter.slot():
            await asyncio.sleep(10)

    holder = asyncio.create_task(hold_forever())
    await asyncio.sleep(0)
    with pytest.raises(ProviderOverloadedError):
        async with limiter.slot():
            pass
    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder


@pytest.mark.asyncio
async def test_waits_then_succeeds_when_slot_frees() -> None:
    limiter = ProviderLimiter(max_concurrent=1, queue_timeout_seconds=1.0)
    results: list[str] = []

    async def hold() -> None:
        async with limiter.slot():
            results.append("first")
            await asyncio.sleep(0.05)

    async def wait() -> None:
        async with limiter.slot():
            results.append("second")

    await asyncio.gather(hold(), wait())
    assert results == ["first", "second"]


@pytest.mark.asyncio
async def test_overload_waits_for_queue_timeout() -> None:
    limiter = ProviderLimiter(max_concurrent=1, queue_timeout_seconds=0.1)

    async def hold() -> None:
        async with limiter.slot():
            await asyncio.sleep(0.5)

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0)
    started = asyncio.get_running_loop().time()
    with pytest.raises(ProviderOverloadedError):
        async with limiter.slot():
            pass
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.4
    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder


@pytest.mark.asyncio
async def test_eager_acquire_fails_fast_before_stream() -> None:
    limiter = ProviderLimiter(max_concurrent=1, queue_timeout_seconds=0.05)
    await limiter.acquire()
    with pytest.raises(ProviderOverloadedError):
        await limiter.acquire()
    limiter.release()
    await limiter.acquire()
    limiter.release()


@pytest.mark.asyncio
async def test_release_returns_slot_for_streaming_generator() -> None:
    limiter = ProviderLimiter(max_concurrent=1, queue_timeout_seconds=1.0)
    await limiter.acquire()

    async def consume() -> None:
        try:
            pass
        finally:
            limiter.release()

    await consume()
    async with limiter.slot():
        pass
