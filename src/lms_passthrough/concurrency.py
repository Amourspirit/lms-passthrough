"""Per-provider concurrency limiting for inference calls."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from lms_passthrough.provider_api import ProviderOverloadedError


class ProviderLimiter:
    """Bound concurrent inference calls to a single provider.

    `max_concurrent` of ``None`` (or a non-positive value) means unlimited and
    the limiter becomes a no-op passthrough. Otherwise inference calls acquire
    a semaphore slot, waiting at most `queue_timeout_seconds` before failing
    with :class:`ProviderOverloadedError` rather than queuing unboundedly.
    """

    def __init__(
        self,
        max_concurrent: int | None = None,
        queue_timeout_seconds: float = 5.0,
    ) -> None:
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None
        )
        self._queue_timeout = queue_timeout_seconds

    async def acquire(self) -> None:
        """Block until a slot is free, failing fast once the queue timeout elapses."""
        if self._semaphore is None:
            return
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._queue_timeout)
        except TimeoutError:
            raise ProviderOverloadedError(
                "Provider is at capacity; too many concurrent inference calls"
            ) from None

    def release(self) -> None:
        """Return a slot acquired via :meth:`acquire`."""
        if self._semaphore is not None:
            self._semaphore.release()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold a concurrency slot for the duration of the ``with`` block."""
        try:
            await self.acquire()
            yield
        finally:
            self.release()
