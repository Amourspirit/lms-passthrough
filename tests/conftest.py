"""Shared pytest fixtures and CLI options for the test suite."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

StubHttp = Callable[[Any, dict[str, int]], None]


@pytest.fixture
def stub_http() -> StubHttp:
    """Point providers' shared httpx clients at mock transports.

    ``stub_http(app, {"lm-studio": 200})`` makes that provider answer every
    request with that status, so upstream-facing behaviour (``/health/ready``)
    is exercisable without a network.
    """

    def apply(app: Any, statuses: dict[str, int]) -> None:
        for name, status in statuses.items():
            app.state.state.http_clients[name] = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _request, status=status: httpx.Response(status)
                )
            )

    return apply


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in live LM Studio conformance flag."""
    parser.addoption(
        "--live-lms",
        action="store",
        default=None,
        metavar="URL",
        help="Run the live LM Studio conformance suite against this base URL "
        "(e.g. http://127.0.0.1:8999). Skipped when not provided.",
    )


@pytest.fixture
def live_lms_url(request: pytest.FixtureRequest) -> str | None:
    """The --live-lms base URL, or None when the flag was not provided."""
    return request.config.getoption("--live-lms")


@pytest.fixture
def require_live_lms(live_lms_url: str | None) -> Iterator[str]:
    """Skip the enclosing test unless --live-lms was provided."""
    if not live_lms_url:
        pytest.skip("live LM Studio suite requires --live-lms <url>")
    yield live_lms_url
