"""Transparent passthrough helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.datastructures import Headers

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def filter_headers(headers: httpx.Headers | Headers) -> dict[str, str]:
    """Remove hop-by-hop and sensitive headers before forwarding."""
    filtered = {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "authorization"
    }
    filtered.pop("x-lms-provider", None)
    return filtered


async def stream_bytes(response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield raw bytes from an upstream streaming response."""
    async for chunk in response.aiter_bytes():
        yield chunk


async def passthrough_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Execute a non-streaming passthrough request."""
    response = await client.request(method, url, headers=headers, json=json_body)
    return response.status_code, filter_headers(response.headers), response.content
