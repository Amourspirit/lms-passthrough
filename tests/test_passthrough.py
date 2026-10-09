from __future__ import annotations

import httpx

from lms_passthrough.api.passthrough import filter_headers


def test_filter_headers_removes_sensitive_and_hop_by_hop() -> None:
    headers = httpx.Headers(
        {
            "authorization": "Bearer secret",
            "connection": "keep-alive",
            "content-type": "application/json",
            "x-lms-provider": "lm-studio",
        }
    )
    filtered = filter_headers(headers)
    assert "authorization" not in filtered
    assert "connection" not in filtered
    assert "x-lms-provider" not in filtered
    assert filtered["content-type"] == "application/json"
