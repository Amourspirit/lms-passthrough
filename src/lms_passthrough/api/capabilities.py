"""Capability validation helpers."""

from __future__ import annotations

from typing import Any

from lms_passthrough.provider_api import Capability, ProviderUnsupportedError


def required_capability_for_endpoint(path: str) -> Capability:
    """Map public endpoint to required provider capability."""
    if path == "/api/v1/chat":
        return Capability.CHAT
    if path == "/v1/chat/completions":
        return Capability.CHAT
    if path == "/v1/responses":
        return Capability.RESPONSES
    if path == "/v1/embeddings":
        return Capability.EMBEDDINGS
    if path in {"/api/v1/models", "/v1/models"}:
        return Capability.MODELS
    if path in {"/api/v1/decisions", "/v1/decisions"}:
        return Capability.DECISIONS
    raise ProviderUnsupportedError(f"Unsupported endpoint: {path}")


def validate_request_features(provider: Any, payload: dict[str, Any]) -> None:
    """Reject request features unsupported by the selected provider."""
    capabilities = provider.info.capabilities
    if payload.get("stream") and Capability.STREAMING not in capabilities:
        raise ProviderUnsupportedError("Provider does not support streaming")
    if payload.get("tools") and Capability.TOOLS not in capabilities:
        raise ProviderUnsupportedError("Provider does not support tools")
    if payload.get("response_format") and Capability.STRUCTURED_OUTPUT not in capabilities:
        raise ProviderUnsupportedError("Provider does not support structured output")
    messages = payload.get("messages") or payload.get("input")
    has_image = False
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict) and item.get("type") == "image":
                has_image = True
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") in {"image", "image_url"}
                for part in content
            ):
                has_image = True
    if has_image and Capability.IMAGES not in capabilities:
        raise ProviderUnsupportedError("Provider does not support images")
