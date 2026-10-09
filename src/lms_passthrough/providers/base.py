"""Provider registry and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lms_passthrough.provider_api import Capability, ProviderInfo


@dataclass(slots=True)
class ProviderRoute:
    """Provider registry entry."""

    name: str
    kind: str
    info: ProviderInfo
    models: dict[str, str] = field(default_factory=dict)


class UnsupportedOperationError(Exception):
    """Raised when a provider does not support an operation."""


def require_capability(info: ProviderInfo, capability: Capability) -> None:
    """Validate provider capability."""
    if capability not in info.capabilities:
        raise UnsupportedOperationError(f"Provider {info.name} does not support {capability.value}")


def serialize_transcript(messages: list[dict[str, Any]], system_prompt: str | None = None) -> str:
    """Serialize canonical messages into a deterministic prompt transcript."""
    lines: list[str] = []
    if system_prompt:
        lines.append(f"System: {system_prompt}")
    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        content = message.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "message"}:
                    text_parts.append(str(item.get("text") or item.get("content") or ""))
                elif isinstance(item, dict):
                    text_parts.append(str(item))
            content = "\n".join(part for part in text_parts if part)
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)
