"""Structured logging with request IDs and redaction."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from lms_passthrough.config import LoggingConfig

REDACTED = "[REDACTED]"

_SECRET_KEYS = frozenset(
    {"authorization", "api_key", "api-key", "apikey", "x-api-key", "token", "password"}
)

_STRUCTURED_FIELDS = (
    "request_id",
    "method",
    "path",
    "provider",
    "status",
    "latency_ms",
    "failover",
)


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (REDACTED if key.lower() in _SECRET_KEYS else redact_value(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


class RedactingFilter(logging.Filter):
    """Redact secret values from structured record fields.

    Reads the ``redact`` extra attribute (a mapping) and replaces known
    secret keys with ``[REDACTED]`` before formatting.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        redact = getattr(record, "redact", None)
        if isinstance(redact, dict):
            record.redact = redact_value(redact)
        return True


class JsonFormatter(logging.Formatter):
    """Emit each record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _STRUCTURED_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        redact = getattr(record, "redact", None)
        if isinstance(redact, dict):
            payload["redact"] = redact_value(redact)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable single-line formatter with redacted extras."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        parts = [super().format(record)]
        for key in _STRUCTURED_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                parts.append(f"{key}={value}")
        redact = getattr(record, "redact", None)
        if isinstance(redact, dict):
            parts.append(f"redact={redact_value(redact)!r}")
        return " ".join(parts)


def setup_logging(config: LoggingConfig) -> None:
    """Configure the application's logging based on the provided config."""
    level = logging.getLevelName(config.level)
    formatter = JsonFormatter() if config.format == "json" else TextFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.addFilter(RedactingFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package namespace."""
    return logging.getLogger(f"lms_passthrough.{name}")
