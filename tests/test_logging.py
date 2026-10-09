from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lms_passthrough import create_app
from lms_passthrough.logging import JsonFormatter, RedactingFilter, redact_value

REQUEST_LOGGER = "lms_passthrough.request"


class CaptureHandler(logging.Handler):
    """Collect emitted records for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def records() -> list[logging.LogRecord]:
    handler = CaptureHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(REQUEST_LOGGER)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield handler.records
    logger.handlers.clear()


def make_config(tmp_path: Path, level: str = "INFO") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
logging:
  level: {level}
  format: json
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
""".strip(),
        encoding="utf-8",
    )
    return path


def test_request_emits_one_structured_line(
    tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.get("/health/live", headers={"X-LMS-Provider": "lm-studio"})
    assert response.status_code == 200
    assert len(records) == 1
    record = records[0]
    assert record.name == REQUEST_LOGGER
    assert record.getMessage() == "request"
    payload: dict[str, Any] = json.loads(JsonFormatter().format(record))
    assert payload["method"] == "GET"
    assert payload["path"] == "/health/live"
    assert payload["provider"] == "lm-studio"
    assert payload["status"] == 200
    assert payload["request_id"]
    assert payload["latency_ms"] >= 0


def test_caller_supplied_request_id_propagates_to_log_and_response(
    tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.get(
        "/v1/models",
        headers={"X-Request-ID": "req-abc-123", "X-LMS-Provider": "lm-studio"},
    )
    assert response.headers["X-Request-ID"] == "req-abc-123"
    record = records[0]
    assert record.request_id == "req-abc-123"  # type: ignore[attr-defined]


def test_generated_request_id_appears_in_response_header(
    tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.get("/health/live")
    request_id = response.headers["X-Request-ID"]
    assert request_id
    assert records[0].request_id == request_id  # type: ignore[attr-defined]


def test_log_never_contains_authorization_or_api_keys(
    tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer supersecrettoken",
            "X-Request-ID": "req-redact",
        },
    )
    for record in records:
        for key, value in record.__dict__.items():
            if key.startswith("redact"):
                continue
            text = str(value)
            assert "supersecrettoken" not in text


def test_prompt_and_completion_text_not_logged(
    tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    client.get("/health/ready")
    for record in records:
        assert "tell me a secret" not in str(record.__dict__)


def test_log_level_is_configurable(tmp_path: Path) -> None:
    config = create_app(make_config(tmp_path, level="DEBUG")).state.state
    assert config.config.logging.level == "DEBUG"


def test_redact_value_redacts_secret_keys() -> None:
    value = {
        "authorization": "Bearer secret",
        "api_key": "xxx",
        "prompt": "hello",
        "nested": {"X-API-Key": "yyy"},
    }
    result = redact_value(value)
    assert result["authorization"] == "[REDACTED]"
    assert result["api_key"] == "[REDACTED]"
    assert result["nested"]["X-API-Key"] == "[REDACTED]"
    assert result["prompt"] == "hello"


def test_redacting_filter_masks_secret_extras() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request",
        args=None,
        exc_info=None,
    )
    record.redact = {"api_key": "supersecret"}  # type: ignore[attr-defined]
    assert RedactingFilter().filter(record)
    assert record.redact == {  # type: ignore[attr-defined]
        "api_key": "[REDACTED]"
    }


def test_setup_logging_installs_redacting_filter(tmp_path: Path) -> None:
    create_app(make_config(tmp_path))
    root = logging.getLogger()
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    assert all(
        any(isinstance(f, RedactingFilter) for f in handler.filters)
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler)
    )


def test_redaction_protects_auth_values_exercised_through_middleware(
    tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer tok1234",
            "api-key": "provkey999",
        },
    )
    for record in records:
        rendered = JsonFormatter().format(record)
        assert "tok1234" not in rendered
        assert "provkey999" not in rendered


def test_upstream_error_log_missing_auth_and_key(
    tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.post(
        "/api/v1/chat",
        headers={
            "Authorization": "Bearer upsec123",
            "X-Request-ID": "req-upstream",
        },
        json={
            "model": "any",
            "messages": [{"role": "user", "content": "secret prompt text"}],
        },
    )
    assert response.status_code == 503
    for record in records:
        rendered = JsonFormatter().format(record)
        assert "upsec123" not in rendered
        assert "secret prompt text" not in rendered
