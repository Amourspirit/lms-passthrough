from pathlib import Path

import pytest

from lms_passthrough.config import load_config, load_config_or_default


def test_load_config_with_env_interpolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEMIN_API_KEY", "secret")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
  - name: 1min-chat
    kind: one_min_chat
    base_url: https://api.1min.ai
    api_key_env: ONEMIN_API_KEY
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.default_provider == "lm-studio"
    assert len(cfg.providers) == 2


def test_load_config_or_default_returns_lm_studio_default(tmp_path: Path) -> None:
    cfg = load_config_or_default(tmp_path / "missing.yaml")
    assert cfg.default_provider == "lm-studio"
    assert cfg.providers[0].kind == "lm_studio"


def test_wildcard_cors_with_credentials_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
security:
  cors_origins: ["*"]
  cors_allow_credentials: true
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wildcard"):
        load_config(path)


def test_wildcard_cors_without_credentials_accepted(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
security:
  cors_origins: ["*"]
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.security.cors_origins == ["*"]


def test_logging_level_is_uppercased(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
logging:
  level: debug
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.logging.level == "DEBUG"


def test_invalid_logging_level_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
logging:
  level: VERBOSE
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid log level"):
        load_config(path)


def test_max_concurrent_defaults_to_unlimited(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.providers[0].max_concurrent is None
    assert cfg.providers[0].queue_timeout_seconds == 5.0


def test_max_concurrent_and_queue_timeout_parsed(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
default_provider: android
providers:
  - name: android
    kind: openai_compatible
    base_url: http://127.0.0.1:8080
    max_concurrent: 4
    queue_timeout_seconds: 0.5
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.providers[0].max_concurrent == 4
    assert cfg.providers[0].queue_timeout_seconds == 0.5


def test_invalid_max_concurrent_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
providers:
  - name: android
    kind: openai_compatible
    base_url: http://127.0.0.1:8080
    max_concurrent: 0
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_concurrent"):
        load_config(path)
