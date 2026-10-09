"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TimeoutConfig(BaseModel):
    """Upstream timeout settings."""

    connect_seconds: float = 5.0
    response_seconds: float = 60.0
    stream_idle_seconds: float = 120.0
    total_seconds: float = 600.0
    catalog_ttl_seconds: float = 300.0
    shutdown_grace_seconds: float = 30.0


class ModelMappingConfig(BaseModel):
    """Public-to-upstream model mapping."""

    public: str
    upstream: str
    type: Literal["llm", "embedding"] = "llm"
    display_name: str | None = None
    capabilities: list[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    """Base provider configuration."""

    name: str
    kind: Literal[
        "lm_studio", "openai_compatible", "one_min_chat", "one_min_code_generator", "system1"
    ]
    base_url: str
    api_key_env: str | None = None
    default_model: str | None = None
    models: list[ModelMappingConfig] = Field(default_factory=list)
    required: bool = False
    health_path: str = "/"
    verify_tls: bool = True
    ca_bundle: str | None = None
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    max_concurrent: int | None = None
    queue_timeout_seconds: float = 5.0
    extra_headers: dict[str, str] | None = None

    @field_validator("max_concurrent")
    @classmethod
    def _validate_max_concurrent(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_concurrent must be a positive integer or null for unlimited")
        return value

    @field_validator("queue_timeout_seconds")
    @classmethod
    def _validate_queue_timeout(cls, value: float) -> float:
        if value < 0:
            raise ValueError("queue_timeout_seconds must be non-negative")
        return value

    def map_model(self, public_model: str) -> str | None:
        """Resolve a public model id to the upstream model id."""
        for mapping in self.models:
            if mapping.public == public_model:
                return mapping.upstream
        if self.default_model and not self.models:
            return self.default_model
        return None

    def synthetic_models(self) -> list[dict[str, object]]:
        """Build native model catalog entries from configured mappings."""
        entries: list[dict[str, object]] = []
        for mapping in self.models:
            entries.append(
                {
                    "type": mapping.type,
                    "publisher": self.name,
                    "key": mapping.public,
                    "display_name": mapping.display_name or mapping.public,
                    "quantization": None,
                    "size_bytes": 0,
                    "params_string": "unknown",
                    "max_context_length": None,
                    "format": None,
                    "loaded_instances": [],
                }
            )
        return entries

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("health_path")
    @classmethod
    def _validate_health_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("health_path must start with /")
        return value


class SecurityConfig(BaseModel):
    """Proxy authentication settings."""

    enabled: bool = False
    token_env: str | None = None
    docs_enabled: bool = True
    cors_origins: list[str] = Field(default_factory=list)
    cors_allow_credentials: bool = False

    @model_validator(mode="after")
    def _validate_cors_policy(self) -> SecurityConfig:
        if "*" in self.cors_origins and self.cors_allow_credentials:
            raise ValueError(
                "cors_origins wildcard '*' cannot be combined with cors_allow_credentials"
            )
        return self


class PersistenceConfig(BaseModel):
    """SQLite persistence settings."""

    path: Path = Path("./state/lms-passthrough.sqlite3")
    retention_days: int = 30


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: Literal["json", "text"] = "json"

    @field_validator("level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log level: {value}")
        return level


class AppConfig(BaseModel):
    """Application configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    default_provider: str = "lm-studio"
    failover: bool = False
    failover_cooldown_seconds: float = 60.0
    readiness_timeout_seconds: float = Field(default=2.0, gt=0)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    providers: list[ProviderConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_provider_names(self) -> AppConfig:
        names = [provider.name for provider in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        if self.default_provider not in set(names):
            raise ValueError("default_provider must name a configured provider")
        return self


class Settings(BaseSettings):
    """Environment settings."""

    model_config = SettingsConfigDict(env_prefix="LMS_PASSTHROUGH_", env_nested_delimiter="__")

    config: Path = Path("./config.yaml")


def _load_dotenv() -> None:
    """Load `.env` from CWD into os.environ if present. Silently no-op if missing."""
    env_path = Path(".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        if env_name not in os.environ:
            raise ValueError(f"Missing environment variable: {env_name}")
        return os.environ[env_name]
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value


def load_config(path: Path) -> AppConfig:
    """Load configuration from YAML with environment interpolation."""
    _load_dotenv()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    resolved = _resolve_env(raw)
    try:
        return AppConfig.model_validate(resolved)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration at {path}: {exc}") from exc


def load_config_or_default(path: Path | None = None) -> AppConfig:
    """Load explicit config, otherwise create a default LM Studio-only config."""
    if path is not None and path.exists():
        return load_config(path)
    return AppConfig(
        providers=[
            ProviderConfig(
                name="lm-studio",
                kind="lm_studio",
                base_url="http://127.0.0.1:1234",
            )
        ]
    )
