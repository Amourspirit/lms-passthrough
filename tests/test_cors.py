from pathlib import Path

from fastapi.testclient import TestClient

from lms_passthrough import create_app


def make_config(tmp_path: Path, security_yaml: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
{security_yaml}providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
""".strip(),
        encoding="utf-8",
    )
    return path


def test_cors_enabled_allows_configured_origin(tmp_path: Path) -> None:
    config = make_config(
        tmp_path, 'security:\n  cors_origins: ["http://localhost:3000"]\n'
    )
    client = TestClient(create_app(config))
    response = client.get("/health/live", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_disabled_by_default_emits_no_headers(tmp_path: Path) -> None:
    client = TestClient(create_app())
    response = client.get("/health/live", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    preflight = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in preflight.headers


def test_cors_preflight_allows_x_lms_provider(tmp_path: Path) -> None:
    config = make_config(
        tmp_path, 'security:\n  cors_origins: ["http://localhost:3000"]\n'
    )
    client = TestClient(create_app(config))
    response = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,x-lms-provider",
        },
    )
    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"].lower()
    assert "x-lms-provider" in allowed
    assert "authorization" in allowed


def test_cors_unknown_origin_not_allowed(tmp_path: Path) -> None:
    config = make_config(
        tmp_path, 'security:\n  cors_origins: ["http://localhost:3000"]\n'
    )
    client = TestClient(create_app(config))
    response = client.get("/health/live", headers={"Origin": "http://evil.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
