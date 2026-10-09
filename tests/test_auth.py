from pathlib import Path

import httpx
import respx
from fastapi.testclient import TestClient

from lms_passthrough import create_app


def make_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
security:
  enabled: true
  token_env: LMS_TOKEN
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
""".strip(),
        encoding="utf-8",
    )
    return path


def test_auth_rejects_missing_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LMS_TOKEN", "secret")
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.get("/v1/models")
    assert response.status_code == 401


def test_auth_allows_correct_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LMS_TOKEN", "secret")
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.get("/health/live", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200


def test_auth_enforced_on_transparent_native_chat(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LMS_TOKEN", "secret")
    app = create_app(make_config(tmp_path))
    with respx.mock(base_url="http://127.0.0.1:1234", assert_all_called=False) as router:
        router.post("/api/v1/chat").mock(
            return_value=httpx.Response(
                200,
                content=b'{"id": "ok"}',
                headers={"content-type": "application/json"},
            )
        )
        client = TestClient(app)
        body = b'{"model": "m"}'
        headers = {"content-type": "application/json"}
        response = client.post("/api/v1/chat", content=body, headers=headers)
        assert response.status_code == 401
        assert router.post("/api/v1/chat").call_count == 0
        authed = client.post(
            "/api/v1/chat",
            content=body,
            headers={**headers, "Authorization": "Bearer secret"},
        )
        assert authed.status_code == 200
        assert router.post("/api/v1/chat").call_count == 1
