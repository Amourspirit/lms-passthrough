from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lms_passthrough import create_app


def make_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:1234
  - name: android
    kind: openai_compatible
    base_url: http://127.0.0.1:8080
    models:
      - public: android-model
        upstream: local-model
""".strip(),
        encoding="utf-8",
    )
    return path


def test_merged_native_models_include_synthetic_mapping(tmp_path: Path) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.get("/api/v1/models", headers={"X-LMS-Provider": "android"})
    assert response.status_code == 200
    keys = [model["key"] for model in response.json()["models"]]
    assert "android-model" in keys


def test_provider_filtered_native_models(tmp_path: Path) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.get("/api/v1/models", headers={"X-LMS-Provider": "android"})
    assert response.status_code == 200
    keys = [model["key"] for model in response.json()["models"]]
    assert "android-model" in keys
