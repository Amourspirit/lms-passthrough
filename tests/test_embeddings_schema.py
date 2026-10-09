"""Typed HTTP boundary for `/v1/embeddings` (OpenAI embeddings)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from lms_passthrough import create_app
from lms_passthrough.api.schemas.embeddings import EmbeddingsRequest, EmbeddingsResponse

MOCK_BASE = "http://mock-upstream:8999"


def make_android_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
default_provider: android
persistence:
  path: {tmp_path}/state.sqlite3
providers:
  - name: android
    kind: openai_compatible
    base_url: http://127.0.0.1:8080
    models:
      - public: m
        upstream: local-model
""".strip(),
        encoding="utf-8",
    )
    return path


def build_lms_proxy(tmp_path: Path, base_url: str = MOCK_BASE):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
default_provider: lm-studio
persistence:
  path: {tmp_path}/state.sqlite3
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: {base_url}
""".strip(),
        encoding="utf-8",
    )
    return create_app(config)


class TestEmbeddingsRequest:
    def test_accepts_string_input(self) -> None:
        request = EmbeddingsRequest.model_validate({"model": "m", "input": "hi"})
        assert request.input == "hi"

    def test_accepts_list_of_strings(self) -> None:
        request = EmbeddingsRequest.model_validate({"model": "m", "input": ["a", "b"]})
        assert request.input == ["a", "b"]

    def test_accepts_list_of_integers(self) -> None:
        request = EmbeddingsRequest.model_validate({"model": "m", "input": [1, 2, 3]})
        assert request.input == [1, 2, 3]

    def test_accepts_list_of_integer_lists(self) -> None:
        request = EmbeddingsRequest.model_validate({"model": "m", "input": [[1, 2], [3, 4]]})
        assert request.input == [[1, 2], [3, 4]]

    def test_accepts_optional_fields(self) -> None:
        request = EmbeddingsRequest.model_validate(
            {
                "model": "m",
                "input": "hi",
                "encoding_format": "base64",
                "dimensions": 3,
                "user": "alice",
            }
        )
        assert request.encoding_format == "base64"
        assert request.dimensions == 3
        assert request.user == "alice"

    def test_rejects_invalid_input_type(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsRequest.model_validate({"model": "m", "input": {"a": 1}})

    def test_rejects_invalid_encoding_format(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsRequest.model_validate(
                {"model": "m", "input": "hi", "encoding_format": "bytes"}
            )

    def test_rejects_missing_model(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsRequest.model_validate({"input": "hi"})

    def test_accepts_unknown_fields(self) -> None:
        request = EmbeddingsRequest.model_validate(
            {"model": "m", "input": "hi", "future_embedding_field": {"x": 1}}
        )
        assert request.model_dump()["future_embedding_field"] == {"x": 1}


class TestEmbeddingsResponse:
    def test_validates_full_response(self) -> None:
        response = EmbeddingsResponse.model_validate(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                ],
                "model": "m",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        )
        assert len(response.data) == 2
        assert response.data[0].embedding == [0.1, 0.2]
        assert response.usage is not None
        assert response.usage.total_tokens == 2

    def test_accepts_base64_embedding(self) -> None:
        response = EmbeddingsResponse.model_validate(
            {"object": "list", "data": [{"index": 0, "embedding": "aGVsbG8="}], "model": "m"}
        )
        assert response.data[0].embedding == "aGVsbG8="


class TestEmbeddingsEndpoint:
    def build_capturing_client(self, tmp_path: Path):
        app = create_app(make_android_config(tmp_path))
        provider = app.state.state.providers["android"]
        seen: list[dict[str, object]] = []

        async def fake_embeddings(request: dict) -> dict:
            seen.append(dict(request))
            return {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "model": request["model"],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }

        provider.embeddings = fake_embeddings  # type: ignore[method-assign]
        return TestClient(app), seen

    def test_malformed_payload_returns_422(self, tmp_path: Path) -> None:
        app = create_app(make_android_config(tmp_path))
        client = TestClient(app)
        response = client.post("/v1/embeddings", json={"input": "hi"})
        assert response.status_code == 422

    def test_invalid_input_type_returns_422(self, tmp_path: Path) -> None:
        app = create_app(make_android_config(tmp_path))
        client = TestClient(app)
        response = client.post("/v1/embeddings", json={"model": "m", "input": {"a": 1}})
        assert response.status_code == 422

    def test_translated_path_forwards_exclude_unset_payload(self, tmp_path: Path) -> None:
        client, seen = self.build_capturing_client(tmp_path)
        response = client.post(
            "/v1/embeddings",
            json={"model": "m", "input": "hi", "dimensions": 3, "user": "alice"},
        )
        assert response.status_code == 200
        assert seen == [
            {"model": "local-model", "input": "hi", "dimensions": 3, "user": "alice"}
        ]

    def test_unset_optional_fields_stay_absent(self, tmp_path: Path) -> None:
        client, seen = self.build_capturing_client(tmp_path)
        response = client.post("/v1/embeddings", json={"model": "m", "input": "hi"})
        assert response.status_code == 200
        assert seen == [{"model": "local-model", "input": "hi"}]

    def test_openapi_describes_embeddings_schema(self, tmp_path: Path) -> None:
        app = create_app(make_android_config(tmp_path))
        client = TestClient(app)
        schema = client.get("/openapi.json").json()
        path_item = schema["paths"]["/v1/embeddings"]["post"]
        request_ref = path_item["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        request_name = request_ref.rsplit("/", 1)[-1]
        properties = schema["components"]["schemas"][request_name]["properties"]
        for field in ["model", "input", "encoding_format", "dimensions", "user"]:
            assert field in properties
        response_ref = (
            path_item["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        )
        response_name = response_ref.rsplit("/", 1)[-1]
        assert response_name == "EmbeddingsResponse"

    def test_transparent_lm_studio_forwards_raw_body_bytes(self, tmp_path: Path) -> None:
        app = build_lms_proxy(tmp_path)
        raw = (
            b'{"model":"embedding-model",\n'
            b' "input": ["caf\xc3\xa9"],\n'
            b' "encoding_format": "float"}\n'
        )
        captured: list[bytes] = []

        def handle(request: httpx.Request) -> httpx.Response:
            captured.append(request.content)
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                    "model": "embedding-model",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
                headers={"content-type": "application/json"},
            )

        with respx.mock(base_url=MOCK_BASE, assert_all_called=False) as router:
            router.post("/v1/embeddings").mock(side_effect=handle)
            client = TestClient(app)
            response = client.post(
                "/v1/embeddings",
                content=raw,
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 200
        assert captured[0] == raw
