"""Typed HTTP boundary for `/v1/audio/transcriptions` (OpenAI speech-to-text)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from lms_passthrough import create_app
from lms_passthrough.provider_api import Capability, TranscriptionResult

MOCK_BASE = "http://mock-upstream:8999"
AUDIO = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"

# Distinct names so a developer's real `.env` can never satisfy a test.
OMLX_ENV = "LMS_PT_TEST_OMLX_KEY"
ONEMIN_ENV = "LMS_PT_TEST_ONEMIN_KEY"


def build_app(tmp_path: Path, kind: str = "openai_compatible", **extra: str) -> Any:
    body = extra.pop("body", "")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
default_provider: upstream
persistence:
  path: {tmp_path}/state.sqlite3
providers:
  - name: upstream
    kind: {kind}
    base_url: {MOCK_BASE}
{body}""".strip(),
        encoding="utf-8",
    )
    return create_app(config)


def write_config(tmp_path: Path, body: str) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"persistence:\n  path: {tmp_path}/state.sqlite3\n{body.strip()}",
        encoding="utf-8",
    )
    return config


def onemin_config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        f"""
default_provider: 1min-chat
providers:
  - name: 1min-chat
    kind: one_min_chat
    base_url: {MOCK_BASE}
    api_key_env: {ONEMIN_ENV}
""",
    )


def post_audio(app: Any, **fields: str) -> httpx.Response:
    data = {"model": "large-v3-tu8rbo", "response_format": "json", **fields}
    return TestClient(app).post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", AUDIO, "audio/wav")},
        data=data,
    )


class TestRequestValidation:
    def test_missing_model_is_422(self, tmp_path: Path) -> None:
        client = TestClient(build_app(tmp_path))
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", AUDIO, "audio/wav")},
            data={"response_format": "json"},
        )
        assert response.status_code == 422

    def test_missing_file_is_422(self, tmp_path: Path) -> None:
        client = TestClient(build_app(tmp_path))
        response = client.post(
            "/v1/audio/transcriptions", data={"model": "m", "response_format": "json"}
        )
        assert response.status_code == 422


class TestResponseFormatGate:
    @pytest.mark.parametrize("fmt", ["srt", "vtt", "verbose_json", "diarized_json"])
    def test_unsupported_formats_rejected_400(self, tmp_path: Path, fmt: str) -> None:
        with respx.mock(base_url=MOCK_BASE, assert_all_called=False) as router:
            route = router.post("/v1/audio/transcriptions").mock(
                return_value=httpx.Response(200, json={"text": "never"})
            )
            response = post_audio(build_app(tmp_path), response_format=fmt)
        assert response.status_code == 400
        assert fmt in response.json()["error"]["message"]
        assert not route.called, "rejection must happen before any upstream call"

    def test_streaming_rejected_400(self, tmp_path: Path) -> None:
        with respx.mock(base_url=MOCK_BASE, assert_all_called=False) as router:
            route = router.post("/v1/audio/transcriptions").mock(
                return_value=httpx.Response(200, json={"text": "never"})
            )
            response = post_audio(build_app(tmp_path), stream="true")
        assert response.status_code == 400
        assert "Streaming" in response.json()["error"]["message"]
        assert not route.called

    @pytest.mark.parametrize("fmt", ["json", "text"])
    def test_supported_formats_reach_the_provider(self, tmp_path: Path, fmt: str) -> None:
        with respx.mock(base_url=MOCK_BASE) as router:
            route = router.post("/v1/audio/transcriptions").mock(
                return_value=httpx.Response(200, json={"text": "hello there"})
            )
            response = post_audio(build_app(tmp_path), response_format=fmt)
        assert response.status_code == 200
        assert route.called


class TestOpenAICompatible:
    def test_json_format_returns_text_object(self, tmp_path: Path) -> None:
        with respx.mock(base_url=MOCK_BASE) as router:
            router.post("/v1/audio/transcriptions").mock(
                return_value=httpx.Response(
                    200,
                    json={"text": "hello there", "language": "english", "duration": 1.5},
                )
            )
            response = post_audio(build_app(tmp_path))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"text": "hello there", "language": "english", "duration": 1.5}

    def test_text_format_returns_bare_string(self, tmp_path: Path) -> None:
        with respx.mock(base_url=MOCK_BASE) as router:
            router.post("/v1/audio/transcriptions").mock(
                return_value=httpx.Response(200, json={"text": "hello there"})
            )
            response = post_audio(build_app(tmp_path), response_format="text")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == "hello there"

    def test_optional_fields_are_forwarded(self, tmp_path: Path) -> None:
        captured: list[bytes] = []
        with respx.mock(base_url=MOCK_BASE) as router:
            router.post("/v1/audio/transcriptions").mock(
                side_effect=lambda request: captured.append(request.content)
                or httpx.Response(200, json={"text": "ok"})
            )
            response = post_audio(
                build_app(tmp_path), language="en", prompt="hello world"
            )
        assert response.status_code == 200
        body = captured[0]
        assert b'name="language"' in body and b"en" in body
        assert b'name="prompt"' in body and b"hello world" in body

    def test_model_id_is_not_translated(self, tmp_path: Path) -> None:
        captured: list[bytes] = []
        with respx.mock(base_url=MOCK_BASE) as router:
            router.post("/v1/audio/transcriptions").mock(
                side_effect=lambda request: captured.append(request.content)
                or httpx.Response(200, json={"text": "ok"})
            )
            response = post_audio(build_app(tmp_path), model="large-v3-tu8rbo")
        assert response.status_code == 200
        assert b"large-v3-tu8rbo" in captured[0]

    def test_api_key_and_extra_headers_reach_upstream(self, tmp_path: Path) -> None:
        os.environ[OMLX_ENV] = "secret-token"
        try:
            config = write_config(
                tmp_path,
                f"""
default_provider: omlx
providers:
  - name: omlx
    kind: openai_compatible
    base_url: {MOCK_BASE}
    api_key_env: {OMLX_ENV}
    extra_headers:
      CF-Access-Client-Id: abc123
""",
            )
            seen: list[httpx.Headers] = []
            with respx.mock(base_url=MOCK_BASE) as router:
                router.post("/v1/audio/transcriptions").mock(
                    side_effect=lambda request: seen.append(request.headers)
                    or httpx.Response(200, json={"text": "ok"})
                )
                response = post_audio(create_app(config))
        finally:
            del os.environ[OMLX_ENV]
        assert response.status_code == 200
        assert seen[0]["authorization"] == "Bearer secret-token"
        assert seen[0]["cf-access-client-id"] == "abc123"
        assert seen[0]["content-type"].startswith("multipart/form-data")

    def test_upstream_4xx_is_not_failed_over(self, tmp_path: Path) -> None:
        with respx.mock(base_url=MOCK_BASE) as router:
            router.post("/v1/audio/transcriptions").mock(
                return_value=httpx.Response(400, json={"error": {"message": "bad model"}})
            )
            response = post_audio(build_app(tmp_path))
        assert response.status_code == 400


class TestCapabilityGate:
    def test_lm_studio_is_never_selected(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""
default_provider: lm-studio
persistence:
  path: {tmp_path}/state.sqlite3
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: {MOCK_BASE}
""".strip(),
            encoding="utf-8",
        )
        response = post_audio(create_app(config))
        assert response.status_code == 400
        assert "does not support transcriptions" in response.json()["error"]["message"]

    def test_code_generator_provider_is_never_selected(self, tmp_path: Path) -> None:
        os.environ[ONEMIN_ENV] = "onemin-key"
        try:
            config = write_config(
                tmp_path,
                f"""
default_provider: 1min-codegen
providers:
  - name: 1min-codegen
    kind: one_min_code_generator
    base_url: {MOCK_BASE}
    api_key_env: {ONEMIN_ENV}
""",
            )
            response = post_audio(create_app(config))
        finally:
            del os.environ[ONEMIN_ENV]
        assert response.status_code == 400
        assert "does not support transcriptions" in response.json()["error"]["message"]

    def test_declared_capability_reaches_the_provider(self, tmp_path: Path) -> None:
        app = build_app(tmp_path)
        assert Capability.TRANSCRIPTIONS in app.state.state.providers["upstream"].info.capabilities


class TestOneMinChatProvider:
    """The 1min path is two hops: Asset API upload, then the feature call."""

    def test_two_hop_call_returns_transcript(self, tmp_path: Path) -> None:
        os.environ[ONEMIN_ENV] = "onemin-key"
        try:
            with respx.mock(base_url=MOCK_BASE) as router:
                upload = router.post("/api/assets").mock(
                    return_value=httpx.Response(
                        200, json={"asset": {"key": "audios/abc_whisper.wav"}}
                    )
                )
                feature = router.post("/api/features").mock(
                    return_value=httpx.Response(
                        200,
                        json={
                            "aiRecord": {
                                "aiRecordDetail": {"resultObject": ["hello from 1min"]}
                            }
                        },
                    )
                )
                response = post_audio(create_app(onemin_config(tmp_path)), model="whisper-1")
        finally:
            del os.environ[ONEMIN_ENV]
        assert response.status_code == 200
        assert response.json() == {"text": "hello from 1min"}
        assert upload.called and feature.called
        body = json.loads(feature.calls[0].request.content)
        assert body["type"] == "SPEECH_TO_TEXT"
        assert body["model"] == "whisper-1"
        assert body["promptObject"]["audioUrl"] == "audios/abc_whisper.wav"
        assert AUDIO in upload.calls[0].request.content

    def test_asset_upload_sends_api_key_not_json_content_type(self, tmp_path: Path) -> None:
        os.environ[ONEMIN_ENV] = "onemin-key"
        try:
            with respx.mock(base_url=MOCK_BASE) as router:
                upload = router.post("/api/assets").mock(
                    return_value=httpx.Response(200, json={"asset": {"key": "audios/a.wav"}})
                )
                router.post("/api/features").mock(
                    return_value=httpx.Response(
                        200,
                        json={"aiRecord": {"aiRecordDetail": {"resultObject": ["t"]}}},
                    )
                )
                post_audio(create_app(onemin_config(tmp_path)), model="whisper-1")
        finally:
            del os.environ[ONEMIN_ENV]
        headers = upload.calls[0].request.headers
        assert headers["api-key"] == "onemin-key"
        assert headers["content-type"].startswith("multipart/form-data")

    def test_missing_asset_key_is_an_upstream_failure(self, tmp_path: Path) -> None:
        os.environ[ONEMIN_ENV] = "onemin-key"
        try:
            with respx.mock(base_url=MOCK_BASE) as router:
                router.post("/api/assets").mock(
                    return_value=httpx.Response(200, json={"asset": {}})
                )
                response = post_audio(create_app(onemin_config(tmp_path)), model="whisper-1")
        finally:
            del os.environ[ONEMIN_ENV]
        assert response.status_code == 503

    async def test_stt_models_join_the_chat_catalog(self, tmp_path: Path) -> None:
        os.environ[ONEMIN_ENV] = "onemin-key"
        listings = {
            "UNIFY_CHAT_WITH_AI": [{"modelId": "chat-model"}],
            "SPEECH_TO_TEXT": [{"modelId": "whisper-1"}, {"modelId": "elevenlabs-stt"}],
        }
        try:
            with respx.mock(base_url=MOCK_BASE) as router:
                router.get("/models").mock(
                    side_effect=lambda request: httpx.Response(
                        200,
                        json={
                            "models": next(
                                (
                                    entries
                                    for feature, entries in listings.items()
                                    if feature in str(request.url)
                                ),
                                [],
                            )
                        },
                        request=request,
                    )
                )
                provider = create_app(onemin_config(tmp_path)).state.state.providers["1min-chat"]
                models = await provider.list_models()
        finally:
            del os.environ[ONEMIN_ENV]
        assert {m["modelId"] for m in models} == {"chat-model", "whisper-1", "elevenlabs-stt"}

    async def test_failing_stt_listing_keeps_chat_models(self, tmp_path: Path) -> None:
        os.environ[ONEMIN_ENV] = "onemin-key"
        try:
            with respx.mock(base_url=MOCK_BASE) as router:
                router.get("/models").mock(
                    side_effect=lambda request: (
                        httpx.Response(200, json={"models": [{"modelId": "chat-model"}]})
                        if "UNIFY_CHAT" in str(request.url)
                        else httpx.Response(500, text="stt listing is down")
                    )
                )
                provider = create_app(onemin_config(tmp_path)).state.state.providers["1min-chat"]
                models = await provider.list_models()
        finally:
            del os.environ[ONEMIN_ENV]
        assert {m["modelId"] for m in models} == {"chat-model"}


class TestResultShaping:
    def test_none_valued_fields_are_omitted(self, tmp_path: Path) -> None:
        app = build_app(tmp_path)
        provider = app.state.state.providers["upstream"]

        async def fake(request: Any) -> TranscriptionResult:
            return TranscriptionResult(model=request.model, text="bare")

        provider.transcriptions = fake  # type: ignore[method-assign]
        assert post_audio(app).json() == {"text": "bare"}
