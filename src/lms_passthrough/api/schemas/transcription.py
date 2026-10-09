"""OpenAI speech-to-text HTTP response schemas.

The request side is a `multipart/form-data` envelope, so FastAPI binds it
straight from `UploadFile` / `Form` parameters — there is no request model to
declare here. Only the outbound shapes are typed, per ADR-0001: the inbound
fields reach the provider as a canonical `TranscriptionRequest`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

#: The only `response_format` values this proxy serves. `srt`, `vtt`,
#: `verbose_json` and `diarized_json` are rejected at the boundary rather than
#: silently degraded — see docs/adr/0003-transcription-response-formats.md.
SUPPORTED_RESPONSE_FORMATS = frozenset({"json", "text"})


class TranscriptionResponse(BaseModel):
    """The ``/v1/audio/transcriptions`` response body for ``response_format=json``."""

    model_config = ConfigDict(extra="allow")

    text: str
    language: str | None = None
    duration: float | None = None
