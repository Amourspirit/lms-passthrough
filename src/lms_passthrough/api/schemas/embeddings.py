"""OpenAI embeddings HTTP request/response schemas.

Typed only at the HTTP boundary per ADR-0001: ``Provider.embeddings`` keeps
its ``dict[str, Any]`` signature and the handler forwards
``EmbeddingsRequest.model_dump(exclude_unset=True)`` unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingsRequest(BaseModel):
    """The ``/v1/embeddings`` request body.

    Field commitments follow the typed HTTP boundary ADR: ``model`` and
    ``input`` (with OpenAI's full accepted shapes) are typed, and
    ``encoding_format`` / ``dimensions`` / ``user`` are typed as first-class.
    Anything else flows through via ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] | None = None
    dimensions: int | None = None
    user: Any = Field(default=None, description="Passed through; not validated.")


class EmbeddingData(BaseModel):
    """One embedding vector returned in ``data``."""

    model_config = ConfigDict(extra="allow")

    object: Literal["embedding"] = "embedding"
    index: int = 0
    embedding: list[float] | str


class EmbeddingsUsage(BaseModel):
    """Token usage of an embeddings call."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    """The ``/v1/embeddings`` response body."""

    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingsUsage | None = None
