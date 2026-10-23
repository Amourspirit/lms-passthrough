"""System 1 JEV ``/v1/systemone`` HTTP request/response schemas.

The request is validated strictly (``extra="forbid"``) so typos in question
definitions fail at the HTTP boundary. The response mirrors the documented
upstream shape; the route returns the upstream body unchanged, so these models
exist for OpenAPI docs rather than response validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Context = str | dict[str, Any] | list[Any]
QuestionType = Literal["noul", "choice", "score", "box"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JevCriteria(_Strict):
    """Optional criteria clarifying what counts as true/false for a question."""

    true: Context | None = None
    false: Context | None = None


class JevQuestion(_Strict):
    """A single typed question about the request ``state``."""

    type: QuestionType
    instructions: Context
    criteria: JevCriteria | None = None


class JevRequest(_Strict):
    """The ``/api/v1/systemone`` request body."""

    model: str = Field(description="Name or alias of the JEV model to use.")
    state: Context = Field(description="Content all questions in this request refer to.")
    questions: dict[str, JevQuestion] = Field(
        description="Question id -> typed question."
    )
    assets: dict[str, Any] | None = Field(
        default=None,
        description="Optional media assets keyed by opaque placeholders.",
    )


class JevAnswer(BaseModel):
    """One answer keyed by the question name supplied in the request.

    ``type`` matches the question type. Extra fields are allowed because the
    upstream may return type-specific values (for example ``choice`` or
    ``score``) beyond the documented ``noul`` probability.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    noul: float | None = None


class JevUsage(BaseModel):
    """Billable token usage reported by the upstream."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int


class JevResponse(BaseModel):
    """The documented JEV response body."""

    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, JevAnswer]
    usage: JevUsage
