"""System 1 JEV ``/v1/systemone`` HTTP request/response schemas.

The request is validated strictly (``extra="forbid"``) against the official
TypeSafe System One schema: each question type has its own ``criteria`` shape.
The route returns the upstream body unchanged, so the response models exist for
OpenAPI docs rather than response validation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Context = str | dict[str, Any] | list[Any]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JevNoulCriteria(_Strict):
    """Criteria clarifying what counts as a yes or no answer."""

    true: Context | None = None
    false: Context | None = None


class JevNoulQuestion(_Strict):
    """A yes/no question or statement, answered with the probability of yes."""

    type: Literal["noul"]
    instructions: Context | None = None
    criteria: JevNoulCriteria | None = None


class JevChoiceQuestion(_Strict):
    """Select one option from the choices in ``criteria``."""

    type: Literal["choice"]
    instructions: Context | None = None
    # Values may be null: the reference server accepts `{"billing": null}`.
    criteria: dict[str, Context | None] = Field(min_length=1, max_length=255)


class JevScoreQuestion(_Strict):
    """Rate the content against ordered levels in ``criteria``."""

    type: Literal["score"]
    instructions: Context | None = None
    criteria: list[Context] = Field(min_length=1, max_length=10)


class JevBoxQuestion(_Strict):
    """Locate what ``instructions`` describe in the request's image."""

    type: Literal["box"]
    instructions: Context | None = None


Question = Annotated[
    JevNoulQuestion | JevChoiceQuestion | JevScoreQuestion | JevBoxQuestion,
    Field(discriminator="type"),
]


class JevAsset(_Strict):
    """An image that replaces its placeholder in ``state`` or a question."""

    type: Literal["image"] | None = None
    format: Literal["png", "jpg", "jpeg"] | None = None
    data_base64: str | None = None
    part: str | None = None


class JevRequest(_Strict):
    """The ``/api/v1/systemone`` request body."""

    model: str = Field(description="Name or alias of the JEV model to use.")
    state: Context = Field(description="Content all questions in this request refer to.")
    questions: dict[str, Question] = Field(
        min_length=1,
        description="Question id -> typed question.",
    )
    assets: dict[str, JevAsset] | None = Field(
        default=None,
        min_length=1,
        description="Optional media assets keyed by opaque placeholders.",
    )


class JevNoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class JevChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class JevScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, Context]
    probabilities: dict[str, float]


class JevBoxCoordinates(BaseModel):
    top: float
    bottom: float
    left: float
    right: float


class JevBoxAnswer(BaseModel):
    type: Literal["box"]
    box: JevBoxCoordinates
    confidence: float | None = None


Answer = Annotated[
    JevNoulAnswer | JevChoiceAnswer | JevScoreAnswer | JevBoxAnswer,
    Field(discriminator="type"),
]


class JevUsage(BaseModel):
    """Billable token usage reported by the upstream."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int


class JevResponse(BaseModel):
    """The documented JEV response body (returned unchanged by the route)."""

    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, Answer]
    usage: JevUsage
