"""System 1 (JEV ``jev_decide``) decisions HTTP request/response schemas.

The request is validated strictly (``extra="forbid"``, mirroring the schema's
``additionalProperties: false``). The response is a pass-through of the
upstream ``{code, message, data}`` envelope; its models exist for OpenAPI
docs and never validate what the upstream returns.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Context = str | dict[str, Any] | list[Any]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulCriteria(_Strict):
    """Optional meaning of a yes/no judgment."""

    true: str | None = None
    false: str | None = None


class NoulQuestion(_Strict):
    """Yes/no judgment answered with a probability in ``[0, 1]``."""

    type: Literal["noul"]
    instructions: Context
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    """Pick one labelled option."""

    type: Literal["choice"]
    instructions: Context
    # Values may be null: the reference server accepts `{"billing": null}`.
    criteria: dict[str, str | None] = Field(description="Option key -> description.")


class ScoreQuestion(_Strict):
    """Ordered levels, lowest to highest; answered with a fractional level index."""

    type: Literal["score"]
    instructions: Context
    criteria: list[str] = Field(min_length=2)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionsRequest(_Strict):
    """The ``/api/v1/decisions`` request body."""

    model: str | None = Field(default=None, description="Jev model identifier. Optional.")
    state: Context = Field(description="Context to evaluate.")
    questions: dict[str, Question] = Field(description="Question id -> typed question.")


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionsData(BaseModel):
    model_config = ConfigDict(extra="allow")

    answers: dict[str, Answer]


class DecisionsResponse(BaseModel):
    """Upstream envelope, returned unchanged. ``code: 0`` means success."""

    model_config = ConfigDict(extra="allow")

    code: int
    message: str
    data: DecisionsData
