"""HTTP surface. One request carries a state and every question about it.

The shape is deliberate: a caller that needs six judgements on the same
passage sends one request, and the state is encoded once for all six.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from brier.config import Settings
from brier.decide import Decider
from brier.encoder import EncoderError, OllamaEncoder
from brier.model import load
from brier.types import MAX_OPTIONS, Question, QuestionKind

MAX_QUESTIONS = 32


class QuestionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    kind: QuestionKind
    prompt: str = Field(min_length=1, max_length=2000)
    options: list[str] = Field(min_length=2, max_length=MAX_OPTIONS)


class DecideRequest(BaseModel):
    state: str = Field(min_length=1, max_length=20000)
    questions: list[QuestionRequest] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @field_validator("questions")
    @classmethod
    def names_must_be_unique(cls, questions: list[QuestionRequest]) -> list[QuestionRequest]:
        """The name is how the caller keys the answer, so two of them lose one.

        Accepting duplicates returned 200 with two answers of the same name, and
        a caller building a dict from the response silently kept one of them.
        """
        names = [question.name for question in questions]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"question names must be unique; repeated: {', '.join(repeated)}")
        return questions


class AnswerResponse(BaseModel):
    name: str
    kind: QuestionKind
    value: str
    confidence: float
    abstained: bool
    probabilities: list[float]


class DecideResponse(BaseModel):
    answers: list[AnswerResponse]
    temperature: float


@lru_cache(maxsize=1)
def provide_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def provide_decider() -> Decider:
    """Build the decider once, and say plainly what is missing when it cannot.

    The encoder here is the plain one, not the disk cache. That cache is built
    for training: it grows without bound and rewrites its whole vector file on
    every miss, so on a serving path it would store every passage a caller ever
    sends and get slower as it did.

    Raises:
        HTTPException: 503 when the weights have not been trained yet.
    """
    settings = provide_settings()
    try:
        parameters = load(settings.weights_path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"no weights at {settings.weights_path}; run scripts/train_model.py first",
        ) from error

    return Decider(
        encoder=OllamaEncoder(settings.ollama_url, settings.encoder_model),
        parameters=parameters,
        min_confidence=settings.min_confidence,
    )


app = FastAPI(
    title="brier",
    description="Calibrated decisions over text, with abstention",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only: answers even with no weights and no encoder."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, object]:
    """Whether this instance can actually answer, and with what.

    Separate from /health on purpose: a load balancer needs to know the
    difference between a process that is up and one that has its weights.

    Raises:
        HTTPException: 503 when the weights are missing or do not match.
    """
    decider = provide_decider()
    return {
        "status": "ready",
        "temperature": decider.parameters.temperature,
        "weights": decider.parameters.provenance.describe(),
        "min_confidence": decider.min_confidence,
    }


@app.post("/decide", response_model=DecideResponse)
def decide(
    request: DecideRequest, decider: Annotated[Decider, Depends(provide_decider)]
) -> DecideResponse:
    """Answer every question about the state in one pass.

    Raises:
        HTTPException: 400 when a question is malformed, 503 when the encoder is
            unreachable.
    """
    try:
        questions = tuple(
            Question(
                name=item.name,
                kind=item.kind,
                prompt=item.prompt,
                options=tuple(item.options),
            )
            for item in request.questions
        )
    except ValueError as error:
        # Only question construction is a client error. Wrapping the whole call
        # in this would report a server-side ValueError, such as a shape
        # mismatch from numpy, as "your request is wrong".
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        answers = decider.decide(request.state, questions)
    except EncoderError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return DecideResponse(
        answers=[
            AnswerResponse(
                name=answer.name,
                kind=answer.kind,
                value=answer.value,
                confidence=round(answer.confidence, 6),
                abstained=answer.abstained,
                probabilities=list(answer.probabilities),
            )
            for answer in answers
        ],
        temperature=decider.parameters.temperature,
    )
