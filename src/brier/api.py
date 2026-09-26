"""HTTP surface. One request carries a state and every question about it.

The shape is deliberate: a caller that needs six judgements on the same
passage sends one request, and the state is encoded once for all six.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from brier.config import Settings
from brier.decide import Decider
from brier.encoder import CachedEncoder, EncoderError, OllamaEncoder
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
    """Build the decider once. Raises at request time, not at import time.

    Raises:
        FileNotFoundError: when the weights have not been trained yet.
    """
    settings = provide_settings()
    encoder = CachedEncoder(
        OllamaEncoder(settings.ollama_url, settings.encoder_model), settings.cache_path
    )
    return Decider(
        encoder=encoder,
        parameters=load(settings.weights_path),
        min_confidence=settings.min_confidence,
    )


app = FastAPI(
    title="brier",
    description="Calibrated decisions over text, with abstention",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
        answers = decider.decide(request.state, questions)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
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
