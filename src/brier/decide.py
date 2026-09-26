"""The public entry point: state in, calibrated answers out.

Several questions about the same state are answered in one pass. That is the
economy of the design: the state is encoded once, and each extra question costs
one small matrix multiply per option, not another forward pass through a large
model.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from brier.encoder import Encoder
from brier.features import pair_features
from brier.model import Parameters, probabilities
from brier.types import Answer, Question

DEFAULT_MIN_CONFIDENCE = 0.6


@dataclass(frozen=True, slots=True)
class Decider:
    """A trained model, ready to answer questions about a state.

    Attributes:
        encoder: turns text into vectors; frozen and not trained here.
        parameters: our weights, including the fitted temperature.
        min_confidence: below this, the answer comes back as an abstention.
    """

    encoder: Encoder
    parameters: Parameters
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be a probability")

    def decide(self, state: str, questions: Sequence[Question]) -> tuple[Answer, ...]:
        """Answer every question about one state.

        Raises:
            ValueError: when the state is blank or no question was asked.
            EncoderError: when the encoder cannot produce vectors.
        """
        if not state.strip():
            raise ValueError("a decision needs a state to be about")
        if not questions:
            raise ValueError("no questions asked")

        texts = tuple(
            dict.fromkeys(
                [state, *(q.prompt for q in questions), *(o for q in questions for o in q.options)]
            )
        )
        lookup = dict(zip(texts, self.encoder.encode(texts), strict=True))
        return tuple(self._answer(lookup, state, question) for question in questions)

    def _answer(self, lookup: dict, state: str, question: Question) -> Answer:
        features = pair_features(
            lookup[state],
            lookup[question.prompt],
            np.asarray([lookup[option] for option in question.options]),
        )
        predicted = probabilities(self.parameters, features)
        index = int(np.argmax(predicted))
        confidence = float(predicted[index])

        return Answer(
            name=question.name,
            kind=question.kind,
            value=question.options[index],
            index=index,
            probabilities=tuple(round(float(value), 6) for value in predicted),
            confidence=confidence,
            abstained=confidence < self.min_confidence,
        )
