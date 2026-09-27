"""What the frozen encoder already answers, with at most one parameter.

A trained head is only worth its 344k parameters if it beats what the encoder
gives away for free. This is that free thing, built to be as strong as it is
allowed to be: the threshold it uses is fitted on the training split, the same
data the head saw.

Two rules cover every question type:

- A question whose options carry content ("quarterly revenue" against "pending
  litigation") is answered by the option closest to the state.
- A question whose options carry none ("no" against "yes") cannot be answered
  that way, so it is answered by how close the state is to the prompt, against
  a fitted threshold.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from brier.data import Example
from brier.encoder import Encoder
from brier.types import QuestionKind

THRESHOLD_GRID = tuple(np.round(np.arange(0.0, 1.0, 0.005), 3))


@dataclass(frozen=True, slots=True)
class CosineBatch:
    """The cosines a one-parameter baseline is allowed to see."""

    name: str
    kind: QuestionKind
    state_option: np.ndarray
    state_prompt: float
    correct: int


def cosine_batches(examples: Sequence[Example], encoder: Encoder) -> tuple[CosineBatch, ...]:
    """Encode once and keep only the cosines."""
    texts = tuple(
        dict.fromkeys(
            text
            for example in examples
            for text in (example.state, example.question.prompt, *example.question.options)
        )
    )
    lookup = dict(zip(texts, encoder.encode(texts), strict=True))
    return tuple(
        CosineBatch(
            name=example.question.name,
            kind=example.question.kind,
            state_option=np.asarray(
                [lookup[example.state] @ lookup[option] for option in example.question.options]
            ),
            state_prompt=float(lookup[example.state] @ lookup[example.question.prompt]),
            correct=example.correct,
        )
        for example in examples
    )


def fit_thresholds(batches: Sequence[CosineBatch]) -> dict[str, float]:
    """Best state-prompt threshold per boolean family, fitted by brute force.

    Raises:
        ValueError: when there is nothing to fit on.
    """
    if not batches:
        raise ValueError("a baseline needs questions to fit on")

    families = {batch.name for batch in batches if batch.kind is QuestionKind.BOOL}
    fitted = {}
    for name in sorted(families):
        rows = [batch for batch in batches if batch.name == name]
        best, best_hits = 0.5, -1
        for threshold in THRESHOLD_GRID:
            hits = sum(int((row.state_prompt >= threshold) == bool(row.correct)) for row in rows)
            if hits > best_hits:
                best, best_hits = float(threshold), hits
        fitted[name] = best
    return fitted


def predict(batch: CosineBatch, thresholds: dict[str, float]) -> int:
    """The baseline's answer for one question."""
    if batch.kind is QuestionKind.BOOL:
        return int(batch.state_prompt >= thresholds.get(batch.name, 0.5))
    return int(np.argmax(batch.state_option))


def accuracy_by_name(
    batches: Sequence[CosineBatch], thresholds: dict[str, float]
) -> dict[str, tuple[int, float]]:
    """Count and accuracy per family, so it lines up with the model's table."""
    tally: dict[str, list[int]] = {}
    for batch in batches:
        counts = tally.setdefault(batch.name, [0, 0])
        counts[0] += 1
        counts[1] += int(predict(batch, thresholds) == batch.correct)
    return {name: (total, hits / total) for name, (total, hits) in sorted(tally.items())}


def majority_by_name(batches: Sequence[CosineBatch]) -> dict[str, tuple[int, float]]:
    """The floor under everything: always answer the commonest option.

    Fitted per family on the same split it is measured on, which makes it the
    most generous version of itself.
    """
    tally: dict[str, list[int]] = {}
    for batch in batches:
        tally.setdefault(batch.name, []).append(batch.correct)
    result = {}
    for name, answers in sorted(tally.items()):
        counts = np.bincount(answers)
        result[name] = (len(answers), float(counts.max() / len(answers)))
    return result
