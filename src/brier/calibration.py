"""Making the reported confidence mean what it says.

A model that is right 70% of the time while reporting 0.95 is worse than
useless downstream, because the caller cannot use the number to decide when to
escalate. So confidence is fitted on held-out data after training, and the
error is measured and published rather than assumed.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from brier.model import Parameters, probabilities, scores, with_temperature
from brier.train import Batch

TEMPERATURE_GRID = tuple(np.round(np.arange(0.25, 6.01, 0.05), 2))
DEFAULT_BINS = 10


@dataclass(frozen=True, slots=True)
class Bin:
    """One slice of the reliability curve."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.accuracy)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Measured, not claimed."""

    temperature: float
    expected_calibration_error: float
    brier: float
    accuracy: float
    mean_confidence: float
    bins: tuple[Bin, ...]

    @property
    def overconfidence(self) -> float:
        """Positive when the model claims more than it delivers."""
        return self.mean_confidence - self.accuracy


def fit_temperature(parameters: Parameters, batches: Sequence[Batch]) -> float:
    """Pick the temperature that minimises held-out cross-entropy.

    A grid search rather than a gradient: it is one scalar, the grid is cheap,
    and it cannot land in a local minimum or diverge.

    Raises:
        ValueError: when there is nothing to fit on.
    """
    if not batches:
        raise ValueError("temperature needs held-out questions to fit on")

    raw = tuple(_raw_scores(parameters, batch) for batch in batches)
    correct = tuple(batch.correct for batch in batches)

    best_temperature, best_loss = 1.0, float("inf")
    for temperature in TEMPERATURE_GRID:
        loss = float(
            np.mean(
                [
                    -np.log(max(_softmax(score / temperature)[answer], 1e-12))
                    for score, answer in zip(raw, correct, strict=True)
                ]
            )
        )
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), loss
    return best_temperature


def assess(
    parameters: Parameters, batches: Sequence[Batch], bin_count: int = DEFAULT_BINS
) -> CalibrationReport:
    """Measure how well the reported confidence matches the outcome.

    Raises:
        ValueError: when there is nothing to measure or the bin count is unusable.
    """
    if not batches:
        raise ValueError("calibration needs questions to measure")
    if bin_count < 2:
        raise ValueError("a reliability curve needs at least two bins")

    predictions = [probabilities(parameters, batch.features) for batch in batches]
    confidences = np.array([float(np.max(p)) for p in predictions])
    hits = np.array(
        [int(np.argmax(p) == batch.correct) for p, batch in zip(predictions, batches, strict=True)]
    )
    brier = float(
        np.mean([_brier(p, batch.correct) for p, batch in zip(predictions, batches, strict=True)])
    )

    bins = _reliability(confidences, hits, bin_count)
    total = len(batches)
    error = sum(b.count / total * b.gap for b in bins)

    return CalibrationReport(
        temperature=parameters.temperature,
        expected_calibration_error=error,
        brier=brier,
        accuracy=float(hits.mean()),
        mean_confidence=float(confidences.mean()),
        bins=bins,
    )


def calibrate(
    parameters: Parameters, batches: Sequence[Batch], bin_count: int = DEFAULT_BINS
) -> tuple[Parameters, CalibrationReport]:
    """Fit the temperature and report what it bought."""
    tempered = with_temperature(parameters, fit_temperature(parameters, batches))
    return tempered, assess(tempered, batches, bin_count)


def _reliability(confidences: np.ndarray, hits: np.ndarray, bin_count: int) -> tuple[Bin, ...]:
    """Bins are built over the range the model actually used.

    Fixed [0, 1] bins leave most of the curve empty for a model whose lowest
    possible confidence is 1/options, and an empty bin contributes nothing but
    a misleading row.
    """
    edges = np.linspace(confidences.min(), 1.0, bin_count + 1)
    result = []
    for index in range(bin_count):
        lower, upper = float(edges[index]), float(edges[index + 1])
        inside = (confidences > lower) & (confidences <= upper) if index else confidences <= upper
        if not inside.any():
            continue
        result.append(
            Bin(
                lower=lower,
                upper=upper,
                count=int(inside.sum()),
                mean_confidence=float(confidences[inside].mean()),
                accuracy=float(hits[inside].mean()),
            )
        )
    return tuple(result)


def _brier(predicted: np.ndarray, correct: int) -> float:
    target = np.zeros_like(predicted)
    target[correct] = 1.0
    return float(np.sum((predicted - target) ** 2))


def _raw_scores(parameters: Parameters, batch: Batch) -> np.ndarray:
    return scores(parameters, batch.features)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - np.max(values))
    return shifted / np.sum(shifted)


@dataclass(frozen=True, slots=True)
class Coverage:
    """What abstaining at one threshold costs and buys."""

    threshold: float
    answered: int
    total: int
    accuracy_when_answered: float

    @property
    def coverage(self) -> float:
        return self.answered / self.total


def coverage_curve(
    parameters: Parameters, batches: Sequence[Batch], thresholds: Sequence[float]
) -> tuple[Coverage, ...]:
    """Accuracy among the questions the model would still answer.

    This is the number a caller actually buys: a model that is 70% right
    overall but 94% right on the two thirds it does not abstain from is usable,
    and one that is 70% right with unshakeable confidence is not.

    Raises:
        ValueError: when there is nothing to measure or a threshold is not a
            probability.
    """
    if not batches:
        raise ValueError("a coverage curve needs questions to measure")
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be probabilities")

    predictions = [probabilities(parameters, batch.features) for batch in batches]
    confidences = np.array([float(np.max(p)) for p in predictions])
    hits = np.array(
        [int(np.argmax(p) == batch.correct) for p, batch in zip(predictions, batches, strict=True)]
    )

    curve = []
    for threshold in thresholds:
        answered = confidences >= threshold
        count = int(answered.sum())
        curve.append(
            Coverage(
                threshold=float(threshold),
                answered=count,
                total=len(batches),
                accuracy_when_answered=float(hits[answered].mean()) if count else float("nan"),
            )
        )
    return tuple(curve)
