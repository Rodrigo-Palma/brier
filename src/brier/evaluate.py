"""Putting a number next to every number: baseline, interval, and floor.

A measurement without a comparison is not a measurement. This module reports
the model beside what the frozen encoder already answers for free, so a reader
can see what the trained weights actually bought, and beside the interval each
rate is consistent with, so a reader can see whether the difference is real.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from brier.baseline import CosineBatch, accuracy_by_name, majority_by_name, predict
from brier.calibration import CalibrationReport
from brier.model import Parameters, probabilities
from brier.statistics import (
    Interval,
    area_under_risk_coverage,
    auroc,
    auroc_interval,
    permutation_p,
    wilson,
)
from brier.train import Batch

BOOTSTRAP_DRAWS = 2000


@dataclass(frozen=True, slots=True)
class FamilyResult:
    """One question family, model against the free alternatives."""

    name: str
    model: Interval
    cosine: Interval
    majority: float

    @property
    def gain(self) -> float:
        return self.model.rate - self.cosine.rate


@dataclass(frozen=True, slots=True)
class Ranking:
    """Whether the confidence knows which answers are right.

    This is what abstention rests on, and unlike a coverage table it is a single
    number with an interval, so it can be compared across runs.
    """

    auroc: float
    low: float
    high: float
    p_value: float
    risk_coverage_area: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Everything the README is allowed to claim."""

    rows: int
    distinct: int
    model: Interval
    cosine: Interval
    gain_low: float
    gain_high: float
    families: tuple[FamilyResult, ...]
    ranking: Ranking
    calibration: CalibrationReport

    @property
    def verdict(self) -> str:
        """What the interval for the gain over cosine actually supports."""
        if self.gain_low > 0.0:
            return "the model beats the baseline"
        if self.gain_high < 0.0:
            return "the model LOSES to the baseline"
        return "indistinguishable from the baseline"


def evaluate(
    parameters: Parameters,
    batches: Sequence[Batch],
    cosines: Sequence[CosineBatch],
    thresholds: dict[str, float],
    calibration: CalibrationReport,
    distinct: int,
    seed: int = 0,
) -> Evaluation:
    """Score the model and the baseline on the same questions.

    Raises:
        ValueError: when the two views disagree about how many questions there are.
    """
    if len(batches) != len(cosines):
        raise ValueError(
            f"{len(batches)} model rows against {len(cosines)} baseline rows: "
            "both views must describe the same questions"
        )

    predictions = [probabilities(parameters, batch.features) for batch in batches]
    model_hits = np.array(
        [int(np.argmax(p) == b.correct) for p, b in zip(predictions, batches, strict=True)]
    )
    cosine_hits = np.array([int(predict(c, thresholds) == c.correct) for c in cosines])
    confidences = np.array([float(np.max(p)) for p in predictions])

    low, high = _gain_interval(model_hits, cosine_hits, seed)
    model_by_name = _by_name(batches, model_hits)
    cosine_by_name = accuracy_by_name(cosines, thresholds)
    majority = majority_by_name(cosines)

    families = tuple(
        FamilyResult(
            name=name,
            model=model_by_name[name],
            cosine=wilson(
                round(cosine_by_name[name][1] * cosine_by_name[name][0]), cosine_by_name[name][0]
            ),
            majority=majority[name][1],
        )
        for name in sorted(model_by_name)
    )

    return Evaluation(
        rows=len(batches),
        distinct=distinct,
        model=wilson(int(model_hits.sum()), len(model_hits)),
        cosine=wilson(int(cosine_hits.sum()), len(cosine_hits)),
        gain_low=low,
        gain_high=high,
        families=families,
        ranking=_ranking(confidences, model_hits, seed),
        calibration=calibration,
    )


def _by_name(batches: Sequence[Batch], hits: np.ndarray) -> dict[str, Interval]:
    tally: dict[str, list[int]] = {}
    for batch, hit in zip(batches, hits, strict=True):
        counts = tally.setdefault(batch.name, [0, 0])
        counts[0] += 1
        counts[1] += int(hit)
    return {name: wilson(got, total) for name, (total, got) in tally.items()}


def _gain_interval(
    model_hits: np.ndarray, cosine_hits: np.ndarray, seed: int
) -> tuple[float, float]:
    """Bootstrap interval for the paired difference in accuracy."""
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = rng.integers(0, len(model_hits), len(model_hits))
        differences.append(model_hits[picked].mean() - cosine_hits[picked].mean())
    return float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))


def _ranking(confidences: np.ndarray, hits: np.ndarray, seed: int) -> Ranking:
    area = auroc(confidences, hits)
    low, high = auroc_interval(confidences, hits, seed=seed)
    return Ranking(
        auroc=area,
        low=low,
        high=high,
        p_value=permutation_p(confidences, hits, seed=seed),
        risk_coverage_area=area_under_risk_coverage(confidences, hits),
    )
