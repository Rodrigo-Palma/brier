"""Intervals and floors, so a number is never reported as if it were exact.

Every rate here comes from a few hundred questions. At that size a difference
of five points is routinely noise, and an expected calibration error of 0.09
is what a perfectly calibrated model produces. Reporting the point estimate
alone is how a measurement turns into a claim it cannot support.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Z_95 = 1.959964


@dataclass(frozen=True, slots=True)
class Interval:
    """A rate with the range it is actually consistent with."""

    rate: float
    low: float
    high: float
    total: int

    def __str__(self) -> str:
        return f"{self.rate:.3f} [{self.low:.3f}, {self.high:.3f}] n={self.total}"


def wilson(hits: int, total: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a proportion.

    Wilson rather than the normal approximation because the normal one is wrong
    in exactly the cases here: small samples and rates near 0 or 1, where it can
    reach past the ends of the scale.

    Raises:
        ValueError: when the counts are impossible.
    """
    if total <= 0:
        raise ValueError("an interval needs at least one observation")
    if not 0 <= hits <= total:
        raise ValueError(f"{hits} hits out of {total} is impossible")

    rate = hits / total
    denominator = 1 + z**2 / total
    centre = (rate + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return Interval(
        rate=rate,
        low=float(max(0.0, centre - spread)),
        high=float(min(1.0, centre + spread)),
        total=total,
    )


def auroc(scores: Sequence[float], outcomes: Sequence[int]) -> float:
    """Probability that a correct answer is more confident than a wrong one.

    This is the statistic abstention actually rests on. A coverage table is five
    correlated rows; this is one number, and 0.5 means the confidence carries no
    information about whether the answer is right.

    Raises:
        ValueError: when there is nothing to compare.
    """
    right = np.asarray([s for s, o in zip(scores, outcomes, strict=True) if o])
    wrong = np.asarray([s for s, o in zip(scores, outcomes, strict=True) if not o])
    if not len(right) or not len(wrong):
        raise ValueError("AUROC needs at least one right and one wrong answer")

    comparisons = right[:, None] - wrong[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def auroc_interval(
    scores: Sequence[float], outcomes: Sequence[int], draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap interval for `auroc`, resampling questions."""
    rng = np.random.default_rng(seed)
    scores, outcomes = np.asarray(scores), np.asarray(outcomes)
    values = []
    for _ in range(draws):
        picked = rng.integers(0, len(scores), len(scores))
        try:
            values.append(auroc(scores[picked], outcomes[picked]))
        except ValueError:
            continue
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def permutation_p(
    scores: Sequence[float], outcomes: Sequence[int], draws: int = 2000, seed: int = 0
) -> float:
    """How often shuffled outcomes reach this AUROC or better."""
    observed = auroc(scores, outcomes)
    rng = np.random.default_rng(seed)
    outcomes = np.asarray(outcomes)
    at_least = 0
    for _ in range(draws):
        shuffled = rng.permutation(outcomes)
        try:
            at_least += int(auroc(scores, shuffled) >= observed)
        except ValueError:
            continue
    return (at_least + 1) / (draws + 1)


def area_under_risk_coverage(confidences: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean error rate over every coverage level, answering most-confident first.

    Lower is better, and it summarises the whole abstention curve instead of
    picking one threshold to quote.

    Raises:
        ValueError: when there is nothing to measure.
    """
    if not len(confidences):
        raise ValueError("a risk-coverage curve needs questions")

    order = np.argsort(-np.asarray(confidences, dtype=float))
    errors = 1 - np.asarray(outcomes, dtype=float)[order]
    cumulative = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(np.mean(cumulative))


def ece_noise_floor(
    confidences: Sequence[float], bin_count: int, draws: int = 400, seed: int = 0
) -> Interval:
    """What ECE a perfectly calibrated model would score, at this sample size.

    ECE is biased upward, and the bias grows as the sample shrinks. Simulating
    outcomes from the reported confidences gives the floor: an ECE at or below
    this number is evidence of nothing at all.

    Returns:
        The mean floor, with the 2.5th and 97.5th percentiles as the range.
    """
    confidences = np.asarray(confidences, dtype=float)
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, bin_count + 1)

    floors = []
    for _ in range(draws):
        drawn = rng.random(len(confidences)) < confidences
        error = 0.0
        for index in range(bin_count):
            lower, upper = edges[index], edges[index + 1]
            inside = (
                (confidences > lower) & (confidences <= upper) if index else confidences <= upper
            )
            if inside.any():
                error += (
                    inside.sum()
                    / len(confidences)
                    * abs(confidences[inside].mean() - drawn[inside].mean())
                )
        floors.append(error)

    return Interval(
        rate=float(np.mean(floors)),
        low=float(np.percentile(floors, 2.5)),
        high=float(np.percentile(floors, 97.5)),
        total=len(confidences),
    )
