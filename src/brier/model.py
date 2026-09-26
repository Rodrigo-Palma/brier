"""The part of the system that is ours: a scorer trained from outcomes.

One function scores a (state, question, option) triple. A question is decided
by scoring each of its options and taking a softmax over the scores, so the
same weights answer a yes/no and a five-level rubric without knowing the
difference. The encoder underneath is frozen and borrowed; everything here is
trained by us, on CPU, in minutes.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

DEFAULT_HIDDEN = 64
INIT_SCALE = 0.05


@dataclass(frozen=True, slots=True)
class Parameters:
    """Weights of the scorer. Frozen: every update returns a new instance."""

    first_weight: np.ndarray
    first_bias: np.ndarray
    second_weight: np.ndarray
    second_bias: np.ndarray
    temperature: float = 1.0


def initialise(feature_size: int, hidden_size: int = DEFAULT_HIDDEN, seed: int = 0) -> Parameters:
    """Small random weights, scaled so early scores sit near zero.

    Raises:
        ValueError: when a size is not positive.
    """
    if feature_size <= 0 or hidden_size <= 0:
        raise ValueError("sizes must be positive")

    rng = np.random.default_rng(seed)
    scale = INIT_SCALE / np.sqrt(feature_size)
    return Parameters(
        first_weight=(rng.standard_normal((feature_size, hidden_size)) * scale).astype(np.float32),
        first_bias=np.zeros(hidden_size, dtype=np.float32),
        second_weight=(rng.standard_normal(hidden_size) * INIT_SCALE).astype(np.float32),
        second_bias=np.float32(0.0),
    )


def scores(parameters: Parameters, features: np.ndarray) -> np.ndarray:
    """One raw score per option, shape (n,)."""
    hidden = np.maximum(features @ parameters.first_weight + parameters.first_bias, 0.0)
    return hidden @ parameters.second_weight + parameters.second_bias


def probabilities(parameters: Parameters, features: np.ndarray) -> np.ndarray:
    """Softmax over the options, after dividing by the calibration temperature.

    The temperature is the whole point of keeping it in the parameters: it is
    fitted after training, on held-out data, and it is what makes a reported
    0.8 mean eight times in ten.
    """
    return softmax(scores(parameters, features) / parameters.temperature)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - np.max(values))
    return shifted / np.sum(shifted)


def with_temperature(parameters: Parameters, temperature: float) -> Parameters:
    """Return a copy carrying a new temperature.

    Raises:
        ValueError: when the temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return replace(parameters, temperature=float(temperature))


def gradients(
    parameters: Parameters, features: np.ndarray, correct: int
) -> tuple[dict[str, np.ndarray], float]:
    """Cross-entropy loss and its gradients for one question.

    Training runs at temperature 1; calibration is a separate, later step, so
    the temperature deliberately does not appear here.

    Raises:
        IndexError: when `correct` is not one of the options.
    """
    if not 0 <= correct < len(features):
        raise IndexError(f"option {correct} is outside the {len(features)} given")

    hidden_raw = features @ parameters.first_weight + parameters.first_bias
    hidden = np.maximum(hidden_raw, 0.0)
    predicted = softmax(hidden @ parameters.second_weight + parameters.second_bias)

    loss = float(-np.log(max(predicted[correct], 1e-12)))

    d_scores = predicted.copy()
    d_scores[correct] -= 1.0
    d_hidden = np.outer(d_scores, parameters.second_weight) * (hidden_raw > 0)

    return {
        "first_weight": features.T @ d_hidden,
        "first_bias": d_hidden.sum(axis=0),
        "second_weight": hidden.T @ d_scores,
        "second_bias": np.float32(d_scores.sum()),
    }, loss


def save(parameters: Parameters, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        first_weight=parameters.first_weight,
        first_bias=parameters.first_bias,
        second_weight=parameters.second_weight,
        second_bias=parameters.second_bias,
        temperature=np.float32(parameters.temperature),
    )


def load(path: Path) -> Parameters:
    """Read weights written by `save`.

    Raises:
        FileNotFoundError: when the file is not there.
    """
    if not path.exists():
        raise FileNotFoundError(f"no weights at {path}")
    stored = np.load(path)
    return Parameters(
        first_weight=stored["first_weight"],
        first_bias=stored["first_bias"],
        second_weight=stored["second_weight"],
        second_bias=np.float32(stored["second_bias"]),
        temperature=float(stored["temperature"]),
    )
