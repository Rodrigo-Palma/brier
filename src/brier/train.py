"""Fitting the scorer. Adam, written out, because the whole point is that the
weights are ours and nothing here is a call into someone else's trainer."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from brier.data import Example
from brier.encoder import Encoder
from brier.features import feature_size, pair_features
from brier.model import Parameters, gradients, initialise, probabilities

BETA_ONE = 0.9
BETA_TWO = 0.999
EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 40
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    hidden_size: int = 64
    seed: int = 0
    patience: int = 6


DEFAULT_CONFIG = TrainingConfig()


@dataclass(frozen=True, slots=True)
class Epoch:
    number: int
    train_loss: float
    holdout_loss: float
    holdout_accuracy: float


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """What the run actually did, kept so the README can quote measurements."""

    epochs: tuple[Epoch, ...]
    best_epoch: int
    parameter_count: int

    @property
    def best(self) -> Epoch:
        return self.epochs[self.best_epoch - 1]


@dataclass(frozen=True, slots=True)
class Batch:
    """One question, already turned into numbers.

    The name travels with the numbers so results can be broken down by family.
    An aggregate accuracy over mixed question types hides the one that is
    failing, which is how a model ships looking better than it is.
    """

    features: np.ndarray
    correct: int
    name: str = ""
    tag: str = ""


def featurise(examples: Sequence[Example], encoder: Encoder) -> tuple[Batch, ...]:
    """Encode every example once, so training epochs are pure arithmetic."""
    texts = tuple(
        dict.fromkeys(
            text
            for example in examples
            for text in (example.state, example.question.prompt, *example.question.options)
        )
    )
    lookup = dict(zip(texts, encoder.encode(texts), strict=True))
    return tuple(
        Batch(
            features=pair_features(
                lookup[example.state],
                lookup[example.question.prompt],
                np.asarray([lookup[option] for option in example.question.options]),
            ),
            correct=example.correct,
            name=example.question.name,
            tag=example.tag,
        )
        for example in examples
    )


def fit(
    training: Sequence[Batch],
    holdout: Sequence[Batch],
    dimension: int,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> tuple[Parameters, TrainingReport]:
    """Train on `training`, stop on `holdout`, return the best weights seen.

    Early stopping reads the held-out loss rather than its accuracy, because a
    model can get more answers right while getting more confident about the
    ones it gets wrong, and confidence is what this model sells.

    Raises:
        ValueError: when either split is empty.
    """
    if not training or not holdout:
        raise ValueError("both splits need at least one question")

    parameters = initialise(feature_size(dimension), config.hidden_size, config.seed)
    moment, velocity, step = _zeros(parameters), _zeros(parameters), 0
    rng = np.random.default_rng(config.seed)

    history: list[Epoch] = []
    best, best_epoch, best_loss = parameters, 1, float("inf")

    for number in range(1, config.epochs + 1):
        total = 0.0
        for index in rng.permutation(len(training)):
            batch = training[int(index)]
            gradient, loss = gradients(parameters, batch.features, batch.correct)
            total += loss
            step += 1
            parameters, moment, velocity = _step(
                parameters, gradient, moment, velocity, step, config
            )

        holdout_loss, holdout_accuracy = evaluate(parameters, holdout)
        history.append(Epoch(number, total / len(training), holdout_loss, holdout_accuracy))

        if holdout_loss < best_loss:
            best, best_epoch, best_loss = parameters, number, holdout_loss
        elif number - best_epoch >= config.patience:
            break

    return best, TrainingReport(
        epochs=tuple(history),
        best_epoch=best_epoch,
        parameter_count=_count(best),
    )


def by_name(parameters: Parameters, batches: Sequence[Batch]) -> dict[str, tuple[int, float]]:
    """Count and accuracy for each question family present in the split."""
    return _tally(parameters, batches, lambda batch: batch.name)


def by_tag(
    parameters: Parameters, batches: Sequence[Batch], name: str
) -> dict[str, tuple[int, float]]:
    """The same, one level deeper, inside a single family.

    A family average hides its own subtypes. Measured on `answerable`: the
    family read 0.68 while the wrong-year subtype sat at exactly zero, with the
    model answering "yes" to every one of them and doing it confidently.
    """
    rows = [batch for batch in batches if batch.name == name]
    return _tally(parameters, rows, lambda batch: batch.tag or "untagged")


def _tally(parameters: Parameters, batches: Sequence[Batch], key) -> dict[str, tuple[int, float]]:
    tally: dict[str, list[int]] = {}
    for batch in batches:
        predicted = probabilities(parameters, batch.features)
        counts = tally.setdefault(key(batch), [0, 0])
        counts[0] += 1
        counts[1] += int(np.argmax(predicted) == batch.correct)
    return {name: (total, hits / total) for name, (total, hits) in sorted(tally.items())}


def evaluate(parameters: Parameters, batches: Sequence[Batch]) -> tuple[float, float]:
    """Mean cross-entropy and accuracy over a split."""
    losses, hits = [], 0
    for batch in batches:
        predicted = probabilities(parameters, batch.features)
        losses.append(-np.log(max(predicted[batch.correct], 1e-12)))
        hits += int(np.argmax(predicted) == batch.correct)
    return float(np.mean(losses)), hits / len(batches)


def _step(
    parameters: Parameters,
    gradient: dict[str, np.ndarray],
    moment: dict[str, np.ndarray],
    velocity: dict[str, np.ndarray],
    step: int,
    config: TrainingConfig,
) -> tuple[Parameters, dict[str, np.ndarray], dict[str, np.ndarray]]:
    updates, new_moment, new_velocity = {}, {}, {}
    for name, raw in gradient.items():
        current = getattr(parameters, name)
        decayed = raw + config.weight_decay * current
        new_moment[name] = BETA_ONE * moment[name] + (1 - BETA_ONE) * decayed
        new_velocity[name] = BETA_TWO * velocity[name] + (1 - BETA_TWO) * decayed**2
        corrected_moment = new_moment[name] / (1 - BETA_ONE**step)
        corrected_velocity = new_velocity[name] / (1 - BETA_TWO**step)
        updates[name] = current - config.learning_rate * corrected_moment / (
            np.sqrt(corrected_velocity) + EPSILON
        )
    return replace(parameters, **updates), new_moment, new_velocity


def _zeros(parameters: Parameters) -> dict[str, np.ndarray]:
    return {
        name: np.zeros_like(getattr(parameters, name))
        for name in ("first_weight", "first_bias", "second_weight", "second_bias")
    }


def _count(parameters: Parameters) -> int:
    return (
        parameters.first_weight.size
        + parameters.first_bias.size
        + parameters.second_weight.size
        + 1
    )
