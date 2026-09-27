import numpy as np
import pytest

from brier.features import feature_size, pair_features
from brier.tasks import generate
from brier.train import Batch, TrainingConfig, by_name, evaluate, featurise, fit


def _separable(count: int, dimension: int, noise: float, seed: int) -> list[Batch]:
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(count):
        correct = int(rng.integers(0, 3))
        options = rng.standard_normal((3, dimension)).astype(np.float32)
        state = options[correct] + noise * rng.standard_normal(dimension).astype(np.float32)
        prompt = np.ones(dimension, dtype=np.float32)
        batches.append(Batch(pair_features(state, prompt, options), correct))
    return batches


def test_featurise_encodes_every_text_once(encoder):
    examples = generate(20, seed=1, held_out=frozenset({"infections"}))

    featurise(examples, encoder)

    asked = encoder.calls[0]
    assert len(encoder.calls) == 1
    assert len(asked) == len(set(asked))


def test_featurise_builds_one_row_per_option(encoder):
    examples = generate(9, seed=2, held_out=frozenset({"infections"}))

    batches = featurise(examples, encoder)

    for batch, example in zip(batches, examples, strict=True):
        assert batch.features.shape[0] == len(example.question.options)
        assert batch.correct == example.correct


def test_training_learns_a_learnable_task():
    training, holdout = _separable(200, 8, 0.1, seed=0), _separable(60, 8, 0.1, seed=1)

    _, report = fit(training, holdout, dimension=8, config=TrainingConfig(epochs=20))

    assert report.best.holdout_accuracy > 0.9


def test_the_report_counts_the_parameters_it_trained():
    training, holdout = _separable(40, 4, 0.5, seed=0), _separable(20, 4, 0.5, seed=1)

    _, report = fit(training, holdout, dimension=4, config=TrainingConfig(epochs=3, hidden_size=8))

    assert report.parameter_count == feature_size(4) * 8 + 8 + 8 + 1


def test_the_returned_weights_are_the_best_epoch_not_the_last():
    training, holdout = _separable(60, 6, 1.5, seed=4), _separable(40, 6, 1.5, seed=5)

    parameters, report = fit(
        training, holdout, dimension=6, config=TrainingConfig(epochs=25, patience=25)
    )

    loss, _ = evaluate(parameters, holdout)
    assert loss == pytest.approx(report.best.holdout_loss, abs=1e-6)
    assert loss <= report.epochs[-1].holdout_loss


def test_patience_stops_before_the_last_epoch_when_the_holdout_stops_improving():
    training, holdout = _separable(80, 6, 1.5, seed=6), _separable(40, 6, 1.5, seed=7)

    _, report = fit(training, holdout, dimension=6, config=TrainingConfig(epochs=80, patience=3))

    assert len(report.epochs) < 80


def test_both_splits_are_required():
    with pytest.raises(ValueError, match="at least one question"):
        fit(_separable(4, 4, 0.1, seed=0), [], dimension=4)


def test_results_are_broken_down_by_question_family(encoder):
    examples = generate(30, seed=8, held_out=frozenset({"infections"}))
    batches = featurise(examples, encoder)
    parameters, _ = fit(
        batches, batches, dimension=encoder.dimension, config=TrainingConfig(epochs=2)
    )

    breakdown = by_name(parameters, batches)

    assert set(breakdown) == {"answerable", "relevance", "subject", "tone"}
    assert sum(count for count, _ in breakdown.values()) == len(examples)
    assert all(0.0 <= accuracy <= 1.0 for _, accuracy in breakdown.values())
