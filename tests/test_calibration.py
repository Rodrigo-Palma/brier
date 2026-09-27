import numpy as np
import pytest

from brier.calibration import assess, calibrate, coverage_curve, fit_temperature
from brier.features import feature_size, pair_features
from brier.model import initialise, with_temperature
from brier.train import Batch, TrainingConfig, evaluate, fit


def _noisy(count: int, dimension: int, noise: float, seed: int) -> list[Batch]:
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(count):
        correct = int(rng.integers(0, 3))
        options = rng.standard_normal((3, dimension)).astype(np.float32)
        state = options[correct] + noise * rng.standard_normal(dimension).astype(np.float32)
        batches.append(
            Batch(pair_features(state, np.ones(dimension, np.float32), options), correct)
        )
    return batches


@pytest.fixture
def trained():
    training, holdout = _noisy(250, 8, 1.4, seed=0), _noisy(80, 8, 1.4, seed=1)
    parameters, _ = fit(training, holdout, dimension=8, config=TrainingConfig(epochs=25))
    return parameters, _noisy(200, 8, 1.4, seed=2)


def test_the_fitted_temperature_does_not_raise_the_holdout_loss(trained):
    parameters, validation = trained

    temperature = fit_temperature(parameters, validation).value

    before, _ = evaluate(parameters, validation)
    after, _ = evaluate(with_temperature(parameters, temperature), validation)
    assert after <= before + 1e-9


def test_calibration_brings_confidence_close_to_accuracy(trained):
    parameters, validation = trained

    _, report = calibrate(parameters, validation)

    assert abs(report.overconfidence) < 0.05


def test_the_bins_account_for_every_question(trained):
    parameters, validation = trained

    report = assess(parameters, validation)

    assert sum(one.count for one in report.bins) == len(validation)


def test_an_untrained_model_is_near_chance_and_says_so():
    parameters = initialise(feature_size(8), hidden_size=8, seed=0)

    report = assess(parameters, _noisy(150, 8, 1.0, seed=3))

    assert report.mean_confidence == pytest.approx(1 / 3, abs=0.02)
    assert report.accuracy.low < report.accuracy.rate < report.accuracy.high


def test_the_report_carries_the_error_a_perfect_model_would_score(trained):
    """An ECE below the floor is evidence of nothing."""
    parameters, validation = trained

    report = assess(parameters, validation)

    assert report.noise_floor.rate > 0
    assert report.beats_the_floor is (report.expected_calibration_error > report.noise_floor.high)


def test_a_temperature_that_lands_on_the_grid_edge_says_so(trained):
    parameters, validation = trained

    sharp = fit_temperature(with_temperature(parameters, 1.0), validation)

    assert isinstance(sharp.at_grid_edge, bool)


def test_abstaining_raises_accuracy_among_the_answers_that_remain(trained):
    parameters, validation = trained
    tempered, _ = calibrate(parameters, validation)

    curve = coverage_curve(tempered, validation, (0.0, 0.9))

    assert curve[0].coverage == 1.0
    assert curve[1].coverage < 1.0
    assert curve[1].accuracy_when_answered > curve[0].accuracy_when_answered


def test_coverage_never_grows_as_the_threshold_rises(trained):
    parameters, validation = trained

    curve = coverage_curve(parameters, validation, (0.4, 0.6, 0.8, 0.95))

    coverages = [point.coverage for point in curve]
    assert coverages == sorted(coverages, reverse=True)


def test_a_threshold_nothing_clears_reports_nan_rather_than_a_made_up_accuracy(trained):
    parameters, validation = trained

    point = coverage_curve(parameters, validation, (1.0,))[0]

    assert point.answered == 0
    assert np.isnan(point.accuracy_when_answered)


def test_thresholds_must_be_probabilities(trained):
    parameters, validation = trained

    with pytest.raises(ValueError, match="probabilities"):
        coverage_curve(parameters, validation, (1.5,))


def test_measuring_nothing_is_an_error(trained):
    parameters, _ = trained

    with pytest.raises(ValueError, match="needs questions"):
        assess(parameters, [])


def test_a_reliability_curve_needs_more_than_one_bin(trained):
    parameters, validation = trained

    with pytest.raises(ValueError, match="at least two bins"):
        assess(parameters, validation, bin_count=1)
