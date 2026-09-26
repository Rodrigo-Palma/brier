import numpy as np
import pytest

from brier.model import (
    Parameters,
    gradients,
    initialise,
    load,
    probabilities,
    save,
    scores,
    with_temperature,
)


@pytest.fixture
def parameters() -> Parameters:
    return initialise(feature_size=20, hidden_size=8, seed=1)


@pytest.fixture
def features() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((3, 20)).astype(np.float32)


def test_probabilities_sum_to_one(parameters, features):
    assert probabilities(parameters, features).sum() == pytest.approx(1.0)


def test_one_score_per_option(parameters, features):
    assert scores(parameters, features).shape == (3,)


def test_a_higher_temperature_flattens_the_distribution(parameters, features):
    sharp = probabilities(parameters, features)
    flat = probabilities(with_temperature(parameters, 5.0), features)

    assert flat.max() < sharp.max()


def test_temperature_must_be_positive(parameters):
    with pytest.raises(ValueError, match="positive"):
        with_temperature(parameters, 0.0)


def test_with_temperature_leaves_the_original_alone(parameters):
    with_temperature(parameters, 3.0)

    assert parameters.temperature == 1.0


def test_analytic_gradient_matches_finite_differences(parameters, features):
    analytic, _ = gradients(parameters, features, correct=1)
    step, index = 1e-4, 3

    def loss_with(value: float) -> float:
        weights = parameters.second_weight.copy()
        weights[index] = value
        moved = Parameters(
            parameters.first_weight, parameters.first_bias, weights, parameters.second_bias
        )
        return gradients(moved, features, correct=1)[1]

    original = float(parameters.second_weight[index])
    numeric = (loss_with(original + step) - loss_with(original - step)) / (2 * step)

    assert analytic["second_weight"][index] == pytest.approx(numeric, abs=1e-3)


def test_gradients_reject_an_option_that_does_not_exist(parameters, features):
    with pytest.raises(IndexError, match="outside"):
        gradients(parameters, features, correct=7)


def test_initialise_rejects_a_zero_size():
    with pytest.raises(ValueError, match="positive"):
        initialise(feature_size=0)


def test_weights_survive_a_round_trip(parameters, features, tmp_path):
    tempered = with_temperature(parameters, 1.7)
    path = tmp_path / "nested" / "weights.npz"

    save(tempered, path)
    restored = load(path)

    assert restored.temperature == pytest.approx(1.7)
    np.testing.assert_allclose(probabilities(restored, features), probabilities(tempered, features))


def test_load_says_which_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no weights at"):
        load(tmp_path / "absent.npz")
