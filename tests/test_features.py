import numpy as np
import pytest

from brier.features import FEATURE_BLOCKS, feature_size, pair_features


def test_one_row_per_option_and_five_blocks_per_row():
    state, prompt = np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    options = np.eye(3, 4, dtype=np.float32)

    built = pair_features(state, prompt, options)

    assert built.shape == (3, FEATURE_BLOCKS * 4) == (3, feature_size(4))


def test_identical_options_build_identical_rows():
    state, prompt = np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    options = np.tile(np.array([0.5, 0.5, 0.5], dtype=np.float32), (2, 1))

    built = pair_features(state, prompt, options)

    np.testing.assert_allclose(built[0], built[1])


def test_the_absolute_difference_block_is_zero_when_option_equals_state():
    state = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    built = pair_features(state, state, state[None, :])

    difference_block = built[0, 3 * 3 : 4 * 3]

    np.testing.assert_allclose(difference_block, np.zeros(3), atol=1e-7)


def test_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError, match="share a dimension"):
        pair_features(np.ones(4), np.ones(4), np.eye(2, 5))


def test_rejects_a_single_option_vector_that_is_not_a_matrix():
    with pytest.raises(ValueError, match="matrix"):
        pair_features(np.ones(4), np.ones(4), np.ones(4))
