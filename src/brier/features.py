"""How a (state, question, option) triple becomes one row of numbers.

The row is built so that a linear layer can already express the useful
comparisons: how close the option is to the state, where they disagree, and
how the option relates to what was asked.

The last block is the one that makes yes/no questions work at all. When the
options carry content ("quarterly revenue" against "pending litigation") the
state-option blocks decide it. When they do not -- "no" against "yes", or the
levels of a rubric -- the only thing that separates the options is how the
state answers the prompt, and that is a three-way interaction. Without it the
model can only learn a prior over the words "no" and "yes", which is to say it
guesses.
"""

import numpy as np

FEATURE_BLOCKS = 7


def pair_features(state: np.ndarray, prompt: np.ndarray, options: np.ndarray) -> np.ndarray:
    """Build one feature row per option.

    Args:
        state: the shared state vector, shape (d,).
        prompt: the question vector, shape (d,).
        options: one vector per option, shape (n, d).

    Returns:
        Shape (n, 7 * d).

    Raises:
        ValueError: when the vectors do not share a dimension.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    prompt = np.asarray(prompt, dtype=np.float32).reshape(-1)
    options = np.asarray(options, dtype=np.float32)
    if options.ndim != 2:
        raise ValueError("options must be a matrix of one vector per option")
    if not state.shape[0] == prompt.shape[0] == options.shape[1]:
        raise ValueError("state, prompt and options must share a dimension")

    repeated_state = np.repeat(state[None, :], len(options), axis=0)
    repeated_prompt = np.repeat(prompt[None, :], len(options), axis=0)
    return np.hstack(
        [
            repeated_state,
            options,
            repeated_state * options,
            np.abs(repeated_state - options),
            repeated_prompt * options,
            repeated_state * repeated_prompt * options,
            (state @ prompt) * options,
        ]
    ).astype(np.float32)


def feature_size(dimension: int) -> int:
    return FEATURE_BLOCKS * dimension
