import numpy as np
import pytest

from brier.decide import Decider
from brier.features import feature_size
from brier.model import initialise, with_temperature
from brier.types import Abstained, boolean, choice, score
from tests.conftest import FAKE_DIMENSION


@pytest.fixture
def decider(encoder) -> Decider:
    parameters = initialise(feature_size(FAKE_DIMENSION), hidden_size=8, seed=0)
    return Decider(encoder=encoder, parameters=parameters)


def test_one_answer_per_question_in_the_order_asked(decider):
    questions = (
        boolean("relevance", "Does it answer?"),
        score("tone", "How does it read?", ("bad", "ok", "good")),
    )

    answers = decider.decide("revenue grew", questions)

    assert [answer.name for answer in answers] == ["relevance", "tone"]


def test_the_state_is_encoded_once_for_every_question(decider, encoder):
    questions = tuple(boolean(f"q{index}", f"Is it about {index}?") for index in range(6))

    decider.decide("revenue grew", questions)

    assert len(encoder.calls) == 1
    assert encoder.calls[0].count("revenue grew") == 1


def test_probabilities_line_up_with_the_options(decider):
    question = choice("subject", "Which one?", ("revenue", "headcount", "rainfall"))

    answer = decider.decide("revenue grew", (question,))[0]

    assert len(answer.probabilities) == 3
    assert answer.value == question.options[answer.index]
    assert sum(answer.probabilities) == pytest.approx(1.0, abs=1e-5)


def test_an_untrained_model_abstains_because_it_is_near_chance(decider):
    answer = decider.decide("revenue grew", (boolean("relevance", "Does it answer?"),))[0]

    assert answer.abstained is True
    with pytest.raises(Abstained):
        _ = answer.as_bool


def test_a_threshold_of_zero_never_abstains(encoder):
    parameters = initialise(feature_size(FAKE_DIMENSION), hidden_size=8, seed=0)
    permissive = Decider(encoder=encoder, parameters=parameters, min_confidence=0.0)

    answer = permissive.decide("revenue grew", (boolean("relevance", "Does it answer?"),))[0]

    assert answer.abstained is False


def test_a_flatter_temperature_pushes_a_borderline_answer_into_abstention(encoder):
    parameters = initialise(feature_size(FAKE_DIMENSION), hidden_size=8, seed=0)
    question = choice("subject", "Which one?", ("revenue", "headcount"))
    sharp = Decider(encoder=encoder, parameters=with_temperature(parameters, 0.01))
    flat = Decider(encoder=encoder, parameters=with_temperature(parameters, 50.0))

    assert sharp.decide("revenue grew", (question,))[0].confidence > 0.5
    assert flat.decide("revenue grew", (question,))[0].abstained is True


def test_a_blank_state_is_refused(decider):
    with pytest.raises(ValueError, match="needs a state"):
        decider.decide("   ", (boolean("relevance", "Does it answer?"),))


def test_asking_nothing_is_refused(decider):
    with pytest.raises(ValueError, match="no questions"):
        decider.decide("revenue grew", ())


def test_the_threshold_must_be_a_probability(encoder):
    with pytest.raises(ValueError, match="must be a probability"):
        Decider(
            encoder=encoder,
            parameters=initialise(feature_size(FAKE_DIMENSION), hidden_size=8),
            min_confidence=1.4,
        )


def test_the_same_state_and_question_always_give_the_same_answer(decider):
    question = choice("subject", "Which one?", ("revenue", "headcount", "rainfall"))

    first = decider.decide("revenue grew", (question,))[0]
    second = decider.decide("revenue grew", (question,))[0]

    assert first == second
    np.testing.assert_allclose(first.probabilities, second.probabilities)
