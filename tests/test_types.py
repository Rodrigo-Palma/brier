import pytest

from brier.types import (
    MAX_OPTIONS,
    Abstained,
    Answer,
    QuestionKind,
    boolean,
    choice,
    score,
)


def test_boolean_puts_yes_at_index_one_so_the_index_is_the_truth_value():
    question = boolean("relevance", "Does it answer?")

    assert question.options == ("no", "yes")
    assert question.kind is QuestionKind.BOOL


def test_score_keeps_the_rubric_order_it_was_given():
    question = score("tone", "How does it read?", ("bad", "fine", "good"))

    assert question.options == ("bad", "fine", "good")


def test_rejects_a_question_with_one_option():
    with pytest.raises(ValueError, match="at least two options"):
        choice("x", "pick", ("only",))


def test_rejects_repeated_options_because_the_softmax_would_split_the_mass():
    with pytest.raises(ValueError, match="repeat"):
        choice("x", "pick", ("same", "same"))


def test_rejects_more_options_than_the_cap():
    with pytest.raises(ValueError, match="over the"):
        choice("x", "pick", tuple(str(index) for index in range(MAX_OPTIONS + 1)))


def test_rejects_a_blank_name():
    with pytest.raises(ValueError, match="name"):
        boolean("  ", "Does it answer?")


def test_as_bool_reads_the_picked_index():
    answered = Answer("relevance", QuestionKind.BOOL, "yes", 1, (0.1, 0.9), 0.9)

    assert answered.as_bool is True


def test_as_level_counts_from_one():
    answered = Answer("tone", QuestionKind.SCORE, "good", 2, (0.1, 0.2, 0.7), 0.7)

    assert answered.as_level == 3


def test_as_bool_refuses_a_score():
    answered = Answer("tone", QuestionKind.SCORE, "good", 2, (0.1, 0.2, 0.7), 0.7)

    with pytest.raises(TypeError, match="not a bool"):
        _ = answered.as_bool


def test_an_abstention_cannot_be_read_as_an_answer():
    answered = Answer("relevance", QuestionKind.BOOL, "yes", 1, (0.45, 0.55), 0.55, abstained=True)

    with pytest.raises(Abstained, match="below the threshold"):
        _ = answered.as_bool


def test_rejects_a_blank_prompt_because_the_model_reads_it():
    with pytest.raises(ValueError, match="needs a prompt"):
        choice("x", "   ", ("a", "b"))
