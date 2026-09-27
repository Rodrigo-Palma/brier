import pytest

from brier.tasks import DOMAINS, generate


def _fragments(domain: str, polarity: str) -> tuple[str, ...]:
    """The stable words of a fact template, with the number slot removed.

    The templates carry a {value} placeholder, so asserting on the raw template
    would match nothing and the test would pass without checking anything. This
    returns the parts either side of the slot instead.
    """
    template = str(DOMAINS[domain][polarity])
    return tuple(part for part in template.split("{value}") if part.strip())


def test_held_out_domains_never_appear_in_the_generated_text():
    held_out = frozenset({"infections", "attendance"})

    examples = generate(120, seed=3, held_out=held_out)

    for example in examples:
        for key in held_out:
            for polarity in ("good", "bad"):
                for fragment in _fragments(key, polarity):
                    assert fragment not in example.state
            assert str(DOMAINS[key]["subject"]) not in example.state


def test_every_family_shows_up():
    names = {example.question.name for example in generate(60, seed=1, held_out=frozenset())}

    assert names == {"relevance", "tone", "subject", "answerable"}


def test_the_same_seed_gives_the_same_examples():
    first = generate(30, seed=7, held_out=frozenset({"revenue"}))
    second = generate(30, seed=7, held_out=frozenset({"revenue"}))

    assert first == second


def test_bad_news_is_labelled_bad_even_when_the_number_went_up():
    """More litigation and more infections are worse, not better."""
    examples = generate(600, seed=11, held_out=frozenset())
    tone = [e for e in examples if e.question.name == "tone"]
    fragment = _fragments("litigation", "bad")[0]

    filed = [e for e in tone if fragment in e.state]
    assert filed, f"expected at least one passage containing {fragment!r}"
    assert all(e.question.options[e.correct] == "negative" for e in filed)


def test_a_relevance_question_about_the_written_domain_is_a_yes():
    examples = generate(200, seed=5, held_out=frozenset())
    relevance = [e for e in examples if e.question.name == "relevance"]
    assert relevance

    for example in relevance:
        asked = example.question.prompt.split("about ")[1].rstrip("?")
        matching = [key for key in DOMAINS if str(DOMAINS[key]["subject"]) == asked]
        assert matching, f"prompt asked about an unknown subject: {asked!r}"
        written = any(
            fragment in example.state
            for key in matching
            for polarity in ("good", "bad")
            for fragment in _fragments(key, polarity)
        )
        assert bool(example.correct) is written


def test_neutral_passages_name_their_subject_so_the_split_separates_them():
    """Fixed neutral strings used to leak: 25 of 75 test rows were training rows."""
    held_out = frozenset({"infections", "attendance"})
    train = generate(300, seed=1, held_out=held_out)
    test = generate(300, seed=2, held_out=frozenset(DOMAINS) - held_out)

    shared = {e.state for e in train} & {e.state for e in test}

    assert shared == set()


def test_needs_two_domains_to_work_with():
    with pytest.raises(ValueError, match="at least two domains"):
        generate(10, seed=0, held_out=frozenset(DOMAINS))


def test_the_hard_negative_family_is_close_to_balanced():
    """A quarter-positive split is scored 75% by answering "no" to everything."""
    examples = [
        example
        for example in generate(600, seed=13, held_out=frozenset())
        if example.question.name == "answerable"
    ]

    positive = sum(example.correct for example in examples) / len(examples)

    assert 0.4 < positive < 0.6


def test_a_hard_negative_shares_its_wording_with_the_positive():
    """The point of the family: rejecting it cannot be done on vocabulary."""
    examples = [
        example
        for example in generate(400, seed=17, held_out=frozenset())
        if example.question.name == "answerable"
    ]
    negatives = [example for example in examples if not example.correct]

    assert negatives
    for example in negatives[:20]:
        assert example.question.prompt.startswith("Does the passage answer this question:")
        assert "annual report" in example.state


def test_the_possessive_reads_as_english_not_as_two_articles():
    prompts = {
        example.question.prompt
        for example in generate(300, seed=19, held_out=frozenset())
        if example.question.name == "answerable"
    }

    assert not any("'s the " in prompt for prompt in prompts)
