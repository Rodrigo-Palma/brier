"""Synthetic decision tasks, built by composition.

Training data is generated rather than collected, which is what makes the
model reproducible from an empty directory. The risk of generated data is that
the model learns the template instead of the meaning, so every family here is
built from a grid of (domain, fact) pairs and the splits hold out whole
combinations: a test example is phrased from a pairing the training set never
showed.
"""

import random
from collections.abc import Iterator

from brier.data import Example
from brier.types import boolean, choice, score

DOMAINS: dict[str, dict[str, str]] = {
    "revenue": {
        "subject": "quarterly revenue",
        "good": "revenue grew to 94.9 billion dollars",
        "bad": "revenue fell to 81.4 billion dollars",
    },
    "headcount": {
        "subject": "the number of employees",
        "good": "the company added 12,000 people without raising cost per head",
        "bad": "the company reduced its workforce to 121,000 employees",
    },
    "litigation": {
        "subject": "pending litigation",
        "good": "the remaining antitrust claims were dismissed",
        "bad": "three new class actions were filed against the company",
    },
    "rainfall": {
        "subject": "rainfall in the region",
        "good": "rainfall returned to the seasonal average after two dry years",
        "bad": "rainfall stopped at 31 millimetres, the driest season on record",
    },
    "attendance": {
        "subject": "match attendance",
        "good": "the stadium sold out for eleven consecutive matches",
        "bad": "attendance dropped below nine thousand per match",
    },
    "infections": {
        "subject": "reported infections",
        "good": "reported infections fell for the ninth straight week",
        "bad": "reported infections doubled over six weeks",
    },
}

SENTIMENT_LEVELS = ("negative", "neutral", "positive")

# Hard negatives. A question about the right subject and the wrong year, or the
# right subject and another company, is close enough in wording that similarity
# alone cannot reject it. Without these, the relevance task is isomorphic to
# cosine similarity and a model trained on it just learns to be cosine.
YEARS = ("2019", "2021", "2023", "2024", "2025")
COMPANIES = ("Apple", "Petrobras", "Vale", "Siemens", "Toyota")

HEDGES = (
    "According to the filing, {}.",
    "The report states that {}.",
    "Over the period covered, {}.",
    "Management noted that {}.",
    "In the year under review, {}.",
)

NEUTRAL_LINES = (
    "The figures are presented in the accompanying tables.",
    "Comparative periods are restated on the same basis.",
    "The accounting policy is unchanged from the prior year.",
    "Amounts are stated in millions unless indicated otherwise.",
)


def generate(count: int, seed: int, held_out: frozenset[str]) -> tuple[Example, ...]:
    """Build `count` examples using only the domains outside `held_out`.

    Args:
        count: how many examples to produce.
        seed: makes the result reproducible.
        held_out: domain keys this split must not touch.

    Returns:
        Examples in a shuffled order, spread evenly across the families.

    Raises:
        ValueError: when the held-out set leaves fewer than two domains.
    """
    usable = tuple(key for key in DOMAINS if key not in held_out)
    if len(usable) < 2:
        raise ValueError("at least two domains must remain outside the held-out set")

    rng = random.Random(seed)
    families = (_relevance, _sentiment, _subject, _answerable)
    examples = [next(families[index % len(families)](rng, usable)) for index in range(count)]
    rng.shuffle(examples)
    return tuple(examples)


def _relevance(rng: random.Random, usable: tuple[str, ...]) -> Iterator[Example]:
    """Does this passage answer this question? The bridge to retrieval."""
    while True:
        asked, written = rng.choice(usable), rng.choice(usable)
        passage = _passage(rng, written)
        question = boolean(
            "relevance",
            f"Does the passage state something about {DOMAINS[asked]['subject']}?",
        )
        yield Example(state=passage, question=question, correct=int(asked == written))


def _sentiment(rng: random.Random, usable: tuple[str, ...]) -> Iterator[Example]:
    """A rubric with three levels, decided by the same weights as the boolean."""
    while True:
        domain = rng.choice(usable)
        polarity = rng.choice(("good", "bad", "flat"))
        passage = (
            rng.choice(NEUTRAL_LINES) if polarity == "flat" else _passage(rng, domain, polarity)
        )
        question = score(
            "tone",
            "How does the passage read for the business?",
            SENTIMENT_LEVELS,
        )
        correct = {"bad": 0, "flat": 1, "good": 2}[polarity]
        yield Example(state=passage, question=question, correct=correct)


def _subject(rng: random.Random, usable: tuple[str, ...]) -> Iterator[Example]:
    """A plain choice over the domains, so the model must name the subject."""
    while True:
        domain = rng.choice(usable)
        options = tuple(sorted({domain, *rng.sample(usable, k=min(3, len(usable)))}))
        question = choice(
            "subject",
            "Which of these is the passage about?",
            tuple(DOMAINS[key]["subject"] for key in options),
        )
        yield Example(
            state=_passage(rng, domain),
            question=question,
            correct=options.index(domain),
        )


def _answerable(rng: random.Random, usable: tuple[str, ...]) -> Iterator[Example]:
    """Does this passage answer THIS question, not merely mention the subject?

    The passage names a company, a year and a fact. The question asks about a
    company, a year and a subject. Everything has to line up, and the wrong
    pairings are the ones a similarity score cannot reject.
    """
    while True:
        domain = rng.choice(usable)
        company, year = rng.choice(COMPANIES), rng.choice(YEARS)
        subject = DOMAINS[domain]["subject"]
        fact = DOMAINS[domain][rng.choice(("good", "bad"))]
        passage = f"In {company}'s {year} annual report: {fact}."

        # Half the examples line up. Drawing one mismatch out of four would leave
        # a quarter of them positive, and a model can score 75% on that by
        # answering "no" to everything.
        mismatch = "none" if rng.random() < 0.5 else rng.choice(("year", "company", "subject"))
        asked_year = rng.choice([other for other in YEARS if other != year])
        asked_company = rng.choice([other for other in COMPANIES if other != company])
        asked_subject = DOMAINS[rng.choice([key for key in usable if key != domain])]["subject"]

        wanted = {
            "none": (company, year, subject),
            "year": (company, asked_year, subject),
            "company": (asked_company, year, subject),
            "subject": (company, year, asked_subject),
        }[mismatch]

        question = boolean(
            "answerable",
            f"Does the passage answer this question: what was {wanted[0]}'s "
            f"{_bare(wanted[2])} in {wanted[1]}?",
        )
        yield Example(state=passage, question=question, correct=int(mismatch == "none"))


def _bare(subject: str) -> str:
    """Drop a leading article, so the possessive reads as English."""
    return subject.removeprefix("the ")


def _passage(rng: random.Random, domain: str, polarity: str | None = None) -> str:
    """One sentence about a domain. Polarity is what it means for the business,
    not which way the number moved: more litigation and more infections are bad
    news, so a model trained on direction alone would be learning noise."""
    fact = DOMAINS[domain][polarity or rng.choice(("good", "bad"))]
    return rng.choice(HEDGES).format(fact)
