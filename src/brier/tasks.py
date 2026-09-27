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

from brier.data import Example, identity
from brier.types import boolean, choice, score

ATTEMPTS_PER_EXAMPLE = 400

DOMAINS: dict[str, dict[str, object]] = {
    "revenue": {
        "subject": "quarterly revenue",
        "good": "revenue grew to {value}",
        "bad": "revenue fell to {value}",
        "values": ("94.9 billion dollars", "88.2 billion dollars", "81.4 billion dollars"),
    },
    "headcount": {
        "subject": "the number of employees",
        "good": "the workforce grew to {value}",
        "bad": "the workforce shrank to {value}",
        "values": ("164,000 people", "147,500 people", "121,000 people"),
    },
    "litigation": {
        "subject": "pending litigation",
        "good": "{value} were dismissed",
        "bad": "{value} were filed against the company",
        "values": ("three antitrust claims", "two class actions", "four consumer suits"),
    },
    "rainfall": {
        "subject": "rainfall in the region",
        "good": "rainfall recovered to {value}",
        "bad": "rainfall dropped to {value}",
        "values": ("240 millimetres", "112 millimetres", "31 millimetres"),
    },
    "attendance": {
        "subject": "match attendance",
        "good": "attendance climbed to {value}",
        "bad": "attendance sank to {value}",
        "values": ("48,000 per match", "23,500 per match", "9,000 per match"),
    },
    "infections": {
        "subject": "reported infections",
        "good": "reported infections fell to {value}",
        "bad": "reported infections doubled to {value}",
        "values": ("1,200 cases", "4,800 cases", "9,600 cases"),
    },
    "emissions": {
        "subject": "carbon emissions",
        "good": "emissions were cut to {value}",
        "bad": "emissions rose to {value}",
        "values": ("2.1 million tonnes", "5.6 million tonnes", "8.9 million tonnes"),
    },
    "downtime": {
        "subject": "service downtime",
        "good": "downtime was reduced to {value}",
        "bad": "downtime increased to {value}",
        "values": ("12 minutes", "94 minutes", "310 minutes"),
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
    "The disclosure confirms that {}.",
    "As set out in the notes, {}.",
    "For the reporting period, {}.",
    "The summary records that {}.",
    "Auditors observed that {}.",
    "Across the twelve months, {}.",
    "The statement shows that {}.",
)

# Neutral passages carry a subject too. Fixed strings that mention no domain
# would appear in every split, and a domain-held-out test set would then contain
# training rows verbatim. Measured before this change: 25 of 75 tone rows in the
# test split were byte-identical to training rows, and the model scored 1.000 on
# exactly those.
NEUTRAL_TEMPLATES = (
    "Figures for {subject} are presented in the accompanying tables.",
    "Comparative periods for {subject} are restated on the same basis.",
    "The accounting policy for {subject} is unchanged from the prior year.",
    "Amounts for {subject} are stated in millions unless indicated otherwise.",
    "Disclosure of {subject} follows the format used in prior years.",
    "The notes describe how {subject} is measured, without restating it.",
)


def generate(
    count: int,
    seed: int,
    held_out: frozenset[str],
    exclude: frozenset[tuple[str, str, tuple[str, ...], int]] = frozenset(),
) -> tuple[Example, ...]:
    """Build `count` distinct examples using only the domains outside `held_out`.

    Every example is distinct from the others and from anything in `exclude`,
    so a caller can build splits that share no rows. The generated space is
    finite, so asking for more than it holds fails loudly rather than returning
    duplicates: a split that silently overlaps its neighbour is worse than a
    split that is too small, because the overlap is invisible in every metric.

    Args:
        count: how many distinct examples to produce.
        seed: makes the result reproducible.
        held_out: domain keys this split must not touch.
        exclude: identities already used by another split.

    Returns:
        Examples in a shuffled order, spread evenly across the families.

    Raises:
        ValueError: when fewer than two domains remain, or when the space cannot
            supply `count` fresh examples.
    """
    usable = tuple(key for key in DOMAINS if key not in held_out)
    if len(usable) < 2:
        raise ValueError("at least two domains must remain outside the held-out set")

    rng = random.Random(seed)
    families = (_relevance, _sentiment, _subject, _answerable)
    streams = [family(rng, usable) for family in families]

    taken: set[tuple[str, str, tuple[str, ...], int]] = set(exclude)
    examples: list[Example] = []
    attempts = 0
    limit = count * ATTEMPTS_PER_EXAMPLE

    while len(examples) < count:
        attempts += 1
        if attempts > limit:
            raise ValueError(
                f"the space of {sorted(usable)} ran out after {len(examples)} of {count} "
                f"distinct examples; ask for fewer or widen the generators"
            )
        candidate = next(streams[len(examples) % len(streams)])
        key = identity(candidate)
        if key in taken:
            continue
        taken.add(key)
        examples.append(candidate)

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
        passage = _neutral(rng, domain) if polarity == "flat" else _passage(rng, domain, polarity)
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
        subject = str(DOMAINS[domain]["subject"])
        fact = _fact(rng, domain)
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
        yield Example(
            state=passage,
            question=question,
            correct=int(mismatch == "none"),
            tag=mismatch,
        )


def _fact(rng: random.Random, domain: str) -> str:
    """The bare fact, without a hedge in front of it."""
    template = str(DOMAINS[domain][rng.choice(("good", "bad"))])
    values = DOMAINS[domain]["values"]
    assert isinstance(values, tuple)
    return template.format(value=rng.choice(values))


def _neutral(rng: random.Random, domain: str) -> str:
    """A passage that states nothing good or bad, but names its subject."""
    return rng.choice(NEUTRAL_TEMPLATES).format(subject=DOMAINS[domain]["subject"])


def _bare(subject: str) -> str:
    """Drop a leading article, so the possessive reads as English."""
    return subject.removeprefix("the ")


def _passage(rng: random.Random, domain: str, polarity: str | None = None) -> str:
    """One sentence about a domain. Polarity is what it means for the business,
    not which way the number moved: more litigation and more infections are bad
    news, so a model trained on direction alone would be learning noise."""
    chosen = polarity or rng.choice(("good", "bad"))
    template = str(DOMAINS[domain][chosen])
    values = DOMAINS[domain]["values"]
    assert isinstance(values, tuple)
    return rng.choice(HEDGES).format(template.format(value=rng.choice(values)))
