"""The question types the model answers, and the shape of an answer.

Every question is a choice between options that are described in words. A
boolean is a choice between two, a score is a choice between the levels of a
rubric, and a plain choice is a choice between whatever the caller listed. One
scorer serves all three, so a new question type costs no new model.
"""

from dataclasses import dataclass
from enum import StrEnum

MAX_OPTIONS = 64


class QuestionKind(StrEnum):
    CHOICE = "choice"
    SCORE = "score"
    BOOL = "bool"


@dataclass(frozen=True, slots=True)
class Question:
    """A question asked against a shared state.

    Attributes:
        name: how the answer is keyed in the response.
        kind: what the caller expects back.
        prompt: the question in words, which the model reads.
        options: the answers it is allowed to give, in order.
    """

    name: str
    kind: QuestionKind
    prompt: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a question needs a name")
        if not self.prompt.strip():
            raise ValueError(f"{self.name}: a question needs a prompt to read")
        if len(self.options) < 2:
            raise ValueError(f"{self.name}: a question needs at least two options")
        if len(self.options) > MAX_OPTIONS:
            raise ValueError(
                f"{self.name}: {len(self.options)} options is over the {MAX_OPTIONS} cap"
            )
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"{self.name}: the options repeat")


def boolean(name: str, prompt: str) -> Question:
    """A yes or no question. ``True`` is the second option, so index equals truth."""
    return Question(name=name, kind=QuestionKind.BOOL, prompt=prompt, options=("no", "yes"))


def score(name: str, prompt: str, levels: tuple[str, ...]) -> Question:
    """A rubric, from the lowest level to the highest."""
    return Question(name=name, kind=QuestionKind.SCORE, prompt=prompt, options=levels)


def choice(name: str, prompt: str, options: tuple[str, ...]) -> Question:
    return Question(name=name, kind=QuestionKind.CHOICE, prompt=prompt, options=options)


@dataclass(frozen=True, slots=True)
class Answer:
    """What the model returns for one question.

    ``probabilities`` lines up with the question's options. ``confidence`` is the
    probability of the option that was picked, which is the number calibration
    is measured against.
    """

    name: str
    kind: QuestionKind
    value: str
    index: int
    probabilities: tuple[float, ...]
    confidence: float
    abstained: bool = False

    @property
    def as_bool(self) -> bool:
        """Raises:
        TypeError: when the question was not a boolean.
        Abstained: when the model declined to answer.
        """
        self._require(QuestionKind.BOOL)
        return self.index == 1

    @property
    def as_level(self) -> int:
        """The rubric level, counting from 1 at the lowest.

        Raises:
            TypeError: when the question was not a score.
            Abstained: when the model declined to answer.
        """
        self._require(QuestionKind.SCORE)
        return self.index + 1

    def _require(self, kind: QuestionKind) -> None:
        """An abstention has to be impossible to read as an answer.

        The picked option is still carried, for logging and for a caller that
        wants to see what the model was leaning towards, but every accessor
        that hands back a usable value refuses while `abstained` is set.
        """
        if self.kind is not kind:
            raise TypeError(f"{self.name} is a {self.kind.value}, not a {kind.value}")
        if self.abstained:
            raise Abstained(
                f"{self.name}: confidence {self.confidence:.3f} was below the threshold"
            )


class Abstained(RuntimeError):
    """Raised when a caller reads a value the model declined to commit to."""
