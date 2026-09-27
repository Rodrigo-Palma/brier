"""A labelled example and how it travels to disk."""

import json
from dataclasses import dataclass
from pathlib import Path

from brier.types import Question, QuestionKind


@dataclass(frozen=True, slots=True)
class Example:
    """One decision whose outcome we know.

    Attributes:
        state: the text the decision is about.
        question: what is being asked.
        correct: index into `question.options` of the option that was right.
        tag: optional subtype, so results can be broken down below the family.
            A family average is an average, and one of its subtypes can sit at
            zero while the family looks merely weak.
    """

    state: str
    question: Question
    correct: int
    tag: str = ""

    def __post_init__(self) -> None:
        if not self.state.strip():
            raise ValueError("an example needs a state")
        if not 0 <= self.correct < len(self.question.options):
            raise ValueError(
                f"correct index {self.correct} is outside the "
                f"{len(self.question.options)} options of {self.question.name!r}"
            )


def identity(example: Example) -> tuple[str, str, tuple[str, ...], int]:
    """What makes two examples the same question with the same answer.

    Splits are checked against each other with this. Generated data can repeat
    itself, and a validation row that is byte-identical to a training row is a
    training row: it measures memorisation and it silently ruins any early
    stopping or temperature fitted on it.
    """
    return (
        example.state,
        example.question.prompt,
        example.question.options,
        example.correct,
    )


def to_jsonl(examples: tuple[Example, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(_as_dict(example), ensure_ascii=False) for example in examples)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def from_jsonl(path: Path) -> tuple[Example, ...]:
    """Read examples written by `to_jsonl`.

    Raises:
        FileNotFoundError: when the file is not there.
        ValueError: when a line is not a valid example.
    """
    if not path.exists():
        raise FileNotFoundError(f"no examples at {path}")

    examples = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(_from_dict(json.loads(line)))
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{number} is not a valid example: {error}") from error
    return tuple(examples)


def _as_dict(example: Example) -> dict:
    return {
        "state": example.state,
        "name": example.question.name,
        "kind": example.question.kind.value,
        "prompt": example.question.prompt,
        "options": list(example.question.options),
        "correct": example.correct,
        "tag": example.tag,
    }


def _from_dict(raw: dict) -> Example:
    question = Question(
        name=raw["name"],
        kind=QuestionKind(raw["kind"]),
        prompt=raw["prompt"],
        options=tuple(raw["options"]),
    )
    return Example(
        state=raw["state"],
        question=question,
        correct=int(raw["correct"]),
        tag=str(raw.get("tag", "")),
    )
