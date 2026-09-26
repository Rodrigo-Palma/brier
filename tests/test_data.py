import pytest

from brier.data import Example, from_jsonl, to_jsonl
from brier.types import boolean, score


def test_examples_survive_a_round_trip(tmp_path):
    examples = (
        Example("revenue grew", boolean("relevance", "About revenue?"), 1),
        Example("nothing happened", score("tone", "How does it read?", ("bad", "ok", "good")), 1),
    )
    path = tmp_path / "split" / "train.jsonl"

    to_jsonl(examples, path)

    assert from_jsonl(path) == examples


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "train.jsonl"
    to_jsonl((Example("text", boolean("relevance", "About it?"), 0),), path)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    assert len(from_jsonl(path)) == 1


def test_rejects_a_correct_index_outside_the_options():
    with pytest.raises(ValueError, match="outside the 2 options"):
        Example("text", boolean("relevance", "About it?"), 2)


def test_rejects_a_blank_state():
    with pytest.raises(ValueError, match="needs a state"):
        Example("   ", boolean("relevance", "About it?"), 0)


def test_a_bad_line_names_the_line_number(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"state": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"train.jsonl:1"):
        from_jsonl(path)


def test_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no examples at"):
        from_jsonl(tmp_path / "absent.jsonl")
