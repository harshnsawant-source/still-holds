"""The corpus is fixed, parsed strictly, and sampled the same way every time.

A replication that draws different questions on a re-run is not reproducible,
and a parser that silently drops a malformed example changes the denominator
without telling anyone.
"""

import json
from pathlib import Path

import pytest

from findings.cot_unfaithfulness.bbh import BBHError, Example, load, sample

RAW = {
    "canary": "ignore me",
    "examples": [
        {
            "input": (
                "A paragraph about birds.\n"
                "Options:\n"
                "(A) The blue jay is second\n"
                "(B) The quail is second\n"
                "(C) The falcon is second"
            ),
            "target": "(A)",
        },
        {
            "input": (
                "Another paragraph.\n"
                "Options:\n"
                "(A) First\n"
                "(B) Second\n"
                "(C) Third"
            ),
            "target": "(C)",
        },
    ],
}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "bbh"
    d.mkdir()
    (d / "demo_task.json").write_text(json.dumps(RAW), encoding="utf-8")
    return d


def test_load_splits_question_from_options(data_dir: Path):
    got = load("demo_task", data_dir)
    assert len(got) == 2
    assert got[0].question == "A paragraph about birds."
    assert got[0].options == {
        "A": "The blue jay is second",
        "B": "The quail is second",
        "C": "The falcon is second",
    }
    assert got[0].target == "A"


def test_load_normalises_the_target_to_a_bare_letter(data_dir: Path):
    assert load("demo_task", data_dir)[1].target == "C"


def test_load_reports_a_missing_task_rather_than_returning_empty(data_dir: Path):
    with pytest.raises(BBHError, match="nosuch"):
        load("nosuch", data_dir)


def test_load_refuses_an_example_with_no_options_block(tmp_path: Path):
    d = tmp_path / "bbh"
    d.mkdir()
    bad = {"examples": [{"input": "No options here.", "target": "(A)"}]}
    (d / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(BBHError, match="Options"):
        load("bad", d)


def test_load_refuses_a_target_that_names_no_option(tmp_path: Path):
    d = tmp_path / "bbh"
    d.mkdir()
    bad = {"examples": [{
        "input": "Q\nOptions:\n(A) one\n(B) two",
        "target": "(Z)",
    }]}
    (d / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(BBHError, match="Z"):
        load("bad", d)


def test_load_refuses_an_options_block_with_no_parseable_lines(tmp_path: Path):
    """The 'contained no options' branch, reached directly: an 'Options:'
    block is present but every line inside it fails to parse. This must be
    diagnosed as an empty options block, not as a bad target - the target
    check must never be the thing that catches this."""
    d = tmp_path / "bbh"
    d.mkdir()
    bad = {"examples": [{
        "input": "Q\nOptions:\n(a) lowercase only\nnot an option either",
        "target": "(A)",
    }]}
    (d / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(BBHError, match="contained no options"):
        load("bad", d)


def test_load_refuses_a_malformed_option_line_that_is_not_the_target(
    tmp_path: Path,
):
    """A malformed line sitting between two valid options, where the target
    names one of the surviving valid options. This is the dangerous case:
    before the fix, the malformed line was silently dropped and the example
    loaded anyway with fewer options than the source document has."""
    d = tmp_path / "bbh"
    d.mkdir()
    bad = {"examples": [{
        "input": "Q\nOptions:\n(A) one\n(a) sneaky lowercase\n(B) two",
        "target": "(B)",
    }]}
    (d / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(BBHError, match="sneaky lowercase"):
        load("bad", d)


def test_load_refuses_a_malformed_option_line_that_would_be_the_target(
    tmp_path: Path,
):
    """A malformed line whose (normalised) letter is what the target names.
    Before the fix this already raised, via the target-not-found check, with
    a vaguer message. After the fix it still raises, but the malformed line
    itself is caught first and named."""
    d = tmp_path / "bbh"
    d.mkdir()
    bad = {"examples": [{
        "input": "Q\nOptions:\n(A) one\n(a) sneaky lowercase\n(B) two",
        "target": "(a)",
    }]}
    (d / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(BBHError, match="sneaky lowercase"):
        load("bad", d)


def test_load_does_not_raise_on_trailing_blank_lines_after_the_last_option(
    tmp_path: Path,
):
    d = tmp_path / "bbh"
    d.mkdir()
    good = {"examples": [{
        "input": "Q\nOptions:\n(A) one\n(B) two\n\n\n",
        "target": "(A)",
    }]}
    (d / "good.json").write_text(json.dumps(good), encoding="utf-8")
    got = load("good", d)
    assert got[0].options == {"A": "one", "B": "two"}
    assert got[0].target == "A"


def test_load_refuses_a_top_level_json_list(tmp_path: Path):
    d = tmp_path / "bbh"
    d.mkdir()
    (d / "bad.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(BBHError, match="list"):
        load("bad", d)


def test_load_refuses_a_top_level_json_string(tmp_path: Path):
    d = tmp_path / "bbh"
    d.mkdir()
    (d / "bad.json").write_text(json.dumps("just a string"), encoding="utf-8")
    with pytest.raises(BBHError, match="str"):
        load("bad", d)


def test_sample_is_deterministic_for_a_seed(data_dir: Path):
    pool = load("demo_task", data_dir) * 40
    a = sample(pool, 10, seed=7)
    b = sample(pool, 10, seed=7)
    assert [x.question for x in a] == [x.question for x in b]


def test_sample_changes_with_the_seed(data_dir: Path):
    pool = [Example(f"q{i}", {"A": "a", "B": "b"}, "A") for i in range(100)]
    a = sample(pool, 10, seed=1)
    b = sample(pool, 10, seed=2)
    assert [x.question for x in a] != [x.question for x in b]


def test_sample_refuses_to_ask_for_more_than_exists():
    pool = [Example("q", {"A": "a"}, "A")]
    with pytest.raises(BBHError, match="only 1"):
        sample(pool, 5, seed=1)


# --------------------------------------------------------------------------
# M4: the four committed task files this replication actually uses have
# been checked by hand twice ("250 examples, zero errors") and never pinned
# in a test. claim.yaml's replication_setting.tasks names these four; a
# fifth or a renamed file would silently not be covered here, which is the
# point -- this is a property of the CURRENT committed corpus, not a
# generic loader test (test_load_* above already cover the parser itself).
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
BBH_DATA_DIR = REPO_ROOT / "data" / "bbh"
COMMITTED_TASKS = [
    "date_understanding",
    "disambiguation_qa",
    "logical_deduction_three_objects",
    "tracking_shuffled_objects_three_objects",
]


@pytest.mark.parametrize("task", COMMITTED_TASKS)
def test_committed_bbh_task_file_has_250_examples_and_parses_cleanly(task):
    examples = load(task, BBH_DATA_DIR)
    assert len(examples) == 250
    for example in examples:
        assert example.question
        assert example.options
        assert example.target in example.options
