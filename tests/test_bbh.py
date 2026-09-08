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
