"""The runner must produce trials that pair up across conditions.

Every question appears exactly twice per model and task, once biased and once
not, with the same target. If that pairing ever breaks, the paired statistics
downstream are silently wrong.
"""

from pathlib import Path

import json

import pytest

from findings.cot_unfaithfulness.bbh import Example
from findings.cot_unfaithfulness.model import Cache
from findings.cot_unfaithfulness.run import (
    _atomic_write,
    main,
    manifest,
    run_one,
    split_trial_fields,
)

EXAMPLES = [
    Example(f"q{i}", {"A": "alpha", "B": "beta", "C": "gamma"}, "B")
    for i in range(4)
]
DEMOS = [
    Example(f"d{i}", {"A": "one", "B": "two", "C": "three"}, "C")
    for i in range(3)
]


def _ask(answer_letter: str):
    def ask(prompt: str, model: str) -> str:
        return f"Some reasoning here. The best answer is: ({answer_letter})"
    return ask


def test_each_question_yields_one_trial_per_condition(tmp_path: Path):
    trials = run_one("m", "t", EXAMPLES, DEMOS, seed=1,
                     cache=Cache(tmp_path), ask=_ask("B"))
    assert len(trials) == len(EXAMPLES) * 2
    for i in range(len(EXAMPLES)):
        pair = [t for t in trials if t.question_index == i]
        assert {t.condition for t in pair} == {"biased", "unbiased"}
        assert len({t.target for t in pair}) == 1


def test_the_answer_is_extracted_into_the_trial(tmp_path: Path):
    trials = run_one("m", "t", EXAMPLES, DEMOS, seed=1,
                     cache=Cache(tmp_path), ask=_ask("C"))
    assert all(t.answer == "C" for t in trials)


def test_an_unparsable_response_records_none_rather_than_crashing(tmp_path: Path):
    def mush(prompt: str, model: str) -> str:
        return "I am not going to answer that."

    trials = run_one("m", "t", EXAMPLES, DEMOS, seed=1,
                     cache=Cache(tmp_path), ask=mush)
    assert all(t.answer is None for t in trials)


def test_the_manifest_records_what_a_reader_needs_to_reproduce_it():
    m = manifest(["m1"], ["t1"], {"seed": 7, "few_shot_k": 6})
    for key in ("models", "tasks", "config", "python", "platform", "created"):
        assert key in m, key
    assert m["config"]["seed"] == 7


# --------------------------------------------------------------------------
# I6: the full response text is committed, not just the extracted letter.
# --------------------------------------------------------------------------

def test_the_trial_carries_the_full_response_text(tmp_path: Path):
    trials = run_one("m", "t", EXAMPLES, DEMOS, seed=1,
                     cache=Cache(tmp_path), ask=_ask("B"))
    assert all(t.response for t in trials)
    assert all("The best answer is: (B)" in t.response for t in trials)


def test_split_trial_fields_keeps_trials_jsonl_free_of_response_text(
    tmp_path: Path,
):
    trial = run_one("m", "t", EXAMPLES[:1], DEMOS, seed=1,
                    cache=Cache(tmp_path), ask=_ask("B"))[0]
    trial_row, response_row = split_trial_fields(trial)

    assert "response" not in trial_row
    assert set(trial_row) == {"model", "task", "question_index", "condition",
                               "target", "answer", "mentions_position",
                               "response_chars"}

    assert response_row["response"] == trial.response
    assert (response_row["model"], response_row["task"],
            response_row["question_index"], response_row["condition"]) == (
        trial_row["model"], trial_row["task"],
        trial_row["question_index"], trial_row["condition"])


# --------------------------------------------------------------------------
# M1: manifest carries model digests, the exact command, and library
# versions, and I10's limited flag.
# --------------------------------------------------------------------------

def test_manifest_carries_model_digests_command_and_package_versions():
    m = manifest(
        ["m1"], ["t1"], {"seed": 7, "few_shot_k": 6},
        model_digests={"m1": "sha256:abc123"},
        command="python -m findings.cot_unfaithfulness.run",
        package_versions={"pyyaml": "6.0.3"},
    )
    assert m["model_digests"] == {"m1": "sha256:abc123"}
    assert m["command"] == "python -m findings.cot_unfaithfulness.run"
    assert m["package_versions"] == {"pyyaml": "6.0.3"}


def test_manifest_defaults_are_present_even_when_not_supplied():
    m = manifest(["m1"], ["t1"], {"seed": 7, "few_shot_k": 6})
    assert m["model_digests"] == {}
    assert m["package_versions"] == {}
    assert m["limited"] is False
    assert isinstance(m["command"], str) and m["command"]


def test_manifest_records_a_limited_run():
    m = manifest(["m1"], ["t1"], {"seed": 7, "few_shot_k": 6}, limited=True)
    assert m["limited"] is True


# --------------------------------------------------------------------------
# I10: --limit 0 must be rejected, not silently treated as "no limit".
# --------------------------------------------------------------------------

def test_limit_zero_is_rejected_not_silently_treated_as_unlimited(capsys):
    exit_code = main(["--limit", "0"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "--limit" in captured.err


def test_negative_limit_is_also_rejected(capsys):
    exit_code = main(["--limit", "-3"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "--limit" in captured.err


# --------------------------------------------------------------------------
# Atomic writes for raw/ (mirrors model.Cache's atomicity tests).
# --------------------------------------------------------------------------

def test_atomic_write_round_trips(tmp_path: Path):
    p = tmp_path / "out.txt"
    _atomic_write(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"


def test_atomic_write_interrupted_before_replace_leaves_no_partial_file(
    tmp_path: Path, monkeypatch
):
    from findings.cot_unfaithfulness import run as run_module

    p = tmp_path / "out.txt"

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(run_module.os, "replace", boom)
    with pytest.raises(OSError):
        _atomic_write(p, "a value that would otherwise land on disk")
    assert not p.exists()
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# I5: the manipulation is on the PROMPT, not just the trial's condition
# label. A run_one that built the same (biased) prompt for both conditions
# would still produce trials labelled "biased" and "unbiased" -- passing the
# existing label-only assertions above -- while testing nothing.
# --------------------------------------------------------------------------

def test_the_biased_and_unbiased_prompts_are_actually_different(tmp_path: Path):
    seen: dict[tuple[int, str], str] = {}

    def recording_ask(prompt: str, model: str) -> str:
        seen[len(seen)] = prompt
        return "The best answer is: (B)"

    run_one("m", "t", EXAMPLES, DEMOS, seed=1, cache=Cache(tmp_path),
            ask=recording_ask)

    prompts = list(seen.values())
    # Two prompts per question (unbiased then biased, per run_one's own
    # loop order), for every one of the 4 questions.
    assert len(prompts) == len(EXAMPLES) * 2
    for i in range(len(EXAMPLES)):
        unbiased_prompt, biased_prompt = prompts[2 * i], prompts[2 * i + 1]
        assert unbiased_prompt != biased_prompt, (
            f"question {i}: biased and unbiased prompts must differ -- "
            f"otherwise the manipulation was never applied"
        )
