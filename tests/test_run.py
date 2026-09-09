"""The runner must produce trials that pair up across conditions.

Every question appears exactly twice per model and task, once biased and once
not, with the same target. If that pairing ever breaks, the paired statistics
downstream are silently wrong.
"""

from pathlib import Path

from findings.cot_unfaithfulness.bbh import Example
from findings.cot_unfaithfulness.model import Cache
from findings.cot_unfaithfulness.run import manifest, run_one

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
