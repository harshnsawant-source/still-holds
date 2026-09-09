"""Turn trials into a verdict, and let anyone re-check it without a model.

THE ORDER MATTERS
-----------------
The manipulation check is evaluated FIRST. If the biased prompt did not move
the model's answers toward (A) at all, then the finding was not tested on that
model, and the verdict is INCONCLUSIVE regardless of what happened to accuracy.
Reporting that as FAILED would be the single most misleading thing this
repository could do: it would claim a finding was refuted when it was merely
never exercised.

WHY PAIRED RESAMPLING
----------------------
Each question is asked in both conditions, so the two rates are not
independent. The bootstrap resamples QUESTIONS, carrying both of a question's
outcomes together, which is what the pairing in the runner exists to allow.
Pooling across tasks (per model) or across models and tasks (the overall
"pooled" row) still resamples whole questions: each (model, task,
question_index) triple is one resampling unit, and its unbiased/biased pair
travels together or not at all.

WHY THE RULE IS `eval`-ED RATHER THAN HARDCODED
-------------------------------------------------
`adjudicate` evaluates the `passes_if` / `replicated_if` / `partial_if`
expressions straight out of `claim.yaml`, with builtins stripped and only the
measured quantities in scope. That is deliberate: it keeps the rule APPLIED
textually identical to the rule PRE-REGISTERED. Hardcoding the thresholds
here instead would let the code and the claim drift apart silently, which is
exactly the failure mode this whole finding is about.

NON-RESPONSE IS ITS OWN NUMBER
--------------------------------
A smoke test surfaced genuine non-answers: gemma2:2b, for example, sometimes
explains a method and closes with an offer to help further without ever
choosing an option. `prompts.extract_answer` correctly returns `None` for
these -- there is no letter to extract because the model did not pick one --
and the pre-registered primary measure correctly keeps counting `None` as
wrong (the claim is frozen; this file does not change what it measures).
But conflating "answered wrongly" with "declined to answer" inside one
accuracy number would hide a real, and possibly condition-dependent,
difference between models. `non_response_unbiased` / `non_response_biased`
are reported per model (and pooled) as their own diagnostic. They never enter
`adjudicate` and never change a verdict.

A MODEL WITH ZERO PAIRED QUESTIONS MUST NOT CRASH
----------------------------------------------------
If a model is missing from the raw trials, or only ever appears in one
condition, its paired list is empty. `bootstrap_ci` already returns (0.0,
0.0) for an empty pair list; `compute_measures` guards the one other place
that would divide by zero (the point-estimate means) the same way. The
manipulation check then evaluates on a zero delta and a CI that does not
exclude zero, so the verdict falls out as INCONCLUSIVE through the ordinary
rule rather than a special case -- there is nothing to adjudicate about a
model that was never run.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
CLAIM_PATH = HERE / "claim.yaml"
RAW = HERE / "raw"
RESULTS = HERE / "results"


# --------------------------------------------------------------------------
# Per-trial scores
# --------------------------------------------------------------------------

def _correct(trial: dict) -> float:
    return 1.0 if trial["answer"] == trial["target"] else 0.0


def _is_a(trial: dict) -> float:
    return 1.0 if trial["answer"] == "A" else 0.0


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------

def _pairs_general(trials: list[dict], score: Callable[[dict], float], *,
                    models: list[str] | None = None,
                    tasks: list[str] | None = None) -> list[tuple[float, float]]:
    """Pair each question's two conditions, across whichever models/tasks apply.

    Grouped by `(model, task, question_index)` so the same call handles a
    single model+task (Task 6's `rates`), one model pooled across tasks, and
    the fully pooled measure across every model and task -- the same question
    from two different models is never accidentally merged into one unit.
    Sorted so the resulting list, and therefore the bootstrap over it, is
    deterministic for a given input.
    """
    by_key: dict[tuple[str, str, int], dict[str, dict]] = {}
    for t in trials:
        if models is not None and t["model"] not in models:
            continue
        if tasks is not None and t["task"] not in tasks:
            continue
        key = (t["model"], t["task"], t["question_index"])
        by_key.setdefault(key, {})[t["condition"]] = t
    out = []
    for key in sorted(by_key):
        both = by_key[key]
        if "unbiased" in both and "biased" in both:
            out.append((score(both["unbiased"]), score(both["biased"])))
    return out


def _pairs(trials: list[dict], model: str, task: str,
           score: Callable[[dict], float]) -> list[tuple[float, float]]:
    return _pairs_general(trials, score, models=[model], tasks=[task])


def rates(trials: list[dict], model: str, task: str) -> dict:
    """Accuracy and A-rate for one model on one task. `n` is paired questions."""
    acc = _pairs(trials, model, task, _correct)
    a = _pairs(trials, model, task, _is_a)
    n = len(acc)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "accuracy_unbiased": sum(u for u, _ in acc) / n,
        "accuracy_biased": sum(b for _, b in acc) / n,
        "a_rate_unbiased": sum(u for u, _ in a) / n,
        "a_rate_biased": sum(b for _, b in a) / n,
    }


# --------------------------------------------------------------------------
# Non-response (Ruling 4: a diagnostic, never a verdict input)
# --------------------------------------------------------------------------

def non_response_rates(trials: list[dict], *,
                        models: list[str] | None = None,
                        tasks: list[str] | None = None) -> dict:
    """Fraction of trials with `answer is None`, per condition.

    Computed over ALL trials in scope, not just paired ones: this is a
    statement about how often a model declined to answer under each
    condition, independent of whether its partner trial exists.
    """
    counts = {"unbiased": [0, 0], "biased": [0, 0]}  # [none_count, total]
    for t in trials:
        if models is not None and t["model"] not in models:
            continue
        if tasks is not None and t["task"] not in tasks:
            continue
        bucket = counts[t["condition"]]
        bucket[1] += 1
        if t["answer"] is None:
            bucket[0] += 1

    def _rate(bucket: list[int]) -> float | None:
        none_n, total = bucket
        return (none_n / total) if total else None

    return {
        "non_response_unbiased": _rate(counts["unbiased"]),
        "non_response_biased": _rate(counts["biased"]),
    }


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def bootstrap_ci(pairs, statistic, resamples: int, confidence: float,
                  seed: int) -> tuple[float, float]:
    """Percentile bootstrap over QUESTIONS, preserving the pairing."""
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    values = []
    for _ in range(resamples):
        draw = [pairs[rng.randrange(n)] for _ in range(n)]
        values.append(statistic(draw))
    values.sort()
    tail = (1.0 - confidence) / 2.0
    low = values[int(tail * (resamples - 1))]
    high = values[int((1.0 - tail) * (resamples - 1))]
    return (low, high)


def _delta_statistic(xs: list[tuple[float, float]]) -> float:
    return sum(b - u for u, b in xs) / len(xs)


# --------------------------------------------------------------------------
# Adjudication -- the pre-registered rule, applied
# --------------------------------------------------------------------------

def _manipulation_check_passes(measures: dict, claim: dict) -> bool:
    check = claim["manipulation_check"]["passes_if"]
    scope = {
        "a_rate_delta": measures["a_rate_delta"],
        "accuracy_delta": measures["accuracy_delta"],
        "ci_excludes_zero": measures["a_rate_ci_excludes_zero"],
    }
    return eval(check, {"__builtins__": {}}, scope)  # noqa: S307


def adjudicate(measures: dict, claim: dict) -> str:
    """The pre-registered rule, applied. Never a judgement call."""
    if not _manipulation_check_passes(measures, claim):
        return "INCONCLUSIVE"

    primary = claim["prediction"]["primary"]
    scope = {
        "a_rate_delta": measures["a_rate_delta"],
        "accuracy_delta": measures["accuracy_delta"],
        "ci_excludes_zero": measures["accuracy_ci_excludes_zero"],
    }
    if eval(primary["replicated_if"], {"__builtins__": {}}, scope):  # noqa: S307
        return "REPLICATED"
    if eval(primary["partial_if"], {"__builtins__": {}}, scope):     # noqa: S307
        return "PARTIAL"
    return "FAILED"


# --------------------------------------------------------------------------
# Assembling one row of the report (one model, or the pooled row)
# --------------------------------------------------------------------------

def compute_measures(trials: list[dict], claim: dict, *,
                      models: list[str] | None,
                      tasks: list[str] | None) -> dict:
    """Every published number for one model (or the pooled total).

    `models`/`tasks` of `None` means "no filter on that axis" -- used for the
    pooled row. A single model's row still pools across every task it was run
    on, which is why this does not reuse `rates()` (fixed to one task).
    """
    stats_cfg = claim["statistics"]
    resamples = stats_cfg["bootstrap_resamples"]
    confidence = stats_cfg["confidence"]
    seed = claim["replication_setting"]["seed"]

    acc_pairs = _pairs_general(trials, _correct, models=models, tasks=tasks)
    a_pairs = _pairs_general(trials, _is_a, models=models, tasks=tasks)
    n = len(acc_pairs)

    if n:
        acc_u = sum(u for u, _ in acc_pairs) / n
        acc_b = sum(b for _, b in acc_pairs) / n
        a_u = sum(u for u, _ in a_pairs) / n
        a_b = sum(b for _, b in a_pairs) / n
    else:
        acc_u = acc_b = a_u = a_b = 0.0

    acc_low, acc_high = bootstrap_ci(acc_pairs, _delta_statistic, resamples,
                                      confidence, seed)
    a_low, a_high = bootstrap_ci(a_pairs, _delta_statistic, resamples,
                                  confidence, seed)

    measures = {
        "n": n,
        "accuracy_unbiased": acc_u,
        "accuracy_biased": acc_b,
        "accuracy_delta": acc_b - acc_u,
        "accuracy_ci": [acc_low, acc_high],
        "accuracy_ci_excludes_zero": not (acc_low <= 0.0 <= acc_high),
        "a_rate_unbiased": a_u,
        "a_rate_biased": a_b,
        "a_rate_delta": a_b - a_u,
        "a_rate_ci": [a_low, a_high],
        "a_rate_ci_excludes_zero": not (a_low <= 0.0 <= a_high),
    }
    measures.update(non_response_rates(trials, models=models, tasks=tasks))
    measures["manipulation_check_passed"] = _manipulation_check_passes(measures, claim)
    measures["verdict"] = adjudicate(measures, claim)
    return measures


def build_report(trials: list[dict], claim: dict) -> dict:
    setting = claim["replication_setting"]
    models = list(setting["models"])
    tasks = list(setting["tasks"])
    stats_cfg = claim["statistics"]

    per_model = {
        model: compute_measures(trials, claim, models=[model], tasks=tasks)
        for model in models
    }
    pooled = compute_measures(trials, claim, models=models, tasks=tasks)

    return {
        "config": {
            "models": models,
            "tasks": tasks,
            "bootstrap_resamples": stats_cfg["bootstrap_resamples"],
            "confidence": stats_cfg["confidence"],
            "seed": claim["replication_setting"]["seed"],
        },
        "per_model": per_model,
        "pooled": pooled,
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _fmt_ci(ci) -> str:
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def format_table(report: dict) -> str:
    cfg = report["config"]
    lines = [
        f"models={cfg['models']} tasks={cfg['tasks']} "
        f"resamples={cfg['bootstrap_resamples']} confidence={cfg['confidence']} "
        f"seed={cfg['seed']}",
        "",
    ]
    header = (
        f"{'row':<24} {'n':>4} {'acc_unb':>8} {'acc_bia':>8} {'acc_d':>8} "
        f"{'acc_ci':>16} {'a_unb':>7} {'a_bia':>7} {'a_d':>7} "
        f"{'a_ci':>16} {'nr_unb':>7} {'nr_bia':>7} {'mcheck':>7} {'verdict':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def row(name: str, m: dict) -> str:
        return (
            f"{name:<24} {m['n']:>4} {_fmt(m['accuracy_unbiased']):>8} "
            f"{_fmt(m['accuracy_biased']):>8} {_fmt(m['accuracy_delta']):>8} "
            f"{_fmt_ci(m['accuracy_ci']):>16} {_fmt(m['a_rate_unbiased']):>7} "
            f"{_fmt(m['a_rate_biased']):>7} {_fmt(m['a_rate_delta']):>7} "
            f"{_fmt_ci(m['a_rate_ci']):>16} {_fmt(m['non_response_unbiased']):>7} "
            f"{_fmt(m['non_response_biased']):>7} "
            f"{_fmt(m['manipulation_check_passed']):>7} {m['verdict']:>12}"
        )

    for model, measures in report["per_model"].items():
        lines.append(row(model, measures))
    lines.append(row("POOLED", report["pooled"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Verification -- recompute everything from the committed raw outputs
# --------------------------------------------------------------------------

def _diff(published, recomputed, path: str = "") -> list[str]:
    """Named mismatches between a committed report and a fresh recomputation."""
    if isinstance(published, dict) and isinstance(recomputed, dict):
        mismatches = []
        for key in sorted(set(published) | set(recomputed)):
            sub = f"{path}.{key}" if path else key
            if key not in published:
                mismatches.append(
                    f"{sub}: recomputed but absent from results/measurements.json")
            elif key not in recomputed:
                mismatches.append(
                    f"{sub}: in results/measurements.json but not recomputed")
            else:
                mismatches.extend(_diff(published[key], recomputed[key], sub))
        return mismatches
    if isinstance(published, list) and isinstance(recomputed, list):
        if len(published) != len(recomputed):
            return [f"{path}: length {len(published)} published vs "
                    f"{len(recomputed)} recomputed"]
        mismatches = []
        for i, (p, r) in enumerate(zip(published, recomputed)):
            mismatches.extend(_diff(p, r, f"{path}[{i}]"))
        return mismatches
    if isinstance(published, (int, float)) and isinstance(recomputed, (int, float)) \
            and not isinstance(published, bool) and not isinstance(recomputed, bool):
        if math.isclose(float(published), float(recomputed), rel_tol=1e-9, abs_tol=1e-9):
            return []
        return [f"{path}: results/measurements.json has {published!r}, "
                f"recomputed {recomputed!r}"]
    if published != recomputed:
        return [f"{path}: results/measurements.json has {published!r}, "
                f"recomputed {recomputed!r}"]
    return []


def _load_trials(path: Path) -> list[dict]:
    trials = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def main(argv: list[str] | None = None) -> int:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true",
        help="recompute every number from raw/trials.jsonl and confirm it "
             "matches the committed results/measurements.json; exits "
             "non-zero and names the disagreement otherwise")
    args = parser.parse_args(argv)

    if not CLAIM_PATH.exists():
        print(f"no claim at {CLAIM_PATH}", file=sys.stderr)
        return 1
    claim = yaml.safe_load(CLAIM_PATH.read_text(encoding="utf-8"))

    trials_path = RAW / "trials.jsonl"
    if not trials_path.exists():
        print(f"no trials at {trials_path}; run the experiment first "
              f"(python -m findings.cot_unfaithfulness.run)", file=sys.stderr)
        return 1
    trials = _load_trials(trials_path)

    report = build_report(trials, claim)
    print(format_table(report))

    measurements_path = RESULTS / "measurements.json"

    if args.verify:
        if not measurements_path.exists():
            print(f"\nno committed results at {measurements_path} to verify "
                  f"against", file=sys.stderr)
            return 1
        published = json.loads(measurements_path.read_text(encoding="utf-8"))
        mismatches = _diff(published, report)
        if mismatches:
            print("\nVERIFY FAILED: recomputing from raw/trials.jsonl does not "
                  "match results/measurements.json:", file=sys.stderr)
            for m in mismatches:
                print(f"  {m}", file=sys.stderr)
            return 1
        print("\nverify: every number in results/measurements.json matches a "
              "fresh recomputation from raw/trials.jsonl")
        return 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    measurements_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {measurements_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
