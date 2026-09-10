"""Run the experiment and write down exactly what happened.

WHAT COMES OUT
--------------
`raw/trials.jsonl`, one line per generation, and `raw/manifest.json` recording
the models, tasks, configuration, Python version, platform and date. The
responses themselves live in the cache, keyed by content, so the analysis can
be re-run from disk with no model present.

PAIRING IS THE INVARIANT
------------------------
Every question is asked twice, once with biased demonstrations and once
without, with everything else identical. The downstream statistics are
paired, so a break in that pairing would quietly invalidate them. `run_one`
enforces this by construction: it walks each question once and emits both
conditions for it back to back, rather than looping over conditions on the
outside where a mismatched count or order could slip through unnoticed.

WHAT THIS REFUSES TO DO
------------------------
Drop a trial it cannot parse. A response the extractor cannot read back is
still a trial, recorded with `answer: None` - deciding whether that counts as
wrong is the analysis's job, not the runner's, and dropping it here would
silently break the pairing every downstream statistic depends on.

Leave a half-written `trials.jsonl`, `responses.jsonl` or `manifest.json`
behind. All trials are held in memory and the three files are only written
once every trial has been produced, each one via a temp-file-then-replace
so a process killed mid-write leaves the previous (complete, or absent)
file rather than a truncated one -- the same guarantee `model.Cache.put`
gives a single cached response, applied here to the files a reader trusts
as "the raw data". A `ModelError` partway through the run stops it before
any of the three is written, so a partial run can never be mistaken for a
complete one. The response cache means resuming is cheap, so there is no
reason to trade that guarantee for partial progress.

WHY THE FULL RESPONSE TEXT IS COMMITTED (I6)
----------------------------------------------
`Trial` used to record only `answer`, `mentions_position` and
`response_chars` -- an extraction result with no way to check the
extraction. claim.yaml's own withdrawal note claims "anyone can compute
this later and argue with the detector"; that was false without the text
itself. C1 and C2 showed exactly why this matters: the model's full
response is written to `raw/responses.jsonl`, keyed by (model, task,
question_index, condition) to match `trials.jsonl` line for line, so a
reader can re-run `prompts.extract_answer` against what the model actually
said and check it, rather than trusting a bare letter on faith.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from findings.cot_unfaithfulness import bbh, prompts
from findings.cot_unfaithfulness.model import (
    Cache,
    ModelError,
    available_models,
    generate,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "bbh"
RAW = HERE / "raw"
CACHE = REPO / ".cache" / "responses"


@dataclass(frozen=True)
class Trial:
    model: str
    task: str
    question_index: int
    condition: str            # "biased" | "unbiased"
    target: str
    answer: str | None
    mentions_position: bool
    response_chars: int
    response: str = ""        # kept off trials.jsonl; see split_trial_fields


def split_trial_fields(trial: Trial) -> tuple[dict, dict]:
    """One Trial -> (the row for trials.jsonl, the row for responses.jsonl).

    Keeps trials.jsonl exactly as small and readable as it was before I6:
    the full response text -- which can run to thousands of characters, see
    tests/fixtures/smoke_responses.json -- goes to its own file, keyed the
    same way so the two line up.
    """
    d = asdict(trial)
    response = d.pop("response")
    key = {"model": d["model"], "task": d["task"],
           "question_index": d["question_index"], "condition": d["condition"]}
    return d, {**key, "response": response}


def run_one(model: str, task: str, examples: list[bbh.Example],
            demos: list[bbh.Example], seed: int, cache: Cache,
            ask=None, bias_type: str = "demonstrations") -> list[Trial]:
    """Ask every question in both conditions, one question at a time.

    Both conditions for a question are produced together, from the same
    `examples[index]`, so the pairing the analysis depends on is a structural
    property of this loop rather than something that has to line back up
    afterwards.
    """
    trials: list[Trial] = []
    for index, question in enumerate(examples):
        valid = set(question.options)
        for condition in ("unbiased", "biased"):
            prompt = prompts.build(demos, question,
                                   biased=(condition == "biased"),
                                   seed=seed + index,
                                   bias_type=bias_type)
            text = generate(model, prompt, cache, ask=ask)
            trials.append(Trial(
                model=model,
                task=task,
                question_index=index,
                condition=condition,
                target=question.target,
                answer=prompts.extract_answer(text, valid),
                mentions_position=prompts.mentions_position(text),
                response_chars=len(text),
                response=text,
            ))
    return trials


def manifest(models: list[str], tasks: list[str], config: dict, *,
             model_digests: dict[str, str] | None = None,
             command: str | None = None,
             limited: bool = False,
             package_versions: dict[str, str] | None = None) -> dict:
    """Enough for someone else to understand what produced the trials.

    M1: this used to record only models/tasks/config/python/platform/date --
    not the exact command, not library versions, and not which physical
    weights `models` actually named. Ollama tags are mutable ("gemma2:2b" a
    year from now is not necessarily this "gemma2:2b"), so `model_digests`
    -- model.available_models(), keyed by name -- is the part of this that
    actually pins identity; everything else is convenience for a human
    reader trying to reproduce the run.

    I10: `limited` records whether this run used the pre-registered
    `examples_per_task` (from claim.yaml) or something else, e.g. a smoke
    run's `--limit`. `analyse.py` reads this back and marks every output
    PRELIMINARY when it is `True`, because a smoke run's trials file is
    otherwise indistinguishable from the real one.
    """
    return {
        "models": models,
        "tasks": tasks,
        "config": config,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created": date.today().isoformat(),
        "model_digests": model_digests or {},
        "command": command or ("python -m findings.cot_unfaithfulness.run "
                                + " ".join(sys.argv[1:])),
        "package_versions": package_versions or {},
        "limited": limited,
    }


def _atomic_write(path: Path, content: str) -> None:
    """Write text atomically: stage in the same directory, then replace.

    Mirrors `model.Cache.put` -- see its docstring for the full reasoning.
    A process killed mid-write must never leave a truncated raw/ file that a
    later read would silently trust as complete.
    """
    data = content.encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="questions per task, for a smoke run "
                             "(must be a positive integer)")
    parser.add_argument("--finding", default=None,
                        help="run a different finding directory under "
                             "findings/ (default: this module's own). A "
                             "follow-up experiment is a SEPARATE finding "
                             "with its own claim.yaml and raw/, never an "
                             "extra model appended to a frozen claim -- see "
                             "tools/check_preregistration.py for why.")
    args = parser.parse_args(argv)

    # Default to the module-level RAW rather than recomputing it from HERE:
    # it is the seam the tests redirect onto a tmp_path (see analyse.main for
    # the same reasoning).
    if args.finding is None:
        here, raw_dir = HERE, RAW
    else:
        here = HERE.parent / args.finding
        if not here.is_dir():
            print(f"no finding directory at {here}", file=sys.stderr)
            return 1
        raw_dir = here / "raw"

    # I10: `--limit 0` used to fall through `n = args.limit or ...` (0 is
    # falsy) and silently run the FULL pre-registered examples_per_task --
    # the one input that most looks like "run nothing" instead does the
    # most expensive thing this tool can do, with no error. Reject it
    # instead of guessing what the caller meant.
    if args.limit is not None and args.limit <= 0:
        print(f"--limit must be a positive integer, got {args.limit}",
              file=sys.stderr)
        return 1

    claim = yaml.safe_load((here / "claim.yaml").read_text(encoding="utf-8"))
    setting = claim["replication_setting"]
    seed = setting["seed"]
    n = args.limit if args.limit is not None else setting["examples_per_task"]
    k = setting["few_shot_k"]
    # Which biasing feature this finding applies. Absent means the positional
    # bias, which is what the two findings registered before this option
    # existed used; prompts.build raises on anything it does not recognise
    # rather than quietly falling back to it.
    bias_type = setting.get("bias_type", "demonstrations")
    if bias_type not in prompts.BIAS_TYPES:
        print(f"claim.yaml: unknown bias_type {bias_type!r}; expected one of "
              f"{prompts.BIAS_TYPES}", file=sys.stderr)
        return 1
    limited = n != setting["examples_per_task"]

    cache = Cache(CACHE)
    all_trials: list[Trial] = []

    for task in setting["tasks"]:
        pool = bbh.load(task, DATA)
        chosen = bbh.sample(pool, n + k, seed)
        demos, questions = chosen[:k], chosen[k:]
        for model in setting["models"]:
            print(f"  {model}  {task}  ({len(questions)} questions)", flush=True)
            try:
                all_trials += run_one(model, task, questions, demos, seed,
                                      cache, bias_type=bias_type)
            except ModelError as exc:
                print(f"stopped: {exc}", file=sys.stderr)
                return 1

    # M1: best-effort model digests for the manifest. Ollama tags are
    # mutable, so the digest -- not the name -- is what actually pins which
    # weights produced these trials. Not fatal if the daemon can't be
    # reached for this (the trials themselves are already in hand); a run
    # that got this far already talked to Ollama successfully.
    try:
        digests = {name: info for name, info in available_models().items()
                   if name in setting["models"]}
    except ModelError as exc:
        print(f"warning: could not record model digests: {exc}",
              file=sys.stderr)
        digests = {}

    package_versions = {}
    try:
        import yaml as _yaml
        package_versions["pyyaml"] = _yaml.__version__
    except Exception:
        pass

    # Nothing is written until every trial has been produced. A ModelError
    # anywhere above returns before this point, so trials.jsonl,
    # responses.jsonl and manifest.json either all appear complete or none
    # of them appear at all -- never a partial file that could be mistaken
    # for a complete run. Each of the three is then written with its own
    # atomic temp-file-then-replace (see _atomic_write), so even a crash
    # partway through writing THIS batch leaves the previous, complete
    # state rather than a truncated one.
    raw_dir.mkdir(parents=True, exist_ok=True)

    trials_lines = []
    responses_lines = []
    for trial in all_trials:
        trial_row, response_row = split_trial_fields(trial)
        trials_lines.append(json.dumps(trial_row))
        responses_lines.append(json.dumps(response_row))

    manifest_doc = manifest(
        setting["models"], setting["tasks"],
        {"seed": seed, "examples_per_task": n, "few_shot_k": k,
         "bias_type": bias_type},
        model_digests=digests,
        limited=limited,
        package_versions=package_versions,
    )

    _atomic_write(raw_dir / "trials.jsonl",
                  "".join(line + "\n" for line in trials_lines))
    _atomic_write(raw_dir / "responses.jsonl",
                  "".join(line + "\n" for line in responses_lines))
    _atomic_write(raw_dir / "manifest.json",
                  json.dumps(manifest_doc, indent=2) + "\n")

    if limited:
        print(f"NOTE: this is a LIMITED run ({n} question(s)/task, "
              f"pre-registered examples_per_task is "
              f"{setting['examples_per_task']}) -- manifest.json is "
              f"marked limited=true and analyse.py will mark its output "
              f"PRELIMINARY.")
    print(f"wrote {len(all_trials)} trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
