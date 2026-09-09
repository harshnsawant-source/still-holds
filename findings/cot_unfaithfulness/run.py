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

Leave a half-written `trials.jsonl` behind. All trials are held in memory and
the file is only written once every trial has been produced; a `ModelError`
partway through stops the run before anything is written, so a partial run
can never be mistaken for a complete one. The response cache means resuming
is cheap, so there is no reason to trade that guarantee for partial progress.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from findings.cot_unfaithfulness import bbh, prompts
from findings.cot_unfaithfulness.model import Cache, ModelError, generate

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


def run_one(model: str, task: str, examples: list[bbh.Example],
            demos: list[bbh.Example], seed: int, cache: Cache,
            ask=None) -> list[Trial]:
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
                                   seed=seed + index)
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
            ))
    return trials


def manifest(models: list[str], tasks: list[str], config: dict) -> dict:
    """Enough for someone else to understand what produced the trials."""
    return {
        "models": models,
        "tasks": tasks,
        "config": config,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created": date.today().isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="questions per task, for a smoke run")
    args = parser.parse_args(argv)

    claim = yaml.safe_load((HERE / "claim.yaml").read_text(encoding="utf-8"))
    setting = claim["replication_setting"]
    seed = setting["seed"]
    n = args.limit or setting["examples_per_task"]
    k = setting["few_shot_k"]

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
                                      cache)
            except ModelError as exc:
                print(f"stopped: {exc}", file=sys.stderr)
                return 1

    # Nothing is written until every trial has been produced. A ModelError
    # anywhere above returns before this point, so trials.jsonl and
    # manifest.json either both appear complete or neither appears at all -
    # never a partial trials.jsonl that could be mistaken for a full run.
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for trial in all_trials:
            handle.write(json.dumps(asdict(trial)) + "\n")
    (RAW / "manifest.json").write_text(
        json.dumps(manifest(setting["models"], setting["tasks"],
                            {"seed": seed, "examples_per_task": n,
                             "few_shot_k": k}), indent=2) + "\n",
        encoding="utf-8")
    print(f"wrote {len(all_trials)} trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
