# still-holds

**Do well-known alignment findings still hold on the open models you can actually run?**

Most influential empirical results in alignment were measured once, on the closed models
available at the time, and never checked again. Whether they still describe the models
people run today is, in most cases, unknown rather than disputed.

This repository checks them, one specific claim at a time, and publishes an honest
verdict — including when the honest verdict is "we could not test this."

Every experiment here runs on free, consumer-accessible hardware. No cluster, no paid
API. The assumption that this work needs a budget is widespread enough to be
self-fulfilling.

---

## The scoreboard

| finding | paper | model(s) | verdict |
| --- | --- | --- | --- |
| [`cot_unfaithfulness`](findings/cot_unfaithfulness/report.md) | Turpin et al. 2023 | `qwen2.5:3b`, `llama3.2:3b`, `gemma2:2b` | **FAILED** |
| [`cot_unfaithfulness_scale`](findings/cot_unfaithfulness_scale/report.md) | Turpin et al. 2023 | `gemma4:31b` | **INCONCLUSIVE** |
| [`cot_unfaithfulness_suggested`](findings/cot_unfaithfulness_suggested/report.md) | Turpin et al. 2023 | `qwen2.5:3b` · `gemma4:31b` | **FAILED** · **INCONCLUSIVE** |

2,880 trials. Every number recomputable from committed raw data with no GPU, no model
and no network.

**Read the verdicts carefully, because they do not mean what they look like.** See
[what these verdicts are worth](#what-these-verdicts-are-worth).

---

## What a verdict means

Three outcomes, and the distinction is the point of the design:

- **INCONCLUSIVE** — the manipulation check failed. The conditions the finding requires
  were not reproduced, so the finding was never put at risk. This says nothing about the
  finding.
- **FAILED** — the manipulation check passed and the prediction did not hold. The
  conditions were reproduced and the predicted consequence did not follow.
- **REPLICATED / PARTIAL** — the prediction held, at full or reduced strength.

The manipulation check is always evaluated **first**. Reporting a finding as refuted when
it was merely never exercised would be the single most misleading thing this repository
could do.

---

## Why you should trust a verdict here

**Pre-registration is enforced from git history, not promised in prose.**
[`tools/check_preregistration.py`](tools/check_preregistration.py) reconstructs from the
commit graph that a claim's *last* modification precedes the *first* appearance of any
raw data or results. It compares commits **topologically, not by date string**, because
committer dates can be set by hand — an earlier version of this tool compared dates and
admitted a backdating attack. It **fails closed**: an unrecognised finding name or a
repository with no git history is an error, never a silent pass.

```console
$ python tools/check_preregistration.py cot_unfaithfulness
ok: cot_unfaithfulness pre-registration holds
```

**The decision rule that is applied is textually the rule that was registered.** The
adjudicator `eval`s the threshold expressions straight out of `claim.yaml`, with builtins
stripped. Hardcoding them would let the code and the claim drift apart silently.

**Every published number is recomputable by anyone.**

```console
$ python -m findings.cot_unfaithfulness.analyse --verify
verify: every number in results/measurements.json matches a fresh recomputation
        from raw/trials.jsonl
```

Raw trials *and* full model response text are committed, so you can re-run the answer
extractor against what the model actually said rather than trusting a bare letter.

**A claim that needs to change after results exist becomes a new claim in a new commit.**
There is no "small correction" exemption. Two of the three findings above are follow-ups
registered exactly this way, and each says so in its own `claim.yaml`.

---

## The bug that would have faked a result

This is here because a scoreboard that reports only verdicts is asking to be trusted.

The first working runner had no stop sequence. Models regularly ran past the test question
and began restating the few-shot demonstrations. Answer extraction took the last option
letter in the output. **Under the biased condition every demonstration answers `(A)` by
construction.** So a model that wandered scored `A` when biased and a mix of letters when
not — from generation length alone, with no bias effect whatsoever.

The signature of that artefact is exactly the signature of the finding being tested. Run
at scale, it would have published a clean **REPLICATED** of an effect that never happened.

The first fix was a keyword blocklist of the observed run-on phrasings. It was tested and
rejected: 3 of 6 run-ons were still mis-scored, and the blocklist missed precisely those
3. It was replaced with a structural fix — a sentinel terminator every demonstration ends
with and the test question does not, passed to the model as a stop sequence.

The 24 pre-fix responses are kept in
[`tests/fixtures/smoke_responses.json`](tests/fixtures/smoke_responses.json) as evidence
of what the failure looked like.

---

## What these verdicts are worth

Less than the word FAILED suggests, and the reports say so at length.

In `cot_unfaithfulness_suggested`, the bias verifiably converted **25 correct answers into
the biased wrong one**. The pre-registered accuracy metric still came out *positive*,
because the same prompt change churned 97 of 240 answers in total and unrelated gains
cancelled the losses.

That is a finding about the measure, not the model:

- **At ~3B:** models are biasable, but answer instability swamps the accuracy metric. The
  bias's cost is real and countable, and undetectable in the registered statistic.
- **At ~31B:** the model is stable enough to measure precisely, and does not take either
  of the paper's biasing features at all.

Neither configuration can test the original claim. **So the honest summary of this
repository's headline finding is "not tested," not "does not hold."** Turpin et al. is not
refuted by anything here.

That conclusion cost three experiments and argues against the repository's own headline.
It is the kind of thing a scoreboard exists to surface.

---

## Reproducing everything

```bash
pip install -r requirements.txt

# No model needed for any of this:
python -m findings.cot_unfaithfulness.analyse --verify
python -m findings.cot_unfaithfulness.analyse --finding cot_unfaithfulness_scale --verify
python -m findings.cot_unfaithfulness.analyse --finding cot_unfaithfulness_suggested --verify
pytest -q          # 179 tests
```

To re-run generation you need [Ollama](https://ollama.com) and the models named in the
relevant `claim.yaml`:

```bash
python tools/check_preregistration.py cot_unfaithfulness    # must pass first
python -m findings.cot_unfaithfulness.run
python -m findings.cot_unfaithfulness.analyse
```

One arm of two findings uses `gemma4:31b-cloud` on Ollama's free tier. It is free, which
is the constraint that keeps this project budget-free, but it is **not local**, unlike
every other model here. Both affected claims record that deviation explicitly.

---

## Repository layout

```
findings/<name>/
    claim.yaml          pre-registered: setting, manipulation check, prediction, thresholds
    raw/                trials.jsonl, responses.jsonl, manifest.json
    results/            measurements.json
    report.md           the written verdict, with its limitations
tools/
    check_preregistration.py
docs/specs/             design document
```

Each finding is a standalone artifact. The shared runner and analyser are a convenience,
not a framework — a finding is a claim plus its data plus its report, and should be
readable without knowing anything else in this repository.

---

## Non-goals

- **Not an evaluation framework.** The unit is a claim from a paper, not a benchmark.
  Mature frameworks exist; where one fits, it is used as a dependency and never
  reimplemented.
- **Not a leaderboard.** Models are chosen to test a claim, not ranked against each other.
- **Not a refutation service.** A FAILED verdict is a result about one configuration. The
  section above is what this repository thinks of its own FAILEDs.

---

## Limitations that apply to everything here

Single implementer: one person wrote the harness, the tests and the reports, and the
integrity failures documented above were found by constructing adverse cases, not by
independent review. Small models and short experiments, by design and by budget. A
replication is not the original experiment — different models, a subset of tasks, and its
own prompt templates.

Each report carries its own limitations section, which is generally longer than its
results section. That ratio is intentional.
