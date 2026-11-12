# still-holds

**Do well-known alignment findings still hold on the open models you can actually run?**

Three pre-registered replications in, the most useful thing this repository has found is
not about any of the papers. It is about the number they are measured with.

---

## The finding

We planted a bias in a prompt and watched it destroy correct answers. The standard
metric reported the model had gotten **better**.

Concretely, on `qwen2.5:3b-instruct`, with a suggested answer stated in the user's turn
("I think the answer is (A) but I'm curious to hear what you think"):

- The bias demonstrably worked. The model moved toward the suggested option by
  **+16.3 points** (95% CI [+8.8, +23.8]).
- On the 177 questions where the suggestion was **wrong**, it abandoned a **correct**
  answer for the suggested one **25 times**. That is real, countable damage.
- The pre-registered accuracy metric came out **+0.054** — *positive*, CI
  [−0.025, +0.133]. No effect detected, in the wrong direction.

The reason is that the same prompt change flipped the correctness of **97 of 240
answers**: 55 became correct and 42 became wrong. On the 177 questions where the
suggestion was wrong, 34 became correct and 33 became wrong. Gains and losses nearly
cancel, so the net accuracy change is small and does not show the 25 correct answers
lost to the suggestion. Accuracy reports the net change correctly; it cannot also say
which answers were damaged. Whether a neutral prompt change would cause the same churn
has not been tested here. That needs a matched control.

For contrast, on the same protocol `gemma4:31b-cloud` changed the correctness of **5 of
240** answers. Stable enough to measure — and it does not take the bias at all, so there
is nothing to measure on it.

**So the two configurations fail as instruments in opposite ways:**

| | biasable? | stable enough to measure? |
| --- | --- | --- |
| ~3B models | **yes** | no — churn swamps the effect |
| ~31B model | no | **yes** |

Neither can test the original claim. If you are running accuracy-delta evals on small
open models, this is worth knowing before you trust a null.

### How much this is worth, precisely

This is a **post-hoc diagnostic**, computed after seeing the verdicts in order to explain
them. It is not pre-registered and it adjudicates nothing. It is two models, one bias
type, one task family, and the instability is measured on one of them.

What it is not: a refutation of Turpin et al. Nothing here refutes that paper — see the
verdicts below, and what they are worth.

The metric it points at — *how often does the bias convert a correct answer into the
biased one* — is paired and per-question, so it does not average against unrelated churn.
It would have caught what accuracy-delta missed. Registering it properly is the obvious
next experiment.

---

## The scoreboard

| finding | paper | model(s) | verdict |
| --- | --- | --- | --- |
| [`cot_unfaithfulness`](findings/cot_unfaithfulness/report.md) | Turpin et al. 2023 | `qwen2.5:3b`, `llama3.2:3b`, `gemma2:2b` | **FAILED** |
| [`cot_unfaithfulness_scale`](findings/cot_unfaithfulness_scale/report.md) | Turpin et al. 2023 | `gemma4:31b` | **INCONCLUSIVE** |
| [`cot_unfaithfulness_suggested`](findings/cot_unfaithfulness_suggested/report.md) | Turpin et al. 2023 | `qwen2.5:3b` · `gemma4:31b` | **FAILED** · **INCONCLUSIVE** |

2,880 trials. Every number recomputable from committed raw data with no GPU, no model
and no network.

**Two FAILEDs, and the honest summary is still "not tested."** Given the section above,
a null on this metric in this regime is worth less than it looks. Turpin et al. is not
refuted by anything here.

Three outcomes, and the distinction is the point of the design:

- **INCONCLUSIVE** — the manipulation check failed. The conditions the finding requires
  were not reproduced, so the finding was never put at risk. Says nothing about it.
- **FAILED** — the manipulation check passed and the prediction did not hold.
- **REPLICATED / PARTIAL** — the prediction held, at full or reduced strength.

The manipulation check is always evaluated **first**. Reporting a finding as refuted when
it was merely never exercised would be the single most misleading thing this repository
could do.

Every experiment runs on free, consumer-accessible hardware. No cluster, no paid API. The
assumption that this work needs a budget is widespread enough to be self-fulfilling.

---

## Why you should trust a verdict here

**Pre-registration is enforced from git history, not promised in prose.**
[`tools/check_preregistration.py`](tools/check_preregistration.py) reconstructs from the
commit graph that a claim's *last* modification precedes the *first* appearance of any
raw data or results. It compares commits **topologically, not by date string**, because
committer dates can be set by hand — an earlier version of this tool compared dates and
admitted a backdating attack. It **fails closed**: an unrecognised finding name or a
repository with no git history is an error, never a silent pass. What this establishes
is the order of the committed files. It cannot prove that nobody saw results before the
plan was committed.

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
No "small correction" exemption. Two of the three findings are follow-ups registered
exactly that way, each saying so in its own `claim.yaml`.

---

## The bug that would have faked a result

This is here because a scoreboard that reports only verdicts is asking to be trusted.

The first working runner had no stop sequence. Models regularly ran past the test question
and began restating the few-shot demonstrations. Answer extraction took the last option
letter in the output. **Under bias every demonstration answers `(A)` by construction.** So
a model that wandered scored `A` when biased and a mix of letters when not — from
generation length alone, with no bias effect whatsoever.

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
readable without knowing anything else here.

---

## Non-goals

- **Not an evaluation framework.** The unit is a claim from a paper, not a benchmark.
  Mature frameworks exist; where one fits, it is used as a dependency and never
  reimplemented.
- **Not a leaderboard.** Models are chosen to test a claim, not ranked against each other.
- **Not a refutation service.** A FAILED verdict is a result about one configuration, and
  the section above is what this repository thinks of its own FAILEDs.

## Limitations that apply to everything here

Single implementer: one person wrote the harness, the tests and the reports, and the
integrity failures documented above were found by constructing adverse cases, not by
independent review. Small models and short experiments, by design and by budget. A
replication is not the original experiment — different models, a subset of tasks, and its
own prompt templates.

Each report carries its own limitations section, generally longer than its results
section. That ratio is intentional.
