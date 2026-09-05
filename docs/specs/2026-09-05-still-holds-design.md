# still-holds: a standing replication scoreboard for alignment findings on open models

Design document. 2026-09-05.

## 1. The problem

Alignment and interpretability research moves fast, and most of its
influential empirical results were established once, on the models available
at the time, and never checked again.

Two gaps follow from that.

The first is temporal. Many well-known findings were measured on closed models
from 2023 and 2024. The open models people actually run today are different in
scale, training data, and post-training. Whether those findings still describe
current models is, in most cases, simply unknown.

The second is economic. Checking a result requires compute, and the assumption
that alignment research needs a cluster is widespread enough to be
self-fulfilling. A large share of the people who would check these results
have a laptop, a free Colab session, and a weekly Kaggle quota. If a finding
cannot be examined on that hardware, it will not be examined by them.

This project addresses both by maintaining a public, continuously-run set of
replications where every experiment is constrained to run on free or consumer
hardware, and where the verdict, including failure, is reported.

## 2. What this is

A repository of **findings**. Each finding takes one specific claim from one
paper, states in advance what result would count as a replication, runs the
experiment against current open models, and publishes an adjudicated verdict
with confidence intervals.

The unit is a claim, not a benchmark. This is deliberately not an evaluation
framework. Mature frameworks exist and are well maintained; where one fits a
finding's needs it is used as a dependency, not reimplemented.

The output is a scoreboard: findings on one axis, models on the other, each
cell a verdict with the compute tier required to reproduce it.

## 3. Scope and non-goals

In scope:

- Claims that can be tested on models of 8B parameters or smaller, or on
  larger models where the experiment is inference-only and short.
- Claims with a measurable, directional effect that a mechanical rule can
  adjudicate.
- Honest reporting of failed and inconclusive replications.

Explicitly not in scope:

- A general-purpose evaluation framework.
- Leaderboards of model quality. This measures whether findings hold, not
  which model is best.
- Claims requiring frontier-scale compute, closed-model access, or
  proprietary datasets.
- Any claim about why a replication succeeded or failed beyond what the
  experiment measures. Speculation is labelled as speculation.

## 4. The finding format

Each finding is a self-contained directory:

```
findings/<slug>/
  claim.yaml        pre-registered claim, prediction, adjudication rule, budget
  run.py            produces raw/ and results/measurements.json
  report.md         narrative, plots, interpretation
  raw/              committed model outputs with provenance stamps
  results/          measurements.json, generated plots
```

`claim.yaml` is the core primitive and carries:

| Field | Meaning |
| --- | --- |
| `paper` | title, authors, year, identifier |
| `claim` | the single claim under test, one sentence |
| `original_setting` | models, scale, and conditions in the source paper |
| `original_effect` | the effect size reported by the source |
| `models` | the open models this replication tests |
| `manipulation_check` | a precondition that must hold for the test to be meaningful |
| `prediction` | direction, and the rule that adjudicates a replication |
| `budget` | tier, wall-clock ceiling, VRAM ceiling |

Example prediction block:

```yaml
prediction:
  direction: decrease
  replicated_if: "effect > 0.5 * original_effect and ci_excludes_null"
  partial_if:    "direction_matches and ci_excludes_null"
```

**Pre-registration is enforced mechanically.** CI checks git history and
fails any finding whose `claim.yaml` was first committed later than, or in the
same commit as, its results. Predictions cannot be quietly fitted to observed
numbers.

## 5. Adjudication

Verdicts are computed by a shared adjudicator from `measurements.json` and the
rule in `claim.yaml`. They are never assigned by hand.

| Verdict | Condition |
| --- | --- |
| `REPLICATED` | effect matches direction, CI excludes null, magnitude satisfies `replicated_if` |
| `PARTIAL` | direction matches and CI excludes null, but magnitude falls short |
| `FAILED` | CI includes null, or the effect runs opposite to the prediction |
| `INCONCLUSIVE` | the manipulation check failed, or the budget was exhausted before adequate power |

`INCONCLUSIVE` exists to prevent a common error. A model too weak to exhibit
the precondition of an experiment has not falsified the finding; it has failed
to test it. Conflating the two would make the scoreboard actively misleading.

Verdicts are per finding and per model. The grid is the result.

## 6. Compute tiers

Every finding declares the minimum tier that can reproduce it. CI fails an
entry whose measured run exceeds its declared budget, which makes the
project's central claim checkable rather than asserted.

| Tier | Hardware | Typical reach |
| --- | --- | --- |
| `cpu` | any laptop, no GPU | prompting experiments, small models |
| `4gb` | 4GB consumer GPU | 1B to 3B, quantized |
| `t4` | free Colab T4 | up to 7B or 8B |
| `kaggle` | weekly free quota | fine-tuning at 7B |

The tier appears next to every scoreboard cell, so a reader can see at a
glance which rows their own hardware can reproduce today.

## 7. Reproduction modes and provenance

Two modes, and the distinction between them is the project's most important
usability decision.

- **`verify`** re-runs the analysis over committed raw model outputs. Seconds,
  no GPU, no model download, no API key. Anyone can audit the statistics.
- **`rerun`** regenerates outputs from the model. Requires the declared tier.

Raw outputs are committed with a provenance manifest recording model
identifier and revision hash, library versions, hardware, date, seed, and the
exact command. Committing raw outputs makes the cost of checking someone
else's arithmetic approximately zero, which is the point.

Where raw outputs would be too large to commit, they are stored as a release
artifact and referenced by content hash from the manifest.

## 8. Architecture

A small shared library, extracted rather than designed up front:

- **model loading with a budget guard** — resolves a tier to a concrete
  configuration, applies quantization, and refuses to proceed when a run would
  silently exceed its declared ceiling.
- **response cache** — content-addressed on model, prompt, generation
  parameters, and seed, so repeated runs cost nothing.
- **statistics** — bootstrap confidence intervals, shared across findings so
  that every entry reports uncertainty the same way.
- **adjudicator** — evaluates a prediction rule against measurements.
- **scoreboard generator** — walks the findings tree and emits the results
  table and the static site.

Each `run.py` is ordinary Python using whatever the experiment needs directly.
The library provides the parts every finding shares and nothing else. It is
expected to be under roughly a thousand lines; growth beyond that is a signal
that framework-building has crept in.

The library is extracted after three findings exist, not before. A shared
abstraction written before it has three real callers is written blind.

## 9. Continuous integration

The repository is public, so CPU minutes are unmetered.

On every pull request:

- schema validation of `claim.yaml`
- the pre-registration ordering check described in section 4
- presence and wellformedness of the budget declaration
- full re-run of every `cpu`-tier finding
- `verify` mode on every finding, confirming committed numbers reproduce from
  committed raw outputs

GPU-tier findings are run manually from a free notebook session and their
outputs committed with the provenance manifest. CI verifies the manifest is
complete and that `verify` reproduces the published numbers. This keeps GPU
results auditable without pretending CI can run them.

## 10. The scoreboard

A static site generated from the findings tree. The landing view is the grid:
findings by model, colour-coded verdicts, tier and runtime per cell. Each cell
links to the finding's report, plot, and the command that produces it.

Failed and inconclusive verdicts are displayed with the same prominence as
replications. A scoreboard that quietly hides its negative results is worth
nothing.

## 11. The first three findings

Chosen to escalate along two axes at once: experimental method, and cost.

### 11.1 Unfaithful chain-of-thought under biasing context

Turpin, Michael, Perez and Bowman, 2023, arXiv:2305.04388. Reordering few-shot
examples so that the correct answer is always the same option biases the
model's answer while its stated reasoning does not mention the bias. Reported
accuracy drop of up to 36% across 13 BIG-Bench Hard tasks, on GPT-3.5 and
Claude 1.0.

Open question: the source models were closed and are two generations old. Open
models at 1B to 8B are trained on far more chain-of-thought than existed then.

Manipulation check required: confirm the model's answers actually shift toward
the biasing feature before measuring whether the reasoning conceals it. A model
that ignores the bias yields `INCONCLUSIVE`, not `FAILED`.

Tier `cpu` to `4gb`. Scoped for v1 to four tasks and three models.

This is first because it can fail only cheaply, and it exercises every part of
the pipeline end to end.

### 11.2 Refusal mediated by a single direction

Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee and Nanda, 2024,
arXiv:2406.11717. A single difference-in-means direction derived from
contrastive harmful and harmless instructions causally mediates refusal across
13 open chat models up to 72B.

This claim is unusual in being both already established on open models and
actively contested. arXiv:2602.02132 argues that refusal is not adequately
described by a single direction. A pre-registered replication that adjudicates
between two published positions is a stronger contribution than a confirmation.

Method note with a safety consequence: the primary causal test is the
**inducing** direction, steering toward refusal on harmless prompts. This tests
the same causal claim as ablation. See section 12.

Tier `4gb` to `t4`.

### 11.3 Self-other overlap reduces deception

Carauleanu, Vaiana, Rosenblatt, Berg and Schwerz de Lucena, 2024,
arXiv:2412.16325. Fine-tuning on contrastive self- and other-referencing
observations reduced deceptive responses in Mistral-7B-Instruct-v0.2 from
73.6% to 17.2%, with comparable reductions reported at 27B and 78B.

The 7B result sits inside the free Kaggle quota, so the headline number is
directly reachable. No independent replication outside the original group is
known to the author at the time of writing.

Tier `kaggle`. This is the entry most likely to slip, which is why it is third.

## 12. Dual-use policy

One class of finding in this area carries a capability externality: methods for
studying a safety behaviour can also be methods for removing it. The refusal
direction is the immediate case, and the policy is set now rather than
improvised later.

Principles:

1. Where a causal claim can be tested in a direction that does not degrade a
   safety property, that direction is the primary test. For refusal, inducing
   refusal on harmless inputs tests the same mediation claim as ablating it.
2. Numbers required to adjudicate a claim are reported, including the
   suppressing direction where the claim depends on it.
3. No turnkey path is shipped. This repository does not provide a command whose
   purpose is to disable a safety behaviour, and does not publish modified
   weights.
4. Where the underlying technique is already public, the policy is not
   concealment. It is a refusal to make the harmful use more convenient than it
   already is.

A finding that cannot satisfy these constraints and still test its claim is not
added.

## 13. Threats to validity

- **Small models may not support the phenomenon.** Several findings depend on
  capabilities that emerge with scale. The manipulation check and the
  `INCONCLUSIVE` verdict exist for this, but the boundary is a judgement call
  and will sometimes be got wrong. Manipulation checks are stated in
  `claim.yaml` and open to challenge.
- **Quantization is a confound.** Running at 4-bit is not running the model. Any
  finding measured on a quantized model records the quantization in its
  manifest, and where budget allows the same finding is run unquantized at a
  higher tier for comparison.
- **A replication is not the original experiment.** Different models, smaller
  scale, and reimplemented code all differ from the source. Reports state what
  differs. A `FAILED` verdict is evidence that the claim does not extend to the
  tested setting, and is not by itself evidence that the original was wrong.
- **Single-implementer bias.** All findings are implemented by the same person,
  so an error in method or taste is correlated across entries. Committed raw
  outputs and `verify` mode exist so that others can check the analysis
  independently of the implementation.
- **Selection of claims.** Which findings get tested is a choice, and choosing
  only fragile-looking claims would produce a misleading scoreboard. Selection
  criteria are recorded in each report.

## 14. Build order and definition of done for v1

1. Finding 11.1 implemented by hand, end to end, with no shared library.
2. Finding 11.2 implemented by hand. Note what was copied.
3. Finding 11.3 implemented. Note what was copied again.
4. Extract the shared library from what repeated across the three.
5. Adjudicator and scoreboard generator.
6. CI: schema, pre-registration ordering, budget enforcement, `verify`.
7. Static site deployed.

v1 is complete when three findings carry adjudicated verdicts, the scoreboard
is generated, `verify` reproduces every published number with no GPU, CI
enforces pre-registration and budgets, and the site is live.

Cadence is unhurried and the order above is not negotiable in one respect: the
library comes after the third finding, never before.
