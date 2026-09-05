# still-holds: do these findings still hold on open models?

Design document. 2026-09-05. Revised the same day to cut premature
infrastructure; see section 9 for what was removed and why.

## 1. The problem

Most influential empirical results in alignment and interpretability were
measured once, on the models available at the time, and never checked again.

Two gaps follow.

The first is temporal. Many well-known findings were established on closed
models from 2023 and 2024. The open models people run today differ in scale,
training data, and post-training. Whether those findings still describe current
models is, in most cases, unknown rather than disputed.

The second is economic. Checking a result costs compute, and the assumption
that this work requires a cluster is widespread enough to be self-fulfilling. A
large share of the people who might check these results have a laptop, a free
notebook session, and a weekly quota. Findings that cannot be examined on that
hardware will not be examined by them.

## 2. What this is

A repository of **replications**. Each one takes a single specific claim from a
single paper, states in advance what result would count as a replication, runs
the experiment against current open models, and publishes an honest verdict.

Each replication is a **standalone artifact**: a directory and a written report
that stand on their own, independent of anything else in the repository. There
is no shared framework, no scoreboard, and no continuous integration in this
version. Section 9 explains why, and what would have to be true before any of
that gets built.

The unit is a claim, not a benchmark. This is not an evaluation framework.
Mature frameworks exist and are well maintained; where one fits a replication's
needs it is used as a dependency and never reimplemented.

## 3. Scope and non-goals

In scope:

- Claims testable on models of 8B parameters or smaller, or on larger models
  where the experiment is inference-only and short.
- Claims with a measurable directional effect that a stated rule can adjudicate.
- Honest reporting of failed and inconclusive results.

Not in scope:

- A general-purpose evaluation framework.
- Leaderboards of model quality. This measures whether findings hold, not which
  model is better.
- Claims requiring frontier-scale compute, closed-model access, or proprietary
  data.
- Explanations of *why* a replication succeeded or failed beyond what the
  experiment measures. Speculation is labelled as speculation.

## 4. The replication format

```
findings/<slug>/
  claim.yaml        pre-registered claim, prediction, adjudication rule
  run.py            produces raw/ and results/measurements.json
  report.md         the standalone writeup: method, plots, verdict, limits
  raw/              committed model outputs with a provenance manifest
  results/          measurements.json, generated plots
```

`claim.yaml` carries:

| Field | Meaning |
| --- | --- |
| `paper` | title, authors, year, identifier |
| `claim` | the single claim under test, one sentence |
| `original_setting` | models, scale, conditions in the source |
| `original_effect` | the effect size the source reported |
| `models` | the open models this replication tests |
| `manipulation_check` | a precondition that must hold for the test to mean anything |
| `prediction` | direction, and the rule that decides the verdict |
| `compute` | what it takes to run: hardware, wall clock, whether a GPU is needed |

Example prediction block:

```yaml
prediction:
  direction: decrease
  replicated_if: "effect > 0.5 * original_effect and ci_excludes_null"
  partial_if:    "direction_matches and ci_excludes_null"
```

**Pre-registration is the one piece of process kept from the start**, because
it is what makes the verdicts worth reading and it costs almost nothing. The
`claim.yaml` for a replication is committed in its own commit, before any
result exists. A short script checks that ordering from git history and is run
before publishing a report. Predictions cannot be quietly fitted to observed
numbers.

## 5. Verdicts

| Verdict | Condition |
| --- | --- |
| `REPLICATED` | direction matches, CI excludes null, magnitude satisfies `replicated_if` |
| `PARTIAL` | direction matches and CI excludes null, magnitude falls short |
| `FAILED` | CI includes null, or the effect runs opposite to the prediction |
| `INCONCLUSIVE` | the manipulation check failed, or the run lacked power |

`INCONCLUSIVE` carries most of the weight here. A model too weak to exhibit the
precondition of an experiment has not falsified the finding; it has failed to
test it. Conflating those two would make this repository a source of
confidently wrong conclusions.

Verdicts are computed from `measurements.json` and the rule in `claim.yaml`,
not assigned by hand. In this version that computation lives in each
replication's own `run.py`. It is not yet shared code.

## 6. Compute

Every replication states what hardware it needs and how long it takes, in
`claim.yaml` and at the top of its report. A reader should be able to tell in
five seconds whether their own machine can reproduce it.

The working constraint is that every replication must run on free or consumer
hardware: a laptop CPU, a 4GB consumer GPU, a free notebook GPU session, or a
weekly free quota. A claim that cannot be tested within that envelope is out of
scope rather than run on borrowed compute.

This is a stated and self-imposed constraint. It is deliberately not enforced
by tooling in this version.

## 7. Reproducibility

Raw model outputs are committed alongside every replication, with a manifest
recording model identifier and revision, library versions, hardware, date,
seed, and the exact command.

Each replication supports re-running its analysis over those committed outputs
without a GPU or a model download. This lets a reader check the statistics in
seconds without trusting the author or spending compute, which matters more
than any other single property of the repository.

Where raw outputs are too large to commit, they are published as a release
artifact and referenced by content hash.

## 8. The replications

Ordered by risk and by how much each one is worth.

### 8.1 Unfaithful chain-of-thought under biasing context

Turpin, Michael, Perez and Bowman, 2023, arXiv:2305.04388. Reordering few-shot
examples so the correct answer is always the same option shifts the model's
answer toward the bias, while its stated reasoning does not mention the bias.
Reported accuracy drop of up to 36% across 13 BIG-Bench Hard tasks on GPT-3.5
and Claude 1.0.

Open question: the source models were closed and are two generations old. Open
models at 1B to 8B are trained on far more chain-of-thought than existed then.

Manipulation check: confirm answers actually shift toward the biasing feature
before measuring whether the reasoning conceals it. A model that ignores the
bias yields `INCONCLUSIVE`.

CPU-feasible. Scoped to four tasks and three models. Roughly a fortnight of
evenings.

First because it is cheap, because it can only fail cheaply, and because
finishing one complete replication end to end is worth more than starting three.

### 8.2 Self-other overlap reduces deception

Carauleanu, Vaiana, Rosenblatt, Berg and Schwerz de Lucena, 2024,
arXiv:2412.16325. Fine-tuning on contrastive self- and other-referencing
observations reduced deceptive responses in Mistral-7B-Instruct-v0.2 from 73.6%
to 17.2%, with comparable reductions reported at 27B and 78B.

**This is the one the project is actually aiming at.** The 7B result sits
inside a free weekly GPU quota, so the headline number is directly reachable.
No independent replication outside the original group is known to the author at
the time of writing. A confirmation gives a neglected method its first outside
support; a failure is itself a result worth publishing. Both outcomes are
useful, which is the mark of a well-chosen experiment.

Requires LoRA fine-tuning, reconstruction of the deception evaluation from the
paper, and a capability-retention measurement so that reduced deception can be
distinguished from degraded competence.

The report is written neutrally whatever the number says, and sent to the
original authors when it is finished.

This is the highest-value and highest-risk item. It gets real budget and it
comes second, once the process is proven on 8.1 and not before.

### 8.3 Refusal mediated by a single direction

Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee and Nanda, 2024,
arXiv:2406.11717. A difference-in-means direction from contrastive harmful and
harmless instructions causally mediates refusal across 13 open chat models.

Attractive because the claim is actively contested: arXiv:2602.02132 argues
refusal is not adequately described by a single direction. A pre-registered
replication adjudicating between two published positions is stronger than a
confirmation.

Optional and third. Attempted only if 8.1 and 8.2 both land.

Method note with a safety consequence: the primary causal test is the
**inducing** direction, steering toward refusal on harmless prompts, which
tests the same mediation claim as ablation. See section 10.

## 9. What was deliberately not built

An earlier draft of this document specified a compute-tier system enforced in
CI, a shared adjudicator library, a generated scoreboard site, and continuous
integration re-running replications on every push.

All of it was removed. That machinery is justified by volume, and at three
replications there is no volume to justify it. Building it first would invert
the project: most of the effort would go into scaffolding rather than into the
experiments the scaffolding exists to hold, and the scaffolding is the more
comfortable work of the two.

The order is fixed. Shared abstractions come after the code they abstract, not
before.

**Graduation criteria.** Shared tooling, a scoreboard, and CI become
appropriate when five or more replications exist, at least three of them share
a mechanism worth extracting, and the manual publishing process has become the
bottleneck. Until all three hold, they stay unbuilt.

## 10. Dual-use policy

Some findings here carry a capability externality: a method for studying a
safety behaviour can also be a method for removing it. The refusal direction is
the immediate case and the policy is set now rather than improvised later.

1. Where a causal claim can be tested in a direction that does not degrade a
   safety property, that direction is the primary test. For refusal, inducing
   refusal on harmless inputs tests the same mediation claim as ablating it.
2. Numbers needed to adjudicate a claim are reported, including a suppressing
   direction where the claim depends on it.
3. No turnkey path is shipped. This repository provides no command whose
   purpose is to disable a safety behaviour, and publishes no modified weights.
4. Where the underlying technique is already public, this is not concealment. It
   is a refusal to make the harmful use more convenient than it already is.

A claim that cannot satisfy these constraints and still be tested is not added.

## 11. Honest limitations

- **Replication is low-status work.** The field rewards novelty. Careful
  confirmation of an existing result may attract little attention, and no amount
  of rigour changes that. The value of a replication that succeeds is real but
  quiet; the value of one that fails is much larger. Which of those happens is
  not under the author's control.
- **Small models may not support the phenomenon.** Several findings depend on
  capabilities that emerge with scale. If a large share of results come back
  `INCONCLUSIVE`, the honest conclusion is that these claims cannot be tested at
  this scale, which is itself worth reporting even though it undercuts the
  premise that this work is broadly accessible on small hardware.
- **A single implementer is weak evidence.** One person reimplementing a method
  and finding it does not replicate is more likely to have implemented it wrong
  than to have overturned the result, and reports must say so. Committed raw
  outputs and reproducible analysis reduce this. They do not remove it.
- **Quantization is a confound.** Running at 4-bit is not running the model. Any
  quantized measurement records that, and is re-run unquantized where the budget
  allows.
- **A replication is not the original experiment.** Different models, smaller
  scale, reimplemented code. A `FAILED` verdict is evidence a claim does not
  extend to the tested setting, not evidence the original was wrong.
- **Choosing which claims to test is itself a choice.** Selecting only
  fragile-looking findings would produce a misleading picture. Selection
  reasoning goes in each report.

## 12. Definition of done

Version one is complete when 8.1 is published as a standalone replication:
pre-registered claim, committed raw outputs, analysis reproducible without a
GPU, an adjudicated verdict, and a report that a reader unfamiliar with this
repository can follow start to finish.

Version two is 8.2, published the same way.

Nothing else is scheduled. Whether this becomes a larger body of work depends
on whether the first two are worth reading.
