# Phase 2 decision: low local error did not preserve task capability

**Decision: stop expansion of the current lambda-0.1 recipe.** E0/E1/E2 are
complete; the task-preservation gate failed, so E3 probes, E4 varied-order
replication and E5 budget-trained k=2 runs were not launched. The reserved final
test and probe-calibration sets remain unopened. This is not a universal rejection
of representation shaping.

## 1. Does D improve on C at the same task budget? Are best/last consistent?

No, on this seed-123 diagnosis-dev set (256 sequences / 8192 true queries).
Checkpoints were selected on original validation, never on diagnosis-dev.

| Group | Best accuracy | Best query CE | Last accuracy | Last query CE |
|---|---:|---:|---:|---:|
| F | 99.8169% | 0.008839 | 99.7803% | 0.012068 |
| M | 60.4980% | 2.920419 | 56.2500% | 3.162334 |
| B | 62.1704% | 2.553363 | 59.5947% | 2.711003 |
| C | 65.2466% | 2.188037 | 65.6372% | 2.186610 |
| D | 0.1099% | 6.931994 | 0.1465% | 6.931659 |

D-C best accuracy difference is -65.1367 percentage points; paired sample-cluster
bootstrap 95% interval [-66.0889, -64.1235]. Last difference is -65.4907 points
[-66.4795, -64.4531]. Thus the direction does not reverse at the last checkpoint.
These intervals describe diagnosis sequences conditional on these trained models,
**not variation across training seeds**. The existing three-seed evidence has
partially learning D seeds 124/125; do not call all D training universally collapsed.

## 2. Does the exact local oracle improve? Does routing realize it?

D is much easier to approximate against its *own* dense local targets, but this
advantage does not preserve task capability.

| Group | Last-layer actual projected relative MSE | Exact k<=1 oracle | Projected target energy | Oracle-subset native accuracy | Injected accuracy |
|---|---:|---:|---:|---:|---:|
| B | 0.300062 | 0.0387172 | 0.429375 | 63.1836% | 94.6289% |
| C | 0.459272 | 0.206645 | 0.247636 | 64.7461% | 98.7305% |
| D | 1.48396e-7 | 7.17967e-8 | 13.908677 | 0.0977% | 0.0977% |

The main local denominator is the identical 1024 true queries (four per sequence),
two heads jointly, last layer, bias-excluded W_o projection. The ratio of D/C
mean oracle relative errors is 3.47440e-7; paired sample-cluster interval
[2.78856e-7, 4.20254e-7]. Absolute oracle MSE is also smaller for D
(9.98119e-7 versus C's 0.0548057), so the result is not solely a larger normalization
energy. It is still a comparison between different native targets/compressors.
First-layer, per-head and nonquery results plus energy/error distributions are
retained; the headline does not select a favorable head or layer.

Labels were not used to choose candidates. Injecting the last-layer joint local
oracle into C gains 33.9844 accuracy points [31.3477, 36.7188] on that same subset;
D gains 0.0000 points [0.0000, 0.0000]. Local output error and task quality are
not interchangeable objectives. These are fixed-QKV, fixed-summary-family
optima, **not full-network or task-optimal oracles**.

Last-layer source selection provides a separate, truth-sourced routing diagnostic
on all 8192 queries:

| Group | Native accuracy | Force source, head 0 | Head 1 | Both heads |
|---|---:|---:|---:|---:|
| M | 60.4980% | 96.7651% | 97.1313% | 99.8291% |
| B | 62.1704% | 94.8486% | 93.0542% | 97.8271% |
| C | 65.2466% | 97.5586% | 97.4243% | 98.9014% |
| D | 0.1099% | 0.1099% | 0.1099% | 0.1099% |

Replay controls match native outputs. The source block **replaces** the old
selection within the one-history-block budget; current-prefix detail cannot be
removed. Source correction reveals substantial recoverable routing error for
M/B/C under their existing upstream representations. It does not recover D in
this last-layer intervention. That narrow result does not prove routing is
irrelevant in earlier layers. Different models' hit-conditioned subsets must not
be treated as causal cross-model comparisons.

## 3. Do equal-capacity fresh summary probes support a representation explanation?

**Unknown: E3 was not admitted or run.** The predeclared joint gate required
oracle improvement *and* task preservation (within 1 accuracy point and 1% CE
of C). D passes the local-error condition but fails both task conditions.
Native C/D oracle differences cannot separate representation from learned
compressor adaptation. No pure representation or optimal-compression claim is
allowed.

## 4. What remains missing relative to F?

Native best C remains 34.5703 accuracy points below F; D is 99.7070 points below F.
Truth-source and local-oracle interventions are not deployable performance points.
No matched-quality read frontier, physical HBM reduction, KV-storage saving,
speedup, real-corpus LM result, novelty or general model-capability claim is
supported. Existing seed 124/125 runs use fixed batch/auxiliary order, not the
plan's optional varied-order recipe; no new runs were relabeled as that recipe.

## 5. What should happen next?

**Stop current-recipe expansion.** The inexpensive tests reject the desired
joint claim for this diagnosis set and seed. Low local approximation error by
itself is not a reason to add capacity, fit probes or retrain at k=2.

A narrowly scoped follow-up could examine separate task/auxiliary gradient norms,
cosines and clipping on fixed calibration batches (E8) before preregistering a
single matched C/D lambda-0.01 pair. That gradient attribution has **not** been
measured here; no claim of auxiliary-gradient dominance is made. Alternatively,
M/B/C's recoverable source-selection loss motivates testing whether their oracle
choices are predictable without truth information. Neither option reopens a
systems/quality claim automatically. A stable same-contract task-preserving
signal is the reopen condition for E3/E4/E5.

## Evidence and scope

- [Frozen protocol](../../experiments/next_round_protocol.md).
- `same_budget_dev.csv`, `task_sample_rows.csv`: main best/last task comparison.
- `route_rows.jsonl.gz`, `route_hit_summary.csv`, `route_strata.csv`: all 32,768
  per-model true-query rows and hit/distance/position stratification.
- `source_intervention.csv`, `source_intervention_sample_rows.csv`: source/replay
  interventions, 256 sample clusters per variant.
- `oracle_rows.jsonl.gz`, `oracle_summary.csv`, `oracle_head_summary.csv`,
  `oracle_distributions.csv`: 12,288 rows (B/C/D x two layers x 1024 true queries
  plus separate 1024 nonqueries), candidate sets/errors/counts/energies.
- `oracle_task_rows.csv`, `oracle_task_summary.csv`, `paired_cluster_intervals.csv`:
  fixed-query output injection and paired sample-level uncertainty.
- `provenance.json`, `correctness.json`, `artifact_manifest.json`: source,
  checkpoint/input hashes, test gates and immutable artifact digests.

Compressed JSONL is lossless; decompression yields the prescribed raw filenames.
Raw checkpoints, training data and operational logs are outside Git. No optimizer
was created during analysis; before/after weights and input hashes agree.
