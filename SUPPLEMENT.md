# Supplementary MQAR replication contract

Date: 2026-10-04. Registered before supplementary measurements.

## Motivation and immutable keeper

The supplied CPU review reproduces ten reference tests and five integration
tests at commit `e7d26ca`; it reports no failure. It does not reproduce the
full-model causal test, CUDA optimizer test, official naive MoBA parity,
training or GPU timing. The completed original seed-123 campaign now reports
C accuracy 0.65440625 and D accuracy 0.00078125. This negative observation is
retained, not tuned away. Training semantics remain unchanged from the public
keeper. The only launcher change forwards an explicit initialization seed.

## Frozen follow-up

- Fresh task checkouts, same pinned upstream and Python/PyTorch/cu130 stack.
- Rerun all 18 reference/integration tests, including the three omitted by the
  CPU reviewer; portable launcher and supervisor regressions are separate.
- Add initialization seeds 124 and 125 to original seed 123. For each seed,
  run F/M/B/C/D for 40,000 steps with 100,000 examples, learning rate 0.001,
  dropout 0.1 and the existing architecture, optimizer and auxiliary schedule.
- Training/validation examples and batch/auxiliary sampling seeds stay fixed.
  These are initialization/dropout-RNG replicates, not independently sampled
  corpora or evidence of general language-model capability.
- Every group in a seed shares initial-state, dataset and batch-order hashes;
  C/D additionally share auxiliary-layer/position schedules. F must reach 90%
  validation accuracy before sparse runs. No further automatic recipe search.
- Select checkpoints by minimum validation CE as before. Retain final metrics,
  complete validation histories, negative runs and existing OOD budget/context
  curves. Three seeds remain a small exploratory sample.
- Primary estimator: paired D/C validation-CE log ratio and D-minus-C accuracy
  (percentage points). Report all seeds plus Student-t 95% intervals with two
  degrees of freedom; the interval assumes approximately normal seed effects.
  Report C/B separately. Do not select whichever metric favors the hypothesis.
- The original matched-quality continuation gate remains <=1% loss increase,
  <=1 percentage-point retrieval decline and >=20% logical-read reduction.
  It cannot be replaced by lower local auxiliary MSE after task collapse.

## Resource and safety boundary

Two separate idle GPUs may run one serialized campaign each. Device UUID locks
and active-process checks apply before/during each child. Preserve all other
workloads. A child has a one-hour cap; failed tests, baseline admission,
nonfinite values, process failure or contention halt that campaign.

Training wall cost includes validation and is not a kernel speedup. Existing
code computes dense logits and retains full KV, so optimized-system timing,
HBM traffic, storage savings, real-corpus LM and budget-trained Pareto points
remain separate unfulfilled gates. Do not relabel a reference cost as a
deployable sparse-attention benchmark.

## Evidence ledger

| Experiment | Hypothesis | Contract | Keeper/candidate | Evidence | Decision |
|---|---|---|---|---|---|
| attention_20261004_full_gpu_gates | CPU omissions are not correctness failures | all 18 tests, same stack | public keeper | supplementary phase0 logs and hashes | designed |
| attention_20261004_mqar_seed_repeats | D's task degradation repeats across initialization seeds | fixed recipe, seeds 123/124/125 | C vs D, F admission, B/M controls | validation histories, batch/aux hashes, checkpoint manifests | designed |
| attention_20261004_ood_budget | more detail may restore quality, without claiming budget-trained Pareto | contexts 512/1024, pairs 16/32/64, histories 0/1/3/7 | each validation-selected checkpoint | identical-data diagnostic rows | designed |

Implementation status, mechanism decision and thesis impact are reported
separately. A negative synthetic MQAR result rejects only the fixed recipe and
task claim; real-corpus/aggregation behavior remains unknown.

## Registered bounded oracle follow-up

After training completes, capture B/C/D Q/K/V at one fixed, distinct held-out
example (data seed 6123, context 512, 32 pairs). At positions 256 and 287 there
are exactly eight closed history blocks. For both layers and heads, enumerate
all 256 subsets with learned summaries, mandatory current prefix and one
softmax. Record the best visited subset of size **at most k**, k=0..8, and the
actual mean-key router on the same row. Require monotone oracle errors,
oracle <= routed error and a dense all-history endpoint. No approximation or
larger search is labeled exact. This is eight rows/model, not a task-quality
oracle, representative corpus estimate, deployable router or latency result.
Record Q/K/V and dense-output RMS and centered K/V token RMS to diagnose, not
prove, representation loss.

Two additional CUDA regressions cover sampled-loop output/backward parity at
zero/one/all history budgets and full-model future-token invariance on GPU.
They supplement rather than replace the original 18 gates. Training is not
changed by these offline diagnostics or tests.

## Completed supplementary evidence

All five groups completed 40,000 steps for each of seeds 123/124/125.
Each run processed 655,360,000 training tokens. The two fresh training
checkouts passed the original 18 gates; the expanded release suite passed
29 tests on the intended GPU stack. Scientific-source/data/order identities,
within-seed initialization identity, cross-seed distinct initialization and
C/D auxiliary-schedule identity passed the fail-closed exporter.

Accuracy is percent correct over 32,000 validation queries at the
minimum-validation-CE checkpoint, not a final independent test. All raw
validation histories, 306 OOD budget/context rows and checkpoint/log hashes
are retained in [the curated artifact](evidence/supplement_3seed.json).

| Initialization seed | F accuracy (%) | M accuracy (%) | B accuracy (%) | C accuracy (%) | D accuracy (%) |
|---|---:|---:|---:|---:|---:|
| 123 | 99.8937 | 61.1812 | 63.1000 | 65.4406 | 0.0781 |
| 124 | 99.8375 | 1.1344 | 39.4344 | 66.3656 | 36.5312 |
| 125 | 99.8250 | 76.6063 | 33.9719 | 62.9344 | 46.2313 |
| Mean | 99.8521 | 46.3073 | 45.5021 | 64.9135 | 27.6135 |

D has higher validation CE and lower accuracy than C in all three observed
seeds. D/C geometric CE ratio is **2.1678** with the
registered t(df=2) 95% interval **[0.8577, 5.4792]**.
Mean accuracy delta is **-37.3000 percentage points**, interval
**[-99.8359, 25.2359]**. These wide intervals include
no difference: the population effect is **inconclusive**, not statistically
established harm or proof that all upstream auxiliary gradients fail.
Near-random retrieval occurs in seed 123; seeds 124/125 learn partially.
C also exceeds B in the observed seeds, but its three-seed interval likewise
includes no difference. M/B seed variability is material and is not hidden.

The fixed k=1 runs do not meet the registered matched-quality continuation
condition relative to F; neither lower local error nor a more favorable
D-vs-M comparison can substitute for that gate. Implementation and regression
status are complete for this bounded supplement. A general representation
advantage remains unestablished; real-corpus behavior remains unknown.

### Bounded exact-oracle observations

All nine B/C/D checkpoints passed exhaustive-subset, at-most-k monotonicity,
oracle<=router and dense-endpoint gates. Raw per-row results are under
[evidence/oracle](evidence/oracle). The following is the mean of **eight
correlated local rows from one example/model**, not a corpus estimate:

| Seed | Group | At-most-1 oracle relative MSE | Mean-key router relative MSE |
|---|---|---:|---:|
| 123 | B | 0.0389069 | 0.258494 |
| 123 | C | 0.443969 | 0.454732 |
| 123 | D | 9.30795e-07 | 9.23365e-06 |
| 124 | B | 0.0147241 | 0.2442 |
| 124 | C | 0.490114 | 0.703808 |
| 124 | D | 0.00171489 | 0.598709 |
| 125 | B | 0.00647564 | 0.187065 |
| 125 | C | 0.559838 | 0.732918 |
| 125 | D | 0.00818969 | 0.673877 |

D's smaller conditional oracle error does not imply retained task information:
its corresponding task quality is worse than C. Targets differ across models,
summary parameters are fixed to each model's learned values, and the metric is
not invariant to downstream reparameterization. Large oracle/router gaps for
seeds 124/125 also prevent interpreting oracle ease as usable cheap routing.
Representation loss or routing difficulty is an inference consistent with
parts of this evidence, not a uniquely proven cause. Do not infer universal
representation collapse from seed 123 or centered-RMS diagnostics alone.

### Claim-evidence ledger and reopen conditions

| Claim | State | Evidence/denominator | Allowed wording |
|---|---|---|---|
| Regression and CUDA integration | measured | 29 tests; correctness_supplement.json | Passed the declared PyTorch regression suite on the tested stack, not sanitizer or architecture-universal proof |
| D vs C in observed seeds | measured | 3 initializations, 32k validation queries each | D is worse on CE/accuracy in each of these fixed-recipe runs |
| Population-level D/C effect | inconclusive | registered paired t(df=2) intervals | Three seeds are insufficient for a precise/general effect claim |
| Exact conditional approximation curves | partial | 8 rows/model, 8 history blocks, 256 subsets | Exact on this bounded row set; not a task-quality or deployable-router oracle |
| Matched-quality read-saving gate | rejected | fixed k=1 task scores vs admitted F | This recipe does not justify its matched-quality continuation claim |
| System/HBM/KV saving | unknown | no optimized kernel or traffic campaign | No speed, physical bandwidth, or storage saving claim |

Reopen only with a preregistered attributable change (for example, objective
weight/control or a different learnable workload) and the same C/D/F controls,
not a favorable subset of these seeds. Expensive real-corpus expansion still
requires novelty, task-quality and system gates. Preserve these negative
results and do not rewrite them into a success.
