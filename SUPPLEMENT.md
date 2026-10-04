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
Record Q/K/V and dense-output RMS to diagnose, not prove, representation loss.

Two additional CUDA regressions cover sampled-loop output/backward parity at
zero/one/all history budgets and full-model future-token invariance on GPU.
They supplement rather than replace the original 18 gates. Training is not
changed by these offline diagnostics or tests.
