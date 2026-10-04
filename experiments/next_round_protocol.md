# Phase 2: source routing and exact conditional approximation

Preregistered before measurements, 2026-10-04. Parent release:
`626f439bfef86b2ffdb49b110355a7235fa09a12`. The supplied plan refers to an
older partial snapshot; all five groups and three initialization seeds now exist.
Historical results remain immutable. This first delivery implements E0/E1/E2;
conditional expansion is not automatic.

## Frozen contract and hypotheses

Two layers, hidden 128, two 64-dimensional heads, vocabulary 2048, context 512,
32 KV pairs, block 32, two summary slots, one *historical* block per query/head
plus the mandatory causal current block. FP32, TF32 disabled, evaluation mode.
Existing training: 100,000 examples, 40,000 steps, batch 32, AdamW lr 0.001,
weight decay 0.1, clip 1, dropout 0.1, auxiliary coefficient 0.1, one layer and
16 uniform positions every eight steps. C detaches only the auxiliary Q/K/V;
D does not. No backbone, compressor capacity, router or sampling change.

E0 tests task preservation and checkpoint stability. E1 tests whether missing
source detail explains errors. E2 tests exact conditional approximation with
fixed original-forward Q/K/V and each model's learned summaries. Native C/D
comparisons confound representation and compressor adaptation; they cannot
establish a pure representation advantage. No new novelty or systems claim is
being advanced. Prior related sparse-attention auxiliary training evidence is
not a proof about this different output-alignment loss.

## Data and checkpoint selection

Training seed 1123 and original validation seed 2123 and their hashes remain
unchanged. Diagnosis-dev: 256 examples, seed **9123** (6123 was already used
by the historical tiny oracle). Reserve calibration 8123 / 4096 examples and
final-test 7123 / 4000 examples; neither is generated in this phase. A scan of
experiment seed declarations found no prior use of the replacement/reserved
seeds. Numeric substring occurrences in metrics are not seed declarations.

E0: all five seed-123 groups, validation-selected minimum-CE `best.pt` and
step-40000 `last.pt`, 8192 true queries. E1: best checkpoints, all 8192 true
queries, M/B/C/D routing and F task reference. E2: B/C/D best checkpoints,
four true queries and four nonquery positions per sample, selected using a
private CPU generator seed 10123 in sample order, identical across models.
Main oracle denominator: 1024 true queries per model, both layers/heads.
Nonqueries are a separate secondary table, never mixed into the main result.
Metadata is required; no group, seed or config inference from filenames.
Known fixed architectural settings are tied to the archived source hashes.

## Measurements and intervention semantics

Source metadata matches query keys only against the unique even-position key
slots in the original 64-token prefix. Random filler collisions cannot select
a source. This annotation consumes no RNG and changes no input or label.
Record per-head selections, last-layer any/both hit, task CE/accuracy and source
distance/query-position bins. Model-specific hit-conditioned subsets are not
causal cross-model comparisons. Query-local error is separate from nonqueries.

Only the last layer is intervened on, only at true query positions, replacing
(rather than adding to) the historical selection for head 0, head 1 or both.
Replay of the original selections must reproduce logits. Truth-sourced forced
selection is diagnostic-only, not a deployable router or a task-optimal oracle.
Earlier-layer representations remain native and may limit recovery.

Oracle candidates are empty and every single closed historical block. The
current prefix is always included; refinement replaces that block's summaries;
one softmax normalizes all retained entries. Enumerate with captured Q/K/V,
not rerun upstream representations. Single-head MSE uses its own dense target.
Joint projected error uses `W_o @ (concat(candidate) - concat(full))`; bias is
excluded from error and target energy. Lexicographic candidate order breaks
ties. Joint last-layer projected relative MSE is primary; retain absolute MSE,
target energy, first-layer and single-head results and logical read counts.
Inject selected last-layer joint outputs at the same 1024 true queries and
measure their task CE/accuracy. Selection uses no labels. This is a local,
fixed-summary-family optimum, not a full-network or global task optimum.

Before/after model-state and input hashes must agree. No optimizer is created.
Restore RNG, model modes, budgets and diagnostic attributes. Invalid future,
duplicate or overbudget selections fail closed. Replay/default paths and every
small-case candidate are checked against the loop reference with FP32
atol 3e-6 / rtol 3e-5. Raw rows and checkpoint/source/data hashes are retained.
Accuracy/error intervals use 2000 paired sample-cluster bootstrap resamples,
seed 11123; the 32 within-sequence queries are not independent resampling units.

## Decision gates and stop rule

Predeclared operational meaning of a promising *joint* E2 signal: D's mean
last-layer joint projected oracle relative error is at most 0.9 times C's, its
paired sample-bootstrap ratio upper 95% bound is below 1, and native D task
accuracy is within 1 percentage point of C with CE no more than 1.01 times C.
Both approximation and task preservation matter. Failure does not prove
universal impossibility; retain any local-only improvement as partial evidence.

Only that gate admits E3's equal-capacity, frozen-backbone, identity-initialized
summary refit (4096 calibration examples, 2000 steps, batch 128 records/layer).
E4/E5 require a supported signal, not a local MSE-only win. Existing three-seed
runs use fixed order 4123 and auxiliary RNG 5123 for all initialization seeds;
they do not satisfy the plan's optional varied-order recipe. No duplicate
training is silently relabeled as satisfying that distinct recipe.

A failed first-phase gate stops expansion at this delivery. E8 gradient
attribution and a possible single matched lambda-0.01 C/D pair remain separate
conditional work, not an automatic blind sweep. Gradient evidence is required
before changing the coefficient. Final-test stays unopened. The decision must
answer the plan's five questions and mark unfitted probes as unknown.

No timing, physical HBM, KV-storage or general model capability claim. Logical
counts are per query/head reads, not physical sparse execution. Raw operational
host state and checkpoints remain outside the public repository.
