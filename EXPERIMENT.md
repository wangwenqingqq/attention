# Frozen screening contract

Date: 2026-10-04. Stage: one-seed synthetic mechanism screening.

## Workload and architecture

- Pinned Zoology: `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`.
- Model: two transformer layers, hidden 128, two MHA heads, head dimension 64,
  vocabulary 2048, tied embedding/output weights, learned positions up to 1024.
- Instantiated parameter count: 806,400, including the same summary parameters
  in all groups (unused for F/M). This is not the proposed larger real-corpus LM.
- Training context 512, 32 independently remapped key/value pairs per example.
- Block 32, summary slots 2, `k_history=1`; current causal prefix is mandatory.
- Float32; TF32 disabled in the screen. No packing, GQA, quantization or RoPE.
- Initial seed 123; data seeds 1123/2123; batch-order seed 4123; auxiliary seed
  5123. Keep generator revision, data hashes, initial state hash and batch-ID
  hashes. Validation has 1,000 examples/32,000 supervised queries.

## Matched larger-data recipe

100,000 training examples; 40,000 steps; batch 32; AdamW lr 1e-3, weight decay
0.1, clipping 1, embedding and task-attention dropout 0.1. All groups receive
655,360,000 input tokens. No pretrained weights or test-based tuning.

Every eight steps, C/D align 16 positions from one sampled layer. Positions
are shared within a batch and sampled by a separate RNG. Auxiliary weight is
0.1, fixed before results. The full target and auxiliary prediction have no
dropout. The target and relative-error denominator are stop-gradient. C blocks
only auxiliary input gradients; it does not freeze the ordinary task path.
D permits them. Both perform the same auxiliary forward computations.

Primary metrics: CE and exact accuracy on labeled query positions. Evaluate
every 250 steps; select the checkpoint with minimum validation CE, not maximum
accuracy or best test score. Preserve initial, best-validation and final states;
the final state also stores optimizer and sampler RNG state.

## Admission and diagnostics

Before training: compare loop/batched outputs, Q/K/V and summary gradients,
all-history dense equivalence, causal future invariance of outputs/routes,
sampled absolute positions including decode query length one, document-segment
isolation and reset, C/D forward equality and auxiliary-gradient separation.
The trainer itself does not support packing; the segmented correctness helper
does not imply an optimized packed or incremental cache implementation.

Run the unchanged official short example first. Then admit sparse comparisons
only after F reaches >=90% validation accuracy on the long task. Stop on loss,
gradient, isolation or process failure; do not reinterpret a failed F recipe as
negative evidence about C/D. F/M/B/C/D use the same initializer/data/order.

Diagnostic contexts: 512/1024; KV pairs: 16/32/64; history budgets: 0/1/3/7.
Record task quality and local relative output error on identical examples, with
logical detail, summary and router counts separate. Budget/context shifts are
OOD. Confirm important points using separately trained budget-matched models.

## Preserved negative evidence

The initial 10,000-example/no-dropout/lr-3e-4/3,000-step F recipe did not learn
the task. A 20,000-step/lr-1e-3 repeat with the same data also failed and showed
worsening validation CE as training loss declined. Sparse comparisons were not
started under either inadequate recipe. The larger-data/dropout intervention is
coupled and does not identify which individual change caused improvement.

## Pending research gates

- Three seeds with intervals: proposed internal continue gate is <=1% relative
  validation-loss increase, <=1 percentage-point retrieval decline and >=20%
  total logical-read reduction at matched quality. One seed cannot pass it.
- Exact oracle: at most eight candidate history blocks, enumerate all sets of
  size <=k. Larger greedy/local-swap curves are heuristic, not lower bounds.
  Error cancellation means adding detail need not reduce error monotonically;
  aggregate the best visited set under an **at-most-k** budget.
- Real-corpus from-scratch LM: document-level split before tokenize/pack; fixed
  corpus and tokenizer revisions, document hashes, data order and task quality.
  Initial larger-model token budgets are screening, not full pretraining proof.
- Deployable system costs: summary writes/build, routing scan, top-k, gathering,
  attention and merging; training forward/backward, prefill and genuine qlen=1
  decode separately. Cache closed summaries instead of rebuilding all history.
- Same-contract optimized full attention, complete NSA branches, MoBA/kconv and
  FlashMoBA comparisons; actual backend/dispatch and supported architecture.
- Separate logical reads, physical HBM bytes, metadata, summaries and KV storage.
  This implementation neither accelerates them nor compresses KV capacity.

Do not claim D's benefit from D-vs-M alone; C and B are essential controls.
Better local MSE with worse task quality can indicate information loss/collapse.
Representation gains without lower total latency are not system-speed evidence.
