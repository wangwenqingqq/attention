# attention

Controlled MQAR experiments on block-summary attention and auxiliary gradient
paths. This is a correctness/training reference, **not an optimized sparse
attention kernel**, a language-model benchmark, or a demonstrated new method.

## Question and controls

At fixed summary capacity, raw-mean-key routing and detail-read budget, does a
sampled full-attention output-alignment objective help when its gradients can
update upstream Q/K/V representations?

| Group | Task path | Auxiliary objective |
|---|---|---|
| F | Full causal attention | None |
| M | Mean-key top-k block attention, MoBA-style reference | None |
| B | Summary + selected detail | None |
| C | Same as B | Q/K/V detached only on the auxiliary path |
| D | Same as C | Auxiliary gradients may update Q/K/V |

All groups train the full task backbone from random initialization. B/C/D use
identical summary parameters and initialization. Selected detail **replaces**
its block summary; all visible entries share one softmax. `k_history` excludes
the mandatory current causal block: one history block means up to two detail
blocks in total. Targets are the same model's current full-attention output,
stop-gradient, not a separately pretrained teacher.

The backbone and MQAR generator come from the pinned
[Zoology repository](https://github.com/HazyResearch/zoology). The M implementation
is not the official optimized MoBA kernel; a regression test compares it with
the [official naive reference](https://github.com/MoonshotAI/MoBA).

## Install and check

Linux, Python 3.12, Git, curl, NVIDIA tooling and `flock` are needed for campaigns.
The tested GPU stack is PyTorch 2.11.0/cu130, torchvision 0.26.0/cu130 on an
RTX PRO 6000 Blackwell Server Edition (SM120), driver 590.48.01. Other GPU stacks
must pass the same gates before use. No external dependency source is vendored.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements.txt
bash prepare_upstream.sh

# The ten original reference tests need only PyTorch and can run on CPU.
.venv/bin/python -m unittest -v test_reference
.venv/bin/python -m unittest -v test_launcher test_campaign
.venv/bin/python -m unittest -v test_oracle test_summary

# The integration tests include CUDA/model checks. Select an allocated idle GPU.
export PYTHONPATH="$PWD:$PWD/upstream:$PWD/upstream/zoology"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v test_reference test_batched

# Complete 42-test release suite (same allocated device and PYTHONPATH).
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v \
  test_reference test_batched test_oracle test_summary test_launcher test_campaign \
  test_diagnostics test_diagnostic_summary
```

The upstream Zoology embedding constructor currently allocates CUDA tensors
even when a downstream caller eventually moves the model to CPU. Therefore the
complete integration suite requires an available CUDA device; the ten original
tests do not. The wrapper and all eight integration tests were checked before
training. Two supplementary regressions additionally test CUDA sampled
forward/backward parity and CUDA full-model causality (ten integration tests
in the current tree). The portable launcher and owned-child termination fallback also have
independent standard-library regression tests; they allocate no GPU or training
process.

## Run the bounded one-seed screen

```bash
ATTENTION_GPU=0 bash launch.sh --tag screen_20261004 \
  --steps 40000 --lr 0.001 --train-examples 100000 --dropout 0.1
```

`ATTENTION_GPU` can be an allocated GPU index or UUID. `ATTENTION_PYTHON` and
`ATTENTION_LOCK_DIR` override the interpreter and cooperative lock directory.
The runner never stops another user's process. It checks for other GPU workloads
before and during each child; contention halts the campaign and retains evidence.
Run one campaign per checkout. Stop its recorded supervisor PID with SIGTERM;
only its dedicated child process group is terminated. Existing outputs are not
silently replaced; choose a new tag for a repeat.

Stages are: 20 correctness/integration tests, unmodified official short MQAR
example, F admission, then M/B/C/D in serial. The official example must exceed
99% validation accuracy; the long-context F baseline must reach at least 90%
before sparse comparisons begin. Each training subprocess has a one-hour cap.

Use `python train_mqar.py` for the original short 3,000-step F recipe, which
failed baseline admission; the command above is the matched larger-data recipe,
not an assertion that default hyperparameters are already optimal.

## Supplementary replication

The [registered supplementary contract](SUPPLEMENT.md) adds initialization
seeds 124 and 125, keeping data, batch order and auxiliary sampling fixed.
Run repeats serially per checkout with unique tags:

```bash
ATTENTION_GPU=0 bash launch.sh --tag repeat124 --seed 124 \
  --steps 40000 --lr 0.001 --train-examples 100000 --dropout 0.1
ATTENTION_GPU=0 bash launch.sh --tag repeat125 --seed 125 \
  --steps 40000 --lr 0.001 --train-examples 100000 --dropout 0.1
```

`summarize_screen.py` accepts three `--campaign` directories containing the
completed group runs. It verifies frozen configuration/runtime, scientific
source, data, initialization, batch/auxiliary sampling, all 161 validation
points and the 32,000-query denominator before aggregating three seeds.
`oracle_diagnostic.py` enumerates all 256 subsets of eight closed history
blocks on eight held-out query/head/layer rows per checkpoint. Its at-most-k
local-error curve is neither a task-quality oracle nor an efficient router.
Both scripts refuse to replace an existing output.

The completed [three-seed artifact](evidence/supplement_3seed.json),
[29-test record](evidence/correctness_supplement.json) and
[exact-oracle rows](evidence/oracle) supplement the original snapshot.
D has lower accuracy and higher CE than C in all three observed seeds;
D/C geometric CE ratio is 2.1678, but its exploratory 95% interval
[0.8577, 5.4792] includes no difference. Do not call this statistically
established harm or universal collapse. See the contract for all group scores,
negative evidence, paired intervals and conditional oracle limitations.

## Phase 2: source-routing and exact conditional oracle

The [phase-2 protocol](experiments/next_round_protocol.md) and
[decision/evidence tables](results/phase2/DECISION.md) execute E0/E1/E2 without
retraining. All seed-123 best/last checkpoint and archived log hashes are verified.
Diagnosis-dev uses 256 fresh sequences (seed 9123), 8192 true queries. The exact
k<=1 oracle uses 1024 identical true queries per B/C/D checkpoint, both layers
and heads; four nonqueries per sequence are reported separately.

**Negative joint result:** native D best accuracy is 0.1099%, C is 65.2466%,
F is 99.8169%. Correct-source last-layer selection raises C to 98.9014% but
leaves D unchanged. D has extremely small native local-oracle error, yet its
last-layer oracle-output injection also does not recover task accuracy. These
truth/full-history diagnostics are not deployable routers or system results.
The predeclared task-preservation gate failed; E3/E4/E5 expansion stopped and
the reserved final test remains unopened. No pure representation claim is made.

```bash
# Select an allocated idle GPU; use a new output directory, never overwrite.
export PYTHONPATH="$PWD:$PWD/upstream:$PWD/upstream/zoology"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python analyze_mqar.py \
  --runs /path/to/completed/seed123/campaign --output results/local_phase2
.venv/bin/python summarize_diagnostics.py results/local_phase2

# Lossless released row traces; their raw digests are in artifact_manifest.json.
gzip -dk results/phase2/route_rows.jsonl.gz results/phase2/oracle_rows.jsonl.gz
```

The analysis CLI requires original seed-123 checkpoint artifacts matching the
published three-seed manifest. Its fresh models preserve weights/input hashes,
caller RNG and diagnostic settings, and create no optimizer. Standalone CLI
invocation requires the caller's GPU allocation/lock policy; it is **not** the
training launcher. Reported intervals resample sequences, not individual queries.
42 tests passed on the declared GPU stack; raw checkpoints/data stay outside Git.

## Evidence and limitations

The original, partial [screening snapshot](evidence/screen_20261004.json) and
[experiment contract](EXPERIMENT.md) distinguish completed observations from
queued work at the time of that snapshot; it is not a live status record.
Two inadequate full-attention recipes are retained as negative evidence rather
than erased. Three initialization seeds on fixed data and validation-selected
checkpoints do not establish a general quality/read frontier, novelty or
statistical advantage.

The batched backend computes all detail logits densely before masking. Logical
detail/summary/router counts are **not physical HBM traffic or GPU speedups**.
It retains the entire KV sequence and rebuilds summaries: no KV-capacity saving,
incremental decode cache, GQA or optimized prefill/decode kernel is claimed.
Budget/context changes after training are OOD diagnostics, not budget-trained
Pareto points. Multi-seed oracle replication beyond the fresh diagnosis-dev set,
real-corpus LM training and optimized external system comparisons remain pending.

Near neighbors include [NSA](https://arxiv.org/abs/2502.11089),
[MoBA](https://arxiv.org/abs/2502.13189),
[SeerAttention](https://arxiv.org/abs/2410.13276), and
[MiniMax Sparse Attention](https://arxiv.org/html/2606.13392v1), whose auxiliary
gradient ablations are material counterevidence to broad representation-shaping
claims. The supplementary fixed-recipe MQAR runs do not establish a C/D
representation advantage; behavior on other workloads remains unknown.

Public files exclude device configurations, raw attachments, environments,
checkpoints, datasets, raw operational logs and private experiment history.
