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

# The integration tests include CUDA/model checks. Select an allocated idle GPU.
export PYTHONPATH="$PWD:$PWD/upstream:$PWD/upstream/zoology"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v test_reference test_batched
```

The upstream Zoology embedding constructor currently allocates CUDA tensors
even when a downstream caller eventually moves the model to CPU. Therefore the
complete integration suite requires an available CUDA device; the ten original
tests do not. The wrapper and all eight integration tests were checked before
training. The portable launcher and owned-child termination fallback also have
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

Stages are: 18 correctness/integration tests, unmodified official short MQAR
example, F admission, then M/B/C/D in serial. The official example must exceed
99% validation accuracy; the long-context F baseline must reach at least 90%
before sparse comparisons begin. Each training subprocess has a one-hour cap.

Use `python train_mqar.py` for the original short 3,000-step F recipe, which
failed baseline admission; the command above is the matched larger-data recipe,
not an assertion that default hyperparameters are already optimal.

## Evidence and limitations

The curated [screening evidence](evidence/screen_20261004.json) and
[experiment contract](EXPERIMENT.md) distinguish completed observations from
queued work. Two inadequate full-attention recipes are retained as negative
evidence rather than erased. One seed and validation-selected checkpoints do
not establish a general quality/read frontier, novelty or statistical advantage.

The batched backend computes all detail logits densely before masking. Logical
detail/summary/router counts are **not physical HBM traffic or GPU speedups**.
It retains the entire KV sequence and rebuilds summaries: no KV-capacity saving,
incremental decode cache, GQA or optimized prefill/decode kernel is claimed.
Budget/context changes after training are OOD diagnostics, not budget-trained
Pareto points. Exact/greedy oracle curves, three seeds, real-corpus LM training
and optimized external system comparisons remain separate pending gates.

Near neighbors include [NSA](https://arxiv.org/abs/2502.11089),
[MoBA](https://arxiv.org/abs/2502.13189),
[SeerAttention](https://arxiv.org/abs/2410.13276), and
[MiniMax Sparse Attention](https://arxiv.org/html/2606.13392v1), whose auxiliary
gradient ablations are material counterevidence to broad representation-shaping
claims. The narrower C/D output-alignment hypothesis remains to be tested.

Public files exclude device configurations, raw attachments, environments,
checkpoints, datasets, raw operational logs and private experiment history.
