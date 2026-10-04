"""Pinned Zoology MQAR screen. No system-speed or general-capability claims."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from zoology.config import ModelConfig, ModuleConfig
from zoology.data.multiquery_ar import multiquery_ar
from zoology.model import LanguageModel

from block_memory_reference import dense_reference, relative_mse
from memory_attention import MemoryMHA


def build_model(group, seed, device="cpu", dropout=0.0):
    torch.manual_seed(seed)
    config = ModelConfig(d_model=128, n_layers=2, vocab_size=2048,
                         max_position_embeddings=1024, embed_dropout=dropout,
                         sequence_mixer=ModuleConfig(name="memory_attention.MemoryMHA",
                                                     kwargs={"num_heads": 2, "dropout": dropout}))
    # The upstream embedding constructor allocates CUDA tensors by default.
    model = LanguageModel(config).to(device)
    model.backbone.embeddings.device = torch.device(device)
    for mixer in mixers(model):
        mixer.group = group
        with torch.no_grad():
            for layer in (mixer.memory.key_summary, mixer.memory.value_summary):
                layer.weight.copy_(torch.eye(64, device=device))
    return model


def mixers(model):
    return [m for m in model.modules() if isinstance(m, MemoryMHA)]


def digest_tensors(*tensors):
    sha = hashlib.sha256()
    for tensor in tensors:
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def dataset(length, pairs, examples, seed):
    torch.manual_seed(seed)
    data = multiquery_ar(vocab_size=2048, num_examples=examples,
                         input_seq_len=length, num_kv_pairs=pairs, seed=seed)
    assert data.inputs.shape == data.labels.shape == (examples, length)
    assert torch.all((data.labels != -100).sum(-1) == pairs)
    return data.inputs, data.labels


@torch.no_grad()
def evaluate(model, x, y, batch=32):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    device = next(model.parameters()).device
    for start in range(0, len(x), batch):
        xb, yb = x[start:start + batch].to(device), y[start:start + batch].to(device)
        logits = model(xb)
        live = yb != -100
        loss += F.cross_entropy(logits[live], yb[live], reduction="sum").item()
        correct += (logits.argmax(-1)[live] == yb[live]).sum().item()
        count += live.sum().item()
    return {"query_ce": loss / count, "query_accuracy": correct / count, "query_count": count}


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


@torch.no_grad()
def diagnostics(model, run):
    """Budget/context changes are OOD diagnostics, not budget-trained Pareto points."""
    modules = mixers(model)
    group = modules[0].group
    for length in (512, 1024):
        for pairs in (16, 32, 64):
            x, y = dataset(length, pairs, 128, 3123 + length + pairs)
            for budget in ((1,) if group == "F" else (0, 1, 3, 7)):
                for m in modules:
                    m.k_history = budget
                row = {"kind": "ood_budget_curve", "context": length, "kv_pairs": pairs,
                       "k_history": budget, **evaluate(model, x, y)}
                # Logical counts and local error on exactly the same captured rows.
                xb = x[:8].to(next(model.parameters()).device)
                for m in modules:
                    m.capture = True
                model(xb)
                counts = {key: 0 for key in ("detail_pairs", "summary_pairs", "router_key_vectors")}
                errors = []
                positions = torch.linspace(0, length - 1, 16, device=xb.device).long()
                for m in modules:
                    q, k, v = m.last_qkv
                    aq = q[:, :, positions]
                    target = dense_reference(aq, k, v, positions)
                    pred, stats = m.memory(aq, k, v, positions, budget, group != "M")
                    if group == "F":
                        pred = target
                        stats = {"detail_pairs": ((positions + 1).sum() * 8 * 2),
                                 "summary_pairs": torch.tensor(0), "router_key_vectors": torch.tensor(0)}
                    errors.append(relative_mse(pred, target).item())
                    for key in counts:
                        counts[key] += stats[key].item()
                    m.last_qkv = None
                    m.capture = False
                row.update(logical_sample_counts=counts, layer_relative_mse=errors,
                           physical_hbm_bytes=None, system_latency=None,
                           dataset_sha256=digest_tensors(x, y))
                append(run / "diagnostics.jsonl", row)
    for m in modules:
        m.k_history = 1


def train(args):
    run = args.output
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_x, train_y = dataset(512, 32, args.train_examples, 1123)
    valid_x, valid_y = dataset(512, 32, 1000, 2123)
    model = build_model(args.group, args.seed, "cuda", args.dropout)
    modules = mixers(model)
    initial_hash = digest_tensors(*model.state_dict().values())
    metadata = {"group": args.group, "seed": args.seed, "steps": args.steps,
                "lr": args.lr, "batch": 32, "aux_weight": 0.1, "aux_interval": 8,
                "train_examples": args.train_examples, "dropout": args.dropout,
                "aux_queries": 16, "aux_layers": 1, "dtype": "float32",
                "initial_state_sha256": initial_hash,
                "train_sha256": digest_tensors(train_x, train_y),
                "valid_sha256": digest_tensors(valid_x, valid_y),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "parameters": sum(p.numel() for p in model.parameters()),
                "torchvision": __import__("torchvision").__version__,
                "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "upstream_commit": "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
                "timing_scope": "training wall-time cost including validation; not kernel speed",
                "packing": False, "incremental_cache": False}
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2))
    torch.save(model.state_dict(), run / "initial.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    order_rng = torch.Generator().manual_seed(4123)
    aux_rng = torch.Generator().manual_seed(5123)
    order, offset, best_ce = torch.randperm(len(train_x), generator=order_rng), 0, math.inf
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        if offset + 32 > len(order):
            order, offset = torch.randperm(len(train_x), generator=order_rng), 0
        ids = order[offset:offset + 32]
        offset += 32
        xb, yb = train_x[ids].cuda(), train_y[ids].cuda()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        selected_layer, sampled_positions = None, None
        for m in modules:
            m.aux_positions = None
        if args.group in ("C", "D") and step % 8 == 0:
            selected_layer = int(torch.randint(len(modules), (1,), generator=aux_rng))
            sampled_positions = torch.randperm(512, generator=aux_rng)[:16].sort().values
            modules[selected_layer].aux_positions = sampled_positions.cuda()
        logits = model(xb)
        task_loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten())
        aux = sum((m.auxiliary_loss for m in modules if m.auxiliary_loss is not None),
                  torch.zeros((), device="cuda"))
        loss = task_loss + 0.1 * aux
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite gradient norm at step {step}")
        optimizer.step()
        # Drop references to the auxiliary graph before validation/checkpointing.
        for m in modules:
            m.aux_positions = None
            m.auxiliary_loss = None
        append(run / "train.jsonl", {"step": step, "task_loss": task_loss.item(),
                                   "aux_loss": aux.item(), "grad_norm_preclip": norm.item(),
                                   "aux_layer": selected_layer,
                                   "aux_positions": sampled_positions.tolist() if sampled_positions is not None else None,
                                   "batch_ids_sha256": digest_tensors(ids)})
        if step == 1 or step % 250 == 0 or step == args.steps:
            metrics = evaluate(model, valid_x, valid_y)
            torch.cuda.synchronize()
            metrics.update(step=step, training_tokens=step * 32 * 512,
                           wall_seconds=time.monotonic() - start,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated())
            append(run / "metrics.jsonl", metrics)
            if metrics["query_ce"] < best_ce:
                best_ce = metrics["query_ce"]
                torch.save({"model": model.state_dict(), "step": step, "metrics": metrics}, run / "best.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": args.steps,
                "order": order, "offset": offset, "order_rng": order_rng.get_state(),
                "aux_rng": aux_rng.get_state(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, run / "last.pt")
    best = torch.load(run / "best.pt", weights_only=False)
    model.load_state_dict(best["model"])
    diagnostics(model, run)
    result = {"status": "completed", "group": args.group, "best_validation": best["metrics"],
              "steps": args.steps, "initial_state_sha256": initial_hash,
              "wall_seconds": time.monotonic() - start,
              "claims": "one-seed synthetic MQAR screening only; no GPU-speed claim"}
    (run / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list("FMBCD"), default="F")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-examples", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("runs/F_seed123"))
    train(parser.parse_args())
