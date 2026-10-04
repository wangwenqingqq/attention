"""Exact at-most-k oracle on at most eight closed blocks; never a GPU benchmark."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from block_memory_reference import BlockMemoryReference, dense_reference, relative_mse


@torch.no_grad()
def exact_curve(module, q, k, v, position):
    if q.shape[:3] != (1, 1, 1) or k.shape[:2] != (1, 1):
        raise ValueError("Oracle operates on one query/head/example at a time")
    closed = position // module.block_size
    if not 0 <= closed <= 8:
        raise ValueError("Exact enumeration is capped at eight closed blocks")
    positions = torch.tensor([position], device=q.device)
    target = dense_reference(q, k, v, positions)
    subsets = []
    for count in range(closed + 1):
        for chosen in itertools.combinations(range(closed), count):
            prediction, _ = module(q, k, v, positions, closed, {(0, 0, 0): chosen})
            subsets.append((relative_mse(prediction, target).item(), chosen))
    points = []
    for budget in range(closed + 1):
        error, chosen = min((row for row in subsets if len(row[1]) <= budget), key=lambda row: row[0])
        prediction, _ = module(q, k, v, positions, budget)
        routed_error = relative_mse(prediction, target).item()
        assert error <= routed_error + 1e-5, "Oracle cannot be worse than an admitted router set"
        points.append({"at_most_k": budget, "oracle_relative_mse": error,
                       "oracle_set": list(chosen), "router_relative_mse": routed_error})
    assert points[-1]["oracle_relative_mse"] < 1e-9, "Full refinement must match dense output"
    return {"position": position, "candidate_history_blocks": closed,
            "enumerated_subsets": len(subsets), "points": points,
            "q_rms": q.square().mean().sqrt().item(), "k_rms": k.square().mean().sqrt().item(),
            "v_rms": v.square().mean().sqrt().item(),
            "k_centered_token_rms": (k - k.mean(-2, keepdim=True)).square().mean().sqrt().item(),
            "v_centered_token_rms": (v - v.mean(-2, keepdim=True)).square().mean().sqrt().item(),
            "dense_output_rms": target.square().mean().sqrt().item()}


def main(args):
    from train_mqar import build_model, dataset, digest_tensors, mixers
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = json.loads((args.run / "metadata.json").read_text())
    model = build_model(metadata["group"], metadata["seed"], "cuda", metadata["dropout"]).eval()
    checkpoint = args.run / "best.pt"
    saved = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(saved["model"])
    x, y = dataset(512, 32, 1, 6123)
    modules = mixers(model)
    for module in modules:
        module.capture = True
    with torch.no_grad():
        model(x.cuda())
    rows = []
    for layer, module in enumerate(modules):
        q, k, v = (tensor.detach().cpu() for tensor in module.last_qkv)
        ref = BlockMemoryReference(64, 32, 2)
        ref.load_state_dict({name: value.detach().cpu() for name, value in module.memory.state_dict().items()})
        for head in range(2):
            for position in (256, 287):
                row = exact_curve(ref, q[:, head:head + 1, position:position + 1],
                                  k[:, head:head + 1, :position + 1],
                                  v[:, head:head + 1, :position + 1], position)
                row.update(layer=layer, head=head)
                rows.append(row)
    result = {"scope": "one held-out example, two layers/heads, positions 256/287; local error only",
              "group": metadata["group"], "seed": metadata["seed"], "data_seed": 6123,
              "dataset_sha256": digest_tensors(x, y), "checkpoint_step": saved["step"],
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "quality_claim": "none; low local error cannot rescue failed task accuracy",
              "rows": rows}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
