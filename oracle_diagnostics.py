"""Exact k<=1 fixed-QKV/fixed-summary candidate enumeration, no task oracle."""
import math
import torch
import torch.nn.functional as F
from block_memory_reference import _validate, dense_reference


@torch.no_grad()
def candidates(memory, q, k, v, positions):
    """One original sample. Candidate index 0=empty, i+1=historical block i.

    Invalid future candidates are masked from minimization. No upstream rerun.
    Candidate-axis vectorization is checked against the loop reference.
    """
    _validate(q, k, v, positions)
    if q.shape[0] != 1:
        raise ValueError("Per-sample positions require exactly one sample here")
    _, heads, queries, dim = q.shape
    length, block, slots = k.shape[2], memory.block_size, memory.summary_slots
    width, closed = block // slots, positions // block
    n = int(closed.max())
    ids = torch.arange(length, device=q.device)
    blocks = torch.arange(n, device=q.device)
    candidate = torch.arange(n + 1, device=q.device)
    selected = ((candidate[:, None, None] - 1) == blocks[None, None, :])
    selected = selected.expand(-1, queries, -1) & (blocks[None, None, :] < closed[None, :, None])
    valid = (candidate[:, None] == 0) | (candidate[:, None] <= closed[None, :])
    exact = ((ids[None, None, :] // block == closed[None, :, None])
             | F.pad(selected, (0, length // block + 1 - n))[..., ids // block])
    exact &= ids[None, None, :] <= positions[None, :, None]
    pool_shape = (1, heads, n, slots, width, dim)
    pk = k[:, :, :n * block].float().reshape(pool_shape).mean(-2)
    pv = v[:, :, :n * block].float().reshape(pool_shape).mean(-2)
    sk = F.linear(pk, memory.key_summary.weight.float()).flatten(2, 3)
    sv = F.linear(pv, memory.value_summary.weight.float()).flatten(2, 3)
    detail = q.float() @ k.float().transpose(-1, -2) * dim ** -0.5
    summary = q.float() @ sk.transpose(-1, -2) * dim ** -0.5 + math.log(width)
    coarse = (blocks[None, None, :] < closed[None, :, None]) & ~selected
    logits = torch.cat((detail.expand(n + 1, -1, -1, -1).masked_fill(~exact[:, None], -torch.inf),
                        summary.expand(n + 1, -1, -1, -1).masked_fill(
                            ~coarse.repeat_interleave(slots, -1)[:, None], -torch.inf)), -1)
    values = torch.cat((v.float(), sv), 2)
    output = logits.softmax(-1) @ values
    stats = dict(detail_pairs=exact.sum(-1), summary_pairs=coarse.sum(-1) * slots,
                 router_key_vectors=closed)
    return output, valid, stats


@torch.no_grad()
def exact_oracle(memory, q, k, v, positions, projection):
    output, valid, counts = candidates(memory, q, k, v, positions)
    if q.shape[1] != 2 or projection.shape != (2 * q.shape[-1], 2 * q.shape[-1]):
        raise ValueError("Joint diagnostic requires two heads and matching projection")
    target = dense_reference(q, k, v, positions)[0]
    actual, stats = memory(q, k, v, positions, 1)
    chosen = stats['selected'][0].to(torch.int64)
    actual_ids = (chosen * (torch.arange(chosen.shape[-1], device=q.device) + 1)).sum(-1)
    error = (output - target[None]).square().mean(-1)
    energy = target.square().mean(-1)
    relative = error / energy.clamp_min(1e-8)
    error_masked = error.masked_fill(~valid[:, None], torch.inf)
    head_best = error_masked.argmin(0)
    n, heads, queries, dim = output.shape
    # Enumerate all head pairs, not just independently optimal heads.
    h0 = output[:, 0, None].expand(n, n, queries, dim)
    h1 = output[None, :, 1].expand(n, n, queries, dim)
    difference = torch.cat((h0 - target[0], h1 - target[1]), -1)
    projected = F.linear(difference, projection.float())
    joint_error = projected.square().mean(-1)
    joint_valid = valid[:, None] & valid[None, :]
    joint_error = joint_error.masked_fill(~joint_valid, torch.inf)
    flat = joint_error.reshape(n * n, queries)
    best = flat.argmin(0)
    joint_ids = torch.stack((best // n, best % n))
    qi = torch.arange(queries, device=q.device)
    best_output = torch.stack([output[joint_ids[h], h, qi] for h in range(2)])
    actual_projection = F.linear((actual[0] - target).transpose(0, 1).flatten(1), projection.float())
    projected_target = F.linear(target.transpose(0, 1).flatten(1), projection.float())
    projected_energy = projected_target.square().mean(-1)
    best_error = flat[best, qi]
    actual_error = actual_projection.square().mean(-1)
    # Fail closed on semantic or numerical violations, using the test contract.
    replay = torch.stack([output[actual_ids[h], h, qi] for h in range(2)])
    torch.testing.assert_close(replay, actual[0], atol=3e-6, rtol=3e-5)
    if not torch.all(best_error <= actual_error + 3e-6 + 3e-5 * actual_error):
        raise AssertionError("Actual router is not inside the oracle candidate family")
    return dict(output=output, valid=valid, counts=counts, target=target,
                head_absolute=error, head_relative=relative, head_energy=energy,
                head_best=head_best, actual_ids=actual_ids, joint_ids=joint_ids,
                best_output=best_output, joint_absolute=best_error,
                joint_relative=best_error / projected_energy.clamp_min(1e-8),
                actual_absolute=actual_error,
                actual_relative=actual_error / projected_energy.clamp_min(1e-8),
                joint_energy=projected_energy,
                zero_absolute=joint_error[0, 0],
                zero_relative=joint_error[0, 0] / projected_energy.clamp_min(1e-8))
