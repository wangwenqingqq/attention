"""Causal block-summary + detail-refinement reference (not an optimized kernel).

This is a research scaffold, NOT the official MoBA/NSA implementation.
Only MHA with equal numbers of Q/K/V heads is supported here.
Layouts: q=[B,H,M,D], k/v=[B,H,T,D], positions=[M] (absolute query positions).
The summary compressor pools each fixed sub-block, then applies a learned
linear map (identity initialized). An unexpanded summary has multiplicity
block_size / summary_slots. A selected block REPLACES its summary.

Python loops/gathers are intentional. Do not use timings from this file to
claim sparse-attention GPU speedups. Logical counters are NOT HBM traffic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReadStats:
    # Sum over all B*H*M query rows (not a unique/shared-memory count).
    detail_pairs: int
    summary_pairs: int
    router_key_vectors: int
    selected_blocks: dict[tuple[int, int, int], tuple[int, ...]]


def _validate(q: Tensor, k: Tensor, v: Tensor, positions: Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have [batch, heads, sequence, dim] layout")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("This reference requires matching MHA heads and dimensions")
    if min(*q.shape, *k.shape) <= 0:
        raise ValueError("All tensor dimensions must be positive")
    if q.device != k.device or k.device != v.device or positions.device != q.device:
        raise ValueError("All inputs, including positions, must be on the same device")
    if not all(x.is_floating_point() for x in (q, k, v)):
        raise ValueError("q/k/v must be floating point")
    if positions.ndim != 1 or positions.numel() != q.shape[2]:
        raise ValueError("positions must contain one absolute position per query")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be an integer tensor")
    if bool(((positions < 0) | (positions >= k.shape[2])).any()):
        raise ValueError("Query position lies outside the KV sequence")


def dense_reference(q: Tensor, k: Tensor, v: Tensor, positions: Tensor) -> Tensor:
    """Explicit FP32 causal attention, including for sampled query positions.

    Do not replace the explicit mask with is_causal=True for non-square
    sampled-query inputs without checking the backend's mask alignment.
    """
    _validate(q, k, v, positions)
    logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    allowed = torch.arange(k.shape[2], device=k.device)[None, :] <= positions[:, None]
    logits = logits.masked_fill(~allowed[None, None, :, :], -torch.inf)
    return (logits.softmax(-1) @ v.float()).to(q.dtype)


def relative_mse(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Local diagnostic only; final model quality must also be evaluated.

    The target and denominator are stop-gradient. This metric alone is not
    invariant to downstream reparameterizations or proof of retained capability.
    """
    reference = target.detach().float()
    squared_error = (prediction.float() - reference).square().mean(-1)
    reference_energy = reference.square().mean(-1).clamp_min(eps)
    return (squared_error / reference_energy).mean()


class BlockMemoryReference(nn.Module):
    def __init__(self, head_dim: int, block_size: int = 128,
                 summary_slots: int = 2) -> None:
        super().__init__()
        if min(head_dim, block_size, summary_slots) <= 0:
            raise ValueError("Dimensions and slot counts must be positive")
        if block_size % summary_slots:
            raise ValueError("block_size must be divisible by summary_slots")
        self.head_dim = head_dim
        self.block_size = block_size
        self.summary_slots = summary_slots
        self.key_summary = nn.Linear(head_dim, head_dim, bias=False)
        self.value_summary = nn.Linear(head_dim, head_dim, bias=False)
        with torch.no_grad():
            self.key_summary.weight.copy_(torch.eye(head_dim))
            self.value_summary.weight.copy_(torch.eye(head_dim))

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, positions: Tensor,
        k_history: int = 3,
        forced_selection: Mapping[tuple[int, int, int], Sequence[int]] | None = None,
    ) -> tuple[Tensor, ReadStats]:
        """Read k_history closed blocks plus the current causal block.

        A fixed, parameter-free router scores q against each closed block's
        raw mean key. Top-k indices receive no gradient. The summary/detail
        output remains differentiable w.r.t. q, k, v and summary parameters.

        forced_selection supplies selected closed-block IDs per (batch,head,
        query-index), for offline oracle/greedy diagnostics. It may use <=k
        blocks. It is never a deployable/free selector.

        This reference rebuilds summaries on each call. A decode implementation
        must cache summaries/centroids of closed blocks and charge their writes.
        """
        _validate(q, k, v, positions)
        if q.shape[-1] != self.head_dim:
            raise ValueError("head_dim does not match the module")
        if k_history < 0:
            raise ValueError("k_history must be nonnegative")
        if q.device != self.key_summary.weight.device:
            raise ValueError("Move the module and inputs to the same device")
        batch, heads, queries, dim = q.shape
        block = self.block_size
        slots = self.summary_slots
        width = block // slots
        full_blocks = k.shape[2] // block

        # Pool in FP32, independently within each block. Future-block summaries
        # may be computed in prefill but cannot be selected/read by an earlier q.
        k_pooled = k[:, :, :full_blocks * block].float().reshape(
            batch, heads, full_blocks, slots, width, dim
        ).mean(-2)
        v_pooled = v[:, :, :full_blocks * block].float().reshape(
            batch, heads, full_blocks, slots, width, dim
        ).mean(-2)
        summary_k = F.linear(k_pooled, self.key_summary.weight.float())
        summary_v = F.linear(v_pooled, self.value_summary.weight.float())
        centroids = k_pooled.mean(-2)
        log_multiplicity = math.log(width)
        scale = dim ** -0.5

        outputs, selections = [], {}
        detail_count = summary_count = router_count = 0
        position_list = positions.detach().cpu().tolist()
        for b in range(batch):
            per_head = []
            for h in range(heads):
                per_query = []
                for j, position in enumerate(position_list):
                    closed = position // block
                    count = min(k_history, closed)
                    row_id = (b, h, j)
                    if forced_selection is not None and row_id in forced_selection:
                        chosen = tuple(sorted(int(x) for x in forced_selection[row_id]))
                        if (len(chosen) > count or len(set(chosen)) != len(chosen)
                                or any(x < 0 or x >= closed for x in chosen)):
                            raise ValueError("Invalid forced closed-block selection")
                    elif count:
                        with torch.no_grad():
                            scores = centroids[b, h, :closed].detach() @ q[b, h, j].detach().float()
                            chosen = tuple(sorted(scores.topk(count).indices.cpu().tolist()))
                    else:
                        chosen = ()
                    selections[row_id] = chosen
                    router_count += closed
                    chosen_set = set(chosen)
                    unchosen = [x for x in range(closed) if x not in chosen_set]

                    # The current block is compulsory, but only its causal prefix.
                    start = closed * block
                    detail_k = [k[b, h, start:position + 1].float()]
                    detail_v = [v[b, h, start:position + 1].float()]
                    for selected_block in chosen:
                        lo, hi = selected_block * block, (selected_block + 1) * block
                        detail_k.append(k[b, h, lo:hi].float())
                        detail_v.append(v[b, h, lo:hi].float())
                    exact_k, exact_v = torch.cat(detail_k), torch.cat(detail_v)
                    keys, values = [exact_k], [exact_v]
                    bias = [torch.zeros(exact_k.shape[0], device=q.device)]
                    detail_count += exact_k.shape[0]
                    if unchosen:
                        coarse_k = summary_k[b, h, unchosen].reshape(-1, dim)
                        coarse_v = summary_v[b, h, unchosen].reshape(-1, dim)
                        keys.append(coarse_k)
                        values.append(coarse_v)
                        bias.append(torch.full((coarse_k.shape[0],), log_multiplicity,
                                               device=q.device))
                        summary_count += coarse_k.shape[0]
                    # One common softmax. No double-counting of selected summaries.
                    all_keys, all_values = torch.cat(keys), torch.cat(values)
                    logits = all_keys @ q[b, h, j].float() * scale + torch.cat(bias)
                    per_query.append(logits.softmax(-1) @ all_values)
                per_head.append(torch.stack(per_query))
            outputs.append(torch.stack(per_head))
        stats = ReadStats(detail_count, summary_count, router_count, selections)
        return torch.stack(outputs).to(q.dtype), stats


def sampled_auxiliary_loss(
    module: BlockMemoryReference, q: Tensor, k: Tensor, v: Tensor,
    positions: Tensor, k_history: int, allow_representation_grad: bool,
) -> Tensor:
    """C/D ablation: stop only the auxiliary input gradient, never the task path.

    In the LM trainer separately compute the normal sparse task loss using
    live q/k/v for BOTH groups. This helper does not include task loss.
    The dense target uses this model's current q/k/v and is stop-gradient;
    it is not a separate frozen pretrained teacher.
    """
    with torch.no_grad():
        target = dense_reference(q, k, v, positions)
    aux_q, aux_k, aux_v = (q, k, v) if allow_representation_grad else (
        q.detach(), k.detach(), v.detach()
    )
    prediction, _ = module(aux_q, aux_k, aux_v, positions, k_history)
    return relative_mse(prediction, target)
