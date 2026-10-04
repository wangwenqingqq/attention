"""Batched correctness/training backend, deliberately NOT a sparse GPU kernel.

All detail logits are computed densely then masked. Logical reads describe the
attention semantics, NOT actual HBM traffic, FLOPs, latency or KV storage.
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from zoology.mixers.attention import MHA
from block_memory_reference import BlockMemoryReference, _validate, dense_reference, relative_mse


class BatchedMemory(BlockMemoryReference):
    def forward(self, q, k, v, positions, k_history=1, summaries=True, dropout_p=0.0,
                forced_selection=None, diagnostic_mode=False):
        _validate(q, k, v, positions)
        if k_history < 0 or q.shape[-1] != self.head_dim:
            raise ValueError("Invalid budget or head dimension")
        if q.device != self.key_summary.weight.device:
            raise ValueError("Module/input device mismatch")
        batch, heads, queries, dim = q.shape
        length, block, slots = k.shape[2], self.block_size, self.summary_slots
        closed_blocks, width = length // block, block // slots
        shape = (batch, heads, closed_blocks, slots, width, dim)
        pk = k[:, :, :closed_blocks * block].float().reshape(shape).mean(-2)
        pv = v[:, :, :closed_blocks * block].float().reshape(shape).mean(-2)
        block_ids = torch.arange(closed_blocks, device=q.device)
        query_block = positions // block
        visible = block_ids[None, :] < query_block[:, None]
        with torch.no_grad():
            scores = q.detach().float() @ pk.detach().mean(-2).transpose(-1, -2)
            scores = scores.masked_fill(~visible, -torch.inf)
            selected = torch.zeros_like(scores, dtype=torch.bool)
            count = min(k_history, closed_blocks)
            if count:
                values, indices = scores.topk(count, dim=-1)
                selected.scatter_(-1, indices, torch.isfinite(values))
        if forced_selection is not None:
            if not diagnostic_mode or dropout_p or self.training:
                raise ValueError("Forced selections require diagnostic evaluation mode")
            pos_to_index = {p: j for j, p in enumerate(positions.tolist())}
            if len(pos_to_index) != queries:
                raise ValueError("Forced positions must be unique")
            for (b, h, position), blocks in forced_selection.items():
                chosen = tuple(blocks)
                if (not 0 <= b < batch or not 0 <= h < heads
                        or position not in pos_to_index
                        or len(chosen) > min(k_history, position // block)
                        or len(set(chosen)) != len(chosen)
                        or any(not isinstance(i, int) or i < 0 or i >= position // block
                               for i in chosen)):
                    raise ValueError("Invalid forced historical selection")
                j = pos_to_index[position]
                selected[b, h, j] = False
                if chosen:
                    selected[b, h, j, list(chosen)] = True
        token_ids = torch.arange(length, device=q.device)
        token_blocks = token_ids // block
        # One sentinel column covers the final incomplete current block.
        table = F.pad(selected, (0, 1), value=False)
        exact = table[..., token_blocks] | (token_blocks[None, :] == query_block[:, None])
        exact = exact & (token_ids[None, :] <= positions[:, None])
        logits = (q.float() @ k.float().transpose(-1, -2)) * dim ** -0.5
        logits = logits.masked_fill(~exact, -torch.inf)
        coarse = visible & ~selected
        if summaries:
            sk = F.linear(pk, self.key_summary.weight.float()).flatten(2, 3)
            sv = F.linear(pv, self.value_summary.weight.float()).flatten(2, 3)
            sl = (q.float() @ sk.transpose(-1, -2)) * dim ** -0.5 + math.log(width)
            sl = sl.masked_fill(~coarse.repeat_interleave(slots, -1), -torch.inf)
            weights = F.dropout(torch.cat((logits, sl), -1).softmax(-1), dropout_p)
            output = weights[..., :length] @ v.float() + weights[..., length:] @ sv
        else:
            output = F.dropout(logits.softmax(-1), dropout_p) @ v.float()
        return output.to(q.dtype), {
            "selected": selected,
            "detail_pairs": exact.sum(),
            "summary_pairs": coarse.sum() * slots if summaries else coarse.sum() * 0,
            "router_key_vectors": visible.sum() * batch * heads,
        }


def segmented_memory(module, q, k, v, lengths, k_history=1):
    """Explicitly separate packed documents; positions and blocks reset per doc.

    The MQAR trainer uses unpacked sequences. This correctness helper is not an
    optimized packed implementation or an incremental KV-cache interface.
    """
    if q.shape != k.shape or k.shape != v.shape or any(n <= 0 for n in lengths):
        raise ValueError("Packing helper requires positive full-query documents")
    if sum(lengths) != q.shape[2]:
        raise ValueError("Document lengths must exactly partition the sequence")
    outputs, offset = [], 0
    for size in lengths:
        sl = slice(offset, offset + size)
        positions = torch.arange(size, device=q.device)
        outputs.append(module(q[:, :, sl], k[:, :, sl], v[:, :, sl], positions, k_history)[0])
        offset += size
    return torch.cat(outputs, 2)


class MemoryMHA(MHA):
    """Zoology-compatible sequence mixer, shared parameter names for all groups."""
    def __init__(self, d_model, num_heads=2, layer_idx=None, **kwargs):
        super().__init__(d_model, num_heads, layer_idx=layer_idx, **kwargs)
        self.memory = BatchedMemory(self.head_dim, 32, 2)
        self.group, self.k_history = "F", 1
        self.aux_positions = None
        self.auxiliary_loss = None
        self.last_qkv = None
        self.capture = False
        self.last_selected = self.last_output = None
        self.forced_selection = self.output_intervention = None
        self.diagnostic_mode = False

    def forward(self, x):
        qkv = rearrange(self.Wqkv(x), "b t (three h d) -> b t three h d",
                        three=3, h=self.num_heads)
        q, k, v = (y.transpose(1, 2) for y in qkv.unbind(2))
        positions = torch.arange(x.shape[1], device=x.device)
        if self.forced_selection is not None or self.output_intervention is not None:
            if not self.diagnostic_mode or self.training or self.group == "F":
                raise ValueError("Interventions require sparse diagnostic evaluation mode")
        selected = None
        if self.group == "F":
            # Preserve the upstream full-attention implementation in the screen.
            out = self.inner_attn(qkv).transpose(1, 2)
        else:
            out, stats = self.memory(q, k, v, positions, self.k_history, self.group != "M",
                                     self.inner_attn.dropout_p if self.training else 0.0,
                                     self.forced_selection, self.diagnostic_mode)
            selected = stats["selected"]
        if self.output_intervention is not None:
            out = out.clone()
            for (b, h, position), value in self.output_intervention.items():
                if (not 0 <= b < out.shape[0] or not 0 <= h < out.shape[1]
                        or not 0 <= position < out.shape[2] or value.shape != (self.head_dim,)
                        or value.device != out.device or value.dtype != out.dtype
                        or not torch.isfinite(value).all()):
                    raise ValueError("Invalid head output intervention")
                out[b, h, position] = value
        self.auxiliary_loss = None
        if self.aux_positions is not None:
            if self.group not in ("C", "D"):
                raise ValueError("Auxiliary sampling is only valid for C/D")
            pos = self.aux_positions
            aq = q[:, :, pos]
            with torch.no_grad():
                target = dense_reference(aq, k, v, pos)
            if self.group == "C":
                aq, ak, av = aq.detach(), k.detach(), v.detach()
            else:
                ak, av = k, v
            prediction, _ = self.memory(aq, ak, av, pos, self.k_history)
            self.auxiliary_loss = relative_mse(prediction, target)
        self.last_qkv = (q, k, v) if self.capture else None
        self.last_selected = selected if self.capture else None
        self.last_output = out if self.capture else None
        return self.out_proj(rearrange(out, "b h t d -> b t (h d)"))
