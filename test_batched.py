"""Regression gates for batched semantics, gradients and Zoology integration."""
import copy
import unittest
import torch

from block_memory_reference import BlockMemoryReference, dense_reference, sampled_auxiliary_loss
from memory_attention import BatchedMemory, MemoryMHA, segmented_memory


class BatchedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_loop_forward_backward_and_routes(self):
        for length in (3, 12, 13):
            for budget in (0, 1, 3, 100):
                q = torch.randn(2, 2, 3, 8, requires_grad=True)
                k = torch.randn(2, 2, length, 8, requires_grad=True)
                v = torch.randn_like(k, requires_grad=True)
                pos = torch.tensor([0, length // 2, length - 1])
                ref, fast = BlockMemoryReference(8, 4, 2), BatchedMemory(8, 4, 2)
                fast.load_state_dict(ref.state_dict())
                a, sa = ref(q, k, v, pos, budget)
                b, sb = fast(q, k, v, pos, budget)
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                self.assertEqual(sa.detail_pairs, sb["detail_pairs"].item())
                self.assertEqual(sa.summary_pairs, sb["summary_pairs"].item())
                self.assertEqual(sa.router_key_vectors, sb["router_key_vectors"].item())
                for row, chosen in sa.selected_blocks.items():
                    self.assertEqual(chosen, tuple(sb["selected"][row].nonzero().flatten().tolist()))
                upstream = torch.randn_like(a)
                ga = torch.autograd.grad((a * upstream).sum(), (q, k, v) + tuple(ref.parameters()), allow_unused=True)
                gb = torch.autograd.grad((b * upstream).sum(), (q, k, v) + tuple(fast.parameters()), allow_unused=True)
                for x, y in zip(ga, gb):
                    if x is None or y is None:
                        if x is not None:
                            self.assertEqual(x.abs().sum().item(), 0)
                        if y is not None:
                            self.assertEqual(y.abs().sum().item(), 0)
                        continue
                    torch.testing.assert_close(x, y, atol=4e-6, rtol=4e-5)

    def test_future_logits_and_routes_and_dense_gradients(self):
        m = BatchedMemory(8, 4, 2)
        q, k, v = (torch.randn(1, 2, 13, 8, requires_grad=True) for _ in range(3))
        pos = torch.arange(13)
        out, stats = m(q, k, v, pos, 1)
        q2, k2, v2 = (x.detach().clone() for x in (q, k, v))
        for x in (q2, k2, v2):
            x[:, :, 8:] = torch.randn_like(x[:, :, 8:]) * 10
        out2, stats2 = m(q2, k2, v2, pos, 1)
        torch.testing.assert_close(out[:, :, :8], out2[:, :, :8])
        self.assertTrue(torch.equal(stats["selected"][:, :, :8], stats2["selected"][:, :, :8]))
        out, _ = m(q, k, v, pos, 100)
        target = dense_reference(q, k, v, pos)
        torch.testing.assert_close(out, target, atol=2e-6, rtol=2e-5)
        for a, b in zip(torch.autograd.grad(out.square().sum(), (q, k, v)),
                        torch.autograd.grad(target.square().sum(), (q, k, v))):
            torch.testing.assert_close(a, b, atol=4e-6, rtol=4e-5)

    def test_decode_absolute_positions(self):
        q, k, v = torch.randn(1, 2, 1, 8), torch.randn(1, 2, 13, 8), torch.randn(1, 2, 13, 8)
        m = BatchedMemory(8, 4, 2)
        out, _ = m(q, k, v, torch.tensor([12]), 100)
        torch.testing.assert_close(out, dense_reference(q, k, v, torch.tensor([12])))

    def test_packing_no_cross_document_gradients(self):
        m = BatchedMemory(8, 4, 2)
        q, k, v = (torch.randn(1, 2, 13, 8, requires_grad=True) for _ in range(3))
        a = segmented_memory(m, q, k, v, [5, 8])
        b = m(q[:, :, 5:], k[:, :, 5:], v[:, :, 5:], torch.arange(8), 1)[0]
        torch.testing.assert_close(a[:, :, 5:], b)
        grads = torch.autograd.grad(a[:, :, 5:].sum(), (q, k, v))
        for g in grads:
            self.assertEqual(g[:, :, :5].abs().sum().item(), 0)

    def test_cd_same_forward_different_auxiliary_gradient(self):
        c = MemoryMHA(16, 2, dropout=0.1)
        d = copy.deepcopy(c)
        c.group, d.group = "C", "D"
        c.aux_positions = d.aux_positions = torch.tensor([64, 95])
        x = torch.randn(1, 96, 16, requires_grad=True)
        torch.manual_seed(99)
        co = c(x)
        torch.manual_seed(99)
        do = d(x)
        torch.testing.assert_close(co, do)
        torch.testing.assert_close(c.auxiliary_loss, d.auxiliary_loss)
        gc = torch.autograd.grad(c.auxiliary_loss, c.Wqkv.weight, allow_unused=True)[0]
        gd = torch.autograd.grad(d.auxiliary_loss, d.Wqkv.weight)[0]
        self.assertIsNone(gc)
        self.assertGreater(gd.abs().sum().item(), 0)
        # The ordinary task path still updates QKV in BOTH groups.
        for module in (c, d):
            module.aux_positions = None
            grad = torch.autograd.grad(module(x).square().mean(), module.Wqkv.weight)[0]
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_model_future_tokens_cannot_change_early_logits(self):
        from train_mqar import build_model
        model = build_model("D", 123).eval()
        ids = torch.randint(2048, (1, 64))
        changed = ids.clone()
        changed[:, 33:] = torch.randint(2048, (1, 31))
        with torch.no_grad():
            torch.testing.assert_close(model(ids)[:, :33], model(changed)[:, :33])

    def test_cuda_training_forward_backward_with_auxiliary(self):
        from train_mqar import build_model, mixers
        model = build_model("D", 123, "cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        ids = torch.randint(2048, (2, 96), device="cuda")
        mixers(model)[0].aux_positions = torch.tensor([64, 95], device="cuda")
        output = model(ids)
        loss = output.square().mean() + 0.1 * mixers(model)[0].auxiliary_loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        optimizer.step()
        model.zero_grad(set_to_none=True)

    def test_moba_matches_official_naive(self):
        from moba_naive import moba_attn_varlen_naive
        q, k, v = (torch.randn(1, 2, 13, 8) for _ in range(3))
        m = BatchedMemory(8, 4, 2)
        out, _ = m(q, k, v, torch.arange(13), 1, False)
        uq, uk, uv = (x[0].transpose(0, 1) for x in (q, k, v))
        official = moba_attn_varlen_naive(uq, uk, uv, torch.tensor([0, 13]), 13, 4, 2)
        torch.testing.assert_close(out[0].transpose(0, 1), official, atol=3e-6, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
