"""CPU correctness checks; no trained-model claims or GPU benchmarks."""
import unittest
import torch
from block_memory_reference import (
    BlockMemoryReference, dense_reference, sampled_auxiliary_loss,
)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def data(self, length=13, dim=8, positions=(0, 3, 4, 7, 8, 12), grad=False):
        q = torch.randn(1, 2, len(positions), dim, requires_grad=grad)
        k = torch.randn(1, 2, length, dim, requires_grad=grad)
        v = torch.randn_like(k).requires_grad_(grad)
        return q, k, v, torch.tensor(positions)

    def test_full_refinement_matches_dense_forward(self):
        q, k, v, pos = self.data()
        module = BlockMemoryReference(8, 4, 2)
        out, info = module(q, k, v, pos, k_history=100)
        torch.testing.assert_close(out, dense_reference(q, k, v, pos), atol=2e-6, rtol=2e-5)
        self.assertEqual(info.summary_pairs, 0)
        self.assertEqual(info.detail_pairs, 2 * int((pos + 1).sum()))

    def test_full_refinement_matches_dense_backward(self):
        q, k, v, pos = self.data(grad=True)
        module = BlockMemoryReference(8, 4, 2)
        out, _ = module(q, k, v, pos, k_history=100)
        upstream = torch.randn_like(out)
        grads = torch.autograd.grad((out * upstream).sum(), (q, k, v))
        ref = dense_reference(q, k, v, pos)
        ref_grads = torch.autograd.grad((ref * upstream).sum(), (q, k, v))
        for actual, expected in zip(grads, ref_grads):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)

    def test_future_kv_cannot_change_earlier_output(self):
        module = BlockMemoryReference(8, 4, 2)
        # Ensure causality still holds after changing learned compressor weights.
        with torch.no_grad():
            module.key_summary.weight.add_(torch.randn(8, 8) * 0.1)
        for position in (0, 2, 3, 4, 6, 7, 8, 11):
            q, k, v, pos = self.data(length=13, positions=(position,))
            before, _ = module(q, k, v, pos, k_history=1)
            k2, v2 = k.clone(), v.clone()
            k2[:, :, position + 1:] = torch.randn_like(k2[:, :, position + 1:]) * 100
            v2[:, :, position + 1:] = torch.randn_like(v2[:, :, position + 1:]) * 100
            after, _ = module(q, k2, v2, pos, k_history=1)
            torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_constant_keys_and_multiplicity_are_exact(self):
        q, k, v, pos = self.data()
        # Different blocks may have different keys. Within each block identical
        # keys make pooling + the multiplicity correction exact at initialization.
        for lo in range(0, k.shape[2], 4):
            k[:, :, lo:lo + 4] = k[:, :, lo:lo + 1].clone()
        module = BlockMemoryReference(8, 4, 2)
        out, _ = module(q, k, v, pos, k_history=0)
        torch.testing.assert_close(out, dense_reference(q, k, v, pos), atol=2e-6, rtol=2e-5)

    def test_zero_history_means_summaries_plus_current_block(self):
        q, k, v, pos = self.data(length=13, positions=(12,))
        _, info = BlockMemoryReference(8, 4, 2)(q, k, v, pos, k_history=0)
        self.assertEqual(info.detail_pairs, 2)  # one local token, two heads
        self.assertEqual(info.summary_pairs, 12)  # 3 blocks * 2 slots * 2 heads
        self.assertEqual(info.router_key_vectors, 6)

    def test_forced_selection_uses_budget_and_replaces_summary(self):
        q, k, v, pos = self.data(length=13, positions=(12,))
        selections = {(0, 0, 0): (1,), (0, 1, 0): (0,)}
        module = BlockMemoryReference(8, 4, 2)
        _, info = module(q, k, v, pos, k_history=1, forced_selection=selections)
        self.assertEqual(info.detail_pairs, 10)  # (4+1) * 2 heads
        self.assertEqual(info.summary_pairs, 8)  # 2 unchosen * 2 slots * 2 heads
        self.assertEqual(info.selected_blocks, selections)
        with self.assertRaises(ValueError):
            module(q, k, v, pos, 1, {(0, 0, 0): (0, 1)})

    def test_C_auxiliary_stops_only_input_gradients(self):
        q, k, v, pos = self.data(length=17, positions=(16,), grad=True)
        module = BlockMemoryReference(8, 4, 2)
        loss = sampled_auxiliary_loss(module, q, k, v, pos, 1, False)
        loss.backward()
        self.assertTrue(all(x.grad is None for x in (q, k, v)))
        self.assertGreater(module.key_summary.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.value_summary.weight.grad.abs().sum().item(), 0)
        # Inputs themselves still require gradients; the normal task path works.
        task_out, _ = module(q, k, v, pos, 1)
        task_out.square().mean().backward()
        self.assertTrue(all(x.grad is not None for x in (q, k, v)))

    def test_D_auxiliary_updates_inputs(self):
        q, k, v, pos = self.data(length=17, positions=(16,), grad=True)
        module = BlockMemoryReference(8, 4, 2)
        loss = sampled_auxiliary_loss(module, q, k, v, pos, 1, True)
        loss.backward()
        self.assertTrue(all(x.grad is not None and x.grad.abs().sum() > 0 for x in (q, k, v)))

    def test_single_partial_block(self):
        q, k, v, pos = self.data(length=3, positions=(0, 1, 2))
        module = BlockMemoryReference(8, 4, 2)
        out, _ = module(q, k, v, pos, 0)
        torch.testing.assert_close(out, dense_reference(q, k, v, pos), atol=2e-6, rtol=2e-5)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            BlockMemoryReference(8, 5, 2)
        q, k, v, pos = self.data()
        module = BlockMemoryReference(8, 4, 2)
        with self.assertRaises(ValueError):
            module(q, k, v, pos, -1)
        with self.assertRaises(ValueError):
            module(q, k, v, torch.full_like(pos, 100), 0)
        with self.assertRaises(ValueError):
            module(q, k[:, :1], v[:, :1], pos, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
