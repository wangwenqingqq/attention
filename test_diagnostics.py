"""Diagnostic legality, semantic reference parity and state-isolation gates."""
import copy
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from analyze_mqar import preserved
from block_memory_reference import BlockMemoryReference, dense_reference
from memory_attention import BatchedMemory
from mqar_metadata import source_rows, sampled_positions
from oracle_diagnostics import candidates, exact_oracle
from train_mqar import build_model, dataset, digest_tensors, mixers


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)
        torch.set_num_threads(1)
        self.m = BatchedMemory(8, 4, 2).eval()
        self.k, self.v = torch.randn(1, 2, 13, 8), torch.randn(1, 2, 13, 8)
        self.pos = torch.tensor([0, 4, 7, 12])
        self.q = torch.randn(1, 2, 4, 8)

    def test_source_metadata_preserves_hash_rng_and_ignores_noise_collisions(self):
        with preserved():
            x, y = dataset(512, 32, 4, 9123)
        # A random nonquery repeats a real key; only the original prefix is valid.
        x[:, 65] = x[:, 0]
        before, rng = digest_tensors(x, y), torch.get_rng_state().clone()
        rows = source_rows(x, y)
        self.assertEqual(len(rows), 128)
        for row in rows:
            s, p = row['sample_id'], row['query_position']
            self.assertLess(row['source_key_position'], 64)
            self.assertEqual(int(x[s, row['source_key_position']]), int(x[s, p]))
        self.assertEqual(before, digest_tensors(x, y))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        samples = sampled_positions(y, torch.Generator().manual_seed(10123))
        self.assertTrue(all(len(q) == len(n) == 4 for q, n in samples))
        broken = x.clone(); broken[0, 2] = broken[0, 0]
        with self.assertRaises(ValueError): source_rows(broken, y)

    def test_none_and_replay_exact_original(self):
        original, stats = self.m(self.q, self.k, self.v, self.pos)
        none = self.m(self.q, self.k, self.v, self.pos, forced_selection=None)[0]
        self.assertTrue(torch.equal(none, original))
        forced = {(0, h, int(p)): torch.where(stats['selected'][0, h, j])[0].tolist()
                  for h in range(2) for j, p in enumerate(self.pos)}
        replay = self.m(self.q, self.k, self.v, self.pos, forced_selection=forced, diagnostic_mode=True)[0]
        self.assertTrue(torch.equal(replay, original))

    def test_invalid_future_duplicate_overbudget_or_current(self):
        for key, value in [((0, 0, 4), [1]), ((0, 0, 7), [0, 0]),
                           ((0, 0, 12), [0, 1]), ((0, 0, 12), [3]),
                           ((1, 0, 12), [0]), ((0, 2, 12), [0]), ((0, 0, 11), [0])]:
            with self.assertRaises(ValueError):
                self.m(self.q, self.k, self.v, self.pos, forced_selection={key: value}, diagnostic_mode=True)

    def test_summary_replacement_full_refinement_and_absolute_qlen1(self):
        q = self.q[:, :, -1:]; pos = self.pos[-1:]
        forced = {(0, h, 12): [0, 1, 2] for h in range(2)}
        out, stats = self.m(q, self.k, self.v, pos, 3, forced_selection=forced, diagnostic_mode=True)
        torch.testing.assert_close(out, dense_reference(q, self.k, self.v, pos), atol=3e-6, rtol=3e-5)
        self.assertEqual(int(stats['detail_pairs']), 26)
        self.assertEqual(int(stats['summary_pairs']), 0)
        _, stats = self.m(q, self.k, self.v, pos, 1, forced_selection={(0, h, 12): [0] for h in range(2)}, diagnostic_mode=True)
        self.assertEqual(int(stats['detail_pairs']), 10)
        self.assertEqual(int(stats['summary_pairs']), 8)

    def test_every_candidate_matches_loop_and_actual_is_in_family(self):
        self.m.key_summary.weight.data.normal_()
        self.m.value_summary.weight.data.normal_()
        outputs, valid, counts = candidates(self.m, self.q, self.k, self.v, self.pos)
        ref = BlockMemoryReference(8, 4, 2).eval(); ref.load_state_dict(self.m.state_dict())
        for i in range(outputs.shape[0]):
            forced = {(0, h, j): ([] if i == 0 or not valid[i, j] else [i-1])
                      for h in range(2) for j in range(len(self.pos))}
            target, _ = ref(self.q, self.k, self.v, self.pos, 1, forced)
            torch.testing.assert_close(outputs[i:i+1], target, atol=3e-6, rtol=3e-5)
        actual, stats = self.m(self.q, self.k, self.v, self.pos)
        for h in range(2):
            for j in range(len(self.pos)):
                indices = torch.where(stats['selected'][0, h, j])[0]
                i = int(indices[0])+1 if len(indices) else 0
                torch.testing.assert_close(outputs[i, h, j], actual[0, h, j], atol=3e-6, rtol=3e-5)

    def test_atmost_one_beats_zero_and_can_choose_empty(self):
        # Identical K/V inside each block makes summaries exact at budget zero.
        self.k.zero_(); self.v.fill_(1)
        result = exact_oracle(self.m, self.q, self.k, self.v, self.pos, torch.eye(16))
        self.assertTrue(torch.all(result['joint_absolute'] <= result['zero_absolute'] + 3e-6))
        self.assertTrue((result['joint_ids'][:, 0] == 0).all())

    def test_joint_oracle_beats_independently_optimal_combination(self):
        projection = torch.randn(16, 16)
        result = exact_oracle(self.m, self.q, self.k, self.v, self.pos, projection)
        best = torch.stack([result['output'][result['head_best'][h], h, torch.arange(4)] for h in range(2)])
        error = F.linear((best-result['target']).transpose(0, 1).flatten(1), projection).square().mean(-1)
        self.assertTrue(torch.all(result['joint_absolute'] <= error + 3e-6))
        torch.testing.assert_close(result['joint_energy'], F.linear(result['target'].transpose(0, 1).flatten(1), projection).square().mean(-1))

    def test_last_layer_output_patch_is_query_and_head_local(self):
        model = build_model('D', 123, 'cuda', .1)
        ids = torch.randint(2048, (1, 96), device='cuda')
        with preserved(model), torch.no_grad():
            baseline = model(ids).clone()
            first = mixers(model)[0].last_output.clone()
            last = mixers(model)[-1]
            original = last.last_output.clone()
            last.output_intervention = {(0, 1, 70): torch.zeros(64, device='cuda')}
            changed = model(ids)
            mask = torch.arange(96, device='cuda') != 70
            torch.testing.assert_close(changed[:, mask], baseline[:, mask], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(mixers(model)[0].last_output, first)
            torch.testing.assert_close(last.last_output[:, 0], original[:, 0])
            self.assertFalse(torch.equal(changed[:, 70], baseline[:, 70]))
            # Earlier-layer patches can propagate to later queries; no locality claim.
            last.output_intervention = None
            mixers(model)[0].output_intervention = {(0, 0, 32): torch.zeros(64, device='cuda')}
            early = model(ids)
            self.assertFalse(torch.equal(early[:, 70], baseline[:, 70]))

    def test_model_state_config_and_caller_rng_are_restored(self):
        model = build_model('C', 123, 'cuda', .1).train()
        before = digest_tensors(*model.state_dict().values())
        cpu, gpu, numpy = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone(), np.random.get_state()
        with preserved(model), torch.no_grad():
            mixers(model)[0].k_history = 0
            model(torch.randint(2048, (1, 96), device='cuda'))
            np.random.rand()
        self.assertTrue(model.training)
        self.assertEqual(mixers(model)[0].k_history, 1)
        self.assertFalse(mixers(model)[0].capture)
        self.assertFalse(mixers(model)[0].diagnostic_mode)
        self.assertEqual(before, digest_tensors(*model.state_dict().values()))
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertTrue(torch.equal(gpu, torch.cuda.get_rng_state()))
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_deploy_mode_rejects_truth_oracle_interventions(self):
        with self.assertRaises(ValueError):
            self.m(self.q, self.k, self.v, self.pos, forced_selection={(0, 0, 12): [0]})
        model = build_model('D', 123, 'cuda').eval()
        last = mixers(model)[-1]
        last.output_intervention = {(0, 1, 70): torch.zeros(64, device='cuda')}
        with self.assertRaises(ValueError): model(torch.zeros(1, 96, dtype=torch.long, device='cuda'))


if __name__ == '__main__':
    unittest.main()
