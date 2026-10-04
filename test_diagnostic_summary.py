"""NumPy-only regressions for cluster uncertainty and row identity validation."""
import unittest
from summarize_diagnostics import paired_interval, validate_ids


class DiagnosticSummaryTests(unittest.TestCase):
    def test_equal_paired_samples(self):
        values = {i: (i + 1) / 256 for i in range(256)}
        value, interval = paired_interval(values, values, 'accuracy_pp')
        self.assertEqual((value, interval), (0., [0., 0.]))
        value, interval = paired_interval(values, values, 'ce_ratio')
        self.assertEqual((value, interval), (1., [1., 1.]))

    def test_missing_mismatched_sample_ids_fail_closed(self):
        values = {i: 1. for i in range(256)}
        with self.assertRaises(ValueError): paired_interval(values, {**values, 256: 1.}, 'accuracy_pp')
        with self.assertRaises(ValueError): paired_interval(values, {i: 1. for i in range(255)}, 'ce_ratio')

    def test_declared_denominators_and_query_schema(self):
        route = [dict(group=g, sample_id=s, query_position=p) for g in 'MBCD' for s in range(2) for p in (64, 66)]
        oracle = [dict(group=g, sample_id=s, query_position=(64 if kind == 'query' else 65), kind=kind, layer=layer)
                  for g in 'BCD' for s in range(2) for kind in ('query', 'nonquery') for layer in range(2)]
        validate_ids(route, oracle, examples=2, queries=2, sampled=1)
        with self.assertRaises(ValueError): validate_ids(route[:-1], oracle, 2, 2, 1)
        changed = [dict(r) for r in oracle]; changed[0]['query_position'] = 66
        with self.assertRaises(ValueError): validate_ids(route, changed, 2, 2, 1)
        changed = [dict(r) for r in oracle]
        for r in changed:
            if r['kind'] == 'nonquery': r['query_position'] = 66
        with self.assertRaises(ValueError): validate_ids(route, changed, 2, 2, 1)


if __name__ == '__main__':
    unittest.main()
