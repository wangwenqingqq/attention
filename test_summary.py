"""Fail closed on unmatched groups and keep the predeclared paired estimator."""
import copy
import math
import unittest
from summarize_screen import paired_interval, validate_contract


class SummaryTest(unittest.TestCase):
    def test_registered_three_seed_interval(self):
        row = paired_interval([-1, 0, 1])
        self.assertEqual(row["mean"], 0)
        self.assertAlmostEqual(row["ci95"][1], 4.302652729696142 / math.sqrt(3))
        with self.assertRaises(ValueError):
            paired_interval([1, 2])

    def test_sampling_mismatch_is_not_published_as_paired(self):
        row = {key: "same" for key in ("initial_state_sha256", "train_sha256", "valid_sha256",
                                      "batch_order_sha256", "aux_schedule_sha256")}
        row["best_validation"] = {"query_accuracy": 0.99}
        groups = {group: copy.deepcopy(row) for group in "FMBCD"}
        validate_contract(groups)
        for key in ("initial_state_sha256", "batch_order_sha256", "aux_schedule_sha256"):
            bad = copy.deepcopy(groups)
            bad["D"][key] = "different"
            with self.assertRaises(AssertionError):
                validate_contract(bad)


if __name__ == "__main__":
    unittest.main()
