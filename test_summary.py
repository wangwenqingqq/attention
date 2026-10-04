"""Fail closed on unmatched groups and keep the predeclared paired estimator."""
import copy
import math
import unittest
from summarize_screen import paired_interval, validate_contract, validate_replicates, validate_history


class SummaryTest(unittest.TestCase):
    def test_complete_history_denominator_and_checkpoint_match(self):
        rows = [{"step": step, "query_count": 32000, "query_ce": 1 / step,
                 "query_accuracy": 0.5, "training_tokens": step * 32 * 512}
                for step in [1, *range(250, 40001, 250)]]
        validate_history(rows, rows[-1])
        with self.assertRaises(AssertionError):
            validate_history(rows[1:], rows[-1])
        bad = copy.deepcopy(rows)
        bad[0]["query_count"] = 100
        with self.assertRaises(AssertionError):
            validate_history(bad, bad[-1])
        best = {**rows[-1], "query_accuracy": 0.6}
        with self.assertRaises(AssertionError):
            validate_history(rows, best)

    def test_cross_seed_data_order_and_source_must_match(self):
        records = [{"seed": seed, "scientific_source_sha256": {"trainer": "same"},
                    "groups": {"F": {"train_sha256": "same", "valid_sha256": "same",
                                     "batch_order_sha256": "same", "initial_state_sha256": str(seed)},
                               "C": {"aux_schedule_sha256": "same"}}}
                   for seed in (123, 124, 125)]
        validate_replicates(records)
        for key in ("train_sha256", "valid_sha256", "batch_order_sha256", "initial_state_sha256"):
            bad = copy.deepcopy(records)
            bad[1]["groups"]["F"][key] = bad[0]["groups"]["F"][key] if key == "initial_state_sha256" else "different"
            with self.assertRaises(AssertionError):
                validate_replicates(bad)
        bad = copy.deepcopy(records)
        bad[1]["scientific_source_sha256"]["trainer"] = "different"
        with self.assertRaises(AssertionError):
            validate_replicates(bad)
        bad = copy.deepcopy(records)
        bad[1]["groups"]["C"]["aux_schedule_sha256"] = "different"
        with self.assertRaises(AssertionError):
            validate_replicates(bad)

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
