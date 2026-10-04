"""The exact oracle must be at-most-k, monotone, and bounded by routing error."""
import unittest
import torch
from block_memory_reference import BlockMemoryReference
from oracle_diagnostic import exact_curve


class OracleTest(unittest.TestCase):
    def test_all_subsets_and_dense_endpoint(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        module = BlockMemoryReference(8, 4, 2)
        row = exact_curve(module, torch.randn(1, 1, 1, 8),
                          torch.randn(1, 1, 33, 8), torch.randn(1, 1, 33, 8), 32)
        self.assertEqual(row["enumerated_subsets"], 256)
        errors = [point["oracle_relative_mse"] for point in row["points"]]
        self.assertEqual(errors, sorted(errors, reverse=True))
        for point in row["points"]:
            self.assertLessEqual(len(point["oracle_set"]), point["at_most_k"])
            self.assertLessEqual(point["oracle_relative_mse"], point["router_relative_mse"] + 1e-5)

    def test_larger_search_is_not_called_exact(self):
        module = BlockMemoryReference(8, 4, 2)
        with self.assertRaises(ValueError):
            exact_curve(module, torch.randn(1, 1, 1, 8),
                        torch.randn(1, 1, 37, 8), torch.randn(1, 1, 37, 8), 36)


if __name__ == "__main__":
    unittest.main()
