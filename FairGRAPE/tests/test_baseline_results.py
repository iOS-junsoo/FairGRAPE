"""Check final-test metric definitions and mask-count denominators without CUDA."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_results import dogscats_metrics, save_dogscats_test_results


class Tensor(np.ndarray):
    """Only the shape, numel and comparison interface is needed by the reporter."""

    def numel(self):
        return self.size


def tensor(values, shape):
    return np.asarray(values).reshape(shape).view(Tensor)


class Model:
    def __init__(self):
        self.layers = {
            "features.0.0": SimpleNamespace(weight=tensor([1, 1], (2, 1, 1, 1)),
                                             mask=tensor([1, 1], (2, 1, 1, 1))),
            "features.1.conv.0.0": SimpleNamespace(weight=tensor([1, 0, 0, 0], (4, 1, 1, 1)),
                                                  mask=tensor([1, 0, 0, 0], (4, 1, 1, 1))),
            "classifier": SimpleNamespace(weight=tensor([1, 1], (1, 2)),
                                           mask=tensor([1, 1], (1, 2))),
        }

    def named_modules(self):
        return self.layers.items()

    def named_parameters(self):
        for name, layer in self.layers.items():
            yield f"{name}.weight", layer.weight
            yield f"{name}.mask", layer.mask
        yield "classifier.bias", tensor([0], (1,))


def predictions():
    # Cell accuracies: 3/4, 1/1, 0/1, 1/2. ACC=5/8, BA=9/16.
    # TPR: 1 vs 1/2; FPR: 1/4 vs 1. EO=(1/2+3/4)/2=5/8.
    return pd.DataFrame({"species": [0, 0, 0, 0, 1, 0, 1, 1],
                         "color": [0, 0, 0, 0, 0, 1, 1, 1],
                         "species_preds_fair": [0, 0, 0, 1, 1, 1, 1, 0]})


class BaselineResultsTests(unittest.TestCase):
    def test_imbalanced_cells_separate_acc_from_group_class_ba(self):
        result = dogscats_metrics(predictions())
        self.assertAlmostEqual(result["ACC"], 5 / 8)
        self.assertAlmostEqual(result["BA"], 9 / 16)
        self.assertAlmostEqual(result["EO"], 5 / 8)
        self.assertEqual(result["ba_valid_cells"], 4)
        self.assertEqual([cell["count"] for cell in result["cells"]], [4, 1, 1, 2])

    def test_missing_cell_has_no_fabricated_eo(self):
        frame = predictions()
        frame = frame.loc[~((frame.color == 1) & (frame.species == 0))]
        result = dogscats_metrics(frame)
        self.assertIsNone(result["EO"])
        self.assertAlmostEqual(result["BA"], (3 / 4 + 1 + 1 / 2) / 3)
        self.assertEqual(result["ba_valid_cells"], 3)
        self.assertEqual(result["cells"][2]["count"], 0)
        self.assertIsNone(result["cells"][2]["accuracy"])

    def test_save_final_test_outputs_and_scope_counts(self):
        frame = predictions()
        args = SimpleNamespace(fg_mode="original", fg_scope="blocks", seed=42,
                               skew_beta=10, prune_rate=.9, no_retrain=True)
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            result = save_dogscats_test_results(frame, Model(), args, tmp_path)
            loaded = json.loads((tmp_path / "final_test_metrics.json").read_text())
            self.assertEqual(loaded, result)
            pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "final_test_predictions.csv"), frame)
            row = pd.read_csv(tmp_path / "final_test_metrics.csv").iloc[0]
        self.assertEqual(row["split"], "test")
        self.assertAlmostEqual(row["ACC"], 5 / 8)
        self.assertIs(loaded["arguments"]["no_retrain"], True)
        counts = result["sparsity"]
        self.assertEqual(counts["total_parameters_excluding_masks"], 9)
        self.assertEqual(counts["remaining_parameters"], 6)
        self.assertAlmostEqual(counts["model_sparsity"], 3 / 9)
        self.assertAlmostEqual(counts["conv_linear_weight_sparsity"], 3 / 8)
        self.assertEqual(counts["target_weights"], 4)
        self.assertAlmostEqual(counts["target_sparsity"], 3 / 4)
        self.assertEqual(counts["target_layer_names"], ["features.1.conv.0.0"])

    def test_all_scope_includes_stem_classifier_and_missing_eo_json(self):
        with tempfile.TemporaryDirectory() as directory:
            result = save_dogscats_test_results(predictions().iloc[:5], Model(), {}, directory,
                                               mode="original", scope="all")
            loaded = json.loads((Path(directory) / "final_test_metrics.json").read_text())
        self.assertEqual(result["sparsity"]["target_weights"], 8)
        self.assertAlmostEqual(result["sparsity"]["target_sparsity"], 3 / 8)
        self.assertIsNone(loaded["EO"])

    def test_rejects_invalid_binary_labels(self):
        for column in ("species", "color", "species_preds_fair"):
            with self.subTest(column=column):
                frame = predictions()
                frame.loc[0, column] = 2
                with self.assertRaisesRegex(ValueError, "binary"):
                    dogscats_metrics(frame)


if __name__ == "__main__":
    unittest.main()
