"""Numerical checks for the requested experiment, without CelebA images."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

import celeba_fourway_smoke as smoke


class FourWayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)

    def test_missing_cells_and_requested_fairness(self):
        # Male0 has TPR=1 FPR=0; Male1 TPR=0 FPR=0.5.
        y = np.array([1, 0, 0, 1, 0, 0])[:, None]
        p = np.array([1, 0, 0, 0, 1, 0])[:, None]
        g = np.array([0, 0, 0, 1, 1, 1])
        row = smoke.classification_metrics(y, p, g, ['attr'])[0]
        self.assertEqual(row['EqualOpportunity'], 1.)
        self.assertEqual(row['EqualizedOdds'], 1.)
        self.assertEqual(row['legacy_EO_mean_gap'], .75)
        self.assertEqual(row['DemographicParity'], 0.)
        self.assertEqual(row['BA'], .625)
        # Positive-only in Male0: its FPR and EO undefined, TPR/DP still valid.
        row = smoke.classification_metrics(y[[0, 3, 4, 5]], p[[0, 3, 4, 5]],
                                           g[[0, 3, 4, 5]], ['attr'])[0]
        self.assertTrue(np.isnan(row['Male0_FPR']))
        self.assertTrue(np.isnan(row['EqualizedOdds']))
        self.assertEqual(row['EqualOpportunity'], 1.)
        self.assertTrue(np.isfinite(row['DemographicParity']))

    def test_group_means_not_totals_signed_then_absolute(self):
        # Signed sample values: group0=(-2,4), group1=(3); means 1 and 3.
        sums = torch.tensor([[2., 100.], [3., 5.]], dtype=torch.float64)
        counts = torch.tensor([2., 1.])
        actual = smoke.gap_from_sums(sums, counts, torch.tensor([True, False]))
        self.assertEqual(float(actual), 2.)
        missing = smoke.gap_from_sums(sums, torch.tensor([0., 1.]), torch.tensor([True, True]))
        self.assertTrue(torch.isnan(missing))

    def test_B_reduces_larger_gap_not_difference_of_gaps(self):
        signals = dict(activation_all=7., activation_cond=np.tile([2., 5.], (39, 1)),
                       contribution_all=np.arange(39.),
                       contribution_cond=np.tile([2., 5.], (39, 1)))
        self.assertEqual(smoke.selection_score(signals, 'A_activation'), 7.)
        self.assertEqual(smoke.selection_score(signals, 'B_activation'), 5.)
        self.assertEqual(smoke.selection_score(signals, 'A_contribution'), 19.)
        self.assertEqual(smoke.selection_score(signals, 'B_contribution'), 5.)
        signals['activation_cond'][0, 1] = np.nan
        with self.assertRaises(ValueError):
            smoke.selection_score(signals, 'B_activation')

    def test_original_weight_importance_matches_independent_calculation(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*2*2, 78))
        x = torch.randn(6, 3, 2, 2)
        labels = torch.randint(0, 2, (6, 40))
        loader = DataLoader(TensorDataset(x, labels), batch_size=3)
        expected, batch_gradients = [], []
        for bx, by in loader:
            model.zero_grad()
            logits = model(bx).reshape(-1, 39, 2)
            # Independent flattened CE has same mean over 39 tasks and batch.
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), by[:, :39].reshape(-1))
            loss.backward()
            grad = model[1].weight.grad.detach().clone()
            batch_gradients.append(grad)
            expected.append((model[1].weight.detach()*grad)**2)
        with patch.object(smoke.prune, 'make_datasets', return_value=(loader, loader)), \
             patch.object(smoke.prune, 'device', torch.device('cpu')), \
             contextlib.redirect_stdout(io.StringIO()):
            actual, _ = smoke.prune.compute_importance(model, None, pd.DataFrame(),
                output_cols_each_task=smoke.SLICES, stop_batch=2, imp_batch_size=3)
        torch.testing.assert_close(actual['1'], torch.stack(expected).mean(0), rtol=2e-5, atol=1e-10)
        wrong = (model[1].weight.detach()*torch.stack(batch_gradients).mean(0))**2
        self.assertGreater(float((wrong-actual['1']).abs().max()), 1e-8)
        # Sensitive labels must not affect task importance.
        alternate = DataLoader(TensorDataset(x, torch.cat((labels[:, :39], 1-labels[:, 39:]), 1)), batch_size=3)
        with patch.object(smoke.prune, 'make_datasets', return_value=(alternate, alternate)), \
             patch.object(smoke.prune, 'device', torch.device('cpu')), \
             contextlib.redirect_stdout(io.StringIO()):
            second, _ = smoke.prune.compute_importance(model, None, pd.DataFrame(),
                output_cols_each_task=smoke.SLICES, stop_batch=2, imp_batch_size=3)
        torch.testing.assert_close(actual['1'], second['1'], rtol=0, atol=0)

    def test_temporary_removal_BN_offsets_and_no_regrowth(self):
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(1280, 78)
        smoke.install_masks(model)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        snapshot = {k:v.clone() for k,v in model.state_dict().items()}
        with smoke.temporary_channels(model, [('features.1', 0), ('features.2', 1)]):
            smoke.assert_masks(model)
            self.assertTrue((model.features[2].conv[2].mask[:, 1] == 0).all())
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v, snapshot[k], rtol=0, atol=0)
        smoke.apply_channels(model, [('features.1', 0), ('features.2', 1)])
        with torch.no_grad():
            baseline = model(x)
            model.features[1].conv[0][1].bias[0] = 20.
            model.features[2].conv[1][1].bias[1] = -20.
            torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        model.train()
        model(x).square().sum().backward()
        optimizer.step()
        smoke.enforce_masks(model)
        smoke.assert_masks(model)

    def test_contribution_matches_closed_form_and_batch_partition(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 2, 1)
                self.fc = torch.nn.Linear(2, 78)

            def forward(self, x):
                return self.fc(self.conv(x).mean((2, 3)))

        model = Toy().eval()
        smoke.install_masks(model)
        x = torch.randn(7, 3, 2, 2)
        g = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        y = torch.tensor([0, 1, 0, 1, 0, 1, 1])[:, None].expand(7, 39)
        labels = torch.cat((y, g[:, None]), dim=1)
        a = model.conv(x).mean((2, 3)).detach()
        p = model(x).reshape(7, 39, 2).softmax(-1)[:, :, 1].detach()
        weights = model.fc.weight.detach().reshape(39, 2, 2)
        # Each spatial derivative includes the global-average-pool factor 1/HW.
        q = a[:, None, :] * (weights[:, 1]-weights[:, 0])[None] * (p*(1-p))[:, :, None] / 4
        expected = (q[g == 0].mean(0)-q[g == 1].mean(0)).abs().mean(-1).numpy()
        observed = []
        with patch.object(smoke, 'reference_layers', return_value={'conv': model.conv}):
            for batch in (2, 3):
                loader = DataLoader(TensorDataset(x, labels), batch_size=batch)
                observed.append(smoke.collect_signals(model, loader, True))
        np.testing.assert_allclose(observed[0]['contribution_all'], expected, rtol=2e-5, atol=1e-9)
        for key in ('activation_all', 'activation_cond', 'contribution_all', 'contribution_cond'):
            np.testing.assert_allclose(observed[0][key], observed[1][key], rtol=2e-5, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
