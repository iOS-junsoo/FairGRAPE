"""Numerical contract checks for layer-local keep-greedy selection."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import celeba_layerwise_experiment as exp


class LayerwiseTests(unittest.TestCase):
    def test_protection_pool_budget_and_signed_compensation(self):
        importance = np.arange(100, dtype=float)
        d = np.zeros(100)
        d[99] = 10
        d[:10] = -1
        d[10:30] = .2
        decision = exp.greedy_keep(importance, np.ones(100, bool), {'activation_all': d}, 'A_activation')
        self.assertEqual(decision['protected'], list(range(99, 89, -1)))
        self.assertEqual(decision['fixed_kept'], list(range(30, 100)))
        self.assertEqual(decision['candidates'], list(range(30)))
        self.assertEqual(len(decision['removed']), 10)
        self.assertEqual(len(decision['choices']), 20)
        self.assertEqual([v['kept_channel'] for v in decision['choices'][:10]], list(range(9, -1, -1)))
        self.assertTrue(set(range(30, 100)) <= set(decision['kept']))
        self.assertAlmostEqual(decision['kept_proxy_score'], 2.)

    def test_22_rounds_preserve_floor_budget_in_small_layers(self):
        original=[32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]
        lives=[np.ones(n,bool) for n in original]
        totals=[]
        for rnd in range(22):
            removed=0
            for n,live in zip(original,lives):
                before=int(live.sum())
                d=exp.greedy_keep(np.arange(n),live,{'activation_all':np.zeros(n)},'A_activation')
                self.assertEqual(d['remove_count'],before//10)
                live[d['removed']]=False
                self.assertGreaterEqual(int(live.sum()),9)
                removed+=len(d['removed'])
            totals.append(removed)
        self.assertEqual(totals[:3],[708,634,569])
        self.assertEqual(totals[-1],75)
        self.assertEqual(sum(int(x.sum()) for x in lives),766)
        self.assertEqual(int(lives[0].sum()),9)

    def test_ties_prefer_higher_importance_then_lower_index(self):
        d = {'activation_all': np.zeros(100)}
        a = exp.greedy_keep(np.arange(100), np.ones(100, bool), d, 'A_activation')
        self.assertEqual(a['removed'], list(range(10)))
        b = exp.greedy_keep(np.ones(100), np.ones(100, bool), d, 'A_activation')
        self.assertEqual(b['protected'], list(range(10)))
        self.assertEqual(b['removed'], list(range(90, 100)))

    def test_three_rounds_current_live_and_floor_counts(self):
        original = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]
        totals = []
        lives = [np.ones(n, bool) for n in original]
        for _ in range(3):
            total = 0
            for n, live in zip(original, lives):
                before = live.copy()
                d = exp.greedy_keep(np.arange(n), live, {'activation_all': np.zeros(n)}, 'A_activation')
                self.assertEqual(d['remove_count'], int(before.sum())//10)
                self.assertTrue(before[d['removed']].all())
                live[d['removed']] = False
                total += len(d['removed'])
            totals.append(total)
        self.assertEqual(totals, [708, 634, 569])

    def test_B_max_not_difference_and_39_attribute_mean(self):
        values = np.tile(np.array([2., -5.])[None, :, None], (39, 1, 3))
        np.testing.assert_allclose(exp.objective(values, 'B_activation'), [5.,5.,5.])
        values[0, :, :] = 44.
        np.testing.assert_allclose(exp.objective(values, 'B_contribution'), [6.,6.,6.])
        vals = np.arange(39.)[:, None]
        np.testing.assert_allclose(exp.objective(vals, 'A_contribution'), [19.])
        # The unconditional activation objective has no task dimension.
        np.testing.assert_allclose(exp.objective(np.array([-2., 4.]), 'A_activation'), [2.,4.])

    def test_group_means_and_missing_cells(self):
        s = torch.tensor([[2., -2.], [3., -3.]], dtype=torch.float64)
        counts = torch.tensor([2.,1.])
        d = exp.signed_difference(s, counts)
        torch.testing.assert_close(d, torch.tensor([-2.,2.], dtype=torch.float64))
        self.assertEqual(float(d.sum().abs()), 0.)
        self.assertTrue(torch.isnan(exp.signed_difference(s, torch.tensor([0.,1.]))).all())

    def test_collector_uses_raw_positive_logit_even_when_probability_saturated(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1, bias=False)
                self.conv.weight.data.fill_(1)
                self.conv.register_buffer('mask', torch.ones_like(self.conv.weight))
            def forward(self, x):
                a = self.conv(x).mean((2,3))
                positive = a*3 + 100
                return torch.stack((a*0, positive), -1).repeat(1,39,1).reshape(-1,78)
        model = Toy()
        x = torch.tensor([1.,3.,5.,9.])[:,None,None,None]
        labels = torch.zeros(4,40, dtype=torch.long)
        labels[2:,39] = 1
        labels[[1,3],:39] = 1
        loader = DataLoader(TensorDataset(x,labels), batch_size=2)
        with patch.object(exp.base, 'reference_layers', return_value={'layer': model.conv}):
            s = exp.collect(model, loader)
        np.testing.assert_allclose(s['deltas']['layer']['activation_all'], [-5.])
        np.testing.assert_allclose(s['deltas']['layer']['contribution_all'], np.full((39,1), -15.))
        np.testing.assert_allclose(s['deltas']['layer']['contribution_cond'][:,0], np.full((39,1), -12.))
        np.testing.assert_allclose(s['deltas']['layer']['contribution_cond'][:,1], np.full((39,1), -18.))


if __name__ == '__main__':
    unittest.main()
