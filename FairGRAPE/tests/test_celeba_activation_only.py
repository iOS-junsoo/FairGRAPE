"""Activation-only selection/evaluation must never request logit gradients."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import celeba_layerwise_full as full
from celeba_layerwise_full import experiment as exp, torch, np
from torch.utils.data import DataLoader, TensorDataset


class ActivationOnlyTests(unittest.TestCase):
    def test_activation_statistics_and_classification_match_without_gradients(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 2, 1, bias=False)
                self.conv.weight.data[:, 0, 0, 0] = torch.tensor([1., -2.])
                self.grad_modes = []
            def forward(self, x):
                self.grad_modes.append(torch.is_grad_enabled())
                a = self.conv(x).mean((2,3))
                z = torch.stack((a[:, 1], a[:, 0]*3+100), -1)
                return z[:,None,:].expand(-1,39,-1).reshape(-1,78)
        model = Toy()
        x = torch.arange(1.,7.).reshape(6,1,1,1).expand(-1,1,2,3).clone()
        y = torch.zeros(6,40,dtype=torch.long)
        y[1::2,:39] = 1
        y[2:,39] = 1  # Unequal group sample counts; every label/group cell exists.
        loader = DataLoader(TensorDataset(x,y),batch_size=2)
        attrs = [f'Attr{t}' for t in range(39)]
        with patch.object(exp.base,'reference_layers',return_value={'layer':model.conv}), \
             patch.object(exp.base,'active_masks',return_value={'layer':torch.ones(2,dtype=torch.bool)}):
            original = exp.evaluate(model,loader,attrs,'dense',0,'dense')
            model.grad_modes.clear()
            with patch.object(torch.autograd,'grad',side_effect=AssertionError('No attribute gradients allowed')):
                signals = exp.collect(model,loader,contribution=False)
                activation = exp.evaluate(model,loader,attrs,'dense',0,'dense',contribution=False)
            self.assertTrue(model.grad_modes)
            self.assertFalse(any(model.grad_modes))
        np.testing.assert_allclose(signals['deltas']['layer']['activation_all'],[-3.,6.])
        self.assertNotIn('contribution_all',signals['deltas']['layer'])
        self.assertEqual(activation['global_activation_gap'],original['global_activation_gap'])
        self.assertEqual(activation['contribution_metrics_status'],'deferred_by_user')
        for a,b in zip(activation['attributes'],original['attributes']):
            for key in a:
                if key.startswith('contribution_'):
                    self.assertTrue(np.isnan(a[key]))
                elif key != 'attribute':
                    np.testing.assert_allclose(a[key],b[key],equal_nan=True)
        exp.base.macro_metrics([activation])
        for key in ('contribution_all','contribution_positive','contribution_negative'):
            self.assertEqual(activation['macro'][key]['n_attributes'],0)
            self.assertIsNone(activation['macro'][key]['value'])
        self.assertEqual(activation['macro']['ACC']['n_attributes'],39)

    def test_full_cached_evaluation_forwards_no_contribution(self):
        class Loader:
            history = [{'samples':4}]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'cache').mkdir()
            with patch.object(full,'signal_loader',return_value=Loader()), \
                 patch.object(exp,'evaluate',return_value={'contribution_metrics_status':'deferred_by_user'}) as evaluate:
                full.evaluate_cached(None,range(4),[],None,'dense',0,'dense',out,'test',contribution=False)
            self.assertIs(evaluate.call_args.kwargs['contribution'],False)


if __name__ == '__main__':
    unittest.main()
