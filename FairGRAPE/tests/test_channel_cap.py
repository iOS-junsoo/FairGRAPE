"""Exercise the actual impt=2 selector without image inference or a GPU."""
import contextlib
import copy
import importlib.util
import io
import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from channel_cap import channel_removal_limits, validate_layer_remove_ratio


class LimitsTest(unittest.TestCase):
    def test_fractional_channels_and_existing_floor(self):
        self.assertEqual(channel_removal_limits({'a':32,'b':100}, {'a':9,'b':100},
                                               {'a':1,'b':95}, .1), {'a':0,'b':5})
        self.assertEqual(channel_removal_limits({'a':32}, {'a':8}, {'a':1}, .2), {'a':1})
        self.assertEqual(channel_removal_limits({'a':32}, {'a':0}, {'a':1}, .2), {'a':0})
        self.assertIsNone(channel_removal_limits({}, {}, {}, None))

    def test_reject_invalid_ratio(self):
        for value in [0, -1, 1.01, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): validate_layer_remove_ratio(value)


class ProductionSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import prune
        cls.torch, cls.prune = torch, prune
        cls.original_module = None
        before = os.environ.get('CHANNEL_CAP_ORIGINAL_SOURCE')
        if before:
            spec = importlib.util.spec_from_file_location('prune_before_strict_cap', before)
            cls.original_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.original_module)

    def model(self):
        nn=self.torch.nn
        model=nn.Module(); model.features=nn.ModuleList([nn.Identity(), nn.Module(), nn.Module()])
        model.features[1].conv=nn.Sequential(nn.Sequential(nn.Conv2d(8,8,3,groups=8)), nn.Conv2d(8,4,1))
        model.features[2].conv=nn.Sequential(nn.Sequential(nn.Conv2d(4,20,1)),
                                           nn.Sequential(nn.Conv2d(20,20,3,groups=20)),nn.Conv2d(20,4,1))
        for layer in model.modules():
            if isinstance(layer,nn.Conv2d): layer.mask=nn.Parameter(self.torch.ones_like(layer.weight),requires_grad=False)
        # One channel was removed in a previous iteration: it must never regrow.
        model.features[2].conv[0][0].mask.data[0]=0
        model.features[2].conv[1][0].mask.data[0]=0
        model.features[2].conv[2].mask.data[:,0]=0
        return model

    def run_selector(self,module,ratio,keep=.85,legacy=None):
        import config
        t=self.torch; model=self.model(); captured=[]
        phi={'features.1.conv.0.0':t.linspace(1,.8,8),
             'features.2.conv.1.0':t.linspace(.1,.01,20)}
        phi['features.2.conv.1.0'][0]=0
        perf={}
        for name,width in [('features.1.conv.0.0',8),('features.2.conv.0.0',20)]:
            layer=dict(model.named_modules())[name]
            v=t.linspace(.001,.002,width)
            v[-1]=1
            if width==20:v[0]=0
            perf[name]=v[:,None,None,None].expand_as(layer.weight).clone()
        with contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(module,'compute_phi_k',return_value=(phi,{}, {},{}, {})), \
             mock.patch.object(module,'compute_importance',return_value=(perf,{})), \
             mock.patch.object(module,'_save_channel_pruning_log',side_effect=lambda **kw:captured.append(kw)), \
             mock.patch.object(module,'IMPT2_LAYER_CAP_MULTIPLIER',legacy), \
             mock.patch.object(module,'IMPT_TYPE2_ALPHA',.7), \
             mock.patch.object(module,'IMPT2_NORM','max'), \
             mock.patch.object(config,'glo_keep_per_iter',keep), \
             mock.patch.object(config,'glo_impt2_layer_max_remove_ratio',ratio):
            masks=module.fairness_grad(model,.5,None,sensitive_classes=2,
                    output_cols_each_task=[(0,2)],col_names=['species','color'],impt_type=2,sensitive_group='color')
        return masks,captured[0]

    def test_redistribution_no_regrowth_and_no_protection_violation(self):
        masks,p=self.run_selector(self.prune,.2)
        keys=p['selected_channels'];counts={b:sum(k[0]==b for k in keys) for b in p['strict_channel_limits']}
        self.assertLessEqual(counts['features.1'],1)
        self.assertGreater(counts['features.2'],0)
        self.assertTrue(set(keys).isdisjoint(p['protected_set']))
        self.assertTrue(all(counts[b]<=v for b,v in p['strict_channel_limits'].items()))
        self.assertEqual(float(masks[2][0].sum()),0)
        self.assertEqual(float(masks[3][0].sum()),0)
        self.assertEqual(float(masks[4][:,0].sum()),0)
        self.assertEqual(p['cap_relaxed_removed'],0)

    def test_deficit_does_not_relax_cap(self):
        masks,p=self.run_selector(self.prune,.1,keep=.05)
        self.assertLess(p['accum_removed'],p['remove_target'])
        self.assertFalse(any(b=='features.1' for b,c in p['selected_channels']))
        self.assertEqual(len(p['selected_channels']),1)
        self.assertEqual(p['cap_relaxed_removed'],0)

    def test_legacy_conflict(self):
        with self.assertRaises(ValueError): self.run_selector(self.prune,.2,legacy=2)

    def test_off_identical_to_prechange_source(self):
        if self.original_module is None:self.skipTest('provide CHANNEL_CAP_ORIGINAL_SOURCE for pre-change comparison')
        for legacy in [None,2.0]:
            old,op=self.run_selector(self.original_module,None,legacy=legacy)
            new,np=self.run_selector(self.prune,None,legacy=legacy)
            self.assertEqual(op['selected_channels'],np['selected_channels'])
            self.assertEqual(op['accum_removed'],np['accum_removed'])
            self.assertTrue(all(self.torch.equal(a,b) for a,b in zip(old,new)))


if __name__=='__main__': unittest.main()
