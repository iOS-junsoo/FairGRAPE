"""Official-split and single-attribute contracts plus production gradient checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import celeba_single as single


class FramesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        from PIL import Image
        attrs = [f'Attr{i}' for i in range(38)] + ['Smiling']
        records = []
        lines = []
        for i in range(12):
            name = f'{i+1:06d}.jpg'
            Image.new('RGB', (8, 8), (i, 0, 0)).save(self.root / name)
            records.append(dict(image_name=name, face_name_align=name,
                                **{a: (i % 4) // 2 for a in attrs}, gender=i % 2))
            lines.append(f'{name} {i // 4}\n')
        self.csv = self.root / 'labels.csv'
        self.frame = pd.DataFrame(records)
        self.frame.to_csv(self.csv, index=False)
        self.partition = self.root / 'partition.txt'
        self.partition.write_text(''.join(lines))
        self.addCleanup(patch.stopall)
        patch.object(single, 'PARTITION_MD5', single.file_hash(self.partition, 'md5')).start()
        patch.object(single, 'SPLIT_SIZES', dict(train=4, val=4, test=4)).start()

    def load(self, **kwargs):
        return single.load_frames(self.csv, self.root, self.partition, kwargs.pop('target', 'Smiling'), **kwargs)

    def test_official_mapping_and_csv_order_invariance(self):
        frames, meta = self.load()
        self.assertEqual(frames['val'].image_name.tolist(), [f'{i:06d}.jpg' for i in range(5, 9)])
        self.frame.sample(frac=1, random_state=93).to_csv(self.csv, index=False)
        second, other = self.load()
        self.assertEqual(meta, other)
        for split in frames:
            self.assertEqual(frames[split].image_name.tolist(), second[split].image_name.tolist())
        _, attr_meta = self.load(target='Attr0')
        self.assertEqual(meta['split_sha256'], attr_meta['split_sha256'])
        self.assertTrue(set(frames['train'].image_name).isdisjoint(frames['test'].image_name))

    def test_unselected_labels_are_irrelevant(self):
        _, before = self.load()
        self.frame['Attr0'] = 1 - self.frame.Attr0
        self.frame.to_csv(self.csv, index=False)
        _, after = self.load()
        self.assertEqual(before, after)

    def test_duplicate_missing_and_invalid_labels_fail(self):
        for altered in (pd.concat([self.frame, self.frame.iloc[:1]]), self.frame.iloc[1:],
                        self.frame.assign(Smiling=2)):
            altered.to_csv(self.csv, index=False)
            with self.assertRaises(ValueError): self.load()

    def test_missing_corrupt_image_and_partition_fail(self):
        image = self.root / '000001.jpg'
        image.write_bytes(b'not an image')
        with self.assertRaises(OSError): self.load()
        image.unlink()
        with self.assertRaises(FileNotFoundError): self.load(check_readable=False)
        self.partition.unlink()
        with self.assertRaises(FileNotFoundError): self.load()

    def test_partition_checksum_duplicate_and_invalid_id(self):
        original = self.partition.read_text()
        for text in (original.replace('000001.jpg 0', '000001.jpg 7'), original + '000001.jpg 0\n'):
            self.partition.write_text(text)
            with self.assertRaises(ValueError): single.read_partition(self.partition)
            with patch.object(single, 'PARTITION_MD5', single.file_hash(self.partition, 'md5')):
                with self.assertRaises(ValueError): single.read_partition(self.partition)

    def test_smoke_checkpoint_contract_is_distinct(self):
        _, full = self.load()
        _, smoke = self.load(smoke_samples=4)
        self.assertNotEqual(full, smoke)
        self.assertEqual(smoke['smoke_samples'], 4)

    def test_invalid_attribute_and_sensitive(self):
        for options in ({'target': 'gender'}, {'target': 'unknown'}, {'sensitive': 'Young'}):
            with self.assertRaises(ValueError): self.load(**options)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.meta = dict(targets=['Smiling'], initialization=single.IMAGENET_WEIGHTS,
                         split_protocol='official', split_sha256={'train': 'abc'}, smoke_samples=0)
        self.ck = dict(experiment_metadata=copy.deepcopy(self.meta), finetuning_complete=True,
                       model_state={'classifier.1.weight': torch.zeros(2, 1280)},
                       stage='dense', prune_iteration=0)

    def test_accept_matching_dense_only(self):
        self.assertIs(single.validate_checkpoint(self.ck, self.meta), self.ck['model_state'])
        for change in ({'targets': ['Smiling', 'Young']}, {'targets': ['Young']},
                       {'initialization': 'random'}, {'split_protocol': 'random'},
                       {'split_sha256': {'train': 'other'}}, {'smoke_samples': 4}):
            ck = copy.deepcopy(self.ck); ck['experiment_metadata'].update(change)
            with self.assertRaises(ValueError): single.validate_checkpoint(ck, self.meta)
        for ck in ({}, self.ck['model_state'], dict(self.ck, finetuning_complete=False)):
            with self.assertRaises(ValueError): single.validate_checkpoint(ck, self.meta)
        self.ck['model_state']['classifier.1.weight'] = torch.zeros(78, 1280)
        with self.assertRaises(ValueError): single.validate_checkpoint(self.ck, self.meta)

    def test_score_only_also_requires_provenance(self):
        self.ck['stage'] = 'pruned'; self.ck['prune_iteration'] = 1
        self.ck['model_state']['classifier.1.mask'] = torch.ones(2, 1280)
        with self.assertRaises(ValueError): single.validate_checkpoint(self.ck, self.meta, allow_pruned=True)
        self.ck['source_checkpoint'] = {'sha256': 'a' * 64}
        single.validate_checkpoint(self.ck, self.meta, allow_pruned=True)
        with self.assertRaises(ValueError): single.validate_checkpoint(self.ck, self.meta)
        self.ck['model_state']['classifier.1.mask'][0, 0] = .5
        with self.assertRaises(ValueError): single.validate_checkpoint(self.ck, self.meta, allow_pruned=True)


class MetricsTest(unittest.TestCase):
    def test_binary_metrics_and_missing_cell(self):
        m = single.binary_metrics([0, 1, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1], .3)
        self.assertEqual({k: m[k] for k in ('accuracy','ba','eo','eopp','dp','loss','samples')},
                         dict(accuracy=.75, ba=.75, eo=.5, eopp=0., dp=.5, loss=.3, samples=4))
        self.assertIsNone(single.binary_metrics([0, 1], [0, 1], [0, 0], 0)['eo'])

    def test_gaps_use_pooled_counts_and_independent_denominators(self):
        # Unequal group sizes: P(pred=1|g0)=3/4, P(pred=1|g1)=1/2.
        m = single.binary_metrics([1,1,0,0,1,0], [1,1,1,0,0,1], [0,0,0,0,1,1], .2)
        self.assertEqual(m['eopp'], 1.)
        self.assertEqual(m['dp'], .25)
        self.assertEqual(m['eo'], .75)
        self.assertEqual(m['groups']['0']['positive_prediction_rate'], .75)
        # No negative samples: EOPP and DP remain defined, EO is undefined.
        m = single.binary_metrics([1,1], [1,0], [0,1], 0)
        self.assertIsNone(m['eo']); self.assertEqual(m['eopp'], 1.); self.assertEqual(m['dp'], 1.)
        # No positive samples: DP remains defined.
        m = single.binary_metrics([0,0], [1,0], [0,1], 0)
        self.assertIsNone(m['eopp']); self.assertIsNone(m['eo']); self.assertEqual(m['dp'], 1.)
        m = single.binary_metrics([0,1], [0,1], [0,0], 0)
        for key in ('eo','eopp','dp'): self.assertIsNone(m[key])

    def test_selection_order(self):
        base = dict(accuracy=.8, eo=.1, loss=.3)
        for better in (dict(base, accuracy=.9, eo=.8), dict(base, eo=.05, loss=.9), dict(base, loss=.2)):
            self.assertGreater(single.selection_key(better), single.selection_key(base))
        self.assertLess(single.selection_key(dict(base, eo=None)), single.selection_key(base))

    def test_schedule_preserves_best_weights_and_uses_eo_before_loss(self):
        import train_and_val as tv
        import config
        model = torch.nn.Linear(1, 1)
        metrics = [dict(accuracy=.8, eo=.1, loss=.4), dict(accuracy=.8, eo=.2, loss=.1)]
        calls = []
        def epoch(candidate, *args, **kwargs):
            idx = len(calls); calls.append(candidate)
            with torch.no_grad(): candidate.weight.fill_(idx + 1)
            candidate.single_validation_metrics = metrics[idx]
            return candidate, .8, metrics[idx]['loss'], {}
        with patch.object(config, 'glo_impt_type', 5), patch.object(tv, 'train_model0', side_effect=epoch), contextlib.redirect_stdout(io.StringIO()):
            result, _ = tv.train(model, None, {}, [1e-4, 1e-5], [1, 1], cfgs=['CelebA'])
        self.assertEqual(result.weight.item(), 1)
        self.assertIsNot(calls[0], calls[1])


class GradientTest(unittest.TestCase):
    def test_production_importance_and_phi_match_single_loss(self):
        import prune
        import config
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 2, 1, bias=False)
                self.classifier = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(2, 2, bias=False))
            def forward(self, x):
                self.activation = self.conv(x)
                self.activation.retain_grad()
                return self.classifier(self.activation.mean((2, 3)))
        torch.manual_seed(7)
        model = Toy()
        x = torch.arange(1., 5.).reshape(4, 1, 1, 1)
        labels = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        loss = torch.nn.functional.cross_entropy(model(x), labels[:, 0]); loss.backward()
        expected_perf = (model.conv.weight.detach() * model.conv.weight.grad).square()
        act = model.activation.detach().flatten(2).mean(2)
        gap = (act[0] - act[1]).abs() + (act[2] - act[3]).abs()
        expected_phi = gap * model.activation.grad.abs().mean((0, 2, 3))
        with patch.object(prune, 'device', torch.device('cpu')), \
             patch.object(config, 'glo_n_groups', 2, create=True), \
             patch.object(prune, 'make_datasets', return_value=([(x, labels)], None)), \
             patch.object(prune, '_get_impt_type2_target_layers', return_value={'conv': model.conv}), \
             contextlib.redirect_stdout(io.StringIO()):
            perf, _ = prune.compute_importance(model, None, pd.DataFrame(), output_cols_each_task=[(0, 2)], col_names=['Smiling','gender'], stop_batch=1)
            phi, _, _, _, _ = prune.compute_phi_k(model, pd.DataFrame(), output_cols_each_task=[(0, 2)], col_names=['Smiling','gender'], stop_batch=1, sensitive_group='gender')
        torch.testing.assert_close(perf['conv'], expected_perf)
        torch.testing.assert_close(phi['conv'], expected_phi)

    def test_impt2_and_5_share_actual_selector(self):
        import prune
        import config
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None, num_classes=2)
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                layer.mask = torch.nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)
        phi = {n: torch.arange(l.out_channels).float() for n,l in prune._get_impt_type2_target_layers(model).items()}
        perf = {n: l.weight.detach().square() for n,l in model.named_modules() if isinstance(l, (torch.nn.Conv2d,torch.nn.Linear))}
        with patch.object(prune, 'device', torch.device('cpu')), \
             patch.object(prune, 'compute_phi_k', return_value=(phi, {}, {}, {}, {})), \
             patch.object(prune, 'compute_importance', return_value=(perf, {})), \
             patch.object(prune, '_save_channel_pruning_log'), \
             patch.object(config, 'glo_keep_per_iter', .99), \
             patch.object(config, 'glo_impt2_layer_max_remove_ratio', None), \
             patch.object(config, 'glo_perf_only', False), \
             patch.object(config, 'glo_phi_analysis', False, create=True), \
             patch.object(config, 'glo_phi_gap_only', False, create=True), contextlib.redirect_stdout(io.StringIO()):
            results = [prune.fairness_grad(copy.deepcopy(model), .01, None, sensitive_classes=2,
                        output_cols_each_task=[(0,2)], col_names=['Smiling','gender'], impt_type=k,
                        network='mobilenetv2', sensitive_group='gender') for k in (2,5)]
        for a,b in zip(*results): self.assertTrue(torch.equal(a,b))


if __name__ == '__main__':
    unittest.main()
