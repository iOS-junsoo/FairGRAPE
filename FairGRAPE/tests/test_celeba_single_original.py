"""Single-task original FairGRAPE scoring, masks and runner regressions."""
import copy
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import prune
import prune_orig_ref as reference
import celeba_single as single
import celeba_single_experiment as runner
import celeba_single_reporting as reporting
import test_celeba_single_runner as base_tests


class OriginalRunnerTest(base_tests.RunnerTest):
    def args(self, name='run', **changes):
        values = dict(impt=6, delta_p=None, fg_scope=None)
        values.update(changes)
        return super().args(name, **values)

    def test_defaults_and_incompatible_options(self):
        args = self.args()
        self.assertEqual(runner.validate_args(args), (None, None))
        self.assertEqual((args.fg_mode, args.fg_scope, args.delta_p, args.para_batch),
                         ('original', 'all', 0, 1))
        for value in (0, 1, 2):
            runner.validate_args(self.args(delta_p=value))
        for changes in (dict(fg_mode='clean'), dict(fg_mode='faithful'), dict(fg_scope='blocks'),
                        dict(delta_p=.5), dict(delta_p=float('nan')), dict(para_batch=0),
                        dict(impt2_alpha=.5), dict(impt2_layer_max_remove_ratio=.1),
                        dict(perf_only=True), dict(phi_analysis=True), dict(phi_gap_only=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.validate_args(self.args(**changes))

    def test_dense_from_impt5_is_reusable_but_pruned_is_not(self):
        runner.run(self.args('five', impt=5, delta_p=.5))
        dense = next((self.root/'five'/'models'/'dense').glob('*.pt'))
        pruned = next((self.root/'five'/'models'/'pruning').glob('BEST_MODEL_final_*.pt'))
        runner.run(self.args('six', checkpoint=str(dense), no_retrain=True))
        for options in (dict(score_only=True), dict(single_attr_eval_only=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runner.run(self.args('wrong_method', checkpoint=str(pruned), **options))

    def test_six_checkpoint_cannot_be_scored_as_five(self):
        runner.run(self.args())
        pruned = next((self.root/'run'/'models'/'pruning').glob('BEST_MODEL_final_*.pt'))
        with self.assertRaises(ValueError):
            runner.run(self.args('wrong_five', impt=5, delta_p=.5,
                                 checkpoint=str(pruned), score_only=True))

    def test_score_only_preserves_loaded_masks_and_input_file(self):
        runner.run(self.args(no_retrain=True, prune_rate=.7))
        pruned = next((self.root/'run'/'models'/'pruning').glob('BEST_MODEL_final_*.pt'))
        checksum = single.file_hash(pruned)
        state = torch.load(pruned, weights_only=True)['model_state']
        model = runner.run(self.args('scores', checkpoint=str(pruned), score_only=True, prune_rate=.01))
        self.assertEqual(single.file_hash(pruned), checksum)
        for name, value in state.items(): self.assertTrue(torch.equal(model.state_dict()[name], value))

    def test_nonzero_masked_checkpoint_weights_are_rejected(self):
        runner.run(self.args(no_retrain=True))
        source = next((self.root/'run'/'models'/'pruning').glob('BEST_MODEL_final_*.pt'))
        checkpoint = torch.load(source, weights_only=True)
        mask = checkpoint['model_state']['classifier.1.mask']
        checkpoint['model_state']['classifier.1.weight'][mask == 0] = 1
        corrupt = self.root/'corrupt.pt'; torch.save(checkpoint, corrupt)
        with self.assertRaisesRegex(ValueError, 'nonzero masked weights'):
            runner.run(self.args('corrupt_run', checkpoint=str(corrupt), score_only=True))
        self.assertFalse((self.root/'corrupt_run').exists())

    def test_original_identity_and_full_target_statistics_are_recorded(self):
        runner.run(self.args(save_mask=True, no_retrain=True))
        import json
        root = self.root/'run'
        info = json.loads((root/'run.json').read_text())
        self.assertEqual((info['impt_type'], info['fg_mode'], info['fg_scope'], info['pruning_unit']),
                         (6, 'original', 'all', 'weight'))
        self.assertIsNone(info['alpha']); self.assertIsNone(info['norm'])
        self.assertTrue((root/'fg_original'/'0_run_info.txt').exists())
        self.assertTrue((root/'models'/'pruning'/'impt6_Smiling_official_masks_001.pt').exists())
        summary = (root/'pruning'/'iter_01'/'BEST_MODEL_SUMMARY_prune_01_val.txt').read_text()
        self.assertIn('Impt Type: 6', summary)
        self.assertIn('FG Scope: all', summary)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.classifier = nn.Sequential(nn.Dropout(.1), nn.Linear(4, 2))

    def forward(self, images):
        return self.classifier(torch.tanh(self.bn(self.conv(images))).mean((2, 3)))


@pytest.fixture
def original_context(monkeypatch, tmp_path):
    monkeypatch.setattr(prune, 'device', torch.device('cpu'))
    monkeypatch.setattr(reference, 'device', torch.device('cpu'))
    monkeypatch.setattr(prune, 'FG_ORIG_MODE', 'clean')  # Six must ignore this default.
    monkeypatch.setattr(config, 'glo_impt_type', 6)
    monkeypatch.setattr(config, 'glo_fg_log_run_dir', str(tmp_path))
    monkeypatch.setattr(config, 'glo_imp_rate', None)
    monkeypatch.setattr(config, 'glo_experiment_metadata', {'targets': ['Smiling']}, raising=False)
    monkeypatch.setattr(reference, 'LAST_GRAD_TARGET', {})
    monkeypatch.setattr(prune, 'FG_ORIG_LAST_STATS', None)
    monkeypatch.setattr(prune, 'FG_ORIG_BLOCK_LAYERS', None)
    torch.manual_seed(31)
    images = torch.randn(8, 3, 3, 3)
    frame = pd.DataFrame(dict(Smiling=[0,1,0,1]*2, gender=[0,0,1,1]*2, Unselected=[0]*8))
    seen_columns, scores = [], []

    def datasets(train, val, testing, batch_size, col_used):
        seen_columns.append(list(col_used))
        labels = torch.tensor(train[col_used].values, dtype=torch.long)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=4)
        return loader, loader

    real_importance = reference.importance_by_class0

    def importance(*args, **kwargs):
        result = real_importance(*args, **kwargs)
        scores.append(copy.deepcopy(result[1]))
        return result

    monkeypatch.setattr(reference, 'make_datasets', datasets)
    monkeypatch.setattr(reference, 'importance_by_class0', importance)
    return frame, seen_columns, scores


def six_masks(model, frame, rate=.5, delta_p=0):
    return prune.fairness_grad(model, rate, frame, sensitive_classes=2,
        output_cols_each_task=[(0,2)], col_names=['Smiling','gender'],
        impt_type=6, stop_batch=2, para_batch=1, delta_p=delta_p,
        network='mobilenetv2', sensitive_group='gender')


@pytest.mark.parametrize('delta_p', [0, 1, 2])
def test_scores_targets_and_masks_match_original(original_context, delta_p):
    frame, columns, scores = original_context
    base = prune.FairGRAPE(TinyNet(), nn.CrossEntropyLoss(), None, [(0,2)]).model
    torch.manual_seed(42)
    expected = reference.fairness_grad(copy.deepcopy(base), .5, frame, sensitive_classes=2,
        output_cols_each_task=[(0,2)], col_names=['Smiling','gender'],
        impt_type=0, stop_batch=2, para_batch=1, delta_p=delta_p)
    targets = copy.deepcopy(reference.LAST_GRAD_TARGET)
    torch.manual_seed(42)
    actual = six_masks(copy.deepcopy(base), frame, delta_p=delta_p)
    for group in (0,1):
        for name in scores[0][group]: assert torch.equal(scores[0][group][name], scores[1][group][name])
    for a, b in zip(actual, expected): assert torch.equal(a, b)
    for stat in prune.FG_ORIG_LAST_STATS:
        np.testing.assert_array_equal(stat['target_prop'], targets[stat['name']])
    assert columns == [['Smiling','gender']]*2


def test_unused_attribute_does_not_change_importance(original_context):
    frame, _, scores = original_context
    base = prune.FairGRAPE(TinyNet(), nn.CrossEntropyLoss(), None, [(0,2)]).model
    torch.manual_seed(9)
    first = six_masks(copy.deepcopy(base), frame)
    changed = frame.copy(); changed['Unselected'] = [1]*8
    torch.manual_seed(9)
    second = six_masks(copy.deepcopy(base), changed)
    for group in (0,1):
        for name in scores[0][group]: assert torch.equal(scores[0][group][name], scores[1][group][name])
    for a, b in zip(first, second): assert torch.equal(a, b)


def test_impt4_original_route_still_matches_reference(original_context, monkeypatch):
    frame, _, scores = original_context
    monkeypatch.setattr(prune, 'FG_ORIG_MODE', 'original')
    monkeypatch.setattr(config, 'glo_impt_type', 4)
    monkeypatch.setattr(config, 'glo_fg_scope', 'all')
    base = prune.FairGRAPE(TinyNet(), nn.CrossEntropyLoss(), None, [(0,2)]).model
    torch.manual_seed(42)
    expected = reference.fairness_grad(copy.deepcopy(base), .5, frame, sensitive_classes=2,
        output_cols_each_task=[(0,2)], col_names=['Smiling','gender'], impt_type=0, stop_batch=2)
    torch.manual_seed(42)
    actual = prune.fairness_grad(copy.deepcopy(base), .5, frame, sensitive_classes=2,
        output_cols_each_task=[(0,2)], col_names=['Smiling','gender'], impt_type=4, stop_batch=2)
    for group in (0,1):
        for name in scores[0][group]: assert torch.equal(scores[0][group][name], scores[1][group][name])
    for a, b in zip(actual, expected): assert torch.equal(a, b)


def test_scoring_isolation_and_retraining_masks(original_context):
    frame, _, _ = original_context
    pruner = prune.FairGRAPE(TinyNet(), nn.CrossEntropyLoss(), None, [(0,2)])
    pruner.update_model(pruner.get_model())
    assert all(p.requires_grad for n,p in pruner.model.named_parameters() if n.endswith('.mask'))
    before = copy.deepcopy(pruner.prun_model.state_dict())
    cfgs = [.5,frame,None,2,True,[(0,2)],['Smiling','gender'],1,6,2,0,'mobilenetv2','gender']
    result = pruner.prune(cfgs)
    for name, value in before.items():
        if name.endswith('.mask'): continue
        mask = result.state_dict().get(name[:-7]+'.mask') if name.endswith('.weight') else None
        expected = value * mask if mask is not None else value
        assert torch.equal(result.state_dict()[name], expected), name
    runner.validate_original_masks(result, .5, {})
    assert not torch.equal(pruner.model.bn.running_mean, result.bn.running_mean)
    trained = copy.deepcopy(result)  # Same copy boundary as train().
    optimizer = torch.optim.Adam([p for p in trained.parameters() if p.requires_grad], lr=.01)
    images = torch.randn(4,3,3,3)
    loss = nn.functional.cross_entropy(trained(images), torch.tensor([0,1,0,1]))
    loss.backward(); optimizer.step()
    previous = {}
    for name, layer in trained.named_modules():
        if hasattr(layer, 'mask'):
            assert torch.equal(layer.mask, result.state_dict()[name+'.mask'])
            assert not layer.mask.requires_grad
            assert torch.count_nonzero(layer.weight[layer.mask == 0]) == 0
            previous[name] = layer.mask.detach().clone()
    pruner.update_model(trained); pruner.update_forward_pass()
    assert all(not p.requires_grad for n,p in pruner.model.named_parameters() if n.endswith('.mask'))
    cfgs[0] = .75
    second = pruner.prune(cfgs)
    runner.validate_original_masks(second, .75, previous)


def test_all_53_layers_and_statistics(original_context, monkeypatch):
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(weights=None, num_classes=2)
    pruner = prune.FairGRAPE(model, nn.CrossEntropyLoss(), None, [(0,2)])
    names = list(prune._get_fg_scope_layers(pruner.model, 'all'))
    assert len(names) == 53
    assert {'features.0.0','features.18.0','classifier.1'} <= set(names)
    # Exercise the full model's layer dispatch with the real original selector;
    # keeping 1% makes this CPU integration check inexpensive.
    def importance(*args, **kwargs):
        h = {g: {n: torch.ones_like(dict(pruner.model.named_modules())[n].weight).cpu() * (g+1)
                 for n in names} for g in (0,1)}
        return None, h
    monkeypatch.setattr(reference, 'importance_by_class0', importance)
    masks = six_masks(pruner.model, original_context[0], rate=.99)
    for name, mask in zip(names, masks):
        assert int(mask.sum()) == int(mask.numel() * (1-.99)), name
        dict(pruner.prun_model.named_modules())[name].mask = nn.Parameter(mask, requires_grad=False)
    stats = reporting.pruning_statistics(pruner.prun_model, 6)
    count = sum(mask.numel() for mask in masks)
    assert f'/ {count:,} (' in stats['프루닝된 파라미터 / 프루닝 대상 파라미터']


def test_cli_defaults_do_not_change_legacy_paths(monkeypatch):
    from types import SimpleNamespace
    import main_test
    for impt, delta, scope in [(5,.5,'blocks'), (6,0,'all')]:
        args = SimpleNamespace(impt=impt, delta_p=None, fg_scope=None)
        with patch.object(runner, 'run', return_value='dispatched'):
            assert main_test.experiment(args) == 'dispatched'
        assert (args.delta_p, args.fg_scope) == (delta, scope)
