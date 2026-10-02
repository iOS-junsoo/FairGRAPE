"""LFWA contracts and real importance paths, without training a full experiment."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
import lfwa
import prune
from lfwa_experiment import planned_iterations
from train_and_val import loss_multi_tasks
from util import make_model

torch.set_num_threads(2)


def test_targets_output_and_loss_do_not_use_young():
    assert len(lfwa.TARGETS) == 39 and 'Male' in lfwa.TARGETS and 'young' not in lfwa.TARGETS
    model = make_model(network='mobilenetv2', n_classes=78, pretrained=False).eval()
    outputs = model(torch.randn(2, 3, 32, 32))
    labels = torch.randint(0, 2, (2, 40))
    assert outputs.shape == (2, 78)
    loss = loss_multi_tasks(outputs, labels, nn.CrossEntropyLoss(), lfwa.OUTPUT_SLICES)
    expected = nn.functional.cross_entropy(outputs.reshape(-1, 2), labels[:, :39].reshape(-1))
    assert torch.allclose(loss.float(), expected)
    changed = labels.clone()
    changed[:, -1] = 1 - changed[:, -1]
    other = loss_multi_tasks(outputs, changed, nn.CrossEntropyLoss(), lfwa.OUTPUT_SLICES)
    assert torch.equal(loss, other)
    assert torch.equal(torch.autograd.grad(loss, outputs, retain_graph=True)[0], torch.autograd.grad(other, outputs)[0])


def test_eo_known_value_and_missing_cell():
    y = np.array([[1], [1], [0], [0], [1], [1], [0], [0]])
    pred = np.array([[1], [1], [1], [0], [1], [0], [0], [0]])
    group = np.array([0] * 4 + [1] * 4)
    table, summary = lfwa.binary_metrics(y, pred, group, ['attr'])
    assert summary['eo'] == .5
    assert summary['ba'] == .75
    assert summary['accuracy'] == .75
    assert summary['eo_valid_attributes'] == 1
    table, summary = lfwa.binary_metrics(y[:6], pred[:6], group[:6], ['attr'])
    assert np.isnan(table.eo.iloc[0]) and summary['eo'] is None
    assert summary['eo_valid_attributes'] == 0


class TinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Sequential(nn.Conv2d(1, 2, 1, bias=False)))

    def forward(self, x):
        return self.conv(x)


class TinyImportanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Identity(), TinyFeatures())
        self.classifier = nn.Sequential(nn.Identity(), nn.Linear(2, 78, bias=False))
        with torch.no_grad():
            self.features[1].conv[0][0].weight.copy_(torch.tensor([1., 2.]).reshape(2, 1, 1, 1))
            self.classifier[1].weight[::2].zero_()
            self.classifier[1].weight[1::2].fill_(.1)

    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


def test_actual_phi_and_performance_paths(monkeypatch):
    model = TinyImportanceModel()
    images = torch.arange(1., 5.).reshape(4, 1, 1, 1)
    labels = torch.tensor([0, 0, 1, 1]).reshape(4, 1).repeat(1, 40)
    labels[:, -1] = torch.tensor([0, 1, 0, 1])
    frame = pd.DataFrame({'split': ['train'] * 4})
    calls = []

    def loader_factory(train, val, give, batch, col_used):
        assert train.equals(frame) and val.equals(frame)
        assert col_used == lfwa.LABEL_COLUMNS
        calls.append(batch)
        # Multiple batches exercise pooled cell statistics, not per-batch gaps.
        loader = DataLoader(TensorDataset(images, labels.clone()), batch_size=2)
        return loader, loader

    monkeypatch.setattr(prune, 'make_datasets', loader_factory)
    monkeypatch.setattr(prune, 'device', torch.device('cpu'))
    monkeypatch.setattr(config, 'glo_n_groups', 2)
    kwargs = dict(new_img_dir=None, output_cols_each_task=lfwa.OUTPUT_SLICES,
                  col_names=lfwa.LABEL_COLUMNS, stop_batch=10000, sensitive_group='young')
    phi, _, gaps, _, gradients = prune.compute_phi_k(model, frame, **kwargs)
    key = 'features.1.conv.0.0'
    assert torch.allclose(gaps[key], torch.tensor([2., 4.], dtype=torch.float64))
    assert torch.allclose(phi[key], (gaps[key] * gradients[key]).float())
    assert phi[key].min() > 0
    state_before = copy.deepcopy(model.state_dict())
    importance, _ = prune.compute_importance(model, None, frame, imp_batch_size=128, **kwargs)
    assert importance and all(torch.isfinite(value).all() for value in importance.values())
    assert all(torch.equal(state_before[k], v) for k, v in model.state_dict().items())
    labels[:, -1] = 0
    no_groups, *_ = prune.compute_phi_k(model, frame, **kwargs)
    assert torch.count_nonzero(no_groups[key]) == 0
    assert calls == [64, 128, 64]
    assert all(not module._forward_hooks for module in model.modules())


def test_pruning_schedule():
    assert planned_iterations(.975, .9) == 91
    assert .975 ** 90 > .1 and .975 ** 91 <= .1
    assert planned_iterations(.9, .9) == 22  # historical CelebA schedule
    with pytest.raises(ValueError):
        planned_iterations(1., .9)


def test_checkpoint_masks_and_metadata(tmp_path):
    model = make_model(network='mobilenetv2', n_classes=78, pretrained=False).eval()
    layer = model.features[1].conv[0][0]
    layer.mask = nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)
    with torch.no_grad():
        layer.mask[0] = 0
        layer.weight[0] = 0
    metadata = {'dataset': 'LFWA', 'targets': lfwa.TARGETS, 'split_sha256': {'train': 'test-fixture'}}
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(inputs)
    checkpoint = lfwa.save_checkpoint(model, tmp_path / 'model.pt', metadata)
    restored, _ = lfwa.load_checkpoint_model(tmp_path / 'model.pt', metadata)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(expected, restored(inputs))
    assert checkpoint['sparsity']['target_removed'] == layer.weight[0].numel()
    wrong = copy.deepcopy(metadata)
    wrong['targets'] = list(reversed(lfwa.TARGETS))
    with pytest.raises(ValueError):
        lfwa.load_checkpoint_model(tmp_path / 'model.pt', wrong)
    with pytest.raises(ValueError):
        lfwa.validate_checkpoint({'model_state': model.state_dict()}, metadata)


def test_real_official_manifest():
    if not Path('csv/LFWA.csv').is_file():
        pytest.skip('Run scripts/prepare_lfwa.py first; required before training')
    frames, metadata = lfwa.load_frames()
    _, labels, official = lfwa.official_annotations('Images/LFWA/raw')
    assert frames['train'].source_index.tolist() == (official['train'][:5000] + 1).tolist()
    assert frames['val'].source_index.tolist() == (official['train'][5000:] + 1).tolist()
    assert frames['test'].source_index.tolist() == (official['test'] + 1).tolist()
    for split, size in lfwa.SPLIT_SIZES.items():
        part = frames[split]
        assert len(part) == size
        assert np.array_equal(part[lfwa.LABEL_COLUMNS], labels.iloc[part.source_index.to_numpy() - 1])
        assert all(Path(p).is_absolute() for p in part.face_name_align)
    modified = frames['all'].copy()
    modified.loc[0, 'young'] ^= 1
    assert lfwa.metadata_for(modified)['manifest_sha256'] != metadata['manifest_sha256']


def test_lfwa_routes_train_validation_and_defers_test(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import lfwa_experiment as experiment
    args = SimpleNamespace(network='mobilenetv2', loss_type='classes', sensitive_group='young', impt=2,
        no_grl=True, drop_race=0, race_binary=False, skew_beta=0, prune_type='FairGRAPE', score_only=False,
        save_impt=False, print_acc=False, perf_only=False, phi_gap_only=False, no_init_train=False,
        checkpoint=None, retrain_iter=5, seed=42, lfwa_csv='fixture', lfwa_image_root='fixture',
        lfwa_run_dir=str(tmp_path), no_pretrained=False, keep_per_iter=.975, impt2_alpha=.3,
        prune_rate=.9, init_pruned=0, prune_iter=1, batch=256, exp_idx=0, para_batch=1,
        stop_batch=10000, delta_p=.5, retrain_lr=1e-5, no_retrain=False, defer_test=True)
    frames = {s: pd.DataFrame({'split': [s]}) for s in ('train', 'val', 'test')}
    metadata = {'dataset': 'LFWA', 'targets': lfwa.TARGETS}
    monkeypatch.setattr(lfwa, 'load_frames', lambda *a: (frames, metadata))
    cpu_device = torch.device('cpu')
    monkeypatch.setattr(experiment, 'torch', SimpleNamespace(device=lambda *a: cpu_device, nn=nn,
                        __version__=torch.__version__, version=torch.version))
    monkeypatch.setattr(experiment, 'make_model', lambda **kw: TinyImportanceModel())
    monkeypatch.setattr(experiment, 'setseed', lambda seed: None)
    calls = []

    def loaders(train, val, give, batch, columns):
        assert train is frames['train'] and val is frames['val']
        assert batch == 256 and columns == lfwa.LABEL_COLUMNS
        return train, val

    def trainer(model, criterion, loaders, lrs, epochs, targets, outputs, cfgs, **kwargs):
        assert loaders['train'] is frames['train'] and loaders['test'] is frames['val']
        assert cfgs[4] is frames['val'] and cfgs[5] is None
        assert targets == lfwa.TARGETS and outputs == lfwa.OUTPUT_SLICES
        assert kwargs['use_debiasing'] is False
        calls.append((lrs, epochs))
        return model, None

    class Pruner:
        def __init__(self, model, *args):
            self.model = model
        def init_mask(self):
            pass
        def prune(self, cfgs, show):
            assert cfgs[1] is frames['train'] and cfgs[2] is None
            assert cfgs[6] == lfwa.LABEL_COLUMNS and cfgs[-1] == 'young'
            return self.model
        def update_model(self, model):
            self.model = model

    evaluations = []
    monkeypatch.setattr(experiment, 'make_datasets', loaders)
    monkeypatch.setattr(experiment, 'train', trainer)
    monkeypatch.setattr(prune, 'FairGRAPE', Pruner)
    monkeypatch.setattr(lfwa, 'evaluate_model', lambda model, frame, *a: evaluations.append(frame))
    # Restore shared global state after orchestration probes.
    with monkeypatch.context() as scope:
        for key in list(vars(config)):
            if key.startswith('glo_'):
                scope.setattr(config, key, getattr(config, key))
        scope.setattr(prune, 'IMPT_TYPE2_ALPHA', prune.IMPT_TYPE2_ALPHA)
        experiment.run(args)
    assert calls == [([1e-4, 1e-5, 1e-6], [8, 1, 1]), ([1e-5], [5]), ([1e-4, 1e-5, 1e-6], [8, 1, 1])]
    assert len(evaluations) == 1 and evaluations[0] is frames['val']


def test_lfwa_keeps_best_across_learning_rate_stages(monkeypatch):
    import train_and_val as training
    model = nn.Linear(1, 1)
    seen = []

    def stage(candidate, loaders, criterion, optimizer, *args, **kwargs):
        assert optimizer.param_groups[0]['params'][0] is candidate.weight
        index = len(seen)
        with torch.no_grad():
            candidate.weight.fill_(index + 1.)
        seen.append(candidate)
        return candidate, [.9, .8, .7][index], [.1, .2, .3][index], {}

    monkeypatch.setattr(training, 'train_model0', stage)
    selected, _ = training.train(model, nn.CrossEntropyLoss(), {}, cfgs=['LFWA'])
    assert selected.weight.item() == 1.0
    assert len({id(candidate) for candidate in seen}) == 3


def test_report_exports_deltas_and_missing_eo(tmp_path):
    from lfwa_report import build_report
    for name, accuracy, eo in [('dense', .8, .2), ('pruned', .79, .1)]:
        folder = tmp_path / 'evaluation' / name
        folder.mkdir(parents=True)
        summary = {'accuracy': accuracy, 'ba': .7, 'eo': eo, 'eo_valid_attributes': 1,
                   'target_sparsity': .9 if name == 'pruned' else 0,
                   'model_sparsity': .7 if name == 'pruned' else 0}
        (folder / 'summary.json').write_text(json.dumps(summary))
        pd.DataFrame({'attribute': ['Male', 'Bald'], 'accuracy': [accuracy, accuracy],
                      'ba': [.7, .6], 'eo': [eo, np.nan]}).to_csv(folder / 'attributes.csv', index=False)
        run = tmp_path / name
        run.mkdir()
        pd.DataFrame({'phase': ['train', 'val'], 'epoch': [1, 1], 'accuracy': [accuracy, accuracy],
                      'loss': [.3, .4], 'eo': [eo, eo]}).to_csv(run / 'epochs.csv', index=False)
    torch.save({'iteration': 1, 'sparsity': {'target_sparsity': .025, 'model_sparsity': .02},
                'validation': {'accuracy': .79, 'eo': .1}}, tmp_path / 'pruned/prune_001.pt')
    build_report(tmp_path)
    output = tmp_path / 'evaluation'
    changes = pd.read_csv(output / 'attribute_changes.csv')
    assert changes.accuracy_delta.iloc[0] == pytest.approx(-.01)
    assert changes.eo_delta.iloc[0] == pytest.approx(-.1)
    assert np.isnan(changes.eo_delta.iloc[1])
    assert (output / 'training_curves.png').stat().st_size > 1000
    assert (output / 'training_curves.pdf').read_bytes().startswith(b'%PDF')
