"""CPU regressions for the direct original-FairGRAPE adapter (no images needed)."""

import copy
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prune_orig_ref as reference
from fairgrape_original import original_fairgrape_masks


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 4, bias=False)
        self.output = nn.Linear(4, 2, bias=False)
        for layer in (self.hidden, self.output):
            layer.mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.forward = types.MethodType(reference.forward_mapping_dict['Linear'], layer)

    def forward(self, inputs):
        return self.output(torch.tanh(self.hidden(inputs)))


@pytest.fixture(autouse=True)
def cpu_reference(monkeypatch):
    monkeypatch.setattr(reference, 'device', torch.device('cpu'))
    monkeypatch.setattr(reference, 'LAST_GRAD_TARGET', {})


def _scores(model, tied=False):
    return {
        group: {
            name: (torch.ones_like(layer.weight) if tied else
                   torch.arange(1, layer.weight.numel() + 1, dtype=torch.float32).reshape_as(layer.weight)
                   .roll(group, dims=0)) * (group + 1)
            for name, layer in model.named_modules() if type(layer).__name__ in reference.supported_layers
        }
        for group in (0, 1)
    }


def _mock_scores(monkeypatch, scores):
    monkeypatch.setattr(reference, 'importance_by_class0',
                        lambda *args, **kwargs: (None, copy.deepcopy(scores)))


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('delta_p', [0, 1, 2])
def test_masks_match_reference_including_exact_ties(monkeypatch, tied, delta_p):
    model = TinyNet()
    _mock_scores(monkeypatch, _scores(model, tied=tied))
    expected = reference.fairness_grad(copy.deepcopy(model), 0.5, None,
                                      sensitive_classes=2, impt_type=0, delta_p=delta_p)
    masks, stats = original_fairgrape_masks(model, 0.5, None, delta_p=delta_p)
    for actual, target, stat in zip(masks, expected, stats):
        assert torch.equal(actual, target)
        assert stat['kept'] == stat['keep_target'] == int(actual.numel() * 0.5)
        assert np.allclose(stat['target_prop'], reference.LAST_GRAD_TARGET[stat['name']])
        assert stat['selected_prop'] is None and stat['picks'] == '-'


def test_second_iteration_matches_reference_and_preserves_scope(monkeypatch):
    model = TinyNet()
    _mock_scores(monkeypatch, _scores(model))
    first, _ = original_fairgrape_masks(model, 0.25, None, scope_layers=['hidden'])
    for layer, mask in zip((model.hidden, model.output), first):
        layer.mask = nn.Parameter(mask.clone(), requires_grad=False)
    expected = reference.fairness_grad(copy.deepcopy(model), 0.5, None,
                                      sensitive_classes=2, impt_type=0)
    second, stats = original_fairgrape_masks(model, 0.5, None, scope_layers=['hidden'])
    assert torch.equal(second[0], expected[0])
    assert torch.equal(second[1], first[1])
    assert torch.all(second[0] <= first[0])
    assert stats[0]['active_before'] == int(first[0].sum())
    assert stats[1]['removed'] == 0
    assert not stats[1]['in_scope']


def test_scope_preserves_masks_from_before_original_scoring(monkeypatch):
    model = TinyNet()
    model.output.mask.data[0, 0] = 0
    before = model.output.mask.detach().clone()
    scores = _scores(model)

    def mutate_masks(*args, **kwargs):
        # Original Adam scoring can turn binary masks into nonbinary values,
        # including formerly zero entries when the mask requires gradients.
        with torch.no_grad():
            model.output.mask.add_(0.2)
        return None, copy.deepcopy(scores)

    monkeypatch.setattr(reference, 'importance_by_class0', mutate_masks)
    masks, stats = original_fairgrape_masks(model, 0.5, None, scope_layers=['hidden'])
    assert torch.equal(masks[1], before)
    assert not torch.equal(masks[1], model.output.mask)
    assert stats[1]['active_before'] == stats[1]['kept'] == 7


def test_full_scoring_matches_reference_and_only_changes_scoring_copy(monkeypatch):
    """Exercise upstream train/Adam/accumulated-grad scoring with two batches."""
    torch.manual_seed(12)
    frame = pd.DataFrame({'species': [0, 1, 1, 0], 'color': [0, 0, 1, 1]})
    labels = torch.tensor(frame[['species', 'color']].values)
    batches = [(torch.randn(4, 4), labels), (torch.randn(4, 4), labels)]
    monkeypatch.setattr(reference, 'make_datasets', lambda *args, **kwargs: (batches, batches))
    actual_model = TinyNet()
    untouched_model = copy.deepcopy(actual_model)
    expected_model = copy.deepcopy(actual_model)
    expected = reference.fairness_grad(
        expected_model, 0.5, frame, sensitive_classes=2, impt_type=0,
        output_cols_each_task=[(0, 2)], col_names=['species', 'color'], stop_batch=2,
    )
    actual, _ = original_fairgrape_masks(
        actual_model, 0.5, frame, output_cols_each_task=[(0, 2)],
        col_names=['species', 'color'], stop_batch=2,
    )
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    for actual_param, expected_param in zip(actual_model.parameters(), expected_model.parameters()):
        assert torch.equal(actual_param, expected_param)
    assert not torch.equal(actual_model.hidden.weight, untouched_model.hidden.weight)
    assert actual_model.training


def test_empty_scope_rejected_before_scoring():
    with pytest.raises(ValueError, match='no supported layers'):
        original_fairgrape_masks(TinyNet(), 0.5, None, scope_layers=[])


def test_local_pruner_original_mode_keeps_live_weights_and_cumulative_masks(monkeypatch, tmp_path):
    """Actual local dispatch, mask application/hooks, repeat round and log writer."""
    import config
    import prune

    monkeypatch.setattr(prune, 'device', torch.device('cpu'))
    monkeypatch.setattr(prune, 'FG_ORIG_MODE', 'original')
    monkeypatch.setattr(prune, 'FG_ORIG_BLOCK_LAYERS', None)
    monkeypatch.setattr(prune, 'FG_ORIG_LAST_STATS', None)
    for key, value in {
        'glo_impt_type': 4,
        'glo_fg_scope': 'all',
        'glo_fg_log_run_dir': str(tmp_path),
        'glo_prune_iter': 0,
        'glo_imp_rate': 0,
        'glo_dataset': 'DogsCats',
        'glo_seed': 42,
    }.items():
        monkeypatch.setattr(config, key, value)

    torch.manual_seed(42)
    frame = pd.DataFrame({'species': [0, 1, 1, 0], 'color': [0, 0, 1, 1]})
    labels = torch.tensor(frame[['species', 'color']].values)
    batches = [(torch.randn(4, 4), labels), (torch.randn(4, 4), labels)]
    monkeypatch.setattr(reference, 'make_datasets', lambda *args, **kwargs: (batches, batches))
    dense_model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Tanh(), nn.Linear(4, 2, bias=False))
    dense_weights = {name: layer.weight.detach().clone() for name, layer in dense_model.named_modules()
                     if isinstance(layer, nn.Linear)}
    pruner = prune.FairGRAPE(dense_model, nn.CrossEntropyLoss(), None, [(0, 2)])
    cfg = [0.5, frame, None, 2, True, [(0, 2)], ['species', 'color'], 1, 4, 2, 0,
           'mobilenetv2', 'color']
    pruned = pruner.prune(cfg)
    first_masks = {}
    for name, layer in pruned.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        assert not layer.mask.requires_grad
        assert torch.all((layer.mask == 0) | (layer.mask == 1))
        assert torch.equal(layer.weight, dense_weights[name] * layer.mask)
        first_masks[name] = layer.mask.detach().clone()
    # Original scoring changed its private copy, while the actual output uses
    # the untouched checkpoint weights multiplied by the selected binary mask.
    assert any(not torch.equal(dict(pruner.model.named_modules())[name].weight, weight)
               for name, weight in dense_weights.items())
    assert all(torch.equal(dict(dense_model.named_modules())[name].weight, weight)
               for name, weight in dense_weights.items())

    optimizer = torch.optim.SGD(pruned.parameters(), lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    nn.CrossEntropyLoss()(pruned(batches[0][0]), labels[:, 0]).backward()
    optimizer.step()
    for name, layer in pruned.named_modules():
        if name in first_masks:
            assert torch.count_nonzero(layer.weight[layer.mask == 0]) == 0
            assert torch.equal(layer.mask, first_masks[name])

    pruner.update_model(pruned)
    cfg[0] = 0.75
    monkeypatch.setattr(config, 'glo_prune_iter', 1)
    second = pruner.prune(cfg)
    for name, layer in second.named_modules():
        if name in first_masks:
            assert torch.all(layer.mask <= first_masks[name])
            assert int(layer.mask.sum()) == int(layer.weight.numel() * 0.25)
            assert torch.count_nonzero(layer.weight[layer.mask == 0]) == 0
            assert not layer.mask.requires_grad

    for filename in ('iter01.txt', 'iter02.txt'):
        log = (tmp_path / filename).read_text(encoding='utf-8')
        assert 'FG_ORIG_MODE : original' in log
        assert 'implementation: prune_orig_ref' in log
        assert 'sparsity_all' in log
    assert all(stat['in_scope'] for stat in prune.FG_ORIG_LAST_STATS)
    assert all(stat['picks'] == '-' for stat in prune.FG_ORIG_LAST_STATS)
