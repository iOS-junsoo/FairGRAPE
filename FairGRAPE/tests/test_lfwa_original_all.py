"""LFWA 39-task labels through the actual original FairGRAPE dispatch."""
from types import SimpleNamespace
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from torch import nn
import config
import lfwa
import lfwa_original_all_experiment as entry
import prune
import prune_orig_ref as reference


def test_lfwa_multitask_original_dispatch(monkeypatch, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args = SimpleNamespace(para_batch=1, stop_batch=2, delta_p=0, fg_scope='all',
                           seed=42, no_pretrained=False, keep_per_iter=.975)
    monkeypatch.setattr(lfwa, 'sparsity', lfwa.sparsity)  # Restore process-local accounting after test.
    entry._configure(args, tmp_path, {})
    assert config.glo_fg_scope == 'all'
    assert config.glo_impt_type == 4 and config.glo_n_groups == 2
    assert prune.FG_ORIG_MODE == 'original' and config.glo_imp_rate is None
    monkeypatch.setattr(prune, 'device', torch.device('cpu'))
    monkeypatch.setattr(reference, 'device', torch.device('cpu'))
    # Tiny network substitutes only the architecture, so all-layer scope applies.
    monkeypatch.setattr(config, 'glo_fg_scope', 'all')
    labels = torch.randint(0, 2, (4, 40))
    labels[:, -1] = torch.tensor([0, 1, 0, 1])
    frame = pd.DataFrame(labels.numpy(), columns=lfwa.LABEL_COLUMNS)
    batches = [(torch.randn(4, 4), labels), (torch.randn(4, 4), labels)]
    monkeypatch.setattr(reference, 'make_datasets', lambda *a, **kw: (batches, batches))
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 78))
    dense = {n: t.clone() for n, t in model.state_dict().items()}
    pruner = prune.FairGRAPE(model, nn.CrossEntropyLoss(), batches, lfwa.OUTPUT_SLICES)
    pruner.init_mask()
    pruner.update_model(pruner.get_model())
    assert all(p.requires_grad for n, p in pruner.model.named_parameters() if n.endswith('.mask'))
    previous = None
    for iteration, rate in enumerate([.25, .5]):
        monkeypatch.setattr(config, 'glo_prune_iter', iteration)
        cfg = entry.prune_configuration(args, rate, frame)
        assert cfg[3] == 2 and cfg[8] == 4 and len(cfg[5]) == 39 and cfg[6][-1] == 'young'
        before = {n: t.clone() for n, t in pruner.prun_model.state_dict().items()}
        pruned = pruner.prune(cfg)
        current = lfwa.assert_masks(pruned)
        for name, layer in pruned.named_modules():
            if name in current:
                assert torch.equal(layer.weight, before[name + '.weight'] * layer.mask)
                assert int(layer.mask.sum()) == int(layer.weight.numel() * (1 - rate))
                if previous:
                    assert torch.all(current[name] <= previous[name])
        # Frozen masked forwards remain valid through multi-task retraining.
        optimizer = torch.optim.Adam(pruned.parameters(), lr=1e-5)
        optimizer.zero_grad()
        reference.loss_multi_tasks(pruned(batches[0][0]), labels, nn.CrossEntropyLoss(), lfwa.OUTPUT_SLICES).backward()
        optimizer.step()
        lfwa.assert_masks(pruned, current)
        previous = current
        pruner.update_model(pruned)
        assert not any(p.requires_grad for n, p in pruner.model.named_parameters() if n.endswith('.mask'))
    assert all(torch.equal(t, model.state_dict()[n]) for n, t in dense.items())


def test_all_layers_budget_and_accounting(monkeypatch):
    import pytest
    from util import make_model
    model = make_model(network='mobilenetv2', n_classes=78, pretrained=False)
    scope = prune._get_fg_scope_layers(model, 'all')
    assert len(scope) == 53
    assert {'features.0.0', 'features.18.0', 'classifier.1'} <= set(scope)
    dense = entry.all_scope_sparsity(model)
    assert dense['target_weights'] == 2289600 and dense['model_parameters'] == 2323790
    previous = {}
    for rate in [.25, .5]:
        stats = []
        removed = 0
        for name, layer in model.named_modules():
            if name not in scope:
                continue
            numel = layer.weight.numel()
            keep = int(numel * (1 - rate))
            mask = torch.zeros_like(layer.weight)
            mask.reshape(-1)[:keep] = 1
            layer.mask = nn.Parameter(mask, requires_grad=False)
            with torch.no_grad():
                layer.weight.mul_(mask)
            removed += numel - keep
            stats.append({'name': name, 'in_scope': True, 'kept': keep, 'keep_target': keep, 'target_prop': [.5, .5]})
        monkeypatch.setattr(prune, 'FG_ORIG_LAST_STATS', stats)
        entry.validate_pruning(model, rate, previous)
        counts = entry.all_scope_sparsity(model)
        assert counts['model_removed'] == counts['target_removed'] == removed
        assert counts['model_sparsity'] == removed / 2323790
        previous = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.endswith('.mask')}
    # Protecting the classifier, as the prior blocks adaptation did, must fail.
    with torch.no_grad():
        model.classifier[1].mask.fill_(1)
    with pytest.raises(AssertionError):
        entry.validate_pruning(model, .5, previous)
