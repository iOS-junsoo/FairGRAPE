"""Independent neuron gates, Taylor math, budgets and serialization."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torchvision.models import mobilenet_v2, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taylor_pruning import (TaylorPruner, attach_gates, conv_bn_pairs,
                            statistics, trainable_weights, validate_masks)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    model = nn.Sequential()
    model.add_module('features', nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4))))
    model.add_module('pool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('classifier', nn.Linear(4, 2))
    return model


def test_full_backbone_counts_identity_and_optimizer_excludes_gates():
    torch.manual_seed(3)
    model = mobilenet_v2(weights=None, num_classes=2).eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = model(inputs)
    original_keys = set(model.state_dict())
    pruner = TaylorPruner(model)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), before, rtol=0, atol=0)
    assert original_keys.issubset(model.state_dict())
    assert len(pruner.pairs) == 52
    assert statistics(model)['initial_neurons'] == 17056
    assert statistics(model)['total_weights'] == 2192320
    assert pruner.target_weights == 1534624
    assert pruner.per_round == 342
    ids = {id(p) for p in trainable_weights(model)}
    assert all(id(bn.taylor_gate) not in ids for _, _, _, bn in pruner.pairs)
    assert id(model.classifier[1].weight) in ids
    assert not any(key.startswith('classifier') and key.endswith('.mask') for key in model.state_dict())


def test_gate_gradient_matches_finite_difference_and_bn_bias_is_blocked():
    torch.manual_seed(4)
    model = attach_gates(tiny_model()).double().eval()
    _, conv, _, bn = conv_bn_pairs(model)[0]
    x = torch.randn(3, 3, 5, 5, dtype=torch.double)
    target = torch.tensor([0, 1, 0])
    loss = nn.functional.cross_entropy(model(x), target)
    loss.backward()
    analytic = bn.taylor_gate.grad[0].item()
    losses = []
    with torch.no_grad():
        for z in (1 + 1e-5, 1 - 1e-5):
            bn.taylor_gate[0] = z
            losses.append(nn.functional.cross_entropy(model(x), target).item())
        bn.taylor_gate[0] = 0
        bn.bias[0] = 7
        conv.mask[0] = 0
        conv.weight[0] = 0
        assert torch.count_nonzero(model.features(x)[:, 0]) == 0
    assert analytic == pytest.approx((losses[0] - losses[1]) / 2e-5, rel=1e-5)


def test_square_then_mean_ema_and_masked_momentum():
    model = tiny_model()
    pruner = TaylorPruner(model, target=.4, fraction=.25, ema=.9)
    _, conv, _, bn = pruner.pairs[0]
    optimizer = torch.optim.SGD(trainable_weights(model), lr=.1, momentum=.9)
    for grad in ([1, 2, 3, 4], [-1, -2, -3, -4]):
        bn.taylor_gate.grad = torch.tensor(grad, dtype=torch.float)
        pruner.accumulate()
    optimizer.state[conv.weight]['momentum_buffer'] = torch.ones_like(conv.weight)
    event = pruner.prune(optimizer)
    assert event['selected'][0]['neuron'] == 0
    torch.testing.assert_close(pruner.scores[0], torch.tensor([1., 4, 9, 16], dtype=torch.double))
    assert torch.count_nonzero(optimizer.state[conv.weight]['momentum_buffer'][0]) == 0
    bn.taylor_gate.grad = torch.tensor([5., 1., 1., 1.])
    pruner.accumulate()
    pruner.prune(optimizer)
    torch.testing.assert_close(pruner.scores[0], torch.tensor([.9, 3.7, 8.2, 14.5], dtype=torch.double))
    classifier_before = model.classifier.weight.detach().clone()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(torch.randn(4, 3, 8, 8)), torch.tensor([0, 1, 0, 1])).backward()
        pruner.mask_gradients()
        assert torch.count_nonzero(conv.weight.grad[conv.mask == 0]) == 0
        optimizer.step()
        pruner.enforce(optimizer)
        validate_masks(model)
    assert not torch.equal(classifier_before, model.classifier.weight)


@pytest.mark.parametrize('name', ['features.0.0', 'features.2.conv.0.0', 'features.2.conv.1.0',
                                  'features.3.conv.2', 'features.18.0'])
def test_each_conv_kind_is_independent_and_residual_survives(name):
    model = mobilenet_v2(weights=None, num_classes=2).eval()
    pruner = TaylorPruner(model, target=1e-7)
    for _, _, _, bn in pruner.pairs:
        bn.taylor_gate.grad = torch.ones_like(bn.taylor_gate)
    selected = next(pair for pair in pruner.pairs if pair[0] == name)
    selected[3].taylor_gate.grad[0] = 0
    pruner.accumulate()
    event = pruner.prune()
    assert len(event['selected']) == 1
    assert event['selected'][0]['layer'] == name
    for candidate, conv, _, bn in pruner.pairs:
        assert int((bn.taylor_gate == 0).sum()) == (1 if candidate == name else 0)
        if candidate != name:
            assert bool((conv.mask == 1).all())
    if name == 'features.3.conv.2':
        inputs = torch.randn(2, 24, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(model.features[3](inputs)[:, 0], inputs[:, 0], atol=0, rtol=0)


def test_global_target_70_percent_ties_floor_and_roundtrip():
    torch.manual_seed(5)
    model = mobilenet_v2(weights=None, num_classes=2).eval()
    pruner = TaylorPruner(model, fraction=1)
    classifier = copy.deepcopy(model.classifier.state_dict())
    for _, _, _, bn in pruner.pairs:
        bn.taylor_gate.grad = torch.ones_like(bn.taylor_gate)
    pruner.accumulate()
    event = pruner.prune()
    assert pruner.reached
    assert .7 <= event['after']['weight_sparsity'] < .70044
    assert len(event['selected']) < pruner.per_round
    assert event['selected'][0]['layer'] == 'features.0.0'
    assert event['selected'][0]['neuron'] == 0
    assert event['after']['removed_weights'] - event['selected'][-1]['removed_weights'] < pruner.target_weights
    assert all(int(bn.taylor_gate.sum()) >= 1 for _, _, _, bn in pruner.pairs)
    for key, value in classifier.items():
        torch.testing.assert_close(model.classifier.state_dict()[key], value, rtol=0, atol=0)
    restored = attach_gates(mobilenet_v2(weights=None, num_classes=2)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    validate_masks(restored)
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    assert statistics(restored) == statistics(model)


def test_impossible_targets_and_nonfinite_scores_fail():
    with pytest.raises(ValueError, match='cannot be reached'):
        TaylorPruner(tiny_model(), target=.9)
    pruner = TaylorPruner(tiny_model(), target=.1)
    pruner.pairs[0][3].taylor_gate.grad = torch.full((4,), float('nan'))
    with pytest.raises(ValueError, match='Non-finite'):
        pruner.accumulate()


def test_resnet18_all_convs_identity_and_independent_shortcut_masks():
    model = resnet18(weights=None, num_classes=2).eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = model(inputs)
    pruner = TaylorPruner(model, target=1e-8)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), before, atol=0, rtol=0)
    assert statistics(model)['initial_neurons'] == 4800
    assert statistics(model)['conv_layers'] == 20
    assert statistics(model)['total_weights'] == 11167936
    assert pruner.per_round == 96
    assert [n for n, _, _, _ in pruner.pairs if 'downsample' in n] == [
        'layer2.0.downsample.0', 'layer3.0.downsample.0', 'layer4.0.downsample.0']
    for _, _, _, bn in pruner.pairs:
        bn.taylor_gate.grad = torch.ones_like(bn.taylor_gate)
    model.layer1[0].bn2.taylor_gate.grad[0] = 0
    pruner.accumulate()
    event = pruner.prune()
    assert event['selected'][0]['layer'] == 'layer1.0.conv2'
    assert len(event['selected']) == 1
    x = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(model.layer1[0](x)[:, 0], x[:, 0].relu(), atol=0, rtol=0)
    assert bool((model.layer1[1].conv1.mask == 1).all())
    assert all(bool((conv.mask == 1).all()) for n, conv, _, _ in pruner.pairs if 'downsample' in n)
    assert not hasattr(model.fc, 'mask')
    model.zero_grad(set_to_none=True)
    model(inputs).sum().backward()
    assert all(bn.taylor_gate.grad is not None for _, _, _, bn in pruner.pairs)
    validate_masks(model)
