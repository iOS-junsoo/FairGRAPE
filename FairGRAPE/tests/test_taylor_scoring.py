import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taylor_pruning import TaylorPruner, trainable_weights, validate_masks
from taylor_scoring import (accumulate_frozen_scores, guarded_prune, probe_information_flow,
                            PruningCollapseError, scoring_options)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def network():
    model = nn.Sequential()
    model.add_module('features', nn.Sequential(nn.Sequential(
        nn.Conv2d(3, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())))
    model.add_module('pool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('dropout', nn.Dropout(.2))
    model.add_module('classifier', nn.Linear(2, 2))
    return model


def test_frozen_scoring_isolated_from_training_gradients_buffers_modes_and_rng():
    torch.manual_seed(12)
    model = network().train()
    pruner = TaylorPruner(model, target=.1)
    reference = copy.deepcopy(model).eval()
    inputs = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    nn.functional.cross_entropy(reference(inputs), labels).backward()
    expected = reference.features[0][1].taylor_gate.grad.double().square()
    for p in model.parameters():
        p.grad = torch.full_like(p, 7)
    state = copy.deepcopy(model.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    accumulate_frozen_scores(pruner, inputs, labels)
    torch.testing.assert_close(pruner.sums[0], expected, atol=0, rtol=0)
    assert pruner.batches == 1
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(g, p.grad) for g, p in zip(gradients, model.parameters()))
    # An added scoring pass leaves the subsequent ordinary SGD step identical.
    reference = copy.deepcopy(model)
    opt = torch.optim.SGD(trainable_weights(model), lr=.01, momentum=.9)
    control_opt = torch.optim.SGD(trainable_weights(reference), lr=.01, momentum=.9)
    for candidate, optimizer in ((model, opt), (reference, control_opt)):
        torch.manual_seed(21)
        candidate.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(candidate(inputs), labels).backward()
        optimizer.step()
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in model.state_dict().items())


def test_rejected_pruning_restores_weights_masks_momentum_and_score_state():
    model = network().train()
    with torch.no_grad():
        model.features[0][0].weight.zero_()
        model.features[0][0].weight[0].fill_(1)
        model.features[0][1].bias[1] = -1
    pruner = TaylorPruner(model, target=.01)
    optimizer = torch.optim.SGD(trainable_weights(model), lr=.01, momentum=.9)
    for p in trainable_weights(model):
        optimizer.state[p]['momentum_buffer'] = torch.ones_like(p)
    pruner.pairs[0][3].taylor_gate.grad = torch.tensor([0., 1.])
    pruner.accumulate()
    before = copy.deepcopy(model.state_dict())
    sums = [s.clone() for s in pruner.sums]
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    images = torch.stack([torch.ones(3, 8, 8), torch.ones(3, 8, 8)*2])
    with pytest.raises(PruningCollapseError) as caught:
        guarded_prune(pruner, optimizer, images)
    assert caught.value.diagnostics['rolled_back']
    assert caught.value.diagnostics['event']['collapse_probe']['after']['batch_bn']['collapsed_layers']
    assert pruner.iteration == 0 and pruner.batches == 1 and pruner.removed_weights == 0
    assert pruner.scores is None
    assert all(torch.equal(a, b) for a, b in zip(sums, pruner.sums))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(bool((s['momentum_buffer'] == 1).all()) for s in optimizer.state.values())
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    validate_masks(model)


def test_guard_accepts_constant_class_with_input_dependent_logits():
    model = network().eval()
    with torch.no_grad():
        model.features[0][0].weight.fill_(1)
        model.classifier.weight.fill_(1)
        model.classifier.bias.copy_(torch.tensor([10., 0.]))
    inputs = torch.stack([torch.ones(3, 8, 8), torch.ones(3, 8, 8)*2])
    assert model(inputs).argmax(1).tolist() == [0, 0]
    result = probe_information_flow(model, inputs)
    assert not result['rejected']
    assert not result['eval']['collapsed_layers']


def test_stale_bn_only_collapse_does_not_reject_or_change_bn_buffers():
    model = network().train()
    with torch.no_grad():
        model.features[0][0].weight.fill_(1)
        model.features[0][1].running_mean.fill_(100)
    state = copy.deepcopy(model.state_dict())
    inputs = torch.stack([torch.ones(3, 8, 8), torch.ones(3, 8, 8)*2])
    result = probe_information_flow(model, inputs)
    assert result['eval']['collapsed_layers']
    assert not result['batch_bn']['collapsed_layers']
    assert not result['rejected']
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())


def test_defaults_preserve_resnet_scoring():
    assert scoring_options(SimpleNamespace(network='mobilenetv2')) == ('frozen_bn', True)
    assert scoring_options(SimpleNamespace(network='resnet18')) == ('train', False)
