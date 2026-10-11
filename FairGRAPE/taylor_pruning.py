"""Taylor-FO-BN gates for MobileNetV2 and torchvision ResNet-18 convolutions.

Reference: Molchanov et al., Importance Estimation for Neural Network Pruning
(CVPR 2019), https://arxiv.org/abs/1906.10771. Gates are independent: no
downstream input column or residual identity is implicitly pruned.
"""
import math

import torch
from torch import nn


METHOD = dict(impt_type=7, unit='neuron', method='taylor_fo_bn',
              scope='all_backbone_convs')


class GatedBatchNorm2d(nn.BatchNorm2d):
    """Preserve BN state keys, adding only a non-optimized binary gate."""

    def forward(self, inputs):
        return super().forward(inputs) * self.taylor_gate.view(1, -1, 1, 1)


def conv_bn_pairs(model):
    """Return Conv/BN pairs in model order, including shortcut projections."""
    modules = dict(model.named_modules())
    pairs = []
    for name, conv in modules.items():
        if not isinstance(conv, nn.Conv2d):
            continue
        parts = name.split('.')
        mobile = len(parts) >= 3 and parts[0] == 'features' and parts[1].isdigit() and 0 <= int(parts[1]) <= 18
        resnet = (name == 'conv1' or (parts[0] in ('layer1', 'layer2', 'layer3', 'layer4')
                  and (parts[-1] in ('conv1', 'conv2') or parts[-2:] == ['downsample', '0'])))
        if not (mobile or resnet):
            raise ValueError(f'Unsupported backbone convolution: {name}')
        parent_name = name.rpartition('.')[0]
        children = list(modules[parent_name].named_children())
        index = next(i for i, (_, child) in enumerate(children) if child is conv)
        if index + 1 >= len(children) or not isinstance(children[index + 1][1], nn.BatchNorm2d):
            raise ValueError(f'Convolution must be immediately followed by BatchNorm2d: {name}')
        bn_key, bn = children[index + 1]
        if bn.num_features != conv.out_channels:
            raise ValueError(f'Conv/BN channel mismatch: {name}')
        pairs.append((name, conv, '.'.join(filter(None, (parent_name, bn_key))), bn))
    if not pairs:
        raise ValueError('No backbone Conv/BN pairs found')
    return pairs


def attach_gates(model):
    """Attach after loading dense weights; safe to call on a gated model again."""
    for name, conv, bn_name, bn in conv_bn_pairs(model):
        if not isinstance(bn, GatedBatchNorm2d):
            gated = GatedBatchNorm2d(bn.num_features, bn.eps, bn.momentum,
                                     bn.affine, bn.track_running_stats)
            gated.to(device=conv.weight.device, dtype=conv.weight.dtype)
            gated.load_state_dict(bn.state_dict())
            gated.train(bn.training)
            gated.register_parameter('taylor_gate', nn.Parameter(conv.weight.new_ones(bn.num_features)))
            parent, _, key = bn_name.rpartition('.')
            model.get_submodule(parent)._modules[key] = gated
        if not hasattr(conv, 'mask'):
            conv.register_buffer('mask', torch.ones_like(conv.weight))
    return model


def trainable_weights(model):
    return [p for name, p in model.named_parameters()
            if p.requires_grad and not name.endswith('.taylor_gate')]


def statistics(model):
    pairs = conv_bn_pairs(model)
    total = sum(layer.weight.numel() for layer in model.modules()
                if isinstance(layer, (nn.Conv2d, nn.Linear)))
    target_total = sum(conv.weight.numel() for _, conv, _, _ in pairs)
    removed = sum(int((conv.mask == 0).sum()) for _, conv, _, _ in pairs if hasattr(conv, 'mask'))
    neurons = sum(conv.out_channels for _, conv, _, _ in pairs)
    removed_neurons = sum(int((bn.taylor_gate == 0).sum()) for _, _, _, bn in pairs
                          if hasattr(bn, 'taylor_gate'))
    original_parameters = sum(p.numel() for name, p in model.named_parameters()
                              if not name.endswith(('.mask', '.taylor_gate')))
    return dict(conv_layers=len(pairs), initial_neurons=neurons,
                removed_neurons=removed_neurons, live_neurons=neurons-removed_neurons,
                total_weights=total, target_weights=target_total, removed_weights=removed,
                weight_sparsity=removed / total, original_parameters=original_parameters,
                max_filter_weights=max(conv.weight[0].numel() for _, conv, _, _ in pairs))


def validate_masks(model):
    for name, conv, _, bn in conv_bn_pairs(model):
        if not isinstance(bn, GatedBatchNorm2d) or not hasattr(conv, 'mask'):
            raise ValueError(f'Missing Taylor gate/mask: {name}')
        gate = bn.taylor_gate.detach()
        if gate.shape != (conv.out_channels,) or not bool(((gate == 0) | (gate == 1)).all()):
            raise ValueError(f'Non-binary or malformed gate: {name}')
        if not bool(gate.any()):
            raise ValueError(f'All neurons pruned in {name}')
        expected = gate.view(-1, 1, 1, 1).expand_as(conv.weight)
        if conv.mask.shape != conv.weight.shape or not torch.equal(conv.mask, expected):
            raise ValueError(f'Mask does not match independent output gate: {name}')
        if bool((conv.weight.detach()[conv.mask == 0] != 0).any()):
            raise ValueError(f'Nonzero masked weights: {name}')
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and hasattr(layer, 'mask'):
            raise ValueError(f'Taylor classifier must not have a pruning mask: {name}')


class TaylorPruner:
    def __init__(self, model, target=0.7, fraction=0.02, ema=0.9):
        if not math.isfinite(target) or not 0 < target < 1:
            raise ValueError('Taylor target must be finite and in (0, 1)')
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('Taylor prune fraction must be in (0, 1]')
        if not math.isfinite(ema) or not 0 <= ema < 1:
            raise ValueError('Taylor EMA must be in [0, 1)')
        self.model = attach_gates(model)
        self.pairs = conv_bn_pairs(model)
        validate_masks(model)
        stats = statistics(model)
        self.target = target
        self.target_weights = math.ceil(target * stats['total_weights'])
        self.removed_weights = stats['removed_weights']
        maximum = sum((conv.out_channels - 1) * conv.weight[0].numel()
                      for _, conv, _, _ in self.pairs)
        if self.target_weights > maximum:
            raise ValueError(f'Target cannot be reached with one neuron per convolution: {maximum}/{stats["total_weights"]}')
        self.per_round = math.ceil(fraction * stats['initial_neurons'])
        self.ema = ema
        self.sums = [torch.zeros_like(bn.taylor_gate, dtype=torch.float64) for _, _, _, bn in self.pairs]
        self.scores = None
        self.batches = 0
        self.iteration = 0

    @property
    def reached(self):
        return self.removed_weights >= self.target_weights

    @torch.no_grad()
    def accumulate(self, gate_gradients=None):
        gradients = ([bn.taylor_gate.grad for _, _, _, bn in self.pairs]
                     if gate_gradients is None else list(gate_gradients))
        if len(gradients) != len(self.pairs):
            raise ValueError('Taylor gradient count does not match gates')
        for (_, _, _, bn), accumulated, gradient in zip(self.pairs, self.sums, gradients):
            if gradient is None:
                raise ValueError('Missing Taylor gate gradient')
            value = (bn.taylor_gate.detach().double() * gradient.detach().double()).square()
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Non-finite Taylor importance')
            accumulated.add_(value)
        self.batches += 1

    @torch.no_grad()
    def mask_gradients(self):
        for _, conv, _, _ in self.pairs:
            if conv.weight.grad is not None:
                conv.weight.grad.mul_(conv.mask)

    @torch.no_grad()
    def enforce(self, optimizer=None):
        for _, conv, _, _ in self.pairs:
            conv.weight.mul_(conv.mask)
            if optimizer is not None:
                momentum = optimizer.state.get(conv.weight, {}).get('momentum_buffer')
                if momentum is not None:
                    momentum.mul_(conv.mask)

    @torch.no_grad()
    def prune(self, optimizer=None):
        if not self.batches:
            raise ValueError('No training batches accumulated for Taylor pruning')
        averages = [value / self.batches for value in self.sums]
        self.scores = averages if self.scores is None else [
            self.ema * old + (1 - self.ema) * new for old, new in zip(self.scores, averages)]
        candidates, live = [], []
        for index, ((name, conv, _, bn), score) in enumerate(zip(self.pairs, self.scores)):
            active = bn.taylor_gate.detach().cpu().tolist()
            live.append(sum(int(x != 0) for x in active))
            for channel, (on, importance) in enumerate(zip(active, score.cpu().tolist())):
                if on:
                    if not math.isfinite(importance):
                        raise ValueError(f'Non-finite Taylor score in {name}')
                    candidates.append((importance, index, channel))
        candidates.sort()
        before = statistics(self.model)
        removed = before['removed_weights']
        selected = []
        for importance, index, channel in candidates:
            if removed >= self.target_weights or len(selected) >= self.per_round:
                break
            if live[index] <= 1:
                continue
            name, conv, bn_name, bn = self.pairs[index]
            cost = conv.weight[0].numel()
            bn.taylor_gate[channel] = 0
            conv.mask[channel] = 0
            live[index] -= 1
            removed += cost
            selected.append(dict(layer=name, bn=bn_name, neuron=channel, importance=importance,
                                 removed_weights=cost))
        if not selected and removed < self.target_weights:
            raise RuntimeError('No removable neurons remain before target sparsity')
        self.enforce(optimizer)
        validate_masks(self.model)
        self.removed_weights = removed
        self.iteration += 1
        result = dict(iteration=self.iteration, scoring_batches=self.batches,
                      target_sparsity=self.target, per_round=self.per_round,
                      selected=selected, before=before, after=statistics(self.model))
        for value in self.sums:
            value.zero_()
        self.batches = 0
        return result
