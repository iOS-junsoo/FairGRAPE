"""Separate fixed-BN scoring and transactional collapse checks for Taylor pruning."""
import copy

import torch
from torch import nn


def scoring_options(args):
    mode = getattr(args, 'taylor_score_mode', 'auto')
    if mode == 'auto':
        mode = 'frozen_bn' if args.network == 'mobilenetv2' else 'train'
    guard = getattr(args, 'taylor_collapse_guard', 'auto')
    enabled = args.network == 'mobilenetv2' if guard == 'auto' else guard == 'on'
    return mode, enabled


def accumulate_frozen_scores(pruner, images, target):
    """Only gate gradients are requested; .grad, weights and BN buffers stay intact.

    Use the same Train minibatch before its ordinary SGD update. Eval mode also
    disables dropout, so scoring neither consumes its RNG nor adds dropout noise.
    Restore every module's mode, including intentionally mixed training modes.
    """
    model = pruner.model
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        loss = nn.functional.cross_entropy(model(images), target)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite frozen-BN scoring loss')
        gates = [bn.taylor_gate for _, _, _, bn in pruner.pairs]
        gradients = torch.autograd.grad(loss, gates)
        pruner.accumulate(gradients)
        return float(loss.detach())
    finally:
        for module, training in modes:
            module.training = training


@torch.no_grad()
def probe_information_flow(model, images):
    """Detect exact input-independent block outputs, not constant class labels.

    Check eval first. If it collapses, also check with minibatch BN statistics:
    stale running statistics alone must not reject an otherwise viable network.
    The small fixed Train probe is a guard, not proof for all possible inputs.
    """
    if len(images) < 2 or not bool((images - images[:1]).abs().max() > 0):
        raise ValueError('Collapse probe requires at least two distinct Train images')
    modes = [(module, module.training) for module in model.modules()]
    bn_buffers = [(module, {key: value.clone() for key, value in module.named_buffers(recurse=False)
                           if value is not None}) for module in model.modules()
                  if isinstance(module, nn.BatchNorm2d)]
    blocks = ([(f'features.{i}', block) for i, block in enumerate(model.features)]
              if hasattr(model, 'features') else
              [(f'layer{i}', getattr(model, f'layer{i}')) for i in range(1, 5)])

    def measure(batch_bn):
        model.eval()
        if batch_bn:
            for module, _ in bn_buffers:
                module.train()
        measurements, hooks = [], []

        def capture(name):
            def hook(module, inputs, output):
                finite = bool(torch.isfinite(output).all())
                delta = float((output - output[:1]).abs().max()) if finite else None
                measurements.append(dict(layer=name, finite=finite, max_input_difference=delta))
            return hook

        try:
            for name, block in blocks:
                hooks.append(block.register_forward_hook(capture(name)))
            output = model(images)
            capture('logits')(model, None, output)
        finally:
            for hook in hooks:
                hook.remove()
        return dict(layers=measurements,
                    collapsed_layers=[m['layer'] for m in measurements if m['max_input_difference'] == 0],
                    nonfinite_layers=[m['layer'] for m in measurements if not m['finite']])

    try:
        evaluation = measure(False)
        training = measure(True) if evaluation['collapsed_layers'] or evaluation['nonfinite_layers'] else None
        collapsed = bool(evaluation['collapsed_layers'] and training and training['collapsed_layers'])
        nonfinite = bool(evaluation['nonfinite_layers'] or (training and training['nonfinite_layers']))
        return dict(rejected=collapsed or nonfinite, samples=len(images), eval=evaluation, batch_bn=training)
    finally:
        for module, buffers in bn_buffers:
            for key, value in buffers.items():
                getattr(module, key).copy_(value)
        for module, training in modes:
            module.training = training


class PruningCollapseError(RuntimeError):
    def __init__(self, diagnostics):
        super().__init__('Taylor information-flow collapse on Train probe; pruning stopped '
                         '(a rejected removal is rolled back). See rejected_iter log.')
        self.diagnostics = diagnostics


def guarded_prune(pruner, optimizer, images):
    """Rollback model, momentum and all score counters if a removal is rejected."""
    before = probe_information_flow(pruner.model, images)
    if before['rejected']:
        raise PruningCollapseError(dict(reason='already_collapsed_before_removal', before=before,
                                       attempted_iteration=pruner.iteration + 1, rolled_back=False))
    model_state = {key: value.detach().clone() for key, value in pruner.model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    previous = dict(scores=pruner.scores, sums=[value.clone() for value in pruner.sums],
                    batches=pruner.batches, iteration=pruner.iteration,
                    removed_weights=pruner.removed_weights)
    try:
        event = pruner.prune(optimizer)
        after = probe_information_flow(pruner.model, images)
        event['collapse_probe'] = dict(before=before, after=after)
        if after['rejected']:
            raise PruningCollapseError(dict(reason='collapse_after_removal', event=event,
                                           attempted_iteration=event['iteration'], rolled_back=True))
        return event
    except Exception:
        pruner.model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        for key, value in previous.items():
            setattr(pruner, key, value)
        raise
