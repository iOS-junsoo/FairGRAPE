#!/usr/bin/env python3
"""Continue paired LFWA pruning trajectories with a hard per-step channel cap.

This isolated runner never edits prune.py. It captures the production selector's
full-precision score/log payload, verifies the uncapped selection, then optionally
reselects with floor(cap_fraction * currently_active_channels) as an unrelaxed
per-block bound. Both arms recompute scores from their own current model.

Typical continuation after the single-step diagnostic::

  python scripts/run_lfwa_cap_trajectory.py \
    --baseline-checkpoint .../baseline/retrained.pt \
    --cap-checkpoint .../strict_cap10/retrained.pt \
    --run-dir lfwa_diagnostics/cap_trajectory --end-step 25

Use --preflight to check source models on CPU without loading image pixels or
training; add --replay-csv for a saved next-step CSV when both sources are equal.
--self-test exercises selector invariants without importing PyTorch. --resume
uses only atomically completed iterations, preserving interrupted directories.
"""

import argparse
import collections
import contextlib
import csv
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def select_channels(candidates, original, active, remove_target, floor_ratio,
                    cap_fraction=None):
    """Stable production score ordering, an original-width floor and strict cap.

    candidates is an insertion-ordered {(block, channel): (score, weight_count)}
    mapping. Protected and previously removed channels must already be absent.
    A budget deficit is reported rather than relaxing either constraint.
    """
    selected, removed = [], 0
    counts, floor_skips, cap_skips = (collections.Counter() for _ in range(3))
    floors = {block: math.ceil(width * floor_ratio) for block, width in original.items()}
    caps = {block: math.floor(active[block] * cap_fraction)
            for block in original} if cap_fraction is not None else {}
    for key, (score, weight) in candidates.items():
        block, channel = key
        if block not in original or not math.isfinite(score) or weight <= 0:
            raise ValueError(f'Invalid candidate: {key} -> {(score, weight)}')
        if isinstance(channel, bool) or channel < 0 or int(channel) != channel:
            raise ValueError(f'Invalid channel index: {key}')
    for key, (_score, weight) in sorted(candidates.items(), key=lambda item: item[1][0]):
        if removed >= remove_target:
            break
        block = key[0]
        if active[block] - counts[block] - 1 < floors[block]:
            floor_skips[block] += 1
            continue
        if cap_fraction is not None and counts[block] >= caps[block]:
            cap_skips[block] += 1
            continue
        selected.append(key)
        counts[block] += 1
        removed += int(weight)
    layers = []
    for block in sorted(original, key=lambda name: int(name.split('.')[1])):
        layers.append({'block': block, 'original_channels': original[block],
                       'active_before': active[block], 'removed_channels': counts[block],
                       'active_after': active[block] - counts[block],
                       'minimum_keep': floors[block], 'strict_channel_cap': caps.get(block),
                       'cap_zero_due_to_rounding': cap_fraction is not None and caps[block] == 0,
                       'skipped_by_floor': floor_skips[block], 'skipped_by_cap': cap_skips[block]})
    return selected, {'removed_weights': removed, 'remove_target': int(remove_target),
                      'budget_deficit': max(0, int(remove_target) - removed),
                      'budget_overshoot': max(0, removed - int(remove_target)), 'layers': layers}


def channel_layout(model):
    import torch
    import prune
    modules = dict(model.named_modules())
    original, active = {}, {}
    for number in range(1, 18):
        block = f'features.{number}'
        first, _, _ = prune._get_impt_type2_block_layer_names(block)
        layer = modules[first]
        width = int(layer.weight.shape[0])
        mask = getattr(layer, 'mask', torch.ones_like(layer.weight))
        original[block] = width
        active[block] = int((mask.reshape(width, -1).sum(1) > 0).sum())
    return original, active


def trusted_torch_load(path):
    """Explicit local-checkpoint loading across torch 1.11 and torch >= 2.6."""
    import torch
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)


def load_model(path, metadata, device):
    import torch
    import types
    import lfwa
    from prune import forward_mapping_dict
    from util import make_model
    checkpoint = trusted_torch_load(path)
    lfwa.validate_checkpoint(checkpoint, metadata)
    state = checkpoint.get('model_state', checkpoint.get('model_state_dict'))
    if state is None:
        raise ValueError(f'No model state in {path}')
    model = make_model(network='mobilenetv2', n_classes=78, pretrained=False)
    modules = dict(model.named_modules())
    for name, value in state.items():
        if name.endswith('.mask'):
            layer = modules[name[:-5]]
            layer.mask = torch.nn.Parameter(torch.ones_like(value), requires_grad=False)
            layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)
    model.load_state_dict(state, strict=True)
    model.to(device)
    lfwa.assert_masks(model)
    return model, checkpoint


def configure(args, directory, metadata, iteration):
    import config
    import prune
    import lfwa_experiment
    directory.mkdir(parents=True, exist_ok=True)
    lfwa_experiment._configure(SimpleNamespace(seed=args.seed, no_pretrained=False,
                                              keep_per_iter=args.keep_per_iter), directory, metadata)
    prune.IMPT_TYPE2_ALPHA = config.glo_imp_rate = args.alpha
    prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER = args.layer_floor
    config.glo_train_stage = 'prune'
    config.glo_prune_iter = iteration - 1
    config.glo_epoch_count = (iteration - 1) * args.retrain_epochs
    config.glo_retrain_epoch = 0
    config.glo_collapse_notes = []
    config.glo_is_checked = False
    config.glo_fairness = {}


def select_from_production(pruner, cfgs, directory, cap_fraction):
    """Capture scores without changing the production scorer or its inputs."""
    import torch
    import prune
    payloads = []
    original_logger = prune._save_channel_pruning_log

    def capture(*args, **kwargs):
        if args:
            raise AssertionError('Production logger signature changed: positional arguments')
        payloads.append(kwargs)

    # The uncapped proposal's console output is kept separately so it cannot be
    # mistaken for the cap arm's actual selected channels or sparsity.
    with (directory / 'production_score_proposal.log').open('w') as handle:
        try:
            prune._save_channel_pruning_log = capture
            with contextlib.redirect_stdout(handle):
                masks = prune.fairness_grad(pruner.model, *cfgs)
        finally:
            prune._save_channel_pruning_log = original_logger
    if len(payloads) != 1:
        raise AssertionError(f'Expected exactly one impt_type=2 score payload, got {len(payloads)}')
    payload = payloads[0]
    if payload['cap_multiplier'] is not None or payload['norm'] != 'max':
        raise AssertionError('This diagnostic requires production cap=None and normalization=max')
    original, active = channel_layout(pruner.model)
    candidates = payload['score_by_channel_dict']
    expected, baseline_audit = select_channels(candidates, original, active,
                                               payload['remove_target'], prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER)
    if expected != payload['selected_channels'] or baseline_audit['removed_weights'] != payload['accum_removed']:
        raise AssertionError('Uncapped replay does not exactly reproduce the production selector')
    # Verify that production masks also match the replayed selected channel set.
    replay_masks = prune._build_channel_mask_list(pruner.model, expected, prune.device)
    if len(masks) != len(replay_masks) or any(not torch.equal(a, b) for a, b in zip(masks, replay_masks)):
        raise AssertionError('Uncapped replay masks differ from production masks')
    del replay_masks
    selected, audit = select_channels(candidates, original, active, payload['remove_target'],
                                      prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER, cap_fraction)
    if cap_fraction is not None:
        masks = prune._build_channel_mask_list(pruner.model, selected, prune.device)
    audit.update({'production_selector_reproduced': True, 'strict_cap_fraction': cap_fraction,
                  'uncapped_proposal_removed_weights': payload['accum_removed'],
                  'uncapped_proposal_selected_channels': len(expected),
                  'selected_channels': [list(key) for key in selected],
                  'candidate_count': len(candidates),
                  'protected_count': len(payload['protected_set'])})
    # JSON retains Python float precision; CSV logging is intentionally only a
    # human-readable view and is never the source for subsequent selection.
    score_records = [{'block': block, 'channel': channel, 'score': float(score), 'weights': int(weight)}
                     for (block, channel), (score, weight) in candidates.items()]
    write_json(directory / 'candidate_scores_full_precision.json', score_records)
    write_json(directory / 'score_components_full_precision.json', {
        'alpha': payload['alpha'], 'normalization': payload['norm'], 'gamma': payload['gamma'],
        'protected_channels': [list(key) for key in sorted(payload['protected_set'])],
        **{field: {layer: vector.detach().cpu().tolist() for layer, vector in payload[field].items()}
           for field in ('phi_by_layer', 'perf_by_layer', 'phi_scaled_by_layer', 'perf_scaled_by_layer')},
    })
    write_json(directory / 'selection_audit.json', audit)
    write_json(directory / 'uncapped_proposal_selection.json', [list(key) for key in expected])
    payload = dict(payload)
    removed = audit['removed_weights']
    selected_weight_counts = collections.Counter()
    for key in selected:
        selected_weight_counts[key[0]] += candidates[key][1]
    payload.update(selected_channels=selected, accum_removed=removed,
                   total_active_after=prune._count_active_weights(pruner.model) - removed,
                   skipped_by_floor={row['block']: row['skipped_by_floor'] for row in audit['layers']},
                   skipped_by_cap={row['block']: row['skipped_by_cap'] for row in audit['layers']},
                   per_layer_removed_w=dict(selected_weight_counts),
                   cap_relaxed_set=set(), cap_relaxed_removed=0)
    payload['model_sparsity'] = 1 - payload['total_active_after'] / payload['total_model_params']
    original_logger(**payload)
    return masks, audit


def evaluate(model, frame, directory, batch):
    import pandas as pd
    import lfwa
    summary = lfwa.evaluate_model(model, frame, directory, batch)
    prediction = pd.read_csv(directory / 'predictions.csv')
    fractions = prediction[[name + '_pred' for name in lfwa.TARGETS]].mean().to_numpy()
    exact, near = (fractions == 0) | (fractions == 1), (fractions <= .01) | (fractions >= .99)
    summary.update(constant_prediction_attributes=int(exact.sum()),
                   near_constant_99pct_attributes=int(near.sum()),
                   constant_prediction_attribute_names=[n for n, flag in zip(lfwa.TARGETS, exact) if flag],
                   near_constant_99pct_attribute_names=[n for n, flag in zip(lfwa.TARGETS, near) if flag])
    write_json(directory / 'summary.json', summary)
    return summary


def preflight(args, frames, metadata, models, checkpoints):
    import torch
    import lfwa
    import prune
    result = {'sources': {}, 'selector_tests': self_test()}
    for name, model in models.items():
        original, active = channel_layout(model)
        result['sources'][name] = {'iteration': int(checkpoints[name]['iteration']),
                                  'sparsity': lfwa.sparsity(model),
                                  'original_channels': original, 'active_channels': active,
                                  'binary_frozen_masks': len(lfwa.assert_masks(model))}
    if args.replay_csv:
        if sha256(args.baseline_checkpoint) != sha256(args.cap_checkpoint):
            raise ValueError('--replay-csv requires identical source checkpoints')
        model = models['baseline']
        original, active = channel_layout(model)
        with Path(args.replay_csv).open() as handle:
            rows = list(csv.DictReader(handle))
        if {int(row['iter']) for row in rows} != {int(checkpoints['baseline']['iteration']) + 1}:
            raise ValueError('Replay CSV must immediately follow the source checkpoint')
        candidates, protected_weight, historical = {}, 0, set()
        for row in rows:
            key = (row['block'], int(row['ch']))
            weight = int(row['weights'])
            if prune._count_channel_weights(model, *key) != weight:
                raise AssertionError(f'Saved CSV channel weights differ: {key}')
            if row['status'] == 'protected':
                protected_weight += weight
            elif weight > 0:
                candidates[key] = (float(row['score']), weight)
            if row['status'].startswith('pruned_now'):
                historical.add(key)
        target = round((sum(value[1] for value in candidates.values()) + protected_weight) * (1 - args.keep_per_iter))
        for name, cap in [('baseline', None), ('cap10', args.cap_fraction)]:
            selected, audit = select_channels(candidates, original, active, target, args.layer_floor, cap)
            if name == 'baseline' and set(selected) != historical:
                raise AssertionError('Saved rounded CSV did not reproduce the historical selection')
            pruner = prune.Prunner(model, torch.nn.CrossEntropyLoss(), None, lfwa.OUTPUT_SLICES)
            pruner.get_mask = lambda unused: prune._build_channel_mask_list(pruner.model, selected, prune.device)
            result_model = pruner.prune(None, False)
            lfwa.assert_masks(result_model)
            difference = lfwa.sparsity(result_model)['target_removed'] - lfwa.sparsity(model)['target_removed']
            if difference != audit['removed_weights']:
                raise AssertionError('Applied mask weight difference does not match selector budget')
            for source_mask, after_mask in zip(lfwa.assert_masks(model).values(), lfwa.assert_masks(result_model).values()):
                if torch.any(after_mask > source_mask):
                    raise AssertionError('Previously removed channels regrew')
            result.setdefault('saved_score_replay', {})[name] = {**audit, 'sparsity': lfwa.sparsity(result_model)}
    write_json(args.run_dir / 'preflight.json', result)
    print(json.dumps({'status': 'preflight_passed', 'output': str(args.run_dir / 'preflight.json')}), flush=True)


def completed_steps(arm_root, source_iteration, end_step):
    completed = []
    for iteration in range(source_iteration + 1, end_step + 1):
        marker = arm_root / f'iteration_{iteration:03d}' / 'complete.json'
        if not marker.is_file():
            break
        row = json.loads(marker.read_text())
        checkpoint = marker.parent / 'retrained.pt'
        if row['iteration'] != iteration or not checkpoint.is_file() or sha256(checkpoint) != row['checkpoint_sha256']:
            raise ValueError(f'Completed checkpoint integrity check failed: {marker}')
        completed.append(row)
    return completed


def run_arm(args, name, source_path, source_iteration, frames, metadata, stop_after=None):
    import torch
    import config
    import lfwa
    import prune
    from dataset import make_datasets
    from train_and_val import train
    from util import setseed
    arm_root = args.run_dir / name
    arm_root.mkdir(parents=True, exist_ok=True)
    completed = completed_steps(arm_root, source_iteration, args.end_step)
    iteration_start = source_iteration + len(completed) + 1
    iteration_end = min(args.end_step, stop_after) if stop_after is not None else args.end_step
    if iteration_start > iteration_end:
        return completed
    checkpoint_path = (arm_root / f'iteration_{completed[-1]["iteration"]:03d}' / 'retrained.pt') if completed else source_path
    model, checkpoint = load_model(checkpoint_path, metadata, 'cuda:0')
    if int(checkpoint['iteration']) != iteration_start - 1:
        raise AssertionError('Resume checkpoint iteration mismatch')
    cap = None if name == 'baseline' else args.cap_fraction
    for iteration in range(iteration_start, iteration_end + 1):
        started = time.monotonic()
        directory = arm_root / f'iteration_{iteration:03d}'
        if directory.exists():
            backup = directory.with_name(directory.name + '_interrupted_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            directory.rename(backup)
        directory.mkdir()
        configure(args, directory, metadata, iteration)
        score_seed, training_seed = args.seed + iteration, args.seed + 10000 + iteration
        write_json(args.run_dir / f'status_{name}.json', {'status': 'running', 'arm': name, 'iteration': iteration,
                                                       'stage': 'score_recomputation', 'updated_at': now()})
        before = lfwa.sparsity(model)
        source_masks = lfwa.assert_masks(model)
        for parameter_name, parameter in model.named_parameters():
            parameter.requires_grad_(not parameter_name.endswith('.mask'))
        setseed(score_seed)
        # The base Prunner preserves loaded masks; FairGRAPE.__init__ resets them.
        pruner = prune.Prunner(model, torch.nn.CrossEntropyLoss(), None, lfwa.OUTPUT_SLICES)
        lfwa.assert_masks(pruner.model, source_masks)
        cumulative_rate = 1 - args.keep_per_iter ** iteration
        # fairness_grad's signature differs from get_mask's positional cfg list:
        # col_names comes after output_cols_each_task, followed by para_batch etc.
        cfgs = [cumulative_rate, frames['train'], None, 78, True, lfwa.OUTPUT_SLICES,
                lfwa.LABEL_COLUMNS, 1, 2, args.stop_batch, False, 1, 'mobilenetv2', 'young']
        masks, audit = select_from_production(pruner, cfgs, directory, cap)
        pruner.get_mask = lambda unused: masks
        model = pruner.prune(None, False)
        del pruner, masks
        frozen = lfwa.assert_masks(model)
        if frozen.keys() != source_masks.keys() or any(torch.any(frozen[key] > source_masks[key]) for key in frozen):
            raise AssertionError('A previously removed weight regrew during pruning')
        del source_masks
        after = lfwa.sparsity(model)
        if after['target_removed'] - before['target_removed'] != audit['removed_weights']:
            raise AssertionError('Applied masks do not match the recorded removal budget')
        _, after_active = channel_layout(model)
        if any(after_active[row['block']] != row['active_after'] for row in audit['layers']):
            raise AssertionError('Applied masks do not match per-block channel selection')
        write_json(args.run_dir / f'status_{name}.json', {'status': 'running', 'arm': name, 'iteration': iteration,
                                                       'stage': 'post_prune_validation', 'updated_at': now()})
        post_prune = evaluate(model, frames['val'], directory / 'post_prune', args.batch)
        setseed(training_seed)
        train_loader, val_loader = make_datasets(frames['train'], frames['val'], True, args.batch, lfwa.LABEL_COLUMNS)
        loaders = {'train': train_loader, 'test': val_loader}
        fit_cfgs = ['LFWA', 'FairGRAPE', 'classes', after['target_sparsity'], frames['val'], None, 78,
                    'mobilenetv2', lfwa.LABEL_COLUMNS, lfwa.OUTPUT_SLICES, 'young', 0, args.seed]
        write_json(args.run_dir / f'status_{name}.json', {'status': 'running', 'arm': name, 'iteration': iteration,
                                                       'stage': 'retrain', 'updated_at': now()})
        model, _ = train(model, torch.nn.CrossEntropyLoss(), loaders,
                         [args.retrain_lr], [args.retrain_epochs], lfwa.TARGETS,
                         lfwa.OUTPUT_SLICES, cfgs=fit_cfgs, use_debiasing=False)
        lfwa.assert_masks(model, frozen)
        if lfwa.sparsity(model) != after:
            raise AssertionError('Sparsity changed during retraining')
        checkpoint_path = directory / 'retrained.pt'
        lfwa.save_checkpoint(model, checkpoint_path, metadata, stage='cap_trajectory', iteration=iteration,
                             arm=name, strict_cap_fraction=cap, score_seed=score_seed, training_seed=training_seed)
        post_retrain = evaluate(model, frames['val'], directory / 'post_retrain', args.batch)
        row = {'iteration': iteration, 'arm': name, 'score_seed': score_seed, 'training_seed': training_seed,
               'source_sparsity': before, 'sparsity': after, 'selection': audit,
               'post_prune': post_prune, 'post_retrain': post_retrain,
               'checkpoint_sha256': sha256(checkpoint_path), 'completed_at': now(),
               'seconds': time.monotonic() - started}
        write_json(directory / 'complete.json', row)
        completed.append(row)
        write_json(arm_root / 'trajectory.json', completed)
        print(json.dumps({'arm': name, 'iteration': iteration, 'sparsity': after,
                          'post_retrain': post_retrain, 'seconds': row['seconds']}), flush=True)
        del frozen, train_loader, val_loader, loaders
        torch.cuda.empty_cache()
    write_json(args.run_dir / f'status_{name}.json', {
        'status': 'complete' if iteration_end == args.end_step else 'waiting_next_iteration',
        'arm': name, 'completed_iteration': iteration_end, 'end_step': args.end_step,
        'updated_at': now(),
    })
    del model
    torch.cuda.empty_cache()
    return completed


def self_test():
    # Equal-score ordering must remain insertion order; a score difference below
    # the six decimal CSV precision must still affect actual selection.
    original = {'features.1': 20, 'features.2': 20}
    active = {'features.1': 10, 'features.2': 20}
    candidates = {('features.1', 4): (.10000004, 5), ('features.1', 2): (.10000003, 5),
                  ('features.2', 7): (.2, 5), ('features.2', 8): (.2, 5)}
    selected, audit = select_channels(candidates, original, active, 15, .03)
    assert selected == [('features.1', 2), ('features.1', 4), ('features.2', 7)]
    selected, audit = select_channels(candidates, original, active, 15, .03, .1)
    assert selected == [('features.1', 2), ('features.2', 7), ('features.2', 8)]
    assert audit['removed_weights'] == 15 and audit['budget_deficit'] == 0
    selected, audit = select_channels(candidates, original, {'features.1': 9, 'features.2': 20}, 25, .03, .1)
    assert selected == [('features.2', 7), ('features.2', 8)]
    assert audit['budget_deficit'] == 15 and audit['layers'][0]['cap_zero_due_to_rounding']
    selected, audit = select_channels(candidates, original, {'features.1': 10, 'features.2': 10}, 10, .5)
    assert selected == [] and audit['budget_deficit'] == 10
    return {'passed': True, 'checks': ['full_precision_order', 'stable_ties', 'strict_cap',
                                      'unrelaxed_infeasible_budget', 'integer_cap_zero', 'layer_floor']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-checkpoint', type=Path)
    parser.add_argument('--cap-checkpoint', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--end-step', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=.7)
    parser.add_argument('--cap-fraction', type=float, default=.1)
    parser.add_argument('--keep-per-iter', type=float, default=.975)
    parser.add_argument('--layer-floor', type=float, default=.03)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--retrain-epochs', type=int, default=5)
    parser.add_argument('--retrain-lr', type=float, default=1e-5)
    parser.add_argument('--stop-batch', type=int, default=10000)
    parser.add_argument('--lfwa-csv', type=Path, default=ROOT / 'csv/LFWA.csv')
    parser.add_argument('--lfwa-image-root', type=Path, default=ROOT / 'Images/LFWA')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--replay-csv', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if not all([args.baseline_checkpoint, args.cap_checkpoint, args.run_dir]):
        parser.error('--baseline-checkpoint, --cap-checkpoint and --run-dir are required')
    if not 0 <= args.alpha < 1:
        parser.error('This user-authorized diagnostic requires 0 <= alpha < 1')
    if not 0 < args.cap_fraction <= 1 or not 0 < args.keep_per_iter < 1 or not 0 <= args.layer_floor <= 1:
        parser.error('Invalid cap, keep fraction or layer floor')
    if min(args.end_step, args.batch, args.retrain_epochs, args.stop_batch) < 1 or args.retrain_lr <= 0:
        parser.error('Steps, batch sizes and learning rate must be positive')
    for field in ('baseline_checkpoint', 'cap_checkpoint', 'run_dir', 'lfwa_csv', 'lfwa_image_root'):
        setattr(args, field, getattr(args, field).resolve())
    if args.preflight:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.chdir(ROOT)
    import torch
    import numpy as np
    import lfwa
    torch.set_num_threads(2)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    marker = args.run_dir / 'run.json'
    sources = {'baseline': args.baseline_checkpoint, 'cap10': args.cap_checkpoint}
    frames, metadata = lfwa.load_frames(args.lfwa_csv, args.lfwa_image_root)
    models, checkpoints = {}, {}
    for name, path in sources.items():
        models[name], checkpoints[name] = load_model(path, metadata, 'cpu')
    iterations = {int(checkpoint['iteration']) for checkpoint in checkpoints.values()}
    if len(iterations) != 1:
        raise ValueError('The paired source checkpoints must have the same iteration')
    source_iteration = iterations.pop()
    if args.end_step <= source_iteration:
        raise ValueError('--end-step must exceed source checkpoint iteration')
    config_fields = ['alpha', 'cap_fraction', 'keep_per_iter', 'layer_floor', 'seed', 'batch',
                     'retrain_epochs', 'retrain_lr', 'stop_batch']
    protocol = {field: getattr(args, field) for field in config_fields}
    protocol.update(source_iteration=source_iteration, experiment_metadata=metadata,
                    sources={name: {'path': str(path), 'sha256': sha256(path)} for name, path in sources.items()},
                    normalization='max', protection_ratio=.005, importance_batch=128, phi_batch=64,
                    evaluation_split='val', final_retraining=False,
                    source_selection='Provided checkpoints; each arm retains its own masks and weights.',
                    score_seed='seed + iteration', training_seed='seed + 10000 + iteration',
                    rng_note='Paired resets differ from the uninterrupted historical experiment RNG schedule.',
                    cap_rounding='floor(active_channels * cap_fraction); no relaxation. Below 10 channels, cap10 cannot remove another channel.',
                    scoring='Production fairness_grad, separately recomputed each iteration and arm; full precision selector inputs retained.')
    files = ['scripts/run_lfwa_cap_trajectory.py', 'prune.py', 'lfwa.py', 'lfwa_experiment.py',
             'train_and_val.py', 'dataset.py', 'util.py', 'config.py', 'adversarial_debiasing.py']
    hashes = {name: sha256(ROOT / name) for name in files}
    if marker.exists():
        if not args.resume:
            raise FileExistsError(f'Use --resume to continue an existing run: {args.run_dir}')
        previous = json.loads(marker.read_text())
        if previous['protocol'] != protocol or previous['source_sha256'] != hashes:
            raise ValueError('Resume protocol/source code differ from the recorded run')
    else:
        write_json(marker, {'created_at': now(), 'protocol': protocol, 'source_sha256': hashes,
                            'initial_end_step': args.end_step, 'python': sys.executable,
                            'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                        'numpy': np.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}})
        for name in files:
            destination = args.run_dir / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
    if args.preflight:
        preflight(args, frames, metadata, models, checkpoints)
        return
    del models, checkpoints
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training; use --preflight for CPU checks')
    old_complete = args.run_dir / 'complete.json'
    if old_complete.exists():
        previous_end = json.loads(old_complete.read_text())['end_step']
        old_complete.replace(args.run_dir / f'complete_through_{previous_end:03d}.json')
    results = {}
    try:
        first_pending = min(source_iteration + len(completed_steps(args.run_dir / name, source_iteration, args.end_step)) + 1
                            for name in sources)
        for iteration in range(first_pending, args.end_step + 1):
            for name, path in sources.items():
                results[name] = run_arm(args, name, path, source_iteration, frames, metadata, stop_after=iteration)
            write_json(args.run_dir / 'results.json', results)
            print(json.dumps({'status': 'paired_iteration_complete', 'iteration': iteration}), flush=True)
        if not results:
            results = {name: completed_steps(args.run_dir / name, source_iteration, args.end_step) for name in sources}
            write_json(args.run_dir / 'results.json', results)
        write_json(args.run_dir / 'complete.json', {'status': 'complete', 'end_step': args.end_step, 'completed_at': now()})
    except Exception as error:
        write_json(args.run_dir / 'failure.json', {'status': 'failed', 'error': repr(error), 'failed_at': now()})
        raise


if __name__ == '__main__':
    main()
