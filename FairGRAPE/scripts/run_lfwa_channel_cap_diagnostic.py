#!/usr/bin/env python3
"""Replay one LFWA pruning step from a reviewed channel-selection plan.

The plan contains a fixed source checkpoint, its next-step channel CSV, and
explicit channel selections for each arm. Scores are not recomputed here.
Each arm receives the same checkpoint and a freshly reset training RNG. Only
Validation is evaluated; the existing LFWA accuracy/EO checkpoint selector is
used during the usual five epochs of retraining.

Minimal plan schema (paths are relative to the repository root):
  {"source_checkpoint": ".../prune_028.pt",
   "source_channel_csv": ".../alpha0.7_iter29_channels.csv",
   "iteration": 29, "alpha": 0.7, "seed": 42,
   "arms": [{"name": "baseline", "reference_selection": true,
             "selected_channels": [["features.2", 1]],
             "expected_removed_weights": 57},
            {"name": "cap10", "strict_channel_cap_fraction": 0.1,
             "selected_channels": [["features.3", 2]],
             "expected_removed_weights": 65}]}

Optional plan fields: lfwa_csv, lfwa_image_root, batch, retrain_epochs,
retrain_lr, keep_per_iter, layer_floor, source_checkpoint_sha256,
source_channel_csv_sha256. --dry-run checks masks and budgets on CPU without
loading image pixels, doing any forward pass, or training.
"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
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


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_plan(path):
    plan = json.loads(path.read_text())
    for field in ('source_checkpoint', 'source_channel_csv', 'iteration', 'arms'):
        if field not in plan:
            raise ValueError(f'Missing plan field: {field}')
    defaults = {'alpha': .7, 'seed': 42, 'batch': 256, 'retrain_epochs': 5,
                'retrain_lr': 1e-5, 'keep_per_iter': .975, 'layer_floor': .03,
                'lfwa_csv': 'csv/LFWA.csv', 'lfwa_image_root': 'Images/LFWA'}
    for field, value in defaults.items():
        plan.setdefault(field, value)
    if not 0 <= plan['alpha'] <= 1 or not 0 < plan['keep_per_iter'] < 1:
        raise ValueError('Invalid alpha or keep_per_iter')
    if not 0 <= plan['layer_floor'] <= 1:
        raise ValueError('Invalid layer_floor')
    for field in ('iteration', 'batch', 'retrain_epochs'):
        if isinstance(plan[field], bool) or int(plan[field]) != plan[field] or plan[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if plan['retrain_lr'] <= 0 or not plan['arms']:
        raise ValueError('Require a positive learning rate and at least one arm')
    names = set()
    for arm in plan['arms']:
        name = arm['name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError(f'Invalid or duplicate arm name: {name}')
        names.add(name)
        channels = []
        for block, channel in arm['selected_channels']:
            if not re.fullmatch(r'features\.(?:[1-9]|1[0-7])', block):
                raise ValueError(f'Invalid pruning block: {block}')
            if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
                raise ValueError(f'Invalid channel: {channel}')
            channels.append((block, channel))
        if len(channels) != len(set(channels)):
            raise ValueError(f'Duplicate selected channels in {name}')
        cap = arm.get('strict_channel_cap_fraction')
        if cap is not None and not 0 < cap <= 1:
            raise ValueError(f'Invalid channel cap in {name}')
        if 'expected_removed_weights' not in arm:
            raise ValueError(f'Missing expected_removed_weights in {name}')
        arm['selected_channels'] = channels
    for field in ('source_checkpoint', 'source_channel_csv', 'lfwa_csv', 'lfwa_image_root'):
        plan[field] = str(resolve_path(plan[field]))
    for field in ('source_checkpoint', 'source_channel_csv'):
        expected = plan.get(field + '_sha256')
        actual = sha256(plan[field])
        if expected is not None and actual != expected:
            raise ValueError(f'Plan input hash changed: {field}')
        plan[field + '_sha256'] = actual
    return plan


def validate_source(model, checkpoint, channel_frame, plan):
    import lfwa
    import prune
    if checkpoint.get('iteration') != plan['iteration'] - 1:
        raise ValueError('Source checkpoint must immediately precede the planned iteration')
    if set(channel_frame['iter'].unique()) != {plan['iteration']}:
        raise ValueError('Channel CSV iteration differs from the plan')
    records = {}
    for row in channel_frame.itertuples(index=False):
        key = (row.block, int(row.ch))
        if key in records:
            raise ValueError(f'Duplicate channel CSV entry: {key}')
        if not re.fullmatch(r'features\.(?:[1-9]|1[0-7])', row.block):
            continue
        count = prune._count_channel_weights(model, *key)
        if count != int(row.weights) or (count == 0) != (row.status == 'pruned_before'):
            raise ValueError(f'CSV weights/activity differ from the source checkpoint: {key}')
        records[key] = row
    original = collections.Counter(key[0] for key in records)
    active = collections.Counter(key[0] for key, row in records.items() if row.weights > 0)
    if set(original) != {f'features.{i}' for i in range(1, 18)}:
        raise ValueError('Channel CSV must cover all 17 pruning blocks')
    if sum(int(row.weights) for row in records.values()) != (
            lfwa.sparsity(model)['target_weights'] - lfwa.sparsity(model)['target_removed']):
        raise ValueError('CSV does not account for all active target weights')
    reference = {key for key, row in records.items() if row.status.startswith('pruned_now')}
    output = {}
    for arm in plan['arms']:
        selected = arm['selected_channels']
        counts = collections.Counter(block for block, _ in selected)
        if arm.get('reference_selection', arm['name'] == 'baseline') and set(selected) != reference:
            raise ValueError('Baseline selection does not exactly match the saved channel CSV')
        rows = []
        for key in selected:
            if key not in records or records[key].weights <= 0:
                raise ValueError(f'Selected channel is absent or already removed: {key}')
            if records[key].status == 'protected':
                raise ValueError(f'Selected a protected channel: {key}')
        expected_removed = sum(int(records[key].weights) for key in selected)
        if expected_removed != arm['expected_removed_weights']:
            raise ValueError(f'Planned weight count differs for {arm["name"]}: {expected_removed}')
        for block in sorted(original, key=lambda name: int(name.split('.')[1])):
            floor = math.ceil(original[block] * plan['layer_floor'])
            cap_fraction = arm.get('strict_channel_cap_fraction')
            cap = math.floor(active[block] * cap_fraction) if cap_fraction is not None else None
            if active[block] - counts[block] < floor:
                raise ValueError(f'Layer floor violation: {arm["name"]}/{block}')
            if cap is not None and counts[block] > cap:
                raise ValueError(f'Strict channel cap violation: {arm["name"]}/{block}')
            rows.append({'block': block, 'original_channels': original[block],
                         'active_before': active[block], 'selected_channels': counts[block],
                         'active_after': active[block] - counts[block], 'minimum_keep': floor,
                         'maximum_remove_this_step': cap})
        output[arm['name']] = {'selected_channels': len(selected),
                               'removed_target_weights': expected_removed, 'layers': rows}
    return output


def apply_selection(model, selected):
    """Use the production mask builder, zeroing, gradient hooks and mask API."""
    import torch
    import prune
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.endswith('.mask'))
    pruner = prune.Prunner(model, torch.nn.CrossEntropyLoss(), None)
    pruner.get_mask = lambda unused: prune._build_channel_mask_list(
        pruner.model, selected, next(pruner.model.parameters()).device)
    return pruner.prune(None, False)


def check_reference_masks(model, arm, plan):
    """Historical retraining must not change the replayed baseline masks."""
    import torch
    if not arm.get('reference_selection', arm['name'] == 'baseline'):
        return {}
    reference = arm.get('reference_checkpoint')
    path = (resolve_path(reference) if reference else Path(plan['source_checkpoint']).with_name(
        f'prune_{plan["iteration"]:03d}.pt'))
    if not path.is_file():
        if reference:
            raise FileNotFoundError(path)
        return {'historical_masks_checked': False}
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model_state', checkpoint.get('model_state_dict'))
    if state is None:
        raise ValueError(f'Historical checkpoint has no model state: {path}')
    expected = {name: value for name, value in state.items() if name.endswith('.mask')}
    actual = {name: value.detach().cpu() for name, value in model.state_dict().items()
              if name.endswith('.mask')}
    if not expected or actual.keys() != expected.keys() or any(
            not torch.equal(actual[name], expected[name]) for name in actual):
        raise AssertionError('Replayed baseline masks differ from the historical next checkpoint')
    return {'historical_masks_checked': True, 'historical_masks_equal': True,
            'historical_mask_count': len(actual), 'historical_checkpoint': str(path),
            'historical_checkpoint_sha256': sha256(path)}


def evaluate(model, frame, directory, batch):
    import pandas as pd
    import lfwa
    summary = lfwa.evaluate_model(model, frame, directory, batch)
    predictions = pd.read_csv(directory / 'predictions.csv')
    fractions = predictions[[name + '_pred' for name in lfwa.TARGETS]].mean().to_numpy()
    exact = (fractions == 0) | (fractions == 1)
    near = (fractions <= .01) | (fractions >= .99)
    summary.update({'constant_prediction_attributes': int(exact.sum()),
                    'near_constant_99pct_attributes': int(near.sum()),
                    'constant_prediction_attribute_names': [
                        name for name, flag in zip(lfwa.TARGETS, exact) if flag],
                    'near_constant_99pct_attribute_names': [
                        name for name, flag in zip(lfwa.TARGETS, near) if flag]})
    write_json(directory / 'summary.json', summary)
    return summary


def configure(plan, arm_root, metadata):
    import config
    import prune
    import lfwa_experiment
    args = SimpleNamespace(seed=plan['seed'], no_pretrained=False,
                           keep_per_iter=plan['keep_per_iter'])
    lfwa_experiment._configure(args, arm_root, metadata)
    prune.IMPT_TYPE2_ALPHA = config.glo_imp_rate = plan['alpha']
    prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER = plan['layer_floor']
    config.glo_train_stage = 'prune'
    config.glo_prune_iter = plan['iteration'] - 1
    config.glo_epoch_count = (plan['iteration'] - 1) * plan['retrain_epochs']
    config.glo_retrain_epoch = 0
    config.glo_collapse_notes = []
    config.glo_is_checked = False
    config.glo_fairness = {}


def run(plan, root, dry_run):
    import pandas as pd
    import torch
    import lfwa
    from dataset import make_datasets
    from train_and_val import train
    from util import setseed

    import numpy as np
    marker = root / ('dry_run_metadata.json' if dry_run else 'diagnostic_run.json')
    run_metadata = json.loads(marker.read_text())
    run_metadata['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                               'numpy': np.__version__,
                               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                               'device': 'cpu' if dry_run else 'cuda:0'}
    write_json(marker, run_metadata)
    torch.set_num_threads(2)
    setseed(plan['seed'])
    frames, metadata = lfwa.load_frames(plan['lfwa_csv'], plan['lfwa_image_root'])
    source, checkpoint = lfwa.load_checkpoint_model(plan['source_checkpoint'], metadata, 'cpu')
    source_sparsity = lfwa.sparsity(source)
    channel_frame = pd.read_csv(plan['source_channel_csv'])
    audits = validate_source(source, checkpoint, channel_frame, plan)
    if not dry_run and not torch.cuda.is_available():
        raise RuntimeError('The production LFWA trainer requires CUDA; use --dry-run for CPU checks')
    device = torch.device('cpu' if dry_run else 'cuda:0')
    source.to(device)
    results = {'source_sparsity': source_sparsity, 'plan_validation': audits, 'arms': {}}
    if not dry_run:
        results['before_prune'] = evaluate(source, frames['val'], root / 'before_prune', plan['batch'])
    for arm in plan['arms']:
        started = time.monotonic()
        name = arm['name']
        arm_root = root / name
        arm_root.mkdir(parents=True, exist_ok=True)
        configure(plan, arm_root, metadata)
        setseed(plan['seed'])
        model = apply_selection(source, arm['selected_channels'])
        masks = lfwa.assert_masks(model)
        current_sparsity = lfwa.sparsity(model)
        actual_removed = current_sparsity['target_removed'] - source_sparsity['target_removed']
        if actual_removed != arm['expected_removed_weights']:
            raise AssertionError(f'Applied masks differ from plan for {name}: {actual_removed}')
        reference_audit = check_reference_masks(model, arm, plan)
        audit = {**audits[name], 'sparsity': current_sparsity, **reference_audit}
        if reference_audit:
            write_json(arm_root / 'historical_mask_verification.json', reference_audit)
        write_json(arm_root / 'mask_audit.json', audit)
        pd.DataFrame(audits[name]['layers']).to_csv(arm_root / 'layer_audit.csv', index=False)
        if dry_run:
            results['arms'][name] = audit
            del model, masks
            continue
        write_json(root / 'status.json', {'status': 'running', 'arm': name, 'stage': 'post_prune'})
        post_prune = evaluate(model, frames['val'], arm_root / 'post_prune', plan['batch'])
        lfwa.save_checkpoint(model, arm_root / 'post_prune.pt', metadata,
                             stage='diagnostic_post_prune', iteration=plan['iteration'], arm=name)
        # Reset immediately before loaders/training, so evaluation/model-loading
        # RNG consumption cannot affect the paired data augmentation or dropout.
        setseed(plan['seed'])
        train_loader, val_loader = make_datasets(
            frames['train'], frames['val'], True, plan['batch'], lfwa.LABEL_COLUMNS)
        loaders = {'train': train_loader, 'test': val_loader}
        rate = current_sparsity['target_sparsity']
        cfgs = ['LFWA', 'FairGRAPE', 'classes', rate, frames['val'], None, 78,
                'mobilenetv2', lfwa.LABEL_COLUMNS, lfwa.OUTPUT_SLICES, 'young', 0, plan['seed']]
        write_json(root / 'status.json', {'status': 'running', 'arm': name, 'stage': 'retrain'})
        model, _ = train(model, torch.nn.CrossEntropyLoss(), loaders,
                         [plan['retrain_lr']], [plan['retrain_epochs']], lfwa.TARGETS,
                         lfwa.OUTPUT_SLICES, cfgs=cfgs, use_debiasing=False)
        lfwa.assert_masks(model, masks)
        if lfwa.sparsity(model) != current_sparsity:
            raise AssertionError(f'Sparsity changed during retraining: {name}')
        lfwa.save_checkpoint(model, arm_root / 'retrained.pt', metadata,
                             stage='diagnostic_retrained', iteration=plan['iteration'], arm=name)
        post_retrain = evaluate(model, frames['val'], arm_root / 'post_retrain', plan['batch'])
        results['arms'][name] = {**audit, 'post_prune': post_prune,
                                 'post_retrain': post_retrain,
                                 'seconds': time.monotonic() - started}
        write_json(arm_root / 'complete.json', {'status': 'complete', **results['arms'][name]})
        write_json(root / 'results.json', results)
        del model, masks, loaders, train_loader, val_loader
        torch.cuda.empty_cache()
    output = 'dry_run.json' if dry_run else 'results.json'
    write_json(root / output, results)
    if not dry_run:
        write_json(root / 'complete.json', {'status': 'complete', 'results': str(root / output)})
        write_json(root / 'status.json', {'status': 'complete'})
    print(json.dumps({'status': 'dry_run_passed' if dry_run else 'complete',
                      'results': str(root / output)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.dry_run:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = args.run_dir.resolve()
    marker = root / ('dry_run_metadata.json' if args.dry_run else 'diagnostic_run.json')
    if marker.exists() or (root / 'complete.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing diagnostic: {root}')
    root.mkdir(parents=True, exist_ok=True)
    plan = read_plan(args.plan.resolve())
    source_files = ['scripts/run_lfwa_channel_cap_diagnostic.py', 'lfwa.py',
                    'lfwa_experiment.py', 'prune.py', 'train_and_val.py',
                    'dataset.py', 'util.py', 'adversarial_debiasing.py', 'config.py']
    hashes = {name: sha256(ROOT / name) for name in source_files}
    write_json(marker, {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        'plan': plan, 'source_sha256': hashes, 'python': sys.executable,
                        'command': sys.argv, 'dry_run': args.dry_run,
                        'score_recomputation': False, 'evaluation_split': 'val',
                        'paired_training_seed': plan['seed'],
                        'rng_note': 'Paired reset; original uninterrupted-run RNG was not checkpointed.',
                        'selection': 'Existing LFWA maximum Validation accuracy, then lower EO/loss ties.'})
    for name in source_files:
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    try:
        run(plan, root, args.dry_run)
    except Exception as error:
        write_json(root / 'status.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
