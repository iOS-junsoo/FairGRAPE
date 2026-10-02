#!/usr/bin/env python3
"""Diagnose recovery capacity of the existing LFWA alpha=.7, whole-60% model.

Two arms start from exactly the same already-pruned checkpoint and train for
ten epochs with LR 1e-5 or 1e-4. No scores are computed and no channels are
removed. Source validation must reproduce the checkpoint's saved metrics before
training. Only validation is evaluated; no test images are used.

--dry-run checks source provenance, sparsity, masks, and both identical CPU
copies without image decoding, forward passes, or GPU allocation. --resume
skips completed arms; an interrupted arm restarts from the original source.
"""

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_lfwa_cap_trajectory import (
    channel_layout, configure, evaluate, load_model, now, sha256, write_json,
)

ARMS = [('lr1e-5', 1e-5), ('lr1e-4', 1e-4)]
EPOCHS, BATCH, SEED = 10, 256, 42
DEFAULT_SOURCE = ROOT / 'lfwa_runs/alpha0p7_seed42_no_final_20260916_143706/prune_060.pt'
METRIC_TOLERANCE = 1e-9


def identical_state(left, right):
    import torch
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def validate_source(model, checkpoint, source_path):
    import lfwa
    actual = lfwa.sparsity(model)
    if actual != checkpoint.get('sparsity'):
        raise ValueError('Actual source sparsity differs from saved checkpoint metadata')
    if int(checkpoint.get('iteration', -1)) != 60:
        raise ValueError('This diagnostic requires the already-retrained iteration-60 checkpoint')
    if not .59 <= actual['model_sparsity'] <= .61:
        raise ValueError(f'Source is not approximately whole-model 60% sparse: {actual}')
    run_path = source_path.parent / 'run.json'
    source_run = json.loads(run_path.read_text())
    if abs(float(source_run['alpha']) - .7) > 1e-12:
        raise ValueError('Source run must use alpha=.7')
    if source_run['experiment_metadata'] != checkpoint['experiment_metadata']:
        raise ValueError('Source run and checkpoint dataset contracts differ')
    if source_run['arguments']['seed'] != SEED:
        raise ValueError('This paired diagnostic expects the original seed42 source')
    reference = checkpoint.get('validation')
    if not reference or any(reference.get(key) is None for key in ('accuracy', 'ba', 'eo')):
        raise ValueError('Require saved source validation accuracy, BA and EO')
    original, active = channel_layout(model)
    return {'iteration': 60, 'source_alpha': .7, 'sparsity': actual,
            'binary_frozen_mask_count': len(lfwa.assert_masks(model)),
            'original_channels': original, 'active_channels': active,
            'saved_validation': reference, 'run_json': str(run_path),
            'run_json_sha256': sha256(run_path)}


def verify_validation(actual, reference):
    differences = {key: float(actual[key]) - float(reference[key]) for key in ('accuracy', 'ba', 'eo')}
    if any(abs(value) > METRIC_TOLERANCE for value in differences.values()):
        raise AssertionError(f'Source validation does not reproduce the saved checkpoint: {differences}')
    return {'passed': True, 'absolute_tolerance': METRIC_TOLERANCE, 'differences': differences}


def run(args, root, source_model, checkpoint, source_audit, frames, metadata):
    import pandas as pd
    import torch
    import config
    import lfwa
    from dataset import make_datasets
    from train_and_val import train
    from util import setseed

    source_masks = lfwa.assert_masks(source_model)
    source_sparsity = lfwa.sparsity(source_model)
    copies = {}
    for name, _ in ARMS:
        candidate = copy.deepcopy(source_model)
        if not identical_state(candidate, source_model):
            raise AssertionError(f'Source copy differs in {name}')
        lfwa.assert_masks(candidate, source_masks)
        if lfwa.sparsity(candidate) != source_sparsity:
            raise AssertionError(f'Source copy sparsity differs in {name}')
        copies[name] = {'identical_source_state': True, 'identical_masks': True,
                        'sparsity': source_sparsity}
        del candidate
    if args.dry_run:
        write_json(root / 'dry_run.json', {'status': 'passed', 'source': source_audit,
                                          'arms': copies, 'forward_passes': 0, 'gpu_used': False})
        print(json.dumps({'status': 'dry_run_passed', 'sparsity': source_sparsity,
                          'output': str(root / 'dry_run.json')}), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; --dry-run performs CPU-only checks')
    source_model.to('cuda:0')
    write_json(root / 'status.json', {'status': 'running', 'stage': 'source_validation', 'updated_at': now()})
    before = evaluate(source_model, frames['val'], root / 'source_validation', BATCH)
    try:
        verification = verify_validation(before, source_audit['saved_validation'])
    except Exception:
        write_json(root / 'source_validation_verification.json', {
            'passed': False, 'actual': before, 'reference': source_audit['saved_validation'],
            'absolute_tolerance': METRIC_TOLERANCE,
        })
        raise
    write_json(root / 'source_validation_verification.json', verification)
    results = {'source': source_audit, 'source_validation': before,
               'source_validation_verification': verification, 'arms': {}}
    source_epoch = int(source_audit['saved_validation'].get('epoch', 300))
    for name, learning_rate in ARMS:
        directory = root / name
        marker = directory / 'complete.json'
        if marker.exists():
            if not args.resume:
                raise FileExistsError(f'Completed arm already exists: {directory}')
            row = json.loads(marker.read_text())
            if sha256(directory / 'recovered.pt') != row['checkpoint_sha256']:
                raise ValueError(f'Completed arm checkpoint changed: {directory}')
            results['arms'][name] = row
            continue
        if directory.exists():
            backup = directory.with_name(directory.name + '_interrupted_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            directory.rename(backup)
        directory.mkdir()
        configure(SimpleNamespace(seed=SEED, keep_per_iter=.975, alpha=.7,
                                  layer_floor=.03, retrain_epochs=5), directory, metadata, 60)
        config.glo_train_stage = 'fixed_mask_recovery'
        config.glo_epoch_count = source_epoch
        model = copy.deepcopy(source_model)
        if not identical_state(model, source_model):
            raise AssertionError(f'GPU source copy differs in {name}')
        for parameter_name, parameter in model.named_parameters():
            parameter.requires_grad_(not parameter_name.endswith('.mask'))
        frozen = lfwa.assert_masks(model, source_masks)
        # Reset immediately before loader creation and training so each arm has
        # identical augmentation/dropout random streams despite its different LR.
        setseed(SEED)
        train_loader, val_loader = make_datasets(frames['train'], frames['val'], True, BATCH, lfwa.LABEL_COLUMNS)
        loaders = {'train': train_loader, 'test': val_loader}
        cfgs = ['LFWA', 'FairGRAPE', 'classes', source_sparsity['target_sparsity'], frames['val'], None,
                78, 'mobilenetv2', lfwa.LABEL_COLUMNS, lfwa.OUTPUT_SLICES, 'young', 0, SEED]
        write_json(root / 'status.json', {'status': 'running', 'arm': name, 'stage': 'recovery_training',
                                         'learning_rate': learning_rate, 'epochs': EPOCHS, 'updated_at': now()})
        started = time.monotonic()
        model, _ = train(model, torch.nn.CrossEntropyLoss(), loaders, [learning_rate], [EPOCHS],
                         lfwa.TARGETS, lfwa.OUTPUT_SLICES, cfgs=cfgs, use_debiasing=False)
        lfwa.assert_masks(model, frozen)
        if lfwa.sparsity(model) != source_sparsity:
            raise AssertionError(f'Fixed-mask recovery changed sparsity: {name}')
        checkpoint_path = directory / 'recovered.pt'
        lfwa.save_checkpoint(model, checkpoint_path, metadata, stage='fixed_mask_recovery', iteration=60,
                             source_checkpoint=str(args.checkpoint), learning_rate=learning_rate,
                             recovery_epochs=EPOCHS, arm=name, source_alpha=.7)
        after = evaluate(model, frames['val'], directory / 'validation', BATCH)
        epoch_frame = pd.read_csv(directory / 'epochs.csv')
        validation_epochs = epoch_frame.loc[epoch_frame.phase == 'val'].copy()
        if len(validation_epochs) != EPOCHS:
            raise AssertionError(f'Expected {EPOCHS} validation epochs in {name}')
        validation_epochs.insert(0, 'recovery_epoch', range(1, EPOCHS + 1))
        validation_epochs.to_csv(directory / 'validation_by_epoch.csv', index=False)
        row = {'status': 'complete', 'arm': name, 'learning_rate': learning_rate,
               'epochs': EPOCHS, 'batch': BATCH, 'paired_seed': SEED,
               'source_state_identical': True, 'masks_unchanged': True, 'sparsity_unchanged': True,
               'sparsity': source_sparsity, 'source_validation': before, 'validation': after,
               'delta': {key: after[key] - before[key] for key in ('accuracy', 'ba', 'eo')},
               'selected_validation': getattr(model, 'lfwa_validation_metrics', None),
               'validation_epochs': validation_epochs.to_dict('records'),
               'checkpoint_sha256': sha256(checkpoint_path), 'completed_at': now(),
               'seconds': time.monotonic() - started}
        write_json(marker, row)
        results['arms'][name] = row
        write_json(root / 'results.json', results)
        print(json.dumps({'arm': name, 'validation': after, 'delta': row['delta']}), flush=True)
        del model, frozen, loaders, train_loader, val_loader
        torch.cuda.empty_cache()
    write_json(root / 'results.json', results)
    write_json(root / 'complete.json', {'status': 'complete', 'completed_at': now(),
                                      'results': str(root / 'results.json')})
    write_json(root / 'status.json', {'status': 'complete', 'updated_at': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--lfwa-csv', type=Path, default=ROOT / 'csv/LFWA.csv')
    parser.add_argument('--lfwa-image-root', type=Path, default=ROOT / 'Images/LFWA')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('CUDA_CACHE_MAXSIZE', '4294967296')
    if args.dry_run:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.chdir(ROOT)
    for field in ('checkpoint', 'run_dir', 'lfwa_csv', 'lfwa_image_root'):
        setattr(args, field, getattr(args, field).resolve())
    import torch
    import numpy as np
    import lfwa
    torch.set_num_threads(2)
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ('dry_run_metadata.json' if args.dry_run else 'run.json')
    frames, metadata = lfwa.load_frames(args.lfwa_csv, args.lfwa_image_root)
    source_model, checkpoint = load_model(args.checkpoint, metadata, 'cpu')
    source_audit = validate_source(source_model, checkpoint, args.checkpoint)
    protocol = {'source_checkpoint': str(args.checkpoint), 'source_sha256': sha256(args.checkpoint),
                'source_audit': source_audit, 'experiment_metadata': metadata,
                'arms': [{'name': name, 'learning_rate': rate} for name, rate in ARMS],
                'epochs_per_arm': EPOCHS, 'batch': BATCH, 'paired_seed': SEED, 'optimizer': 'Adam',
                'new_pruning': False, 'score_recomputation': False, 'grl': False,
                'evaluation_split': 'val', 'source_metric_tolerance': METRIC_TOLERANCE,
                'selection': 'Existing maximum validation accuracy, then lower EO/loss ties, among ten new epochs. Source is a separately reported baseline.',
                'hypothesis': 'Additional recovery at fixed masks; higher LR is a diagnostic, not an assumed improvement.'}
    files = ['scripts/run_lfwa_fixed_mask_recovery.py', 'scripts/run_lfwa_cap_trajectory.py',
             'lfwa.py', 'lfwa_experiment.py', 'prune.py', 'train_and_val.py',
             'dataset.py', 'util.py', 'config.py', 'adversarial_debiasing.py']
    hashes = {name: sha256(ROOT / name) for name in files}
    if marker.exists():
        if not args.resume:
            raise FileExistsError(f'Existing run requires --resume: {root}')
        old = json.loads(marker.read_text())
        if old['protocol'] != protocol or old['source_sha256'] != hashes:
            raise ValueError('Resume protocol or source code differs from the recorded run')
    else:
        write_json(marker, {'created_at': now(), 'protocol': protocol, 'source_sha256': hashes,
                            'python': sys.executable, 'command': sys.argv, 'dry_run': args.dry_run,
                            'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                        'numpy': np.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                        'cuda_cache_maxsize': os.environ.get('CUDA_CACHE_MAXSIZE')}})
        for name in files:
            destination = root / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
    try:
        run(args, root, source_model, checkpoint, source_audit, frames, metadata)
    except Exception as error:
        write_json(root / 'status.json', {'status': 'failed', 'error': repr(error), 'updated_at': now()})
        raise


if __name__ == '__main__':
    main()
