#!/usr/bin/env python3
"""Validation-only, saved-score single-layer cap ablations; no retraining."""
import argparse
import collections
import copy
import csv
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_lfwa_channel_cap_diagnostic import (
    apply_selection, check_reference_masks, evaluate, read_plan, sha256,
    validate_source, write_json,
)


def make_selections(plan, cap_scopes=None):
    """Use archived raw values, not the more coarsely rounded score column."""
    with open(plan['source_channel_csv']) as handle:
        rows = list(csv.DictReader(handle))
    by_layer = collections.defaultdict(list)
    for row in rows:
        by_layer[row['block']].append(row)
    active = {name: sum(int(row['weights']) > 0 for row in part)
              for name, part in by_layer.items()}
    candidates = []
    for name, part in by_layer.items():
        perf_max = max(float(row['perf_raw']) for row in part)
        phi_max = max(float(row['phi_raw']) for row in part)
        for row in part:
            if int(row['weights']) <= 0 or row['status'] == 'protected':
                continue
            score = (plan['alpha'] * float(row['perf_raw']) / (perf_max + 1e-12)
                     - (1 - plan['alpha']) * float(row['phi_raw']) / (phi_max + 1e-12))
            candidates.append((score, name, int(row['ch']), int(row['weights'])))
    candidates.sort(key=lambda row: row[0])
    target = round(sum(int(row['weights']) for row in rows) * (1 - plan['keep_per_iter']))
    floor = {name: math.ceil(len(part) * plan['layer_floor'])
             for name, part in by_layer.items()}

    def select(capped_layers):
        chosen, weights = [], 0
        counts = collections.Counter()
        for score, name, channel, weight in candidates:
            if weights >= target:
                break
            if active[name] - counts[name] - 1 < floor[name]:
                continue
            if name in capped_layers and counts[name] >= math.floor(active[name] * .1):
                continue
            chosen.append((name, channel))
            counts[name] += 1
            weights += weight
        if weights < target:
            raise AssertionError('Strict cap cannot meet the nominal weight budget')
        return {'selected_channels': chosen, 'expected_removed_weights': weights,
                'removed_channels_by_layer': dict(counts)}

    historical = {(row['block'], int(row['ch'])) for row in rows
                  if row['status'].startswith('pruned_now')}
    baseline = select(())
    assert set(baseline['selected_channels']) == historical
    baseline_plan = next(arm for arm in plan['arms'] if arm['name'] == 'baseline')
    assert set(baseline['selected_channels']) == set(baseline_plan['selected_channels'])
    assert baseline['expected_removed_weights'] == baseline_plan['expected_removed_weights']
    arms = []
    if cap_scopes is None:
        cap_scopes = [('features.2',), ('features.13',)]
    for scope in cap_scopes:
        if not scope or len(scope) != len(set(scope)) or not set(scope) <= set(active):
            raise ValueError('Invalid or duplicate cap scope: ' + repr(scope))
        arm = select(scope)
        arm.update(name='features_' + '_'.join(name.split('.')[1] for name in scope) + '_cap10',
                   capped_layers=list(scope),
                   layer_cap_counts={name: math.floor(active[name] * .1) for name in scope},
                   reference_selection=False)
        for name in scope:
            assert arm['removed_channels_by_layer'].get(name, 0) <= arm['layer_cap_counts'][name]
        arms.append(arm)
    return arms, baseline, target, candidates


def run(args):
    import pandas as pd
    import torch
    import lfwa
    import prune
    from util import setseed

    torch.set_num_threads(2)
    # Assemble and audit masks on CPU; only forward evaluation uses CUDA.
    prune.device = torch.device('cpu')
    root = args.run_dir.resolve()
    marker = root / ('dry_run_metadata.json' if args.dry_run else 'run_metadata.json')
    if marker.exists() or (root / 'complete.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed work: {root}')
    root.mkdir(parents=True, exist_ok=True)
    plan = read_plan(args.source_plan.resolve())
    scopes = [tuple(value.split(',')) for value in args.cap_scope] if args.cap_scope else None
    arms, baseline, target, candidates = make_selections(plan, scopes)
    work_plan = copy.deepcopy(plan)
    work_plan['arms'] = arms
    work_plan['batch'] = args.batch
    protocol = {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_plan': str(args.source_plan.resolve()), 'plan': work_plan,
        'purpose': 'Single-layer cap removal-only ablation at the same nominal budget',
        'capped_layers': [arm['capped_layers'] for arm in arms],
        'nominal_removed_weights': target, 'retraining_epochs_executed': 0,
        'evaluation_split': 'val', 'evaluation_batch': args.batch,
        'archived_raw_score_precision': 'CSV scientific notation with seven significant digits; not lossless float storage',
        'score_replay_matches_historical_selected_set': True,
        'score_replay_matches_historical_removed_weight_count': True,
        'baseline_removed_weights': baseline['expected_removed_weights'],
        'source_script_sha256': sha256(__file__), 'helper_script_sha256': sha256(ROOT / 'scripts/run_lfwa_channel_cap_diagnostic.py'),
        'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'CUDA_CACHE_MAXSIZE': os.environ.get('CUDA_CACHE_MAXSIZE'),
        'dry_run': args.dry_run,
        'interpretation': 'Changing one layer cap redistributes removal elsewhere; interactions prevent additive causal percentage attribution.'}
    write_json(marker, protocol)
    write_json(root / 'plan.json', work_plan)
    shutil.copy2(__file__, root / Path(__file__).name)
    pd.DataFrame(candidates, columns=['reconstructed_score', 'block', 'ch', 'weights']).to_csv(
        root / 'candidate_ranking.csv', index=False, float_format='%.17g')
    setseed(plan['seed'])
    print('Loading validated LFWA frames', flush=True)
    frames, metadata = lfwa.load_frames(plan['lfwa_csv'], plan['lfwa_image_root'])
    print('Loading source checkpoint on CPU', flush=True)
    source, checkpoint = lfwa.load_checkpoint_model(plan['source_checkpoint'], metadata, 'cpu')
    source_sparsity = lfwa.sparsity(source)
    source_hash = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    print('Validating CSV activity and selected channel masks', flush=True)
    audits = validate_source(source, checkpoint, pd.read_csv(plan['source_channel_csv']), work_plan)
    # Validate baseline masks against the historical checkpoint without evaluation.
    print('Checking baseline masks against historical checkpoint', flush=True)
    historical_model = apply_selection(source, baseline['selected_channels'])
    baseline_audit = check_reference_masks(historical_model, {'name': 'baseline'}, plan)
    del historical_model
    write_json(root / 'historical_baseline_verification.json', baseline_audit)
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError('GPU validation requested but CUDA is unavailable')
    output = {'source_sparsity': source_sparsity, 'baseline_verification': baseline_audit,
              'nominal_removed_weights': target, 'arms': {}}
    for arm in arms:
        arm_root = root / arm['name']
        arm_root.mkdir(exist_ok=True)
        print('Applying arm: ' + arm['name'], flush=True)
        model = apply_selection(source, arm['selected_channels'])
        sparsity = lfwa.sparsity(model)
        assert sparsity['target_removed'] - source_sparsity['target_removed'] == arm['expected_removed_weights']
        audit = {**audits[arm['name']], 'sparsity': sparsity,
                 'capped_layers': arm['capped_layers'], 'strict_cap_counts': arm['layer_cap_counts']}
        for layer_row in audit['layers']:
            cap_count = arm['layer_cap_counts'].get(layer_row['block'])
            layer_row['maximum_remove_this_step'] = cap_count
            if cap_count is not None:
                assert layer_row['selected_channels'] <= cap_count
        write_json(arm_root / 'mask_audit.json', audit)
        pd.DataFrame(audit['layers']).to_csv(arm_root / 'layer_audit.csv', index=False)
        torch.save({name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
                    if name.endswith('.mask')}, arm_root / 'masks.pt')
        if not args.dry_run:
            model.to('cuda:0')
            audit['post_prune'] = evaluate(model, frames['val'], arm_root / 'post_prune', args.batch)
            lfwa.save_checkpoint(model, arm_root / 'post_prune.pt', metadata,
                                 stage='single_layer_cap_removal_only', iteration=plan['iteration'],
                                 arm=arm['name'], capped_layers=arm['capped_layers'], retraining_epochs=0)
        output['arms'][arm['name']] = audit
        write_json(root / ('dry_run_results.json' if args.dry_run else 'results.json'), output)
        del model
        if not args.dry_run:
            torch.cuda.empty_cache()
    assert all(torch.equal(source.state_dict()[name], value) for name, value in source_hash.items())
    write_json(root / ('dry_run_complete.json' if args.dry_run else 'complete.json'),
               {'status': 'complete', 'evaluation_split': 'val', 'retraining_epochs': 0})
    print(json.dumps({'status': 'complete', 'run_dir': str(root), 'dry_run': args.dry_run}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--cap-scope', action='append', help='Comma-separated blocks; repeat for multiple arms')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.dry_run:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run(args)
