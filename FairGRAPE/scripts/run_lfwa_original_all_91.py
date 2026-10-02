#!/usr/bin/env python3
"""Original FairGRAPE LFWA baseline, all supported layers, LR1e-5, 91 rounds."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD = ROOT / 'lfwa_runs/alpha0p7_seed42_no_final_20260916_143706/run.json'
PRIOR = ROOT / 'lfwa_diagnostics/protection15_alpha05_07_91iter_lr1e5_20260918/analysis/results.json'
FILES = ['scripts/run_lfwa_original_all_91.py', 'lfwa_original_all_experiment.py', 'tests/test_lfwa_original_all.py',
         'lfwa_original_experiment.py', 'tests/test_lfwa_original.py',
         'lfwa_experiment.py', 'lfwa.py', 'prune.py', 'prune_orig_ref.py', 'fairgrape_original.py',
         'train_and_val.py', 'dataset.py', 'util.py', 'config.py', 'adversarial_debiasing.py']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify(root):
    m = read(root / 'manifest.json')
    for path, expected in m['source_sha256'].items():
        assert sha(ROOT / path) == expected, path
    assert sha(m['arguments']['checkpoint']) == m['dense_sha256']
    assert sha(OLD) == m['historical_run_sha256']
    assert sha(PRIOR) == m['historical_results_sha256']
    for path, expected in m['comparison_checkpoint_sha256'].items():
        assert sha(path) == expected, path
    return m


def prepare(root):
    assert not (root / 'manifest.json').exists()
    old = read(OLD)
    args = dict(old['arguments'], impt=4, fg_mode='original', fg_scope='all', impt2_alpha=None,
                delta_p=0, para_batch=1, prune_iter=91, no_retrain=True, defer_test=True,
                lfwa_run_dir=str(root / 'run'), checkpoint=str((ROOT / old['arguments']['checkpoint']).resolve()))
    assert args['retrain_lr'] == 1e-5 and args['retrain_iter'] == 5 and args['keep_per_iter'] == .975
    rows = [r for r in read(PRIOR) if r['gamma'] == .005]
    assert len(rows) == 4
    hashes = {f: sha(ROOT / f) for f in FILES}
    for f in FILES:
        dest = root / 'source' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    write(root / 'manifest.json', {
        'created_at': now(), 'arguments': args, 'source_sha256': hashes,
        'dense_sha256': sha(args['checkpoint']), 'historical_run_sha256': sha(OLD),
        'historical_results_sha256': sha(PRIOR), 'comparison_rows': rows,
        'comparison_checkpoint_sha256': {r['checkpoint']: sha(r['checkpoint']) for r in rows},
        'experiment_metadata': old['experiment_metadata'], 'mode': 'original', 'scope': 'all',
        'reference': 'prune_orig_ref.py; Bernardo1998/FairGRAPE b677eb9',
        'scope_adaptation': None, 'scope_definition': 'All 52 Conv2d and 1 Linear weights; stem, final conv and classifier included',
        'scoring': 'Unmodified original scoring; train mode, Adam1e-4, accumulated gradients, last-group normalization',
        'retraining': 'Same LFWA selected-validation model, Adam1e-5, five epochs per round, no final8+1+1',
        'seed_policy': 'setseed42 once; no per-round reseeding', 'alpha': None, 'protection_ratio': None,
        'comparison': 'Validation at nearest actual whole sparsity .60 and historical final ~.692; original all-scope iteration91 is a separate high-sparsity endpoint',
        'comparison_target_sparsities': {'near60': .6, 'near70_matched': sum(r['model_sparsity'] for r in rows if r['point'] == 'iter91') / 2},
        'user_correction': 'The prior blocks adaptation did not use the requested original full pruning scope',
        'limitation': 'Weight pruning versus channel pruning; original full pruning scope, local LFWA data/training. Different iterations at matched actual sparsity imply different cumulative training epochs.'})
    print('PREPARED', root, flush=True)


def preflight(root):
    m = verify(root)
    import torch
    import lfwa
    import prune
    import lfwa_original_all_experiment as entry
    torch.set_num_threads(2)
    frames, meta = lfwa.load_frames(m['arguments']['lfwa_csv'], m['arguments']['lfwa_image_root'])
    assert meta == m['experiment_metadata']
    assert sorted(frames['train']['young'].unique().tolist()) == [0, 1]
    assert len(lfwa.TARGETS) == 39 and len(lfwa.OUTPUT_SLICES) == 39 and len(lfwa.LABEL_COLUMNS) == 40
    model, ck = lfwa.load_checkpoint_model(m['arguments']['checkpoint'], meta, 'cpu')
    sp = entry.all_scope_sparsity(model)
    assert sp['model_removed'] == 0 and sp['model_parameters'] == 2323790 and sp['target_weights'] == 2289600
    scope = prune._get_fg_scope_layers(model, 'all')
    assert len(scope) == 53
    assert {'features.0.0', 'features.18.0', 'classifier.1'} <= set(scope)
    for row in m['comparison_rows']:
        ck = torch.load(row['checkpoint'], map_location='cpu')
        lfwa.validate_checkpoint(ck, meta)
        assert ck['stage'] == 'prune' and ck['iteration'] == row['iteration']
    # Run meaningful direct-reference and 39-task integration tests in this environment.
    with (root / 'tests.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_fairgrape_original.py',
                        'tests/test_lfwa_original.py', 'tests/test_lfwa_original_all.py'], stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    write(root / 'preflight_complete.json', {'status': 'passed', 'time': now(), 'sparsity': sp,
          'target_layers': scope, 'sensitive_classes': 2, 'target_attributes': 39,
          'reference_and_multitask_tests': 'tests.log', 'comparison_checkpoints_validated': 4})
    print('PREFLIGHT_PASSED', flush=True)


def train_one(root):
    m = verify(root)
    assert (root / 'preflight_complete.json').exists()
    import torch
    import numpy as np
    import random
    import config
    import lfwa
    import prune
    import lfwa_original_all_experiment as entry
    torch.set_num_threads(2)
    original_save = lfwa.save_checkpoint
    original_get_mask = prune.FairGRAPE.get_mask

    def get_mask(self, cfg):
        iteration = config.glo_prune_iter + 1
        write(root / 'status.json', {'status': 'running', 'stage': 'original_scoring_and_selection',
              'iteration': iteration, 'completed_iteration': iteration - 1, 'time': now()})
        # Assert original scoring changes only its disposable copy, not the retraining source.
        before = {n: t.detach().cpu().clone() for n, t in self.prun_model.state_dict().items()}
        masks = original_get_mask(self, cfg)
        assert all(torch.equal(before[n], t.detach().cpu()) for n, t in self.prun_model.state_dict().items())
        write(root / 'status.json', {'status': 'running', 'stage': 'mask_application_and_five_epoch_retraining',
              'iteration': iteration, 'completed_iteration': iteration - 1, 'time': now()})
        return masks

    def save(model, path, metadata, **details):
        ck = original_save(model, path, metadata, **details)
        if Path(path).name.startswith('prune_') and details.get('stage') == 'prune':
            i = details['iteration']
            rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                   'cuda': torch.cuda.get_rng_state_all(), 'iteration': i}
            dest = Path(path).with_suffix('.rng.pt')
            temp = dest.with_suffix('.part')
            torch.save(rng, temp)
            temp.replace(dest)
            write(root / 'progress' / f'iteration_{i:03d}.json', {
                'iteration': i, 'checkpoint': str(path), 'sparsity': ck['sparsity'],
                'validation': ck['validation'], 'seconds': details.get('seconds'), 'time': now()})
            write(root / 'status.json', {'status': 'running', 'stage': 'checkpoint_saved',
                  'completed_iteration': i, 'whole_sparsity': ck['sparsity']['model_sparsity'], 'time': now()})
        return ck

    prune.FairGRAPE.get_mask = get_mask
    lfwa.save_checkpoint = save
    entry.run(SimpleNamespace(**m['arguments']))


def compare(root):
    m = verify(root)
    import torch
    import pandas as pd
    import lfwa
    import lfwa_original_all_experiment as entry
    lfwa.sparsity = entry.all_scope_sparsity
    torch.set_num_threads(2)
    frames, meta = lfwa.load_frames(m['arguments']['lfwa_csv'], m['arguments']['lfwa_image_root'])
    records = [read(root / 'progress' / f'iteration_{i:03d}.json') for i in range(1, 92)]
    choices = {label: min(records, key=lambda r: (abs(r['sparsity']['model_sparsity'] - target), r['iteration']))
               for label, target in m['comparison_target_sparsities'].items()}
    choices['iter91_high_sparsity'] = records[-1]
    rows = [dict(r, method='impt2', lr=1e-5,
                 point='near70_matched' if r['point'] == 'iter91' else 'near60',
                 historical_point=r['point'], pruning_scope='blocks') for r in m['comparison_rows']]
    for label, record in [('dense', {'checkpoint': m['arguments']['checkpoint'], 'iteration': 0}), *choices.items()]:
        dest = root / 'analysis' / 'evaluations' / label
        model, ck = lfwa.load_checkpoint_model(record['checkpoint'], meta, 'cuda:0')
        assert ck['iteration'] == record['iteration']
        if label != 'dense':
            assert ck['stage'] == 'prune'
        summary = lfwa.evaluate_model(model, frames['val'], dest, 256)
        predictions = pd.read_csv(dest / 'predictions.csv')
        rates = predictions[[n + '_pred' for n in lfwa.TARGETS]].mean()
        summary['near_constant_99pct_attributes'] = int(((rates <= .01) | (rates >= .99)).sum())
        summary['constant_prediction_attributes'] = int(((rates == 0) | (rates == 1)).sum())
        write(dest / 'summary.json', summary)
        rows.append({'method': 'dense' if label == 'dense' else 'FairGRAPE_original_all', 'alpha': None,
                     'gamma': None, 'lr': None if label == 'dense' else 1e-5, 'pruning_scope': 'all', 'point': label, 'iteration': record['iteration'],
                     'checkpoint': record['checkpoint'], 'evaluation': str(dest), **summary})
        del model, ck
        torch.cuda.empty_cache()
        write(root / 'analysis' / 'results.json', rows)
    pd.DataFrame(rows).to_csv(root / 'analysis' / 'metrics.csv', index=False)
    lines = ['# LFWA original FairGRAPE 전체 레이어 실험', '',
             '동일 dense·split, 재학습 LR1e-5/5epoch. 원본 전체53개 Conv/Linear층 프루닝. 추가8+1+1 없음. Validation 비교.',
             'FairGRAPE original은 개별 가중치를 제거하며 α·보호 비율이 없다. impt2는 채널 단위이므로 제거 단위도 다르다.', '',
             '| 방법 | 지점 | α | iter | 전체 희소도 | Acc | BA | EO↓ | 근상수 속성 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['point']} | {r['alpha']} | {r['iteration']} | {r['model_sparsity']:.4%} | {r['accuracy']:.6f} | {r['ba']:.6f} | {r['eo']:.6f} | {r['near_constant_99pct_attributes']} |")
    lines.extend(['', '단일 seed. 원본 전체 프루닝 범위를 사용한 LFWA baseline. 실제 희소도를 맞추면 iter와 누적 재학습 에폭은 다르다. 원본91iter(~88.7%)는 기존91iter(~69.2%)와 희소도가 달라 별도 끝점으로 표기한다.',
                  'EO 하락은 BA 및 근상수 속성 수와 함께 해석한다. 점수 규칙만의 효과와 가중치/채널 단위 효과를 이 비교만으로 분리할 수 없다.'])
    (root / 'analysis' / 'report.md').write_text('\n'.join(lines) + '\n')


def queue(root):
    m = verify(root)
    assert (root / 'preflight_complete.json').exists()
    assert not (root / 'run' / 'run.json').exists(), 'Do not overwrite an interrupted run; RNG-aware recovery required'
    write(root / 'status.json', {'status': 'starting', 'time': now()})
    with (root / 'train.log').open('ab', buffering=0) as log:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root), '--train'],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(root / 'status.json', {'status': 'evaluating_comparisons', 'time': now()})
    compare(root)
    write(root / 'complete.json', {'status': 'complete', 'time': now(), 'report': str(root / 'analysis' / 'report.md')})
    write(root / 'status.json', {'status': 'complete', 'time': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    for flag in ['prepare', 'preflight', 'train', 'compare']:
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if args.preflight:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        if args.prepare:
            prepare(root)
        elif args.preflight:
            preflight(root)
        elif args.train:
            train_one(root)
        elif args.compare:
            compare(root)
        else:
            queue(root)
    except Exception as error:
        write(root / ('failure_train.json' if args.train else 'failure.json'), {'error': repr(error), 'time': now()})
        write(root / 'status.json', {'status': 'failed', 'error': repr(error), 'time': now()})
        raise
