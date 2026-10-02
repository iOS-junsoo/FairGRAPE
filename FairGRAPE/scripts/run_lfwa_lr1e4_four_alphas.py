#!/usr/bin/env python3
"""Approved LFWA impt2 LR-only sweep: .9, .7, .5, .3; gamma.005,91iter."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import run_lfwa_lr1e4_gamma005_91 as adapter
from run_lfwa_protection15_91 import read, write, sha, now

PREPARED = ROOT / 'lfwa_diagnostics/lr1e4_gamma005_alpha07_05_91iter_20260919'
INDEX = ROOT / 'lfwa_diagnostics/protection15_alpha05_07_91iter_lr1e5_20260918/baseline_checkpoint_index.json'
FAIRGRAPE = ROOT / 'lfwa_diagnostics/original_fairgrape_all_91iter_lr1e5_20260919/analysis/results.json'
BASELINES = {
    .9: ROOT / 'lfwa_runs/alpha0p9_seed42_no_final_20260916_143657',
    .7: ROOT / 'lfwa_runs/alpha0p7_seed42_no_final_20260916_143706',
    .5: ROOT / 'lfwa_runs/seed42_alpha0p5_20260916_022453/pruned',
    .3: ROOT / 'lfwa_runs/seed42_20260915_073604/pruned',
}


def verify(root):
    m = adapter.verify(root)
    assert [c['alpha'] for c in m['conditions']] == [.9, .7, .5, .3]
    assert sha(FAIRGRAPE) == m['fairgrape_all_results_sha256']
    assert sha(INDEX) == m['historical_index_sha256']
    return m


def prepare(root):
    assert not (root / 'manifest.json').exists()
    m = copy.deepcopy(adapter.verify(PREPARED))
    conditions = []
    shared_dense = m['conditions'][0]['checkpoint']
    shared_hash = m['conditions'][0]['checkpoint_sha256']
    for alpha, old in BASELINES.items():
        info = read(old / 'run.json')
        args = info['arguments']
        assert info['alpha'] == alpha and info['normalization'] == 'max' and info['protection_ratio'] == .005
        assert info['prune_iterations'] == 91 and args['impt'] == 2
        assert args['retrain_lr'] == 1e-5 and args['retrain_iter'] == 5 and args['keep_per_iter'] == .975
        assert info['layer_floor'] == .03 and info['layer_cap'] is None
        assert sha(args['checkpoint']) == shared_hash
        name = 'alpha' + str(alpha).replace('.', 'p')
        new_args = dict(args, checkpoint=shared_dense, lfwa_run_dir=str(root / 'runs' / name),
                        retrain_lr=1e-4, no_retrain=True, defer_test=True)
        conditions.append({'name': name, 'alpha': alpha, 'baseline': str(old),
                           'baseline_manifest_sha256': sha(old / 'run.json'), 'checkpoint': shared_dense,
                           'checkpoint_sha256': shared_hash, 'arguments': new_args})
    # All four runs differ only in alpha and the output directory.
    normalized = [{k: v for k, v in c['arguments'].items() if k not in ['impt2_alpha', 'lfwa_run_dir']} for c in conditions]
    assert all(a == normalized[0] for a in normalized)
    files = list(m['source_sha256']) + ['scripts/run_lfwa_lr1e4_four_alphas.py']
    hashes = {f: sha(ROOT / f) for f in files}
    for f in files:
        dest = root / 'source' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    m.update(created_at=now(), conditions=conditions, source_sha256=hashes,
             supersedes_unlaunched_plan=str(PREPARED), historical_index_sha256=sha(INDEX),
             fairgrape_all_results=str(FAIRGRAPE), fairgrape_all_results_sha256=sha(FAIRGRAPE),
             scope='Four independent impt2 runs: alpha.9,.7,.5,.3; LR1e-4, protection.005, unchanged channel scope',
             comparison='Each alpha: LR1e-4 versus LR1e-5 at protection.005; actual nearest whole60 and iteration91. No final8+1+1.')
    write(root / 'manifest.json', m)
    print('PREPARED_FOUR_ALPHAS', root, flush=True)


def preflight(root):
    m = verify(root)
    # Reuse the frozen adapter's LR-only argument and implementation checks.
    adapter.preflight(root)
    preliminary = read(root / 'preflight_complete.json')
    write(root / 'adapter_preflight_complete.json', preliminary)
    (root / 'preflight_complete.json').unlink()
    import torch
    import lfwa
    torch.set_num_threads(2)
    _, metadata = lfwa.load_frames(ROOT / 'csv/LFWA.csv', ROOT / 'Images/LFWA')
    cached = read(INDEX)
    indexes = {}
    for c in m['conditions']:
        if c['name'] in cached:
            records = copy.deepcopy(cached[c['name']])
            assert len(records) == 91
        else:
            records = []
            for i in range(1, 92):
                path = Path(c['baseline']) / f'prune_{i:03d}.pt'
                ck = torch.load(path, map_location='cpu')
                lfwa.validate_checkpoint(ck, metadata)
                assert ck['stage'] == 'prune' and ck['iteration'] == i
                records.append({'iteration': i, 'checkpoint': str(path), 'sha256': sha(path),
                                'sparsity': ck['sparsity'], 'validation': ck['validation']})
        assert all(a['sparsity']['model_removed'] <= b['sparsity']['model_removed'] for a, b in zip(records, records[1:]))
        choices = [min(records, key=lambda x: (abs(x['sparsity']['model_sparsity'] - .6), x['iteration'])), records[-1]]
        for rec in choices:
            assert sha(rec['checkpoint']) == rec['sha256']
            ck = torch.load(rec['checkpoint'], map_location='cpu')
            lfwa.validate_checkpoint(ck, metadata)
            assert ck['stage'] == 'prune' and ck['iteration'] == rec['iteration']
        indexes[c['name']] = records
        print('BASELINE_INDEX_READY', c['name'], flush=True)
    write(root / 'baseline_checkpoint_index.json', indexes)
    write(root / 'preflight_complete.json', dict(preliminary, time=now(), alphas=[.9, .7, .5, .3],
          full_baseline_trajectories=4, selected_historical_checkpoints_verified=8,
          index_sha256=sha(root / 'baseline_checkpoint_index.json')))
    print('FOUR_ALPHA_PREFLIGHT_PASSED', flush=True)


def compare(root):
    m = verify(root)
    import torch
    import pandas as pd
    import lfwa
    torch.set_num_threads(2)
    frames, metadata = lfwa.load_frames(ROOT / 'csv/LFWA.csv', ROOT / 'Images/LFWA')
    assert sha(root / 'baseline_checkpoint_index.json') == read(root / 'preflight_complete.json')['index_sha256']
    old = read(root / 'baseline_checkpoint_index.json')
    rows = []
    for c in m['conditions']:
        new = [read(root / 'progress' / c['name'] / f'iteration_{i:03d}.json') for i in range(1, 92)]
        for lr, records in [(1e-5, old[c['name']]), (1e-4, new)]:
            choices = {'near60': min(records, key=lambda r: (abs(r['sparsity']['model_sparsity'] - .6), r['iteration'])),
                       'iter91': records[-1]}
            for point, rec in choices.items():
                cached = [r for r in m['reused_rows'] if lr == 1e-5 and r['gamma'] == .005
                          and r['alpha'] == c['alpha'] and r['point'] == point]
                if cached:
                    row = dict(cached[0], method='impt2', setting='lr1e5')
                    assert row['checkpoint'] == rec['checkpoint']
                else:
                    if 'sha256' in rec:
                        assert sha(rec['checkpoint']) == rec['sha256']
                    dest = root / 'analysis' / 'evaluations' / c['name'] / f'{lr:g}_{point}'
                    model, ck = lfwa.load_checkpoint_model(rec['checkpoint'], metadata, 'cuda:0')
                    assert ck['stage'] == 'prune' and ck['iteration'] == rec['iteration']
                    summary = lfwa.evaluate_model(model, frames['val'], dest, 256)
                    pred = pd.read_csv(dest / 'predictions.csv')
                    rates = pred[[n + '_pred' for n in lfwa.TARGETS]].mean()
                    summary['near_constant_99pct_attributes'] = int(((rates <= .01) | (rates >= .99)).sum())
                    summary['constant_prediction_attributes'] = int(((rates == 0) | (rates == 1)).sum())
                    write(dest / 'summary.json', summary)
                    row = dict(alpha=c['alpha'], gamma=.005, lr=lr, method='impt2',
                               setting='lr1e5' if lr == 1e-5 else 'lr1e4', point=point,
                               iteration=rec['iteration'], checkpoint=rec['checkpoint'], evaluation=str(dest), **summary)
                    del model, ck
                    torch.cuda.empty_cache()
                rows.append(row)
                write(root / 'analysis' / 'results.json', rows)
    rows.sort(key=lambda r: (r['point'], -r['alpha'], r['lr']))
    write(root / 'analysis' / 'results.json', rows)
    pd.DataFrame(rows).to_csv(root / 'analysis' / 'metrics.csv', index=False)
    lines = ['# LFWA impt2: 네 알파의 재학습률 단독 변경', '',
             '동일 dense·split·seed42, 보호0.5%, max, 단계당5epoch,91iter. 추가8+1+1 없는 Validation 비교.',
             '각 알파의 LR1e-5와 LR1e-4를 비교한다. 이 표의 모든 행은 사용자 impt2 방법이다.', '',
             '| 지점 | α | LR | iter | 전체 희소도 | Acc | BA | EO↓ | 근상수 속성 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['point']} | {r['alpha']} | {r['lr']:g} | {r['iteration']} | {r['model_sparsity']:.4%} | {r['accuracy']:.6f} | {r['ba']:.6f} | {r['eo']:.6f} | {r['near_constant_99pct_attributes']} |")
    contrasts = []
    for point in ['near60', 'iter91']:
        for alpha in BASELINES:
            pair = {r['lr']: r for r in rows if r['point'] == point and r['alpha'] == alpha}
            delta = {k: pair[1e-4][k] - pair[1e-5][k] for k in ['accuracy', 'ba', 'eo', 'model_sparsity', 'near_constant_99pct_attributes']}
            contrasts.append({'alpha': alpha, 'point': point, 'comparison': 'lr1e4_minus_lr1e5', 'delta': delta})
    write(root / 'analysis' / 'contrasts.json', contrasts)
    refs = [r for r in read(FAIRGRAPE) if r['method'] == 'dense' or (r['method'] == 'FairGRAPE_original_all' and r['point'] in ['near60', 'near70_matched'])]
    write(root / 'analysis' / 'fairgrape_all_reference.json', refs)
    lines += ['', '원본 FairGRAPE의 전체 레이어 baseline은 fairgrape_all_reference.json에 별도로 둔다. impt2 대조군과 혼동하지 않는다.',
              '단일seed 결과. EO는 BA 및 근상수 속성과 함께 판단한다. LR 변경 후 반복 과정의 마스크도 달라질 수 있으므로 고정마스크 재학습 실험은 아니다.']
    (root / 'analysis' / 'report.md').write_text('\n'.join(lines) + '\n')


def queue(root):
    m = verify(root)
    assert (root / 'preflight_complete.json').exists()
    for c in m['conditions']:
        out = Path(c['arguments']['lfwa_run_dir'])
        if (out / 'complete.json').exists():
            assert read(out / 'run.json')['arguments'] == c['arguments']
            assert (root / 'progress' / c['name'] / 'iteration_091.json').exists()
            continue
        assert not (out / 'run.json').exists(), 'Interrupted run needs RNG-aware recovery; do not overwrite'
        write(root / 'status.json', {'status': 'starting', 'condition': c['name'], 'time': now()})
        with (root / (c['name'] + '.log')).open('ab', buffering=0) as log:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root), '--train', c['name']],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(root / 'status.json', {'status': 'evaluating_comparisons', 'time': now()})
    compare(root)
    write(root / 'complete.json', {'status': 'complete', 'time': now(), 'alphas': [.9, .7, .5, .3], 'new_runs': 4,
                                 'report': str(root / 'analysis' / 'report.md')})
    write(root / 'status.json', {'status': 'complete', 'time': now()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--train')
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if a.preflight:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        if a.prepare:
            prepare(root)
        elif a.preflight:
            preflight(root)
        elif a.train:
            verify(root)
            adapter.train_one(root, a.train)
        else:
            queue(root)
    except Exception as error:
        write(root / ('failure_' + a.train + '.json' if a.train else 'failure.json'), {'error': repr(error), 'time': now()})
        write(root / 'status.json', {'status': 'failed', 'error': repr(error), 'time': now()})
        raise
