#!/usr/bin/env python3
"""Fixed dense, original impt2 protocol; change pruning/retraining seed to 100."""
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

REPO = Path(__file__).resolve().parents[1]
PRIOR = REPO / 'lfwa_diagnostics/lr1e4_gamma005_alpha09_07_05_03_91iter_20260920'
CORE = ['lfwa_experiment.py', 'lfwa.py', 'prune.py', 'train_and_val.py', 'dataset.py',
        'util.py', 'config.py', 'adversarial_debiasing.py', 'channel_cap.py']
ALPHAS = [.9, .7, .5, .3]

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.part')
    q.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    q.replace(p)

def prepare(root):
    assert not (root / 'manifest.json').exists()
    prior = read(PRIOR / 'manifest.json')
    assert read(PRIOR / 'complete.json')['status'] == 'complete'
    hashes = {}
    for f in CORE:
        assert sha(REPO / f) == prior['source_sha256'][f], f
        dest = root / 'source' / f; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, dest); hashes[f] = sha(dest)
    runner = Path(__file__).resolve()
    shutil.copy2(runner, root / 'source' / runner.name)
    conditions = []
    dense = prior['conditions'][0]['checkpoint']
    dense_hash = prior['conditions'][0]['checkpoint_sha256']
    for c in prior['conditions']:
        baseline = read(Path(c['baseline']) / 'run.json')
        old = baseline['arguments']
        assert baseline['protection_ratio'] == .005 and baseline['normalization'] == 'max'
        assert baseline['layer_floor'] == .03 and baseline['layer_cap'] is None
        assert old['seed'] == 42 and old['retrain_lr'] == 1e-5 and old['impt'] == 2
        assert sha(old['checkpoint']) == dense_hash
        args = dict(old, seed=100, checkpoint=dense,
                    lfwa_run_dir=str(root / 'runs' / c['name']), no_retrain=True, defer_test=True)
        diff = {k: {'old': old.get(k), 'new': v} for k, v in args.items() if old.get(k) != v}
        assert set(diff) <= {'seed', 'checkpoint', 'lfwa_run_dir', 'no_retrain', 'defer_test'}
        assert args['retrain_iter'] == 5 and args['keep_per_iter'] == .975
        conditions.append(dict(name=c['name'], alpha=c['alpha'], arguments=args,
                               baseline=c['baseline'], baseline_manifest_sha256=sha(Path(c['baseline']) / 'run.json'),
                               differences_from_seed42=diff))
    assert [c['alpha'] for c in conditions] == ALPHAS
    comparable = [{k: v for k, v in c['arguments'].items() if k not in ['impt2_alpha', 'lfwa_run_dir']} for c in conditions]
    assert all(x == comparable[0] for x in comparable)
    for f in ['baseline_checkpoint_index.json']:
        shutil.copy2(PRIOR / f, root / f)
    shutil.copy2(PRIOR / 'analysis/results.json', root / 'seed42_lr_reference.json')
    files = ['csv/LFWA.csv', 'csv/LFWA.metadata.json']
    write(root / 'manifest.json', dict(created_at=now(), conditions=conditions, source_sha256=hashes,
          runner_path=str(runner), runner_sha256=sha(runner), dense_checkpoint=dense, dense_sha256=dense_hash,
          data_sha256={f: sha(REPO / f) for f in files},
          baseline_index_sha256=sha(root / 'baseline_checkpoint_index.json'),
          reference_sha256=sha(root / 'seed42_lr_reference.json'),
          seed=100, dense_training_seed=42, protection_ratio=.005, retrain_lr=1e-5,
          strict_cap_ratio=None, epoch_observer=None, iterations=91, final_extra_epochs=0,
          seed_policy='Same seed42-trained dense for all alphas; setseed(100) once per independent pruning run. No per-iteration reseeding.',
          source_policy='Import byte-identical frozen core modules from this run/source; leave shared core unchanged.',
          comparison='Primary: seed100 vs seed42 at LR1e-5. Fixed dense and official split. Near whole60 and iteration91 (~69%). Not a new dense-training seed replicate.'))
    print('PREPARED', root, flush=True)

def verify(root):
    m = read(root / 'manifest.json')
    assert sha(m['runner_path']) == m['runner_sha256']
    for f, h in m['source_sha256'].items(): assert sha(root / 'source' / f) == h, f
    for f, h in m['data_sha256'].items(): assert sha(REPO / f) == h, f
    assert sha(m['dense_checkpoint']) == m['dense_sha256']
    assert sha(root / 'baseline_checkpoint_index.json') == m['baseline_index_sha256']
    assert sha(root / 'seed42_lr_reference.json') == m['reference_sha256']
    for c in m['conditions']:
        assert sha(Path(c['baseline']) / 'run.json') == c['baseline_manifest_sha256']
        a = c['arguments']
        assert a['seed'] == 100 and a['retrain_lr'] == 1e-5 and a['retrain_iter'] == 5
        assert a['checkpoint'] == m['dense_checkpoint'] and a['no_retrain'] and a['defer_test']
    return m

def activate(root):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(root / 'source'))
    import torch, lfwa, lfwa_experiment, config, prune
    torch.set_num_threads(2)
    for f in CORE:
        name = Path(f).stem
        if name in sys.modules:
            assert Path(sys.modules[name].__file__).resolve() == root / 'source' / f, name
    assert config.glo_impt2_layer_max_remove_ratio is None and config.glo_epoch_observer is None
    return torch, lfwa, lfwa_experiment, config, prune

def check_config(args, config, prune):
    assert config.glo_seed == 100 and args.seed == 100
    assert config.glo_impt2_layer_max_remove_ratio is None and config.glo_epoch_observer is None
    assert prune.IMPT2_NORM == 'max' and prune.IMPT2_PROTECTION_RATIO == .005
    assert prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER == .03 and prune.IMPT2_LAYER_CAP_MULTIPLIER is None
    assert args.retrain_lr == 1e-5 and args.retrain_iter == 5

def preflight(root):
    m = verify(root)
    torch, lfwa, entry, config, prune = activate(root)
    frames, meta = lfwa.load_frames()
    assert {k: len(frames[k]) for k in ['train', 'val', 'test']} == {'train': 5000, 'val': 1263, 'test': 6880}
    model, ck = lfwa.load_checkpoint_model(m['dense_checkpoint'], meta, 'cpu')
    assert lfwa.sparsity(model)['model_removed'] == 0
    assert lfwa.sparsity(model)['model_parameters'] == 2323790
    assert len(lfwa.TARGETS) == 39
    for c in m['conditions']:
        a = SimpleNamespace(**c['arguments'])
        assert read(Path(c['baseline']) / 'run.json')['experiment_metadata'] == meta
        assert entry.planned_iterations(a.keep_per_iter, a.prune_rate) == 91
        d = root / 'preflight_configuration' / c['name']; d.mkdir(parents=True, exist_ok=True)
        entry._configure(a, d, meta); check_config(a, config, prune)
    write(root / 'preflight_complete.json', dict(status='passed', time=now(), seed=100, dense_training_seed=42,
          dense_sha256=m['dense_sha256'], split_metadata=meta, frozen_source_import_verified=True,
          original_protocol_and_disabled_cap_verified=True, torch=torch.__version__, cuda=torch.version.cuda))
    print('PREFLIGHT_PASSED', flush=True)

def train_one(root, name):
    m = verify(root); assert read(root / 'preflight_complete.json')['status'] == 'passed'
    c = next(c for c in m['conditions'] if c['name'] == name)
    torch, lfwa, entry, config, prune = activate(root)
    import random, numpy as np
    original_config = entry._configure
    def configure(args, directory, metadata):
        original_config(args, directory, metadata); check_config(args, config, prune)
        write(directory / 'seed_protocol.json', dict(pruning_seed=100, dense_training_seed=42,
              dense_sha256=m['dense_sha256'], strict_cap_ratio=None, epoch_observer=None,
              retrain_lr=1e-5, protection_ratio=.005, frozen_source=str(root / 'source')))
    entry._configure = configure
    original_save = lfwa.save_checkpoint
    def save(model, path, metadata, **details):
        ck = original_save(model, path, metadata, **details)
        if details.get('stage') == 'prune' and Path(path).name.startswith('prune_'):
            check_config(SimpleNamespace(**c['arguments']), config, prune)
            assert prune.IMPT_TYPE2_ALPHA == c['alpha']
            i = details['iteration']
            write(root / 'progress' / name / f'iteration_{i:03d}.json', dict(iteration=i, checkpoint=str(path),
                  sparsity=ck['sparsity'], validation=ck['validation'], seconds=details.get('seconds'), time=now()))
            rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                       cuda=torch.cuda.get_rng_state_all(), iteration=i)
            dest = Path(path).with_suffix('.rng.pt'); temp = dest.with_suffix('.part')
            torch.save(rng, temp); temp.replace(dest)
            write(root / 'status.json', dict(status='running', condition=name, completed_iteration=i,
                  whole_sparsity=ck['sparsity']['model_sparsity'], time=now()))
        return ck
    lfwa.save_checkpoint = save
    entry.run(SimpleNamespace(**c['arguments']))
    info = read(Path(c['arguments']['lfwa_run_dir']) / 'run.json')
    assert info['arguments'] == c['arguments'] and info['prune_iterations'] == 91

def compare(root):
    m = verify(root); torch, lfwa, _, _, _ = activate(root)
    import pandas as pd
    frames, meta = lfwa.load_frames()
    rows = [dict(x, seed=42, setting='seed42_lr1e5') for x in read(root / 'seed42_lr_reference.json') if x['lr'] == 1e-5]
    assert len(rows) == 8
    for c in m['conditions']:
        recs = [read(root / 'progress' / c['name'] / f'iteration_{i:03d}.json') for i in range(1, 92)]
        for point, rec in [('near60', min(recs, key=lambda x: abs(x['sparsity']['model_sparsity'] - .6))), ('iter91', recs[-1])]:
            dest = root / 'analysis/evaluations' / c['name'] / point
            model, ck = lfwa.load_checkpoint_model(rec['checkpoint'], meta, 'cuda:0')
            assert ck['stage'] == 'prune' and ck['iteration'] == rec['iteration']
            summary = lfwa.evaluate_model(model, frames['val'], dest, 256)
            for k in ['accuracy', 'ba', 'eo']: assert abs(summary[k] - rec['validation'][k]) < 1e-7
            pred = pd.read_csv(dest / 'predictions.csv')
            rates = pred[[n + '_pred' for n in lfwa.TARGETS]].mean()
            summary['near_constant_99pct_attributes'] = int(((rates <= .01) | (rates >= .99)).sum())
            summary['constant_prediction_attributes'] = int(((rates == 0) | (rates == 1)).sum())
            write(dest / 'summary.json', summary)
            rows.append(dict(alpha=c['alpha'], seed=100, gamma=.005, lr=1e-5, setting='seed100_lr1e5',
                             point=point, iteration=rec['iteration'], checkpoint=rec['checkpoint'], **summary))
            write(root / 'analysis/results.json', rows)
            del model, ck; torch.cuda.empty_cache()
    rows.sort(key=lambda r: (r['point'], -r['alpha'], r['seed']))
    write(root / 'analysis/results.json', rows)
    pd.DataFrame(rows).to_csv(root / 'analysis/metrics.csv', index=False)
    lines = ['# LFWA impt2: 기존 dense 고정, 프루닝·재학습 seed42 대 seed100', '',
             '공통 LR1e-5, 보호0.5%, max, 제거 상한 OFF, 단계당5epoch,91iter. 추가8+1+1 없는 Validation 결과.',
             'Dense는 seed42에서 학습된 동일 체크포인트다. Dense 학습부터 시드를 바꾼 반복 실험과 구분한다.', '',
             '| 지점 | α | seed | iter | 전체 희소도 | Acc | BA | EO↓ | 근상수 속성 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"| {x['point']} | {x['alpha']} | {x['seed']} | {x['iteration']} | {x['model_sparsity']:.4%} | {x['accuracy']:.6f} | {x['ba']:.6f} | {x['eo']:.6f} | {x['near_constant_99pct_attributes']} |")
    (root / 'analysis/report.md').write_text('\n'.join(lines) + '\n')

def queue(root):
    m = verify(root); assert (root / 'preflight_complete.json').exists()
    for c in m['conditions']:
        out = Path(c['arguments']['lfwa_run_dir'])
        if (out / 'complete.json').exists():
            assert read(out / 'run.json')['arguments'] == c['arguments']
            assert (root / 'progress' / c['name'] / 'iteration_091.json').exists()
            continue
        assert not (out / 'run.json').exists(), 'Interrupted run requires RNG-aware recovery'
        write(root / 'status.json', dict(status='starting', condition=c['name'], time=now()))
        with (root / (c['name'] + '.log')).open('ab', buffering=0) as log:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root), '--train', c['name']],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(root / 'status.json', dict(status='evaluating_comparisons', time=now()))
    compare(root)
    write(root / 'complete.json', dict(status='complete', time=now(), alphas=ALPHAS, seed=100,
          report=str(root / 'analysis/report.md')))
    write(root / 'status.json', dict(status='complete', time=now()))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--preflight', action='store_true'); p.add_argument('--train')
    a = p.parse_args(); root = a.root.resolve(); root.mkdir(parents=True, exist_ok=True); os.chdir(REPO)
    if a.preflight: os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        if a.prepare: prepare(root)
        elif a.preflight: preflight(root)
        elif a.train: train_one(root, a.train)
        else: queue(root)
    except Exception as error:
        write(root / ('failure_' + a.train + '.json' if a.train else 'failure.json'), dict(error=repr(error), time=now()))
        write(root / 'status.json', dict(status='failed', error=repr(error), time=now()))
        raise
