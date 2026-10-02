"""Full CelebA four-method run using the original impt_type=2 training loop.

No sample limits: complete train/validation/test splits, original augmentation,
original train_model0, cuDNN disabled. The agreed layerwise selector is unchanged.
Completed stages and epochs can be resumed with --resume.
--no_retrain preserves the original meaning: skip final additional training,
NOT the five retraining epochs inside each pruning round. This runner already
omits final additional training by protocol; the flag records that explicitly.
"""
import argparse
import ast
import contextlib
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import sys
import threading
import time
from unittest.mock import patch

import celeba_layerwise_experiment as experiment
from celeba_layerwise_experiment import base, torch, np, pd, dataset, models, ROOT, METHODS, SLICES
import config
import prune
import train_and_val as legacy
from util import setseed
_report_spec = importlib.util.spec_from_file_location('celeba_official_report', ROOT/'scripts/report_celeba_legacy_eo.py')
official_report = importlib.util.module_from_spec(_report_spec)
_report_spec.loader.exec_module(official_report)

SOURCES = ('celeba_layerwise_full.py', 'celeba_layerwise_experiment.py',
           'celeba_fourway_smoke.py', 'dataset.py', 'prune.py', 'train_and_val.py',
           'util.py', 'config.py', 'adversarial_debiasing.py')
DEFINITIONS = experiment.DEFINITIONS
DEFINITIONS = DEFINITIONS.replace(
    '(square BEFORE averaging batches), using cached selection evaluation transforms.',
    '(square BEFORE averaging batches), using the ORIGINAL augmented train loader.\n'
    'All 162,079 training images, batch128, stop_batch=10000 (all 1267 batches).')
DEFINITIONS = DEFINITIONS.replace(
    'five COMPLETE epochs over the recorded retraining subset per round,',
    'original train_and_val.train_model0, five COMPLETE epochs over the FULL train split per round,')
DEFINITIONS = DEFINITIONS.replace(
    'The runner preserves original math but enables deterministic cuDNN kernels;\n'
    'legacy train_model0 disables cuDNN. FP32, no AMP or TF32. This backend difference\n'
    'is shared across all methods. Checkpoints after each epoch and each round.',
    'The ORIGINAL train_model0 runs unchanged: cuDNN DISABLED, original loss,\n'
    'optimizer steps, zero-parameter gradient/weight masks (including initially\n'
    'zero BN parameters), per-batch statistics and validation best-epoch rule.\n'
    'No AMP. Original util.setseed is used; backend defaults are recorded.\n'
    'Additional observers/loaders only record progress and checkpoint state.\n'
    'On resume, the original round-start zero-parameter masks are restored.\n'
    'Checkpoints include optimizer, RNG state and best model after each epoch.')
DEFINITIONS = DEFINITIONS.replace(
    'Small smoke samples are not a full-data accuracy/fairness experiment.',
    'FULL DATA: train/selection=162079, validation=20260, held-out test=20260.\n'
    'Selection signals also use the original TRAIN augmentation, batch64;\n'
    'evaluation uses the original evaluation transform, batch64. No rare-cell\n'
    'oversampling, no cached 520-image selection set, no subset or step limit.\n'
    'Shared round-start seeds make input transforms comparable across methods.\n'
    'Round-one importance/signals may be reused ONLY for the identical dense\n'
    'checkpoint, phase seed, full data and source fingerprint. Later rounds\n'
    'recompute on each method\'s own best retrained model.\n'
    'User overrides of legacy defaults: Male never enters task/GRL loss;\n'
    'GRL=False; features.1..17 only; agreed 10/30/10 keep-greedy; exactly\n'
    '3 rounds x 5 epochs, with no legacy extra training after round three.')


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(base.clean(obj), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


class Progress:
    """Heartbeat bypasses the original training function's stdout capture."""
    def __init__(self, out, interval=30):
        self.out, self.interval = Path(out), interval
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.state = dict(status='running', phase='initializing', pid=os.getpid())
        self.stream = sys.stdout
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def write(self):
        with self.lock:
            self.state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            atomic_json(self.out/'status.json', self.state)

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.write()
            with self.lock:
                s = self.state.copy()
            print('PROGRESS '+json.dumps(s, ensure_ascii=False), file=self.stream, flush=True)

    def __enter__(self):
        self.write()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.write()


class TrackedLoader:
    """Wrap an unchanged repository DataLoader; no truncation or reordering."""
    def __init__(self, loader, progress, phase, epoch_offset=0):
        self.loader, self.progress, self.phase = loader, progress, phase
        self.dataset, self.batch_size = loader.dataset, loader.batch_size
        self.epoch, self.history = epoch_offset, []

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.epoch += 1
        row = dict(epoch=self.epoch, samples=0, batches=0, first_batch_sha256=None)
        self.progress.update(phase=self.phase, epoch=self.epoch, samples=0,
                             total_samples=len(self.dataset), batches=0, total_batches=len(self))
        started = time.monotonic()
        for x, y in self.loader:
            if row['first_batch_sha256'] is None:
                row['first_batch_sha256'] = hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
            row['samples'] += len(y)
            row['batches'] += 1
            self.progress.update(samples=row['samples'], batches=row['batches'])
            yield x, y
        assert row['samples'] == len(self.dataset), 'A full pass must consume the entire split'
        row['seconds'] = time.monotonic()-started
        self.history.append(row)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(), imgaug=dataset.ia.random.get_global_rng().state)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    dataset.ia.random.get_global_rng().state = state['imgaug']


def load_own(path):
    # Only this run's locally generated caches/checkpoints are read here.
    return torch.load(path, map_location='cpu', weights_only=False)


def report(out, result, start, elapsed_before=0):
    result['elapsed_seconds'] = elapsed_before + time.monotonic()-start
    deferred = result['metadata']['arguments'].get('defer_contribution', False)
    rounds = result['metadata']['arguments']['rounds']
    definitions = DEFINITIONS.replace('Three rounds ONLY.', f'{rounds} rounds.').replace(
        '3 rounds x 5 epochs', f'{rounds} rounds x 5 epochs').replace('third round','final configured round')
    base.DEFINITIONS = definitions + (
        '\nACTIVATION-ONLY OVERRIDE / 사용자 요청: A_activation, B_activation 먼저 실행.\n'
        'No attribute-logit gradients in selection OR evaluation. Contribution metrics\n'
        'are DEFERRED, shown as N/A (n=0/39); this is distinct from empty label/group cells.\n'
        f'Existing full-data performance importance, {rounds} rounds and 5 epochs/round.\n'
        if deferred else '')
    temp = out/'.report'
    temp.mkdir(exist_ok=True)
    def comparison(records):
        evaluated = official_report.transform({'evaluations':base.clean(records)})['evaluations']
        return official_report.comparison(evaluated, True, final_round=rounds)
    with patch.object(base,'comparison',side_effect=comparison):
        base.write_report(temp, result)
    text = (temp/'results.txt').read_text(encoding='utf-8').replace(
        'CELEBA FOUR-WAY THREE-ROUND SMOKE / 39속성 채널 프루닝',
        ('CELEBA FULL-DATA ACTIVATION A/B / 예측 기여 계산 연기' if deferred else
         'CELEBA FULL-DATA FOUR-WAY / 원본 재학습 설정·전체 데이터'))
    text = text.replace('덴스 및 3회 재학습 후 비교', f'덴스 및 {rounds}회 재학습 후 비교')
    (temp/'results.txt').write_text(text, encoding='utf-8')
    for n in ('results.json', 'results.txt'):
        os.replace(temp/n, out/n)


def restore_nan_records(result):
    for ev in result['evaluations']:
        for row in ev['attributes']:
            for k, v in row.items():
                if v is None and k != 'attribute':
                    row[k] = float('nan')
    return result


def full_frames(args):
    attrs, annotation = base.validate_annotations(args.csv, args.original_attributes)
    frames = dataset.make_frame(str(args.csv), str(args.image_dir), seven_races=False)
    assert frames['train'].columns[2:41].tolist() == attrs
    names = {k: set(frames[k].image_name) for k in ('train', 'val', 'test')}
    assert names['train'].isdisjoint(names['val']) and names['train'].isdisjoint(names['test'])
    assert names['val'].isdisjoint(names['test'])
    return attrs, annotation, frames


def legacy_loaders(train_frame, val_frame, attrs, progress, offset=0):
    train, val = dataset.make_datasets(train_frame, val_frame, True, 64, attrs+['gender'])
    assert train.num_workers == val.num_workers == 8
    assert not train.drop_last and not val.drop_last
    return dict(train=TrackedLoader(train, progress, 'retrain', offset),
                test=TrackedLoader(val, progress, 'validation', offset))


def original_retrain(model, frames, attrs, args, method, rnd, out, progress, fingerprint,
                     on_epoch=None):
    """Execute the original train_model0, preserving its masks and best rule."""
    started = time.monotonic()
    original_stdout = sys.stdout
    seed = args.seed+1000*rnd
    setseed(seed)
    loaders = legacy_loaders(frames['train'], frames['val'], attrs, progress)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    epoch_path = out/f'{method}_round{rnd}_latest_epoch.pt'
    history, best_state, best_key, best_epoch = [], None, None, None
    fixed_masks = {k: v.detach().clone() for k, v in legacy.create_pruning_masks(model).items()}
    offset, elapsed_before = 0, 0.
    if epoch_path.exists():
        saved = load_own(epoch_path)
        assert saved['fingerprint'] == fingerprint
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        history, best_state = saved['history'], saved['best_model_state']
        best_key, best_epoch = tuple(saved['best_key']), saved['best_epoch']
        offset, elapsed_before = len(history), saved['elapsed_seconds']
        fixed_masks = {k: v.to(args.device) for k, v in saved['round_zero_masks'].items()}
        loaders['train'].epoch = loaders['test'].epoch = offset
        restore_rng(saved['rng_state'])
    updates = [0]
    def count_step(*_):
        assert not torch.backends.cudnn.enabled
        updates[0] += 1
        progress.update(optimizer_steps_this_invocation=updates[0])
    step_hook = optimizer.register_step_post_hook(count_step)
    native_dir = out/'original_training'/method/f'round{rnd}'
    native_dir.mkdir(parents=True, exist_ok=True)
    model_dir = native_dir/'models'
    model_dir.mkdir(exist_ok=True)
    def observer(current, local_epoch, metrics):
        nonlocal best_state, best_key, best_epoch
        epoch = offset+local_epoch
        train_row, val_row = loaders['train'].history[-1], loaders['test'].history[-1]
        assert train_row['samples'] == len(frames['train'])
        assert val_row['samples'] == len(frames['val'])
        assert updates[0] == local_epoch*len(loaders['train'])
        base.assert_masks(current)
        validation = dict(ACC=float(metrics['accuracy']), legacy_EO=float(metrics['eo']),
                          loss=float(metrics['loss']), BA=float(metrics['ba']), samples=val_row['samples'])
        key = (-validation['ACC'], validation['legacy_EO'], validation['loss'])
        if best_key is None or key < best_key:
            best_state, best_key, best_epoch = experiment.cpu_state(current), key, epoch
        row = dict(epoch=epoch, seed=seed, samples=train_row['samples'],
                   optimizer_steps=train_row['batches'], first_batch_sha256=train_row['first_batch_sha256'],
                   validation_first_batch_sha256=val_row['first_batch_sha256'], validation=validation,
                   seconds=train_row['seconds']+val_row['seconds'])
        history.append(row)
        saved = dict(fingerprint=fingerprint, model_state=experiment.cpu_state(current),
                     optimizer_state=optimizer.state_dict(), best_model_state=best_state,
                     best_key=best_key, best_epoch=best_epoch, history=history, attributes=attrs,
                     method=method, round=rnd, rng_state=rng_state(),
                     round_zero_masks={k: v.cpu() for k, v in fixed_masks.items()},
                     elapsed_seconds=elapsed_before+time.monotonic()-started)
        experiment.atomic_save(saved, epoch_path)
        atomic_json(native_dir/'epochs.json', history)
        print(f'FULL EPOCH COMPLETE {method} round={rnd} epoch={epoch}/5 '
              f'train={train_row["samples"]} steps={train_row["batches"]} '
              f'val={val_row["samples"]} ACC={validation["ACC"]:.7f}', file=progress.stream, flush=True)
        if on_epoch:
            on_epoch(history, best_epoch)
    cfgs = ['CelebA', 'layerwise_keep', 'classes', 1-.9**rnd, frames['test'], str(args.image_dir),
            78, 'mobilenetv2', attrs+['gender'], SLICES, 'gender', 0, args.seed]
    settings = dict(glo_use_grl=False, glo_epoch_observer=observer, glo_prune_iter=rnd-1,
                    glo_results_run_dir=str(native_dir), glo_model_run_dir=str(model_dir),
                    glo_n_groups=2, glo_impt_type=2, glo_dataset='CelebA', glo_seed=args.seed,
                    glo_keep_per_iter=.9, glo_imp_rate=None)
    try:
        if offset < 5:
            # Returning the recorded masks is identical to the original call on
            # a fresh round and prevents new masks being inferred after a resume.
            with patch.multiple(config, **settings), patch.object(
                    legacy, 'create_pruning_masks', return_value=fixed_masks):
                legacy.train_model0(model, loaders, torch.nn.CrossEntropyLoss(), optimizer,
                    num_epochs=5-offset, col_used_training=attrs, output_cols_each_task=SLICES,
                    cfgs=cfgs, use_debiasing=False)
        assert len(history) == 5
        model.load_state_dict(best_state, strict=True)
        model.eval()
        base.assert_masks(model)
        return dict(epochs=history, best_epoch=best_epoch, samples_per_epoch=len(frames['train']),
                    total_optimizer_steps=sum(h['optimizer_steps'] for h in history),
                    seconds=elapsed_before+time.monotonic()-started,
                    implementation='unchanged train_and_val.train_model0; original DataLoaders')
    finally:
        sys.stdout = original_stdout
        step_hook.remove()


def importance(model, frame, attrs, args, progress, log_path, rnd):
    setseed(args.seed+rnd)
    original_make = prune.make_datasets
    traced = []
    def make(*a, **kw):
        train, val = original_make(*a, **kw)
        loader = TrackedLoader(train, progress, 'performance_importance')
        traced.append(loader)
        return loader, val
    with patch.object(prune, 'make_datasets', side_effect=make), \
            open(log_path, 'w', buffering=1) as log, contextlib.redirect_stdout(log):
        scores, _ = prune.compute_importance(model, None, frame, new_img_dir=None,
            masked_grads=True, output_cols_each_task=SLICES, col_names=attrs+['gender'],
            stop_batch=10000, network='mobilenetv2', sensitive_group='gender',
            sensitive_classes=78, imp_batch_size=prune.IMPT_TYPE2_IMPORTANCE_BATCH_SIZE)
    assert traced[0].history[0]['samples'] == len(frame)
    return {n: v.detach().cpu().numpy() for n,v in base.aggregate_importance(scores).items()}, traced[0].history[0]


def signal_loader(frame, attrs, progress, phase, training=False):
    train, val = dataset.make_datasets(frame, frame, True, 64, attrs+['gender'])
    return TrackedLoader(train if training else val, progress, phase)


def cached_stage(path, fingerprint, operation):
    if path.exists():
        saved = load_own(path)
        assert saved['fingerprint'] == fingerprint, f'Cache source mismatch: {path}'
        return saved['data']
    data = operation()
    experiment.atomic_save(dict(fingerprint=fingerprint, data=data), path)
    return data


def put_evaluation(result, ev):
    key = (ev['method'], ev['round'], ev['stage'])
    previous = [i for i,e in enumerate(result['evaluations']) if (e['method'],e['round'],e['stage']) == key]
    if previous:
        result['evaluations'][previous[0]] = ev
    else:
        result['evaluations'].append(ev)


def evaluate_cached(model, frame, attrs, progress, method, rnd, stage, out, fingerprint, contribution=True):
    path = out/'cache'/f'eval_{method}_{rnd}_{stage}.pt'
    def run():
        loader = signal_loader(frame, attrs, progress, 'evaluate_'+stage)
        record = experiment.evaluate(model, loader, attrs, method, rnd, stage, contribution=contribution)
        assert loader.history[0]['samples'] == len(frame)
        record['evaluated_samples'] = len(frame)
        return record
    return cached_stage(path, fingerprint, run)


def reuse_initial_stages(source, out, result, fingerprint):
    """Import only immutable dense stages after checking their actual inputs/code."""
    source = Path(source).resolve()
    old = json.loads((source/'results.json').read_text())['metadata']
    new = result['metadata']
    for key in ('checkpoint_sha256', 'csv_sha256', 'annotation', 'split_sizes',
                'sample_sizes', 'analysis_layers', 'environment'):
        assert old[key] == new[key], f'Initial-stage mismatch: {key}'
    excluded = {'methods', 'output_dir', 'resume', 'preflight_only',
                'defer_contribution', 'reuse_initial_stages_from', 'no_retrain', 'continue_from'}
    assert {k:v for k,v in old['arguments'].items() if k not in excluded} == {
        k:v for k,v in new['arguments'].items() if k not in excluded}, 'Initial-stage argument mismatch'
    for key in ('selection', 'retrain', 'validation', 'evaluation'):
        assert old['sample_files'][key]['sha256'] == new['sample_files'][key]['sha256'], key
    changed = {'celeba_layerwise_full.py', 'celeba_layerwise_experiment.py'}
    for name in SOURCES:
        assert base.sha(source/'source'/name) == old['source_sha256'][name], f'Changed old source: {name}'
        if name not in changed:
            assert old['source_sha256'][name] == new['source_sha256'][name], f'Changed dependency: {name}'
    # These are the exact data loader, seed and importance operations behind the caches.
    names = {'full_frames', 'importance', 'signal_loader', 'legacy_loaders', 'TrackedLoader'}
    def selected_ast(path, wanted=None, excluded=None):
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if (wanted is None or getattr(n,'name',None) in wanted)
                 and getattr(n,'name',None) not in (excluded or set())]
        return [ast.dump(n,include_attributes=False) for n in nodes]
    assert selected_ast(source/'source/celeba_layerwise_full.py',names) == selected_ast(ROOT/'celeba_layerwise_full.py',names)
    assert selected_ast(source/'source/celeba_layerwise_experiment.py',excluded={'evaluate'}) == selected_ast(ROOT/'celeba_layerwise_experiment.py',excluded={'evaluate'})
    imported = []
    for name, suffix in [('eval_dense_0_dense.pt',''), ('dense_round1_performance.pt',':dense_round1')]:
        path = source/'cache'/name
        if not path.exists():
            continue
        saved = load_own(path)
        assert saved['fingerprint'] == old['fingerprint']+suffix, f'Invalid cache: {path}'
        data = copy.deepcopy(saved['data'])
        if not suffix:
            assert data['evaluated_samples'] == new['sample_sizes']['evaluation']
            data.pop('macro',None)
            for row in data['attributes']:
                for key in ('contribution_all','contribution_positive','contribution_negative'):
                    row[key] = float('nan')
            for layer in data['per_layer_signal_gaps'].values():
                for key in list(layer):
                    if key.startswith('contribution_'):
                        del layer[key]
            data['contribution_metrics_status'] = 'deferred_by_user'
        else:
            assert data['loader_trace']['samples'] == new['sample_sizes']['selection']
        data['reused_from'] = str(path)
        experiment.atomic_save(dict(fingerprint=fingerprint+suffix,data=data),out/'cache'/name)
        imported.append(dict(source=str(path),source_sha256=base.sha(path),
                             target=str(out/'cache'/name),original_seconds=data['seconds']))
    new['initial_stage_reuse'] = dict(source=str(source),source_fingerprint=old['fingerprint'],
        checks='same checkpoint, labels, ordered splits, arguments, backends and computation source',
        stages=imported,elapsed_time_note='Original stage durations retained; new wall time excludes reused computation')
    print('REUSED INITIAL STAGES', [Path(s['source']).name for s in imported],flush=True)


def continue_previous_run(source, out, result, fingerprint):
    source = Path(source).resolve()
    assert source != out.resolve()
    with (source/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((source/'results.json').read_text())
        assert previous['status'] in ('interrupted','complete'), 'Stop source run before continuation'
        old,new = previous['metadata'],result['metadata']
        assert new['arguments']['rounds'] > old['arguments']['rounds']
        excluded={'rounds','output_dir','resume','preflight_only','continue_from'}
        assert {k:v for k,v in old['arguments'].items() if k not in excluded} == {
            k:v for k,v in new['arguments'].items() if k not in excluded}, 'Continuation argument mismatch'
        for key in ('checkpoint_sha256','csv_sha256','annotation','split_sizes','sample_sizes','analysis_layers','environment'):
            assert old[key] == new[key], f'Continuation mismatch: {key}'
        for key in ('selection','retrain','validation','evaluation'):
            assert old['sample_files'][key]['sha256'] == new['sample_files'][key]['sha256'], key
        for name in SOURCES:
            assert base.sha(source/'source'/name) == old['source_sha256'][name], f'Changed old source: {name}'
            if name not in ('celeba_layerwise_full.py','celeba_layerwise_experiment.py'):
                assert old['source_sha256'][name] == new['source_sha256'][name], f'Changed compute dependency: {name}'
        old_exp=(source/'source/celeba_layerwise_experiment.py').read_text()
        extended_exp=old_exp.replace('if n < 10 or pool_n < remove_n or protected_n + pool_n > n:',
                                    'if n < 1 or pool_n < remove_n or protected_n + pool_n > n:')
        assert extended_exp == (ROOT/'celeba_layerwise_experiment.py').read_text(), 'Changed selection/evaluation math'
        # The only selector change permits floor(.1*N)=0 when a late layer reaches N=9.
        # It cannot change any completed source decision, whose layer sizes are >=10.
        assert all(d['live_before']>=10 for e in previous['rounds'] for d in e['layers'].values())
        protected={'Progress','TrackedLoader','rng_state','restore_rng','load_own','restore_nan_records',
                   'full_frames','legacy_loaders','original_retrain','importance','signal_loader',
                   'cached_stage','put_evaluation','evaluate_cached','fresh'}
        def compute_ast(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in protected}
        assert compute_ast(source/'source/celeba_layerwise_full.py') == compute_ast(ROOT/'celeba_layerwise_full.py'), 'Changed training/RNG/data/evaluation function'
        files=[]
        for path in sorted(list((source/'cache').glob('*.pt'))+list(source.glob('*.pt'))):
            saved=load_own(path)
            if 'fingerprint' in saved:
                original=saved['fingerprint']
                assert original==old['fingerprint'] or original.startswith(old['fingerprint']+':'), path
                saved['fingerprint']=fingerprint+original[len(old['fingerprint']):]
            target=out/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            experiment.atomic_save(saved,target)
            files.append(dict(source=str(path),source_sha256=base.sha(path),target=str(target)))
        # Logs remain available at their original path. Copy epoch summaries needed by status readers.
        for path in (source/'original_training').glob('*/round*/epochs.json'):
            target=out/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        result['rounds']=copy.deepcopy(previous['rounds'])
        result['evaluations']=copy.deepcopy(previous['evaluations'])
        restore_nan_records(result)
        completed_epochs=0
        for entry in result['rounds']:
            path=out/f'{entry["method"]}_round{entry["round"]}_latest_epoch.pt'
            if path.exists():
                state=load_own(path)
                assert state['method']==entry['method'] and state['round']==entry['round']
                entry['epochs_completed']=len(state['history'])
                entry['current_best_epoch']=state['best_epoch']
                completed_epochs+=len(state['history'])
        new['continuation']=dict(source=str(source),source_fingerprint=old['fingerprint'],
            original_rounds=old['arguments']['rounds'],requested_rounds=new['arguments']['rounds'],
            completed_epochs_preserved=completed_epochs,files=files,
            checks='identical checkpoint, ordered splits, arguments, backends, scoring/training/evaluation code; optimizer/RNG/best/round masks preserved',
            selector_extension='when a layer reaches fewer than 10 live channels, preserve existing floor(10%*N)=0 removal rule',
            previous_elapsed_seconds=previous.get('elapsed_seconds',0.))
        print('CONTINUED SAVED RUN',str(source),'epochs preserved=',completed_epochs,'target rounds=',new['arguments']['rounds'],flush=True)
        return previous.get('elapsed_seconds',0.)


def full_summary(out, result):
    text = ['FULL CELEBA / 원본 재학습 설정·전체 데이터',
            '학습·선택 162,079장 / 검증 20,260장 / 평가 20,260장; 표본 제한 없음.',
            f'{result["metadata"]["arguments"]["rounds"]}회 프루닝 × 회차당 5 epoch × {len(result["metadata"]["arguments"]["methods"])} 방법. batch64, Adam lr=1e-5, cuDNN OFF.',
            '재학습은 원본 train_and_val.train_model0 직접 호출. GRL/Male task loss 없음.',
            '상위 10% 보호, 상위 약70% 고정, 하위30% 후보에서 keep-greedy로 생존10% 제거.',
            ('활성화는 부호 유지 공간 평균. 예측 기여 계산은 사용자 요청으로 연기(N/A, n=0).'
             if result['metadata']['arguments'].get('defer_contribution') else
             '활성화는 부호 유지 공간 평균, 예측 기여는 양성 raw logit gradient × activation.'),
            '원본 성능 중요도 계산과 채널 집계 유지. 선택 자료=전체 train, 평가=별도 test.',
            '', official_report.comparison(official_report.transform(base.clean(result))['evaluations'],True,final_round=result['metadata']['arguments']['rounds']), '',
            'EO=(|TPR0-TPR1|+|FPR0-FPR1|)/2; EqualOpportunity=|TPR0-TPR1|; DP=|PPR0-PPR1|.',
            'N/A는 제외하고 모든 방법에 동일한 지표별 유효 속성 집합을 사용.',
            '39속성·그룹 전체 지표, 모든 중간 평가 및 악화 사례는 results.txt 참조.',
            f'활성 실행 시간: {result["elapsed_seconds"]:.1f}초 (중단 시간 제외).',
            '', '실행 명령:', 'cd '+str(ROOT), result['metadata']['command']]
    (out/'summary_ko.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt')
    p.add_argument('--csv', type=Path, default=ROOT/'csv/CelebA.csv')
    p.add_argument('--image-dir', type=Path, default=ROOT/'Images/CelebA/img_align_celeba')
    p.add_argument('--original-attributes', type=Path, default=ROOT.parent/'Balancing_Act/data/celeba/list_attr_celeba.txt')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--rounds', type=int, choices=[3,22], default=3)
    p.add_argument('--retrain-epochs', type=int, choices=[5], default=5)
    p.add_argument('--no_retrain', '--no-retrain', dest='no_retrain', action='store_true',
                   help='Original semantics: omit FINAL extra training; preserve 5 epochs in EACH pruning round. This runner always omits final extra training.')
    p.add_argument('--device', choices=['cuda:0'], default='cuda:0')
    p.add_argument('--defer-contribution', action='store_true',
                   help='Run activation methods without any attribute-logit gradients, including evaluation')
    p.add_argument('--reuse-initial-stages-from', type=Path,
                   help='Validated reuse of completed dense evaluation and first-round performance only')
    p.add_argument('--continue-from', type=Path, help='Continue an identical stopped run with a larger round count; verifies data, code, optimizer and RNG compatibility')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    assert len(set(args.methods)) == len(args.methods)
    if args.continue_from and args.reuse_initial_stages_from:
        raise ValueError('Choose continuation or initial-stage reuse, not both')
    if args.defer_contribution and any('contribution' in m for m in args.methods):
        raise ValueError('--defer-contribution allows activation methods only')
    if args.reuse_initial_stages_from and not args.defer_contribution:
        raise ValueError('Initial-stage migration is restricted to activation-only runs')
    args.batch_size, args.importance_batch_size, args.signal_batch_size = 64, 128, 64
    args.train_samples = args.val_samples = args.eval_samples = args.selection_samples = 0
    args.retrain_lr, args.workers = 1e-5, 8
    for key in ('checkpoint','csv','image_dir','original_attributes'):
        if not getattr(args,key).exists():
            raise FileNotFoundError(f'Missing --{key.replace("_","-")}: {getattr(args,key)}')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out/'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    started, elapsed_before = time.monotonic(), 0.
    source_hash = {n:base.sha(ROOT/n) for n in SOURCES}
    identity_args = {k:v for k,v in base.clean(vars(args)).items() if k not in ('resume','preflight_only')}
    identity = dict(arguments=identity_args, source_sha256=source_hash,
                    checkpoint_sha256=base.sha(args.checkpoint), csv_sha256=base.sha(args.csv))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.resume:
        result = restore_nan_records(json.loads((out/'results.json').read_text()))
        assert result['metadata']['fingerprint'] == fingerprint, 'Resume configuration/source/checkpoint mismatch'
        elapsed_before = result.get('elapsed_seconds',0.)
        result.pop('error', None)
        if result['status'] == 'complete':
            print('Already complete:', out/'results.txt')
            return
    else:
        result = dict(status='preparing', metadata=dict(identity, fingerprint=fingerprint,
            run_kind='full_original_training', arguments=base.clean(vars(args)), working_directory=str(ROOT),
            command=shlex.join([sys.executable,str(Path(__file__).resolve())]+(sys.argv[1:] if argv is None else argv)),
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), rounds=[], evaluations=[])
        snap = out/'source'
        snap.mkdir()
        for n in SOURCES:
            shutil.copy2(ROOT/n,snap/n)
    (out/'cache').mkdir(exist_ok=True)
    with Progress(out) as progress:
        progress.update(target_rounds=args.rounds, epochs_per_round=5, total_retraining_epochs=5*args.rounds*len(args.methods))
        try:
            setseed(args.seed)
            torch.backends.cudnn.enabled = False
            attrs, annotation, frames = full_frames(args)
            sizes = {k:len(frames[k]) for k in ('train','val','test')}
            assert sizes == {'train':162079,'val':20260,'test':20260}, f'Unexpected original split sizes: {sizes}'
            for label, split in [('selection','train'),('retrain','train'),('validation','val'),('evaluation','test')]:
                path = out/f'{label}_samples.csv'
                if not path.exists():
                    frames[split].to_csv(path,index=False)
            state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            assert state['classifier.1.weight'].shape == (78,1280)
            assert not any('mask' in n for n in state)
            def fresh():
                m = models.mobilenet_v2(weights=None)
                m.classifier[1] = torch.nn.Linear(1280,78)
                m.load_state_dict(state,strict=True)
                m.to(args.device)
                base.install_masks(m)
                return m.eval()
            model = fresh()
            result['metadata'].update(annotation=annotation, split_sizes=sizes, split_seed=42,
                sample_sizes=dict(selection=sizes['train'],retrain=sizes['train'],validation=sizes['val'],evaluation=sizes['test']),
                sample_files={k:dict(path=str(out/f'{k}_samples.csv'),sha256=base.sha(out/f'{k}_samples.csv')) for k in ('selection','retrain','validation','evaluation')},
                analysis_layers={n:m.out_channels for n,m in base.reference_layers(model).items()},
                checkpoint_format='plain dense state_dict, 78 logits; attribute order follows verified project CSV',
                training_batches_per_epoch=math.ceil(sizes['train']/64), validation_batches=math.ceil(sizes['val']/64),
                importance_batches=math.ceil(sizes['train']/128), selection_signal_batches=math.ceil(sizes['train']/64),
                environment=dict(python=sys.version, torch=torch.__version__, torchvision=base.torchvision.__version__,
                    numpy=np.__version__, pandas=pd.__version__, gpu=torch.cuda.get_device_name(),
                    cuda=torch.version.cuda, torch_threads=torch.get_num_threads(), cudnn_enabled=torch.backends.cudnn.enabled,
                    cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
                training_implementation='unchanged original train_model0, Adam, original data loaders/augmentation/masks',
                per_round_retraining_epochs=5, additional_retraining_epochs=0,
                total_retraining_epochs=5*args.rounds*len(args.methods),
                no_retrain_semantics='main_test.py original: skip only final additional training; keep 5 epochs in each pruning round',
                performance_implementation='unchanged original compute_importance with full augmented training loader',
                selection_transform='original TRAIN ImgAugTransform, Resize224, tensor, ImageNet normalize',
                evaluation_transform='original deterministic validation/test transform',
                contribution_metrics_status='deferred_by_user' if args.defer_contribution else 'computed',
                selection_attribute_logit_gradients=not args.defer_contribution,
                evaluation_attribute_logit_gradients=not args.defer_contribution,
                phase_seed_rule='importance: seed+round; selection: seed+100+round; retrain: seed+1000*round; original RNG stream within 5 epochs',
                resume='completed stages cached; interrupted stage/epoch restarts; optimizer/RNG/global best/round zero masks restored')
            for label, split in [('selection','train'),('validation','val'),('evaluation','test')]:
                labels = torch.tensor(frames[split][attrs+['gender']].to_numpy(dtype=np.int64))
                counts = base.membership(labels).sum(1)[2:].reshape(39,2,2)
                result['metadata'][f'{label}_cell_counts_y_by_Male'] = counts.tolist()
                if label == 'selection':
                    assert (counts>0).all(), 'Full train lacks a required B-selection cell'
            if args.continue_from and not args.resume:
                elapsed_before = continue_previous_run(args.continue_from,out,result,fingerprint)
            if args.reuse_initial_stages_from and not args.resume:
                reuse_initial_stages(args.reuse_initial_stages_from, out, result, fingerprint)
            result['status'] = 'preflight_complete' if args.preflight_only else 'running'
            report(out,result,started,elapsed_before)
            print('FULL DATA VERIFIED',sizes,'batch64, 2533 train batches/epoch, original train_model0, cuDNN OFF',flush=True)
            if args.preflight_only:
                progress.update(status='preflight_complete',phase='preflight_complete')
                return
            put_evaluation(result,evaluate_cached(model,frames['test'],attrs,progress,'dense',0,'dense',out,fingerprint,contribution=not args.defer_contribution))
            report(out,result,started,elapsed_before)
            del model
            for method in args.methods:
                removed = []
                for rnd in range(1,args.rounds+1):
                    round_start = time.monotonic()
                    progress.update(method=method,round=rnd,epoch=None)
                    model = fresh()
                    if rnd>1:
                        previous = load_own(out/f'{method}_round{rnd-1}.pt')
                        model.load_state_dict(previous['model_state'],strict=True)
                        removed = list(map(tuple,previous['removed_channels']))
                    cache_tag = 'dense_round1' if rnd==1 else f'{method}_round{rnd}'
                    tag_fingerprint = fingerprint+':'+cache_tag
                    def perf_op():
                        t=time.monotonic()
                        p,trace=importance(model,frames['train'],attrs,args,progress,out/f'{cache_tag}_importance.txt',rnd)
                        return dict(performance=p,loader_trace=trace,seconds=time.monotonic()-t)
                    per = cached_stage(out/'cache'/f'{cache_tag}_performance.pt',tag_fingerprint,perf_op)
                    def signals_op():
                        setseed(args.seed+100+rnd)
                        loader=signal_loader(frames['train'],attrs,progress,'selection_signals',training=True)
                        s=experiment.collect(model,loader,contribution=(not args.defer_contribution and (rnd==1 or 'contribution' in method)))
                        return dict(deltas=s['deltas'],cell_counts=s['cell_counts'],seconds=s['seconds'],loader_trace=loader.history[0])
                    signals = cached_stage(out/'cache'/f'{cache_tag}_signals.pt',tag_fingerprint,signals_op)
                    existing=next((e for e in result['rounds'] if e['method']==method and e['round']==rnd),None)
                    if existing is None:
                        perf=per['performance']
                        entry=dict(method=method,round=rnd,layers={},importance_seconds=per['seconds'],
                            importance_sha256=hashlib.sha256(b''.join(v.tobytes() for v in perf.values())).hexdigest(),
                            importance_loader=per['loader_trace'],signals_seconds=signals['seconds'],signal_loader=signals['loader_trace'],
                            shared_identical_dense_statistics=(rnd==1))
                        t=time.monotonic()
                        live=base.active_masks(model)
                        picks=[]
                        for n in base.reference_layers(model):
                            d=experiment.greedy_keep(perf[n],live[n].cpu().numpy(),signals['deltas'][n],method)
                            entry['layers'][n]=d
                            picks.extend((f'features.{base.block_number(n)}',c) for c in d['removed'])
                        entry.update(greedy_seconds=time.monotonic()-t,removed_this_round=picks,
                                     removed_cumulative=removed+picks)
                        result['rounds'].append(entry)
                    else:
                        entry=existing
                        picks=list(map(tuple,entry['removed_this_round']))
                    removed=list(map(tuple,entry['removed_cumulative']))
                    base.apply_channels(model,picks)
                    base.assert_masks(model)
                    entry.update(live_channels_after=sum(int(v.sum()) for v in base.active_masks(model).values()),
                        masked_weights=sum(int((m.mask==0).sum()) for m in base.weight_layers(model).values()))
                    selection_path=out/f'{method}_round{rnd}_selection.pt'
                    if not selection_path.exists():
                        experiment.atomic_save(dict(performance=per['performance'],signed_differences=signals['deltas']),selection_path)
                    print(f'FULL PRUNE {method} round={rnd}: removed={len(picks)} remaining={entry["live_channels_after"]}',flush=True)
                    report(out,result,started,elapsed_before)
                    put_evaluation(result,evaluate_cached(model,frames['test'],attrs,progress,method,rnd,'pre',out,tag_fingerprint+method+':pre',contribution=not args.defer_contribution))
                    report(out,result,started,elapsed_before)
                    final_path=out/f'{method}_round{rnd}.pt'
                    if final_path.exists():
                        final=load_own(final_path)
                        assert final['fingerprint']==tag_fingerprint+method
                        model.load_state_dict(final['model_state'],strict=True)
                        entry['retraining']=final['retraining']
                    else:
                        def checkpoint_report(history,best_epoch):
                            entry['epochs_completed']=len(history)
                            entry['current_best_epoch']=best_epoch
                            report(out,result,started,elapsed_before)
                        entry['retraining']=original_retrain(model,frames,attrs,args,method,rnd,out,progress,
                                                            tag_fingerprint+method,on_epoch=checkpoint_report)
                        experiment.atomic_save(dict(fingerprint=tag_fingerprint+method,model_state=experiment.cpu_state(model),
                            removed_channels=removed,attributes=attrs,method=method,round=rnd,
                            retraining=entry['retraining'],arguments=base.clean(vars(args))),final_path)
                    # Preserve the paired input-stream check across every method.
                    for other in result['rounds']:
                        if other['round']==rnd and other['method']!=method and 'retraining' in other:
                            assert [h['first_batch_sha256'] for h in other['retraining']['epochs']] == [h['first_batch_sha256'] for h in entry['retraining']['epochs']]
                    put_evaluation(result,evaluate_cached(model,frames['test'],attrs,progress,method,rnd,'post',out,tag_fingerprint+method+':post',contribution=not args.defer_contribution))
                    entry['round_seconds']=entry.get('round_seconds',0.)+time.monotonic()-round_start
                    report(out,result,started,elapsed_before)
                    del model
            assert base.sha(args.checkpoint)==identity['checkpoint_sha256']
            result['status']='complete'
            report(out,result,started,elapsed_before)
            full_summary(out,result)
            final_table=official_report.comparison(official_report.transform(base.clean(result))['evaluations'],True,final_round=result['metadata']['arguments']['rounds'])
            (out/'comparison.txt').write_text(final_table+'\n')
            print('\nFULL FINAL COMPARISON / original mean EO, EqualOpportunity, DemographicParity\n'+final_table,flush=True)
            if args.methods:
                progress.update(phase='auditing',status='verifying')
                torch.set_num_threads(2)
                spec=importlib.util.spec_from_file_location('full_audit',ROOT/'scripts/audit_celeba_layerwise_experiment.py')
                audit=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(audit)
                with patch.object(sys,'argv',['audit',str(out)]):
                    audit.main()
            progress.update(status='complete',phase='complete')
        except BaseException as exc:
            result.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
            progress.update(status=result['status'],error=repr(exc))
            report(out,result,started,elapsed_before)
            raise
    lock.close()


if __name__=='__main__':
    main()
