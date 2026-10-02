#!/usr/bin/env python3
"""Run the approved alpha=.7 cap=.10 full experiment via unchanged main_test.

Adds isolated diagnostics, RNG snapshots and read-only cap/mask checks. Uses
main_test's continuous seed-42 stream, production scoring, training and selection.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import runpy
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / 'trained_model/unpruned/DogsCats_unpruned_species_bycolor_mobilenetv2_scratch_beta10.pt'
SOURCES = ['main_test.py', 'prune.py', 'channel_cap.py', 'config.py', 'train_and_val.py',
           'util.py', 'dataset.py', 'adversarial_debiasing.py', 'baseline_results.py',
           'scripts/run_dogscats_alpha07_cap10_full.py', 'scripts/run_dogscats_layer_cap_smoke.py']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(out):
    import torch
    state = torch.load(CHECKPOINT, map_location='cpu')
    for key in ('model_state', 'model_state_dict'):
        if key in state:
            state = state[key]
            break
    assert not any(name.endswith('.mask') for name in state), 'Expected a dense checkpoint'
    rows, total, minimum = [], 0, 0
    for i in range(1, 18):
        prefix = 'features.' + str(i)
        names = ([prefix+'.conv.0.0.weight', prefix+'.conv.1.weight'] if i == 1 else
                 [prefix+'.conv.0.0.weight', prefix+'.conv.1.0.weight', prefix+'.conv.2.weight'])
        width = int(state[names[0]].shape[0])
        weights = sum(state[name].numel() for name in names)
        assert weights % width == 0
        floor = max(9, math.ceil(.03 * width))
        rows.append(dict(layer=i, original_channels=width, minimum_channels=floor,
                         weights_per_channel=weights // width))
        total += weights
        minimum += weights // width * floor
    assert sum(r['original_channels'] for r in rows) == 7136
    report = dict(time=now(), checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT),
                  layers=rows, target_original_weights=total, structural_minimum_weights=minimum,
                  structural_maximum_sparsity=1-minimum/total,
                  note='Structural bound only; it does not guarantee 90% sparsity within 91 iterations.')
    assert report['structural_maximum_sparsity'] > .9
    write_json(out / 'preflight.json', report)
    print('CPU_PREFLIGHT_PASSED', json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(4)
    prepared = preflight(out)
    if args.preflight_only:
        return
    if (out / 'started.json').exists():
        raise RuntimeError('Refusing to overwrite or implicitly restart an existing full run')
    assert torch.cuda.is_available(), 'CUDA GPU required'
    torch.backends.cudnn.enabled = False
    import numpy as np
    import imgaug as ia
    import config
    import prune
    import train_and_val
    import util
    from channel_cap import channel_removal_limits
    from baseline_results import _mask_sparsity, dogscats_metrics
    from run_dogscats_layer_cap_smoke import check_masks, layout

    assert prune.IMPT2_NORM == 'max'
    assert prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER == .03
    assert prune.IMPT2_PROTECTION_RATIO == .005
    assert prune.IMPT2_LAYER_CAP_MULTIPLIER is None
    command = ['main_test.py', '--dataset', 'DogsCats', '--network', 'mobilenetv2',
               '--prune_type', 'FairGRAPE', '--loss_type', 'species', '--sensitive_group', 'color',
               '--skew_beta', '10', '--prune_rate', '0.9', '--prune_iter', '91',
               '--keep_per_iter', '0.975', '--impt', '2', '--impt2_alpha', '0.7',
               '--impt2_layer_max_remove_ratio', '0.10', '--batch', '256',
               '--retrain_iter', '5', '--retrain_lr', '1e-5', '--stop_batch', '10000',
               '--no_grl', '--no_retrain', '--no_pretrained', '--skip_readable_check',
               '--seed', '42', '--exp_idx', '20260920', '--checkpoint', str(CHECKPOINT), '--no_init_train']
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    manifest = dict(time=now(), alpha=.7, cap=.1, beta=10, iterations=91,
                    retrain_epochs=5, learning_rate=1e-5, seed_policy='main_test continuous seed 42',
                    command=command, checkpoint=str(CHECKPOINT), checkpoint_sha256=prepared['checkpoint_sha256'],
                    source_sha256=source_hashes, baseline='historical alpha=.7 continuous OFF; no new OFF run',
                    torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
                    cudnn_enabled=False, cpu_threads=4,
                    cuda_cache_path=os.environ.get('CUDA_CACHE_PATH'),
                    cuda_cache_maxsize=os.environ.get('CUDA_CACHE_MAXSIZE'))
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'started.json', dict(time=now(), pid=os.getpid()))
    for key, folder in [('glo_results_run_dir', 'history'), ('glo_model_run_dir', 'models'),
                        ('glo_channel_log_run_dir', 'channels')]:
        (out / folder).mkdir(exist_ok=True)
        setattr(config, key, str(out / folder))
    write_json(out / 'models' / 'run_info.json', manifest)
    state = dict(iteration=0, stage='initializing', started=time.monotonic())
    epochs, summaries, audits = [], [], {}

    def status(stage, **extra):
        state.update(stage=stage, **extra)
        record = {k: v for k, v in state.items() if k != 'started'}
        record.update(time=now(), elapsed_seconds=time.monotonic()-state['started'], pid=os.getpid())
        write_json(out / 'status.json', record)
        print('FULL_STATUS ' + json.dumps(record), flush=True)

    def rng(path):
        torch.save(dict(python=random.getstate(), numpy=np.random.get_state(),
                        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                        imgaug=copy.deepcopy(ia.random.get_global_rng().state)), path)

    score_function = prune.fairness_grad
    def score(*positional, **keywords):
        iteration = int(config.glo_prune_iter) + 1
        directory = out / 'iterations' / ('iteration_{:03d}'.format(iteration))
        directory.mkdir(parents=True, exist_ok=True)
        if iteration == 1:
            frame = positional[2] if len(positional) > 2 else keywords['test_csv']
            assert len(frame) == 8841
            frame.to_csv(out / 'train_frame.csv', index=False)
        state['iteration_started'] = time.monotonic()
        status('scoring_and_pruning', iteration=iteration, epoch=0)
        rng(directory / 'rng_before_score.pt')
        return score_function(*positional, **keywords)
    prune.fairness_grad = score

    original_logger = prune._save_channel_pruning_log
    def log_selection(**payload):
        iteration = int(payload['prune_iter']) + 1
        assert float(payload['alpha']) == .7 and payload['strict_cap_ratio'] == .1
        before = layout(payload['model'])
        original = {key: value['original'] for key, value in before.items()}
        active = {key: value['active'] for key, value in before.items()}
        floors = {key: math.ceil(value*.03) for key, value in original.items()}
        limits = channel_removal_limits(original, active, floors, .1)
        assert limits == payload['strict_channel_limits']
        selected = payload['selected_channels']
        assert len(selected) == len(set(selected))
        assert not set(selected).intersection(payload['protected_set'])
        assert payload['cap_relaxed_removed'] == 0
        rows = []
        for layer, value in before.items():
            count = sum(candidate[0] == layer for candidate in selected)
            assert count <= limits[layer] and active[layer]-count >= floors[layer]
            rows.append(dict(layer=layer, **value, removed=count,
                             active_after=active[layer]-count, limit=limits[layer]))
        audit = dict(iteration=iteration, cap=.1, layers=rows,
                     selected_channels=selected, remove_target=int(payload['remove_target']),
                     removed_weights=int(payload['accum_removed']),
                     deficit=max(0, int(payload['remove_target']-payload['accum_removed'])))
        directory = out / 'iterations' / ('iteration_{:03d}'.format(iteration))
        write_json(directory / 'selection.json', audit)
        scores = [dict(layer=layer, channel=int(channel), score=float(value[0]), weights=int(value[1]))
                  for (layer, channel), value in payload['score_by_channel_dict'].items()]
        write_json(directory / 'scores_full_precision.json', dict(scores=scores,
                   phi={k: v.detach().cpu().tolist() for k,v in payload['phi_by_layer'].items()},
                   perf={k: v.detach().cpu().tolist() for k,v in payload['perf_by_layer'].items()}))
        audits[iteration] = audit
        original_logger(**payload)
    prune._save_channel_pruning_log = log_selection

    def observe(model, epoch, metrics):
        iteration = int(config.glo_prune_iter) + 1
        record = dict(iteration=iteration, epoch=epoch,
                      **{key: float(metrics[key]) for key in ('accuracy', 'eo', 'loss', 'ba')})
        epochs.append(record)
        write_json(out / 'epochs.json', epochs)
        status('retraining', iteration=iteration, epoch=epoch,
               latest_epoch_accuracy=record['accuracy'], latest_epoch_eo=record['eo'])
    config.glo_epoch_observer = observe

    original_train = train_and_val.train
    def train(model, *positional, **keywords):
        iteration = int(config.glo_prune_iter) + 1
        directory = out / 'iterations' / ('iteration_{:03d}'.format(iteration))
        if iteration == 1:
            loaders = positional[1] if len(positional) > 1 else keywords['dataloaders']
            validation = loaders['test'].dataset.data_frame
            assert len(validation) == 1236
            assert list(validation.groupby(['species', 'color']).size()) == [309] * 4
            validation.to_csv(out / 'val_frame.csv', index=False)
        expected = audits[iteration]
        current = layout(model)
        assert all(current[r['layer']]['active'] == r['active_after'] for r in expected['layers'])
        masks = check_masks(model)
        rng(directory / 'rng_before_train.pt')
        status('retraining', iteration=iteration, epoch=0)
        result = original_train(model, *positional, **keywords)
        check_masks(result[0], masks)
        rng(directory / 'rng_after_train.pt')
        checkpoints = list((out / 'models').glob('BEST_MODEL_prune_{:02d}_*.pt'.format(iteration)))
        assert len(checkpoints) == 1
        checkpoint = torch.load(checkpoints[0], map_location='cpu')
        chosen = max([r for r in epochs if r['iteration'] == iteration],
                     key=lambda r: (r['accuracy'], -r['eo'], -r['loss']))
        assert abs(chosen['accuracy'] - checkpoint['best_acc']) < 1e-8
        assert abs(chosen['eo'] - checkpoint['best_eo']) < 1e-8
        summary = dict(iteration=iteration, accuracy=float(checkpoint['best_acc']),
                       eo=float(checkpoint['best_eo']), best_epoch=chosen['epoch'],
                       checkpoint=str(checkpoints[0]), checkpoint_sha256=sha256(checkpoints[0]),
                       sparsity=_mask_sparsity(result[0], 'blocks', None),
                       deficit=expected['deficit'],
                       elapsed_seconds=time.monotonic()-state['iteration_started'])
        summaries.append(summary)
        write_json(directory / 'complete.json', summary)
        write_json(out / 'results.json', summaries)
        status('iteration_complete', iteration=iteration, accuracy=summary['accuracy'], eo=summary['eo'])
        return result
    train_and_val.train = train

    original_save_model, original_save_output = util.save_model, util.save_output
    def save_model(model, cfgs):
        value = original_save_model(model, cfgs)
        torch.save(dict(model_state_dict=model.state_dict(), manifest=manifest), out / 'final_model.pt')
        return value
    def save_output(model, cfgs, csv_savedir='fair_dfs', save_file=True):
        status('final_test_evaluation', iteration=91)
        frame = original_save_output(model, cfgs, str(out / 'final_predictions'), save_file)
        write_json(out / 'final_test_metrics.json', dogscats_metrics(frame))
        return frame
    util.save_model, util.save_output = save_model, save_output

    status('initializing')
    try:
        sys.argv = command
        runpy.run_path(str(ROOT / 'main_test.py'), run_name='__main__')
        assert len(summaries) == 91 and len(epochs) == 455
        assert sha256(CHECKPOINT) == prepared['checkpoint_sha256']
        assert all(sha256(ROOT / name) == digest for name, digest in source_hashes.items())
        status('complete', iteration=91, completed_iterations=len(summaries))
    except BaseException as error:
        status('failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
