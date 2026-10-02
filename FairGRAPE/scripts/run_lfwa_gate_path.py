#!/usr/bin/env python3
"""Post hoc gate-path diagnosis of four fixed LFWA channels; no training or policy."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
PREVIOUS = ROOT / 'lfwa_diagnostics/alpha0p7_iter13_fairness_alignment_20260917/run'
CASES = [('features.1', 4), ('features.2', 20)]
REPRESENTATIVES = [('features.7', 99), ('features.15', 820)]


def extended_metrics(frame, prediction):
    import numpy as np
    import lfwa
    from run_lfwa_fairness_alignment import soft_statistics
    truth = frame[lfwa.TARGETS].to_numpy()
    stats = soft_statistics(truth, frame.young.to_numpy(), prediction['probabilities'])
    p = np.asarray(prediction['probabilities'], dtype=np.float64)
    clip_epsilon = 1e-12
    clipped = np.clip(p, clip_epsilon, 1 - clip_epsilon)
    ce = -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)).mean()
    means, valid = stats['means'], stats['valid']
    soft_ba = np.concatenate([means[valid, 1, :], 1 - means[valid, 0, :]], axis=1).mean()
    return {'ce_mean_39_attributes': float(ce),
            'ce_definition': 'Mean binary 2-class cross entropy from saved/new float32 positive probabilities, averaged over images and 39 attributes.',
            'ce_clip_epsilon': clip_epsilon,
            'ce_clipped_probability_entries': int((p != clipped).sum()),
            'soft_ba': float(soft_ba),
            'soft_ba_definition': 'Mean across attributes of the four group/class probability-based recall rates.',
            'soft_ba_valid_attributes': int(valid.sum()),
            'soft_eo': stats['value']}


def prediction_from_archive(frame, directory):
    import numpy as np
    import pandas as pd
    import lfwa
    from run_lfwa_fairness_alignment import sha256
    path = directory / 'predictions.csv'
    columns = [name + '_probability' for name in lfwa.TARGETS]
    saved = pd.read_csv(path, dtype={column: np.float32 for column in columns})
    assert len(saved) == len(frame)
    for column in ('source_index', 'image_name', 'identity', 'split', 'young'):
        assert np.array_equal(saved[column].to_numpy(), frame[column].to_numpy()), column
    truth = saved[[name + '_true' for name in lfwa.TARGETS]].to_numpy()
    assert np.array_equal(truth, frame[lfwa.TARGETS].to_numpy()), 'Archived truth rows differ'
    summary = json.loads((directory / 'summary.json').read_text())
    result = {'probabilities': saved[columns].to_numpy(dtype=np.float32),
              'predictions': saved[[name + '_pred' for name in lfwa.TARGETS]].to_numpy(),
              'input_sha256': summary['input_sha256']}
    return result, {'directory': str(directory), 'predictions_sha256': sha256(path),
                    'summary_sha256': sha256(directory / 'summary.json'),
                    'input_sha256': summary['input_sha256'], 'manifest_rows_exact': True}


def metric_sanity():
    import numpy as np
    import pandas as pd
    import lfwa
    rows = 16
    truth = np.tile(np.array([0, 0, 1, 1] * 4)[:, None], (1, 39))
    group = np.array([0, 1, 0, 1] * 4)
    frame = pd.DataFrame(truth, columns=lfwa.TARGETS)
    frame['young'] = group
    probability = np.where(truth == 1, .8, .2)
    result = extended_metrics(frame, {'probabilities': probability})
    assert abs(result['ce_mean_39_attributes'] + np.log(.8)) < 1e-12
    assert abs(result['soft_ba'] - .8) < 1e-12
    assert abs(result['soft_eo']) < 1e-12
    return result


def run(args, root):
    import numpy as np
    import pandas as pd
    import torch
    import lfwa
    from util import setseed
    from run_lfwa_fairness_alignment import (
        CHECKPOINT, CHANNEL_CSV, EXPECTED_CHECKPOINT_SHA256,
        ProjectionGates, assert_frozen, dump, evaluate_arrays,
        memory_record, now, predict, sha256, state_hashes)
    torch.set_num_threads(2)
    setseed(42)
    assert sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256
    previous_run = json.loads((PREVIOUS / 'run.json').read_text())
    previous_complete = json.loads((PREVIOUS / 'complete.json').read_text())
    assert previous_complete['status'] == 'complete'
    assert previous_run['source_checkpoint_sha256'] == sha256(CHECKPOINT)
    assert previous_run['source_channel_sha256'] == sha256(CHANNEL_CSV)
    frames, metadata = lfwa.load_frames()
    assert metadata['manifest_sha256'] == previous_run['manifest_sha256']
    source, checkpoint = lfwa.load_checkpoint_model(CHECKPOINT, metadata, 'cpu')
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    frozen = state_hashes(source)
    assert frozen == previous_run['source_parameter_and_buffer_hashes']
    info = {'created_at': now(), 'python': sys.executable, 'torch': torch.__version__,
            'cuda_build': torch.version.cuda, 'previous_run': str(PREVIOUS),
            'source_checkpoint': str(CHECKPOINT), 'source_checkpoint_sha256': sha256(CHECKPOINT),
            'manifest_sha256': metadata['manifest_sha256'],
            'cases': [{'block': b, 'channel': c, 'gates': [1., .99, .9, .5, 0.]}
                      for b, c in CASES],
            'representatives': [{'block': b, 'channel': c, 'gates': [1., 0.]}
                                for b, c in REPRESENTATIVES],
            'case_selection': 'Post hoc mechanism tracing after viewing prior validation effects; four deliberately selected cases, not an unbiased channel sample.',
            'data': {'train': 5000, 'val': 1263},
            'g1_reused_from_prior_predictions': True, 'val_g0_reused_from_prior_predictions': True,
            'batch': args.batch, 'workers': args.workers, 'allocator_gib': args.allocator_gib,
            'cuda_cache_maxsize': os.environ['CUDA_CACHE_MAXSIZE'],
            'cuda_cache_path': os.environ['CUDA_CACHE_PATH'],
            'cpu_check': args.cpu_check, 'device': args.device,
            'no_optimizer': True, 'no_bn_update': True, 'no_retraining': True,
            'no_new_pruning_policy': True,
            'ce_note': 'For strict reuse comparability, all CE values use float32 model probabilities (old CSV round-tripped back to float32), accumulated in float64, with epsilon1e-12 clipping counts reported.',
            'source_parameter_and_buffer_hashes': frozen,
            'source_sha256': {name: sha256(ROOT / name) for name in
                             ['scripts/run_lfwa_gate_path.py', 'scripts/run_lfwa_fairness_alignment.py',
                              'lfwa.py', 'dataset.py', 'prune.py', 'util.py']}}
    dump(root / 'run.json', info)
    for name in info['source_sha256']:
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    dump(root / 'metric_sanity.json', metric_sanity())
    baselines, baseline_audits = {}, {}
    for split, directory in [('train', PREVIOUS / 'train_baseline'),
                             ('val', PREVIOUS / 'source_validation')]:
        prediction, audit = prediction_from_archive(frames[split], directory)
        summary = evaluate_arrays(frames[split], prediction, root / 'baseline' / split)
        summary.update(extended_metrics(frames[split], prediction))
        original = json.loads((directory / 'summary.json').read_text())
        for metric in ('accuracy', 'ba', 'eo', 'soft_eo'):
            assert abs(summary[metric] - original[metric]) < 1e-10, (split, metric)
        dump(root / 'baseline' / split / 'summary.json', summary)
        baselines[split] = summary
        baseline_audits[split] = audit
    dump(root / 'baseline_reuse_audit.json', baseline_audits)
    q_table = pd.read_csv(PREVIOUS / 'channel_gradients.csv')
    q_lookup = {(row.block, int(row.ch)): float(row.q_soft_eo_gate)
                for row in q_table.itertuples(index=False)}
    if args.cpu_check:
        assert_frozen(source, frozen)
        dump(root / 'complete.json', {'status': 'cpu_preflight_passed',
                                      'source_state_unchanged': True,
                                      'baseline_hard_and_soft_metrics_reproduced': True,
                                      'metric_formula_sanity_passed': True, 'no_gpu_used': True})
        print('GATE_PATH_CPU_PREFLIGHT_PASSED', root, flush=True)
        return
    assert torch.cuda.is_available()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.allocator_gib * 1024**3 / total, .95), 0)
    torch.cuda.reset_peak_memory_stats()
    source.to(args.device)
    gates = ProjectionGates(source)
    results = []
    metrics = ['accuracy', 'ba', 'eo', 'soft_eo', 'ce_mean_39_attributes', 'soft_ba',
               'constant_attributes', 'near_constant_99pct_attributes']
    cases = [(b, c, [1., .99, .9, .5, 0.]) for b, c in CASES]
    cases += [(b, c, [1., 0.]) for b, c in REPRESENTATIVES]
    for block, channel, values in cases:
        for value in values:
            for split in ('train', 'val'):
                directory = root / 'cases' / (block + '_ch' + str(channel)) / (
                    'gate_' + str(value).replace('.', 'p')) / split
                directory.mkdir(parents=True, exist_ok=True)
                reused = value == 1 or (split == 'val' and value == 0)
                dump(root / 'status.json', {'status': 'running', 'block': block, 'channel': channel,
                                           'gate': value, 'split': split, 'reusing': reused,
                                           'updated_at': now()})
                if value == 1:
                    summary = dict(baselines[split])
                    audit = baseline_audits[split]
                elif split == 'val' and value == 0:
                    archive = PREVIOUS / 'channel_validation' / (block + '_ch' + str(channel))
                    prediction, audit = prediction_from_archive(frames[split], archive)
                    assert prediction['input_sha256'] == baselines[split]['input_sha256']
                    summary = evaluate_arrays(frames[split], prediction, directory)
                    summary.update(extended_metrics(frames[split], prediction))
                    old = json.loads((archive / 'summary.json').read_text())
                    for metric in ('accuracy', 'ba', 'eo', 'soft_eo'):
                        assert abs(summary[metric] - old[metric]) < 1e-10
                else:
                    gates.set([(block, channel)], value)
                    prediction = predict(source, frames[split], args,
                                         block + '_' + str(channel) + '_g' + str(value) + '_' + split)
                    assert prediction['input_sha256'] == baselines[split]['input_sha256']
                    summary = evaluate_arrays(frames[split], prediction, directory)
                    summary.update(extended_metrics(frames[split], prediction))
                    audit = {'reused': False, 'input_sha256': prediction['input_sha256']}
                    gates.reset()
                    assert_frozen(source, frozen)
                dump(directory / 'summary.json', summary)
                if reused:
                    dump(directory / 'reused_from.json', audit)
                delta = {metric: summary[metric] - baselines[split][metric] for metric in metrics}
                result = {'block': block, 'channel': channel, 'gate': value, 'split': split,
                          'reused': reused, 'q_train_soft_eo_at_g1': q_lookup[(block, channel)],
                          'local_predicted_train_soft_eo_delta': -(1 - value) * q_lookup[(block, channel)],
                          'first_order_prediction_applies_only_to_train_and_small_gate_decrement': True,
                          'summary': summary, 'delta_from_g1': delta}
                results.append(result)
                dump(root / 'results.json', results)
                flat = [{'block': x['block'], 'channel': x['channel'], 'gate': x['gate'],
                         'split': x['split'], 'reused': x['reused'],
                         'q_train_soft_eo_at_g1': x['q_train_soft_eo_at_g1'],
                         **{k: x['summary'][k] for k in metrics},
                         **{'delta_' + k: v for k, v in x['delta_from_g1'].items()},
                         'ce_clipped_probability_entries': x['summary']['ce_clipped_probability_entries']}
                        for x in results]
                pd.DataFrame(flat).to_csv(root / 'results.csv', index=False)
                print('GATE_PATH_POINT', block, channel, value, split, json.dumps(delta), flush=True)
    gates.reset()
    assert_frozen(source, frozen)
    final = {'status': 'complete', 'completed_at': now(), 'points': len(results),
             'new_inference_points': sum(not x['reused'] for x in results),
             'source_state_unchanged': True, 'baseline_and_val_gate0_reused': True,
             'input_hashes_match_prior_run': True, 'memory': memory_record(torch),
             'post_hoc_cases_not_representative': True}
    dump(root / 'complete.json', final)
    dump(root / 'status.json', {'status': 'complete', 'updated_at': now()})
    gates.close()
    print('GATE_PATH_COMPLETE', json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--allocator-gib', type=float, default=3.5)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or not 1 <= args.allocator_gib < 4:
        parser.error('Invalid batch, workers or allocator limit')
    if args.cpu_check:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('CUDA_CACHE_MAXSIZE', '4294967296')
    os.environ.setdefault('CUDA_CACHE_PATH', '/tmp/fairgrape_lfwa_cuda_cache')
    Path(os.environ['CUDA_CACHE_PATH']).mkdir(parents=True, exist_ok=True)
    args.device = 'cpu' if args.cpu_check else 'cuda:0'
    root = args.run_dir.resolve()
    if (root / 'run.json').exists():
        raise FileExistsError('Refusing to overwrite ' + str(root))
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    try:
        run(args, root)
    except Exception as error:
        from run_lfwa_fairness_alignment import dump, now
        dump(root / 'status.json', {'status': 'failed', 'error': repr(error), 'updated_at': now()})
        raise


if __name__ == '__main__':
    main()

