"""Small real-data/GPU validation of impt=6; writes isolated reviewable artifacts."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from celeba_single import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    output = (args.output_dir or ROOT/'smoke_results'/f'impt6_single_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = {'status': 'running', 'output': str(output), 'torch': str(torch.__version__),
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'attribute': 'Smiling', 'samples_per_split': 4, 'runs': {}}

    def save():
        (output/'validation.json').write_text(json.dumps(evidence, indent=2))

    def run(name, extra=(), *, impt=6):
        command = [sys.executable, str(ROOT/'main_test.py'), '--dataset', 'CelebA',
                   '--network', 'mobilenetv2', '--prune_type', 'FairGRAPE', '--impt', str(impt),
                   '--target_attr', 'Smiling', '--sensitive_group', 'gender',
                   '--single_attr_smoke_samples', '4', '--batch', '4', '--stop_batch', '1',
                   '--prune_iter', '2', '--prune_rate', '.75', '--keep_per_iter', '.5',
                   '--retrain_iter', '1', '--single_attr_run_dir', str(output/name), *extra]
        print(f'START {name}', flush=True)
        started = datetime.datetime.now(datetime.timezone.utc)
        with (output/f'{name}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS='2'), check=True)
        directory = output/name
        complete = json.loads((directory/'complete.json').read_text())
        evidence['runs'][name] = dict(command=command, complete=complete,
            elapsed_seconds=(datetime.datetime.now(datetime.timezone.utc)-started).total_seconds())
        save()
        print(f'PASS {name}', flush=True)
        return directory, complete

    def check_pruning(directory, iteration, rate, previous=None):
        path = next((directory/'models'/'pruning').glob(f'BEST_MODEL_prune_{iteration:02d}_*.pt'))
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        masks = {name[:-5]: value for name, value in checkpoint['model_state'].items() if name.endswith('.mask')}
        assert len(masks) == 53
        assert {'features.0.0', 'features.18.0', 'classifier.1'} <= masks.keys()
        assert checkpoint['pruning_method'] == dict(impt_type=6, unit='weight', mode='original', scope='all')
        for name, mask in masks.items():
            assert bool(((mask == 0) | (mask == 1)).all()), name
            assert int(mask.sum()) == int(mask.numel() * (1-rate)), name
            assert torch.count_nonzero(checkpoint['model_state'][name+'.weight'][mask == 0]) == 0, name
            if previous is not None: assert bool((mask <= previous[name]).all()), name
        saved = torch.load(directory/'models'/'pruning'/f'impt6_Smiling_official_masks_{iteration:03d}.pt',
                           map_location='cpu', weights_only=True)
        assert all(torch.equal(saved[name+'.mask'], mask) for name, mask in masks.items())
        log = (directory/'fg_original'/f'iter{iteration:02d}.txt').read_text()
        assert 'impt_type=6' in log and 'prune_orig_ref' in log
        assert 'Smiling' in log and '(53 layers)' in log
        report = directory/'pruning'/f'iter_{iteration:02d}'/f'BEST_MODEL_SUMMARY_prune_{iteration:02d}_val.txt'
        text = report.read_text()
        assert 'Impt Type: 6' in text and 'FG Scope: all' in text
        metrics = json.loads((directory/'pruning'/f'iter_{iteration:02d}'/'val'/'metrics.json').read_text())
        assert metrics['impt_type'] == 6
        for key in ('accuracy','loss','ba','eo','eopp','dp'):
            assert metrics[key] is None or math.isfinite(metrics[key])
        return masks

    try:
        for name, extra in [('grl_on', ['--save_mask']), ('grl_off', ['--save_mask','--no_grl'])]:
            directory, complete = run(name, extra)
            first = check_pruning(directory, 1, .5)
            second = check_pruning(directory, 2, .75, first)
            final = torch.load(complete['checkpoint'], map_location='cpu', weights_only=True)
            assert all(torch.equal(final['model_state'][name+'.mask'], mask) for name, mask in second.items())
            assert all(torch.count_nonzero(final['model_state'][name+'.weight'][mask == 0]) == 0
                       for name, mask in second.items())
            assert final['experiment_metadata']['grl_enabled'] == (name == 'grl_on')
            assert (directory/'pruning'/'final'/'test'/'predictions.csv').exists()
            evidence['runs'][name]['checked_layers'] = 53
            evidence['runs'][name]['mask_budgets_and_monotonicity'] = True
            save()
        # A real impt=5 dense checkpoint must be usable by impt=6.
        five, _ = run('impt5_dense', ['--prune_iter','0','--defer_test'], impt=5)
        dense = next((five/'models'/'dense').glob('*.pt'))
        reloaded, _ = run('dense_reload', ['--checkpoint',str(dense),'--prune_iter','1',
            '--prune_rate','.99','--keep_per_iter','.01','--retrain_iter','0','--no_retrain',
            '--defer_test','--save_mask'])
        check_pruning(reloaded, 1, .99)
        assert not list(reloaded.rglob('Total_info*.txt'))
        assert not list(reloaded.rglob('FINAL_TEST_SUMMARY.txt'))
        source = Path(evidence['runs']['grl_on']['complete']['checkpoint'])
        source_hash = file_hash(source)
        scores, _ = run('score_only', ['--checkpoint',str(source),'--score_only','--prune_rate','.99'])
        assert not list(scores.rglob('*.pt'))
        assert not list(scores.rglob('predictions.csv'))
        assert (scores/'fg_original'/'iter03.txt').exists()
        evaluation, _ = run('evaluation_only', ['--checkpoint',str(source),'--single_attr_eval_only'])
        assert not list(evaluation.rglob('Total_info*.txt'))
        assert not list(evaluation.rglob('*.pt'))
        assert (evaluation/'pruning'/'final'/'test'/'predictions.csv').exists()
        assert file_hash(source) == source_hash
        evidence.update(status='passed', source_checkpoint_unchanged=True,
                        limitations=['Small execution validation only; no full-scale performance experiment.'])
        save()
        print(f'VALIDATION PASSED: {output}', flush=True)
    except Exception as error:
        evidence.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
