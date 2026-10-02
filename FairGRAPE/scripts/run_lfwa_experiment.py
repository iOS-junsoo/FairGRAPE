"""Run the fixed seed-42 LFWA experiment, in separately recorded stages."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, data):
    partial = path.with_suffix('.part')
    partial.write_text(json.dumps(data, indent=2))
    os.replace(partial, path)


def resource_snapshot():
    """Read counters without importing torch or creating a CUDA context."""
    result = subprocess.run(
        ['nvidia-smi', '--id=0', '--query-gpu=memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    free, utilization = [int(value.strip()) for value in result.stdout.strip().split(',')]
    jobs = subprocess.run(
        ['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=10)
    memory = {line.split(':')[0]: int(line.split()[1])
              for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    return {'free_mib': free, 'gpu_utilization_percent': utilization,
            'gpu_processes': [int(line) for line in jobs.stdout.splitlines() if line.strip()],
            'cpu_cores': len(os.sched_getaffinity(0)), 'cpu_load': list(os.getloadavg()),
            'ram_available_mib': memory['MemAvailable'] // 1024}


def idle_reasons(state, minimum, max_cpu_load_per_core):
    reasons = []
    if state['free_mib'] < minimum:
        reasons.append('insufficient GPU memory')
    if state['gpu_processes']:
        reasons.append('another GPU compute process is present')
    if state['gpu_utilization_percent'] > 10:
        reasons.append('GPU utilization exceeds 10%')
    if max(state['cpu_load'][:2]) > state['cpu_cores'] * max_cpu_load_per_core:
        reasons.append('CPU 1/5-minute load exceeds the limit')
    if state['ram_available_mib'] < 32768:
        reasons.append('less than 32 GiB RAM is available')
    return reasons


def wait_for_gpu(root, minimum, args=None):
    if args is not None and args.require_idle:
        minimum = max(minimum, args.minimum_free_mib)
        idle_since = None
        while True:
            try:
                state = resource_snapshot()
                reasons = idle_reasons(state, minimum, args.max_cpu_load_per_core)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                state, reasons = {}, [f'resource check failed: {error}']
            now = time.monotonic()
            idle_since = None if reasons else (now if idle_since is None else idle_since)
            idle_elapsed = 0 if idle_since is None else now - idle_since
            state.update(status='waiting_for_resources', reasons=reasons, required_mib=minimum,
                         idle_seconds=round(idle_elapsed, 1), required_idle_seconds=args.idle_seconds,
                         max_cpu_load_per_core=args.max_cpu_load_per_core,
                         updated_at=datetime.datetime.now().isoformat())
            write_json(root / 'status.json', state)
            print('LFWA resource check:', json.dumps(state), flush=True)
            if not reasons and idle_elapsed >= args.idle_seconds:
                return
            time.sleep(args.resource_poll_seconds)
    while True:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        free = int(result.stdout.strip())
        if free >= minimum:
            return
        state = {'status': 'waiting_for_gpu', 'free_mib': free, 'required_mib': minimum,
                 'updated_at': datetime.datetime.now().isoformat()}
        write_json(root / 'status.json', state)
        print(f'LFWA waiting for GPU: {free}/{minimum} MiB free; existing jobs are left running.', flush=True)
        time.sleep(30)


def execute(command, log):
    print('RUN:', ' '.join(map(str, command)), flush=True)
    with log.open('w') as output:
        subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)


def verify_roundtrip(checkpoint, root):
    import torch
    import lfwa
    from dataset import make_datasets
    frames, metadata = lfwa.load_frames()
    model, data = lfwa.load_checkpoint_model(checkpoint, metadata, 'cpu')
    model.eval()
    _, loader = make_datasets(frames['val'].iloc[:4], frames['val'].iloc[:4], True, 4, lfwa.LABEL_COLUMNS)
    inputs, labels = next(iter(loader))
    torch.set_num_threads(2)
    with torch.no_grad():
        before = model(inputs)
    target = root / 'roundtrip.pt'
    lfwa.save_checkpoint(model, target, metadata, stage='roundtrip')
    restored, _ = lfwa.load_checkpoint_model(target, metadata, 'cpu')
    restored.eval()
    with torch.no_grad():
        after = restored(inputs)
    if not torch.equal(before, after):
        raise AssertionError('Saved/reloaded model predictions differ')
    if data['sparsity']['target_removed'] <= 0:
        raise AssertionError('Smoke test did not remove any target weights')
    write_json(root / 'smoke_verification.json', {
        'prediction_roundtrip_exact': True, 'mask_integrity': True, 'images': len(labels),
        'sparsity': data['sparsity']})


def evaluate(root):
    import torch
    import lfwa
    from lfwa_report import build_report
    frames, metadata = lfwa.load_frames()
    for name in ('dense', 'pruned'):
        model, _ = lfwa.load_checkpoint_model(root / name / 'final.pt', metadata, 'cuda:0')
        lfwa.evaluate_model(model, frames['test'], root / 'evaluation' / name, 256)
        del model
        torch.cuda.empty_cache()
    build_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--stage', choices=['all', 'dense', 'smoke', 'prune', 'evaluate'], default='all')
    parser.add_argument('--minimum-free-mib', type=int, default=24576,
                        help='Wait for sufficient GPU 0 memory before starting a training subprocess')
    parser.add_argument('--impt2-alpha', type=float, default=0.3)
    parser.add_argument('--dense-from', type=Path,
                        help='Reuse the completed dense/ directory of this LFWA run')
    parser.add_argument('--require-idle', action='store_true',
                        help='Wait for no GPU compute jobs and low CPU load before each stage, including preflight')
    parser.add_argument('--idle-seconds', type=float, default=120)
    parser.add_argument('--max-cpu-load-per-core', type=float, default=0.5)
    parser.add_argument('--resource-poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if not 0 <= args.impt2_alpha <= 1:
        parser.error('--impt2-alpha must be in [0, 1]')
    if args.idle_seconds < 0 or args.max_cpu_load_per_core <= 0 or args.resource_poll_seconds <= 0:
        parser.error('Invalid resource wait limits')
    os.environ.setdefault('CUDA_CACHE_MAXSIZE', '4294967296')
    os.environ.setdefault('CUDA_CACHE_PATH', str(Path(tempfile.gettempdir()) / 'fairgrape_lfwa_cuda_cache'))
    Path(os.environ['CUDA_CACHE_PATH']).mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    root = (args.run_dir or ROOT / 'lfwa_runs' / datetime.datetime.now().strftime('seed42_%Y%m%d_%H%M%S')).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = ['lfwa.py', 'lfwa_experiment.py', 'lfwa_report.py', 'main_test.py', 'prune.py',
               'train_and_val.py', 'dataset.py', 'util.py', 'adversarial_debiasing.py', 'config.py',
               'scripts/prepare_lfwa.py', 'scripts/run_lfwa_experiment.py', 'scripts/check_lfwa_runtime.py']
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    if (root / 'pipeline.json').exists():
        previous = json.loads((root / 'pipeline.json').read_text())
        if previous['source_sha256'] != hashes:
            raise ValueError('Source files changed since this run; use a new run directory')
        if previous.get('alpha', 0.3) != args.impt2_alpha:
            raise ValueError('Alpha differs from the recorded run; use a new run directory')
    else:
        write_json(root / 'pipeline.json', {'seed': 42, 'alpha': args.impt2_alpha, 'source_sha256': hashes,
                   'dense_from': str(args.dense_from.resolve()) if args.dense_from else None,
                   'resource_policy': {'require_idle': args.require_idle, 'minimum_free_mib': args.minimum_free_mib,
                                       'idle_seconds': args.idle_seconds,
                                       'max_cpu_load_per_core': args.max_cpu_load_per_core,
                                       'poll_seconds': args.resource_poll_seconds},
                   'created_at': datetime.datetime.now().isoformat(), 'python': sys.executable,
                   'order': ['dense', 'smoke', 'prune', 'evaluate'], 'prune_iterations': 91,
                   'pruning_retrain_epochs': 455, 'final_retrain_epochs': 10,
                   'cuda_cache_path': os.environ['CUDA_CACHE_PATH'],
                   'cuda_cache_maxsize': os.environ['CUDA_CACHE_MAXSIZE']})
        for name in sources:
            target = root / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
    for name in ['LFWA.csv', 'LFWA.metadata.json', 'LFWA.cell_counts.csv']:
        shutil.copy2(ROOT / 'csv' / name, root / name)
    if args.dense_from:
        source = args.dense_from.resolve() / 'dense'
        if not (source / 'complete.json').is_file():
            raise ValueError('The source Dense run is incomplete')
        checkpoint_hash = hashlib.sha256((source / 'final.pt').read_bytes()).hexdigest()
        destination = root / 'dense'
        if destination.exists():
            if hashlib.sha256((destination / 'final.pt').read_bytes()).hexdigest() != checkpoint_hash:
                raise ValueError('The destination Dense checkpoint differs from --dense-from')
        else:
            shutil.copytree(source, destination)
        write_json(destination / 'complete.json', {'status': 'complete', 'checkpoint': str(destination / 'final.pt')})
        write_json(destination / 'reuse.json', {'source': str(source), 'checkpoint_sha256': checkpoint_hash})
    # Protect frame validation and CPU checkpoint loading as well as GPU training.
    if args.require_idle:
        wait_for_gpu(root, args.minimum_free_mib, args)
    import lfwa
    _, metadata = lfwa.load_frames()
    stages = ['dense', 'smoke', 'prune', 'evaluate'] if args.stage == 'all' else [args.stage]
    common = [sys.executable, '-u', 'main_test.py', '--dataset', 'LFWA', '--network', 'mobilenetv2',
              '--loss_type', 'classes', '--sensitive_group', 'young', '--impt', '2', '--impt2_alpha', str(args.impt2_alpha),
              '--keep_per_iter', '0.975', '--prune_rate', '0.9', '--retrain_iter', '5', '--retrain_lr', '1e-5',
              '--batch', '256', '--seed', '42', '--stop_batch', '10000', '--no_grl', '--defer_test']
    try:
        if not (root / 'preflight.json').exists():
            wait_for_gpu(root, 2048, args)
            write_json(root / 'status.json', {'status': 'running', 'stage': 'preflight'})
            execute([sys.executable, '-u', 'scripts/check_lfwa_runtime.py', '--output', str(root / 'preflight.json')], root / 'preflight.log')
        preflight = json.loads((root / 'preflight.json').read_text())
        if preflight.get('status') != 'passed' or preflight.get('manifest_sha256') != metadata['manifest_sha256']:
            raise ValueError('Preflight did not pass for this LFWA manifest')
        for stage in stages:
            folder = root / ('pruned' if stage == 'prune' else stage)
            if (folder / 'complete.json').exists() and stage != 'evaluate':
                import torch
                lfwa.validate_checkpoint(torch.load(folder / 'final.pt', map_location='cpu'), metadata)
                if stage == 'smoke' and not (folder / 'smoke_verification.json').exists():
                    verify_roundtrip(folder / 'final.pt', folder)
                print(f'Already complete: {stage}', flush=True)
                continue
            if stage in ('smoke', 'prune') and not (root / 'dense' / 'complete.json').exists():
                raise FileNotFoundError('Complete the Dense stage first')
            if stage == 'prune' and not (root / 'smoke' / 'smoke_verification.json').exists():
                raise FileNotFoundError('Complete and verify the smoke stage first')
            if stage == 'evaluate' and not (root / 'pruned' / 'complete.json').exists():
                raise FileNotFoundError('Complete full pruning before Test evaluation')
            wait_for_gpu(root, args.minimum_free_mib, args)
            current_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
            if current_hashes != hashes:
                raise ValueError('Source changed during the experiment; refusing to mix implementations')
            write_json(root / 'status.json', {'status': 'running', 'stage': stage})
            if stage == 'evaluate':
                evaluate(root)
            else:
                command = common + ['--lfwa_run_dir', str(folder), '--prune_type', 'Full' if stage == 'dense' else 'FairGRAPE']
                if stage != 'dense':
                    command += ['--checkpoint', str(root / 'dense' / 'final.pt'), '--no_init_train']
                if stage == 'smoke':
                    command += ['--prune_iter', '1', '--no_retrain']
                execute(command, root / (stage + '.log'))
                if stage == 'smoke':
                    verify_roundtrip(folder / 'final.pt', folder)
            print(f'LFWA stage complete: {stage}', flush=True)
        write_json(root / 'status.json', {'status': 'complete' if args.stage == 'all' else 'stage_complete', 'stages': stages})
    except Exception as error:
        write_json(root / 'status.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
