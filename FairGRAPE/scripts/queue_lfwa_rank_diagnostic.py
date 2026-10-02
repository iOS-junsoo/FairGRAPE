#!/usr/bin/env python3
"""Launch the reviewed rank diagnostic once the paired step25 run releases CUDA."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'lfwa_diagnostics/alpha0p7_cap10_trajectory_seed42_20260917'
RANK = ROOT / 'lfwa_diagnostics/alpha0p7_iter13_rank_norm_seed42_20260917'
STATE = ROOT / 'lfwa_diagnostics/autonomous_state_20260917.json'
PYTHON = '/workspace/miniconda/envs/fairgrape/bin/python'
MAIN_PID = 3794173


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.queue-part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def status(value):
    dump(RANK / 'queue_status.json', {'updated_at': now(), **value})


def state_update(value):
    state = json.loads(STATE.read_text())
    state.update(value)
    state['updated_at'] = now()
    dump(STATE, state)


def main():
    plan = RANK / 'plan.json'
    assert json.loads(plan.read_text())['normalization'] == 'rank'
    assert (RANK / 'dry_run/dry_run.json').exists()
    inputs = [plan, ROOT / 'scripts/run_lfwa_rank_norm_diagnostic.py',
              ROOT / 'scripts/run_lfwa_channel_cap_diagnostic.py',
              ROOT / 'scripts/build_lfwa_signed_phi_plan.py',
              ROOT / 'scripts/run_lfwa_single_layer_cap_ablation.py', ROOT / 'prune.py']
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    dump(RANK / 'queue_input_hashes.json', hashes)
    if (RANK / 'run').exists() or (RANK / 'launcher.json').exists():
        raise FileExistsError('Rank execution already exists; no duplicate launch')
    status({'status': 'waiting_for_paired_step25', 'queue_pid': os.getpid(), 'main_pid': MAIN_PID})
    while True:
        if (MAIN / 'failure.json').exists():
            raise RuntimeError('Paired trajectory failed; stop queue for review')
        ready = (MAIN / 'complete.json').exists() and all(
            (MAIN / arm / 'iteration_025/complete.json').exists() for arm in ['baseline', 'cap10'])
        process = Path('/proc') / str(MAIN_PID) / 'cmdline'
        main_alive = process.exists() and b'run_lfwa_cap_trajectory.py' in process.read_bytes()
        if ready and not main_alive:
            memory = subprocess.check_output(
                ['nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            if int(memory.strip()) >= 26000:
                break
            status({'status': 'waiting_for_gpu_memory', 'free_mib': int(memory.strip()), 'queue_pid': os.getpid()})
        elif not ready and not main_alive:
            raise RuntimeError('Main process ended without completed step25 pair')
        time.sleep(15)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Reviewed input changed before launch: ' + path)
    argv = [PYTHON, '-u', str(ROOT / 'scripts/run_lfwa_rank_norm_diagnostic.py'),
            '--plan', str(plan), '--run-dir', str(RANK / 'run')]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_CACHE_MAXSIZE='4294967296',
                       CUDA_CACHE_PATH='/tmp/fairgrape_lfwa_cuda_cache', OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (RANK / 'training.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    record = {'pid': child.pid, 'argv': argv, 'started_at': now(), 'queue_pid': os.getpid(),
              'cuda_cache_maxsize': environment['CUDA_CACHE_MAXSIZE'], 'after_completed_pair': 25,
              'reviewed_input_hashes': hashes}
    dump(RANK / 'launcher.json', record)
    status({'status': 'running', **record})
    state_update({'current_experiment': RANK.name, 'current_pid': child.pid,
                  'current_comparison': 'One-step rank_shared vs saved paired max controls, same source12 and five-epoch training.',
                  'queued_rank_experiment': None, 'trajectory_paused_at_completed_pair': 25})
    while child.poll() is None:
        time.sleep(15)
    complete = child.returncode == 0 and (RANK / 'run/complete.json').exists()
    status({'status': 'complete' if complete else 'failed', 'returncode': child.returncode, 'pid': child.pid,
            'queue_pid': os.getpid()})
    if not complete:
        raise RuntimeError('Rank diagnostic did not finish successfully; preserve artifacts for review')
    state_update({'current_pid': None, 'last_completed_experiment': RANK.name})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status({'status': 'failed', 'error': repr(error), 'queue_pid': os.getpid()})
        raise
