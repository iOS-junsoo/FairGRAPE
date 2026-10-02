"""The LFWA queue must not launch during another experiment or a brief idle gap."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_lfwa_experiment.py'
spec = importlib.util.spec_from_file_location('lfwa_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ResourceGateTests(unittest.TestCase):
    def setUp(self):
        self.free = dict(free_mib=32000, gpu_utilization_percent=0, gpu_processes=[],
                         cpu_cores=112, cpu_load=[12., 20., 80.], ram_available_mib=100000)
        self.args = SimpleNamespace(require_idle=True, minimum_free_mib=24576,
                                    idle_seconds=120, max_cpu_load_per_core=.5,
                                    resource_poll_seconds=30)

    def test_idle_resources_are_allowed(self):
        self.assertEqual(runner.idle_reasons(self.free, 24576, .5), [])

    def test_gpu_job_blocks_even_at_zero_utilization(self):
        state = dict(self.free, gpu_processes=[123])
        self.assertIn('another GPU compute process is present', runner.idle_reasons(state, 24576, .5))

    def test_recent_or_sustained_cpu_load_blocks(self):
        for load in ([57, 20, 20], [10, 57, 20]):
            with self.subTest(load=load):
                self.assertTrue(runner.idle_reasons(dict(self.free, cpu_load=load), 24576, .5))

    def test_memory_and_gpu_utilization_block(self):
        for change in ({'free_mib': 10000}, {'ram_available_mib': 10000}, {'gpu_utilization_percent': 11}):
            with self.subTest(change=change):
                self.assertTrue(runner.idle_reasons(dict(self.free, **change), 24576, .5))

    def run_gate(self, snapshots):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner, 'resource_snapshot', side_effect=snapshots), \
                 patch.object(runner.time, 'monotonic', side_effect=[30*i for i in range(len(snapshots))]), \
                 patch.object(runner.time, 'sleep') as sleep, patch('builtins.print'):
                runner.wait_for_gpu(Path(tmp), 2048, self.args)
            return json.loads((Path(tmp) / 'status.json').read_text()), sleep.call_count

    def test_idle_must_be_continuous_for_two_minutes(self):
        snapshots = [copy.deepcopy(self.free) for _ in range(8)]
        snapshots[2]['gpu_processes'] = [123]
        state, sleeps = self.run_gate(snapshots)
        self.assertEqual(sleeps, 7)
        self.assertEqual(state['idle_seconds'], 120)
        # Preflight's 2 GiB minimum must still honor the requested 24 GiB guard.
        self.assertEqual(state['required_mib'], 24576)

    def test_failed_query_does_not_count_as_idle(self):
        snapshots = [copy.deepcopy(self.free) for _ in range(7)]
        snapshots[1] = subprocess.TimeoutExpired('nvidia-smi', 10)
        state, sleeps = self.run_gate(snapshots)
        self.assertEqual(sleeps, 6)
        self.assertEqual(state['idle_seconds'], 120)

    def test_parse_gpu_and_cpu_counters_without_cuda(self):
        responses = [SimpleNamespace(stdout='10000, 0\n'), SimpleNamespace(stdout='1472759\n')]
        with patch.object(runner.subprocess, 'run', side_effect=responses), \
             patch.object(runner.os, 'sched_getaffinity', return_value=set(range(112))), \
             patch.object(runner.os, 'getloadavg', return_value=(274., 148., 123.)), \
             patch.object(Path, 'read_text', return_value='MemAvailable: 104857600 kB\n'):
            state = runner.resource_snapshot()
        self.assertEqual(state['gpu_processes'], [1472759])
        self.assertEqual(state['ram_available_mib'], 102400)
        self.assertEqual(state['cpu_cores'], 112)
        self.assertEqual(len(runner.idle_reasons(state, 24576, .5)), 3)


    def test_smoke_uses_requested_alpha_and_copied_dense(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'code'
            root.mkdir()
            sources = ['lfwa.py', 'lfwa_experiment.py', 'lfwa_report.py', 'main_test.py', 'prune.py',
                       'train_and_val.py', 'dataset.py', 'util.py', 'adversarial_debiasing.py', 'config.py',
                       'scripts/prepare_lfwa.py', 'scripts/run_lfwa_experiment.py', 'scripts/check_lfwa_runtime.py']
            for name in sources + ['csv/LFWA.csv', 'csv/LFWA.metadata.json', 'csv/LFWA.cell_counts.csv']:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('fixture')
            baseline = Path(tmp) / 'baseline'
            (baseline / 'dense').mkdir(parents=True)
            (baseline / 'dense' / 'complete.json').write_text('{"status":"complete"}')
            (baseline / 'dense' / 'final.pt').write_bytes(b'same dense weights')
            output = Path(tmp) / 'run'
            output.mkdir()
            (output / 'preflight.json').write_text('{"status":"passed","manifest_sha256":"test"}')
            args = ['runner', '--run-dir', str(output), '--stage', 'smoke', '--impt2-alpha', '0.5',
                    '--dense-from', str(baseline), '--require-idle']
            fake_lfwa = SimpleNamespace(load_frames=lambda: (None, {'manifest_sha256': 'test'}))
            with patch.object(runner, 'ROOT', root), patch.object(runner.sys, 'argv', args), \
                 patch.object(runner.os, 'chdir'), patch.dict(runner.sys.modules, {'lfwa': fake_lfwa}), \
                 patch.object(runner, 'execute') as execute, patch.object(runner, 'wait_for_gpu') as wait, \
                 patch.object(runner, 'verify_roundtrip'), patch('builtins.print'):
                runner.main()
            command = execute.call_args.args[0]
            self.assertEqual(command[command.index('--impt2_alpha') + 1], '0.5')
            self.assertEqual(command[command.index('--checkpoint') + 1], str(output / 'dense/final.pt'))
            self.assertIn('--no_init_train', command)
            self.assertEqual((output / 'dense/final.pt').read_bytes(), b'same dense weights')
            self.assertEqual(json.loads((output / 'pipeline.json').read_text())['alpha'], .5)
            self.assertEqual(wait.call_count, 2)


if __name__ == '__main__':
    unittest.main()
