#!/usr/bin/env python3
"""Read-only LFWA channel fairness-direction diagnosis, with external projection gates.

No optimizer, retraining, BN calibration, changed pruning policy, or alpha sweep.
CPU preflight must pass and be reviewed before the separate GPU run is launched.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / 'lfwa_runs/alpha0p7_seed42_no_final_20260916_143706'
CHECKPOINT = SOURCE_RUN / 'prune_012.pt'
CHANNEL_CSV = SOURCE_RUN / 'channel_pruning/alpha0.7_iter13_channels.csv'
EXPECTED_CHECKPOINT_SHA256 = '654e84357fd8ace65db1778cece5df0cee2733217853e284e3cd20e6852b0e80'
EXPECTED_CHANNEL_SHA256 = 'df437068250bb9362fca210fcee729da46cf4d47e9056560aad7327cc50fb3d9'
sys.path.insert(0, str(ROOT))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str((tuple(value.shape), str(value.dtype))).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_hashes(model):
    return {name: tensor_hash(value) for name, value in model.state_dict().items()}


def assert_frozen(model, expected):
    assert not model.training
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert state_hashes(model) == expected, 'Model weights, masks or BN buffers changed'


def soft_statistics(truth, sensitive, probabilities):
    import numpy as np
    truth, sensitive = np.asarray(truth), np.asarray(sensitive)
    p = np.asarray(probabilities, dtype=np.float64)
    assert p.shape == truth.shape and sensitive.shape == (len(p),)
    n_attr = p.shape[1]
    counts = np.zeros((n_attr, 2, 2), dtype=np.int64)
    means = np.zeros((n_attr, 2, 2), dtype=np.float64)
    for j in range(n_attr):
        for y in (0, 1):
            for a in (0, 1):
                mask = (truth[:, j] == y) & (sensitive == a)
                counts[j, y, a] = int(mask.sum())
                if counts[j, y, a]:
                    means[j, y, a] = p[mask, j].mean()
    valid = (counts > 0).all(axis=(1, 2))
    if not valid.any():
        raise ValueError('No attribute has all four label/group cells')
    gaps = means[:, :, 0] - means[:, :, 1]
    coefficient = np.zeros_like(p)
    for j in np.flatnonzero(valid):
        for y in (0, 1):
            sign = np.sign(gaps[j, y])
            for a in (0, 1):
                mask = (truth[:, j] == y) & (sensitive == a)
                coefficient[mask, j] = sign * (1 if a == 0 else -1) / (
                    2 * valid.sum() * counts[j, y, a])
    return {'value': float(np.abs(gaps[valid]).mean()),
            'valid': valid, 'counts': counts, 'means': means, 'gaps': gaps,
            'coefficient': coefficient}


def soft_json(stats):
    return {'soft_eo': stats['value'], 'valid_attributes': int(stats['valid'].sum()),
            'valid_attribute_mask': stats['valid'].tolist(),
            'cell_counts': stats['counts'].tolist(), 'cell_means': stats['means'].tolist(),
            'signed_group_gaps': stats['gaps'].tolist(),
            'zero_gap_subgradient': 0.0}


class ProjectionGates:
    """External parameters: never registered in model.state_dict()."""

    def __init__(self, model):
        import torch
        self.model = model
        self.gates, self.handles = {}, []
        modules = dict(model.named_modules())
        for block in range(1, 18):
            name = 'features.' + str(block)
            projection = name + ('.conv.1' if block == 1 else '.conv.2')
            layer = modules[projection]
            assert isinstance(layer, torch.nn.Conv2d)
            assert layer.kernel_size == (1, 1) and layer.groups == 1
            self.gates[name] = torch.nn.Parameter(
                torch.ones(layer.in_channels, dtype=layer.weight.dtype, device=layer.weight.device),
                requires_grad=True)
            self.handles.append(layer.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inputs):
            assert len(inputs) == 1 and inputs[0].shape[1] == len(self.gates[name])
            return (inputs[0] * self.gates[name].view(1, -1, 1, 1),)
        return hook

    def reset(self):
        import torch
        with torch.no_grad():
            for gate in self.gates.values():
                gate.fill_(1.)

    def set(self, selected, value):
        import torch
        self.reset()
        with torch.no_grad():
            for block, channel in selected:
                self.gates[block][channel] = value

    def parameters(self):
        return list(self.gates.values())

    def dict_gradients(self, gradients):
        return {name: gradient.detach().cpu().double().numpy()
                for name, gradient in zip(self.gates, gradients)}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def loader_for(frame, args):
    import torch
    import dataset
    import lfwa
    _, original = dataset.make_datasets(frame, frame, True, args.batch, lfwa.LABEL_COLUMNS)
    return torch.utils.data.DataLoader(original.dataset, batch_size=args.batch, shuffle=False,
                                      num_workers=args.workers, drop_last=False)


def check_labels(labels, frame, offset):
    import numpy as np
    import lfwa
    expected = frame.iloc[offset:offset + len(labels)][lfwa.LABEL_COLUMNS].to_numpy()
    assert np.array_equal(labels.detach().cpu().numpy(), expected), 'Manifest row order changed'


def forward_probabilities(model, images):
    logits = model(images).reshape(-1, 39, 2)
    return logits, logits.softmax(-1)[:, :, 1]


def predict(model, frame, args, progress_name):
    import numpy as np
    import torch
    arrays, hard = [], []
    digest = hashlib.sha256()
    seen = 0
    model.eval()
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        for batch_idx, (images, labels) in enumerate(loader_for(frame, args)):
            check_labels(labels, frame, seen)
            digest.update(images.numpy().tobytes())
            logits, p = forward_probabilities(model, images.to(args.device))
            arrays.append(p.cpu().numpy())
            hard.append(logits.argmax(-1).cpu().numpy())
            seen += len(images)
            if batch_idx % 40 == 0:
                print('FORWARD', progress_name, batch_idx, seen, '/', len(frame), flush=True)
    assert seen == len(frame)
    return {'probabilities': np.concatenate(arrays), 'predictions': np.concatenate(hard),
            'input_sha256': digest.hexdigest()}


def evaluate_arrays(frame, prediction, directory):
    import numpy as np
    import pandas as pd
    import lfwa
    truth, sensitive = frame[lfwa.TARGETS].to_numpy(), frame.young.to_numpy()
    per_attribute, summary = lfwa.binary_metrics(truth, prediction['predictions'], sensitive)
    stats = soft_statistics(truth, sensitive, prediction['probabilities'])
    summary['soft_eo'] = stats['value']
    fraction = prediction['predictions'].mean(axis=0)
    constant, near = (fraction == 0) | (fraction == 1), (fraction <= .01) | (fraction >= .99)
    summary.update({'constant_attributes': int(constant.sum()),
                    'near_constant_99pct_attributes': int(near.sum()),
                    'constant_attribute_names': [name for name, flag in zip(lfwa.TARGETS, constant) if flag],
                    'near_constant_attribute_names': [name for name, flag in zip(lfwa.TARGETS, near) if flag],
                    'ba_le_055_attributes': int((per_attribute.ba <= .55).sum()),
                    'input_sha256': prediction['input_sha256'],
                    'split': frame.split.iloc[0]})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = frame[['source_index', 'image_name', 'identity', 'split', 'young']].copy()
    for j, name in enumerate(lfwa.TARGETS):
        output[name + '_true'] = truth[:, j]
        output[name + '_pred'] = prediction['predictions'][:, j]
        output[name + '_probability'] = prediction['probabilities'][:, j]
    output.to_csv(directory / 'predictions.csv', index=False)
    per_attribute.to_csv(directory / 'attributes.csv', index=False, na_rep='N/A')
    dump(directory / 'summary.json', summary)
    dump(directory / 'soft_cells.json', soft_json(stats))
    return summary


def production_pruned(pristine, channels, device):
    import torch
    import prune
    original_device = prune.device
    prune.device = torch.device(device)
    try:
        candidate = copy.deepcopy(pristine).to(device)
        for name, parameter in candidate.named_parameters():
            parameter.requires_grad_(not name.endswith('.mask'))
        pruner = prune.Prunner(candidate, torch.nn.CrossEntropyLoss(), None)
        pruner.get_mask = lambda unused: prune._build_channel_mask_list(
            pruner.model, channels, torch.device(device))
        result = pruner.prune(None, False)
    finally:
        prune.device = original_device
    result.eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def equivalence_check(model, pristine, gates, images, channels):
    import torch
    audit = []
    gates.reset()
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        reference = copy.deepcopy(pristine).to(images.device).eval()
        expected = reference(images)
        actual = model(images)
        delta = float((actual - expected).abs().max())
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
        audit.append({'case': 'all_gates_one', 'max_abs_logit_error': delta,
                      'same_hard_predictions': bool(torch.equal(actual.reshape(-1, 39, 2).argmax(-1),
                                                               expected.reshape(-1, 39, 2).argmax(-1)))})
        del reference
        cases = [[channel] for channel in channels] + [channels]
        for selection in cases:
            gates.set(selection, 0.)
            actual = model(images)
            pruned = production_pruned(pristine, selection, images.device)
            expected = pruned(images)
            delta = float((actual - expected).abs().max())
            equal = bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4))
            item = {'case': 'gate_zero_vs_production_mask', 'channels': selection,
                    'max_abs_logit_error': delta, 'allclose_atol_1e5_rtol_1e4': equal,
                    'same_hard_predictions': bool(torch.equal(actual.reshape(-1, 39, 2).argmax(-1),
                                                             expected.reshape(-1, 39, 2).argmax(-1)))}
            audit.append(item)
            assert equal, item
            del pruned, actual, expected
    gates.reset()
    return audit


def toy_gradient_check():
    """Exact two-pass chain rule vs direct differentiable global EO in float64."""
    import numpy as np
    import torch
    generator = torch.Generator().manual_seed(42)
    count, attributes, channels = 53, 3, 7
    features = torch.randn(count, attributes, channels, dtype=torch.float64, generator=generator)
    offset = torch.randn(count, attributes, dtype=torch.float64, generator=generator)
    truth = np.array([[(i // (j + 1)) % 2 for j in range(attributes)] for i in range(count)])
    sensitive = np.array([(i // 4) % 2 for i in range(count)])
    gate = torch.ones(channels, dtype=torch.float64, requires_grad=True)
    probabilities = (offset + (features * gate).sum(-1)).sigmoid()
    stats = soft_statistics(truth, sensitive, probabilities.detach().numpy())
    direct_terms = []
    for j in range(attributes):
        for y in (0, 1):
            m0 = torch.tensor((truth[:, j] == y) & (sensitive == 0))
            m1 = torch.tensor((truth[:, j] == y) & (sensitive == 1))
            direct_terms.append((probabilities[m0, j].mean() - probabilities[m1, j].mean()).abs())
    direct = torch.stack(direct_terms).mean()
    expected, = torch.autograd.grad(direct, gate)
    accumulated = torch.zeros_like(gate)
    for start in range(0, count, 8):
        p = (offset[start:start + 8] + (features[start:start + 8] * gate).sum(-1)).sigmoid()
        local = (p * torch.tensor(stats['coefficient'][start:start + 8])).sum()
        gradient, = torch.autograd.grad(local, gate)
        accumulated += gradient
    error = float((expected - accumulated).abs().max())
    assert error < 1e-12, error
    return {'samples': count, 'batch': 8, 'last_batch': count % 8,
            'direct_vs_two_pass_max_abs_error': error}


def source_audit(model, checkpoint, frame):
    import numpy as np
    import prune
    assert checkpoint['iteration'] == 12
    assert len(frame) == 7136 and set(frame['iter']) == {13}
    assert int((frame.status == 'pruned_now').sum()) == 154
    for row in frame.itertuples(index=False):
        actual = prune._count_channel_weights(model, row.block, int(row.ch))
        assert actual == int(row.weights)
        assert (actual == 0) == (row.status == 'pruned_before')
    phi_norm = []
    for row in frame.itertuples(index=False):
        maximum = frame.loc[frame.block == row.block, 'phi_raw'].max()
        phi_norm.append(float(row.phi_raw / (maximum + 1e-12)))
    frame = frame.copy()
    frame['historical_phi_norm_reconstructed'] = phi_norm
    chosen = frame[frame.status == 'pruned_now']
    representatives = []
    for block in ('features.1', 'features.2', 'features.13', 'features.17'):
        rows = chosen[chosen.block == block].sort_values(
            ['historical_phi_norm_reconstructed', 'ch'], ascending=[False, True])
        assert len(rows)
        representatives.append((block, int(rows.iloc[0].ch)))
    return frame, representatives


def gradient_pass(model, gates, frame, reference_prediction, args, root):
    import numpy as np
    import torch
    import lfwa
    stats = soft_statistics(frame[lfwa.TARGETS].to_numpy(), frame.young.to_numpy(),
                            reference_prediction['probabilities'])
    accumulated = {name: np.zeros(len(gate), dtype=np.float64)
                   for name, gate in gates.gates.items()}
    seen = 0
    input_digest = hashlib.sha256()
    max_probability_error = 0.
    gates.reset()
    with torch.backends.cudnn.flags(enabled=False):
        for batch_idx, (images, labels) in enumerate(loader_for(frame, args)):
            check_labels(labels, frame, seen)
            input_digest.update(images.numpy().tobytes())
            _, p = forward_probabilities(model, images.to(args.device))
            reference = reference_prediction['probabilities'][seen:seen + len(images)]
            error = float(np.max(np.abs(p.detach().cpu().numpy() - reference)))
            max_probability_error = max(max_probability_error, error)
            coefficients = torch.tensor(stats['coefficient'][seen:seen + len(images)],
                                        dtype=p.dtype, device=p.device)
            scalar = (p * coefficients).sum()
            gradients = torch.autograd.grad(scalar, gates.parameters(), allow_unused=False)
            for name, gradient in gates.dict_gradients(gradients).items():
                assert np.isfinite(gradient).all(), 'Nonfinite gate gradient in ' + name
                accumulated[name] += gradient
            seen += len(images)
            del scalar, gradients, p, coefficients
            if batch_idx % 20 == 0:
                print('GLOBAL_GRADIENT', batch_idx, seen, '/', len(frame), flush=True)
                dump(root / 'status.json', {'status': 'running', 'stage': 'global_gradient',
                                           'images': seen, 'total': len(frame), 'updated_at': now()})
    assert seen == len(frame) == 5000
    assert input_digest.hexdigest() == reference_prediction['input_sha256']
    assert max_probability_error <= 1e-6, max_probability_error
    return accumulated, {'images': seen, 'input_sha256': input_digest.hexdigest(),
                         'max_second_pass_probability_error': max_probability_error,
                         'coefficient_counts': stats['counts'].tolist(),
                         'gradient_objective': 'full-train mean absolute probability group gaps conditional on label',
                         'global_soft_eo': stats['value'],
                         'nonempty_valid_attributes': int(stats['valid'].sum()),
                         'coefficient_batch_averaging': False,
                         'zero_gap_subgradient': 0.}


def choose_panel(channel_frame, gradients, args, root):
    import numpy as np
    import pandas as pd
    table = channel_frame.copy()
    table['q_soft_eo_gate'] = [float(gradients[row.block][int(row.ch)])
                              for row in table.itertuples(index=False)]
    table['numerical_sign_class'] = np.where(
        table.q_soft_eo_gate > args.sign_floor, 'positive',
        np.where(table.q_soft_eo_gate < -args.sign_floor, 'negative', 'near_zero'))
    table['sign_floor'] = args.sign_floor
    table.to_csv(root / 'channel_gradients.csv', index=False, na_rep='N/A')
    removed = table[table.status == 'pruned_now'].copy()
    panel = []
    for sign in ('positive', 'negative'):
        subset = removed[removed.numerical_sign_class == sign].copy()
        subset['block_number'] = subset.block.str.split('.').str[-1].astype(int)
        subset = subset.sort_values(['historical_phi_norm_reconstructed', 'block_number', 'ch'],
                                    ascending=[False, True, True]).head(8)
        for row in subset.itertuples(index=False):
            panel.append({'block': row.block, 'channel': int(row.ch),
                          'numerical_sign': sign, 'q_soft_eo_gate': float(row.q_soft_eo_gate),
                          'old_phi_raw': float(row.phi_raw),
                          'old_phi_normalized': float(row.historical_phi_norm_reconstructed),
                          'old_perf_raw': float(row.perf_raw),
                          'old_perf_normalized_rounded': float(row.perf_n),
                          'removed_weights': int(row.weights)})
    assert len(panel) <= 16
    manifest = {'created_at': now(),
                'rule': 'Among the 154 historical pruned_now channels, up to 8 per numerical q sign, descending historical normalized phi; stable block/channel tie break.',
                'selection_data': 'train gradients and historical train-derived scores only',
                'validation_counterfactual_evaluations_started': False,
                'sign_floor': args.sign_floor,
                'sign_warning': 'Numerical sign above a floor is not a confidence statement; train-cell sampling and surrogate uncertainty remain.',
                'old_phi_view': 'Historical augmented train batches; exact old augmentation RNG was not stored.',
                'new_gradient_view': 'Deterministic existing evaluation transforms on the same 5000 train images.',
                'historically_removed_sign_counts': removed.numerical_sign_class.value_counts().to_dict(),
                'panel': panel}
    dump(root / 'panel_manifest.json', manifest)
    pd.DataFrame(panel).to_csv(root / 'panel.csv', index=False)
    return panel, manifest


def batch_finite_difference(model, gates, batch, panel, device, epsilon):
    import numpy as np
    import torch
    images, labels = batch
    images = images.to(device)
    labels = labels.numpy()
    gates.reset()
    with torch.backends.cudnn.flags(enabled=False):
        _, probabilities = forward_probabilities(model, images)
        stats = soft_statistics(labels[:, :39], labels[:, -1], probabilities.detach().cpu().numpy())
        scalar = (probabilities * torch.tensor(stats['coefficient'],
                                              dtype=probabilities.dtype, device=device)).sum()
        gradients = gates.dict_gradients(torch.autograd.grad(scalar, gates.parameters()))
        del probabilities, scalar
        selected = []
        for sign in ('positive', 'negative'):
            selected.extend([item for item in panel if item['numerical_sign'] == sign][:2])
        output = []
        for item in selected:
            block, channel = item['block'], item['channel']
            q = float(gradients[block][channel])
            for step in (epsilon, epsilon / 2):
                gates.set([(block, channel)], 1 - step)
                with torch.no_grad():
                    _, p = forward_probabilities(model, images)
                actual = soft_statistics(labels[:, :39], labels[:, -1], p.cpu().numpy())['value']
                slope = (stats['value'] - actual) / step
                output.append({'block': block, 'channel': channel, 'epsilon': step,
                               'batch_soft_eo_before': stats['value'], 'batch_soft_eo_after': actual,
                               'batch_q': q, 'one_sided_difference_slope': slope,
                               'absolute_derivative_error': abs(slope - q),
                               'numerical_direction_agrees': bool(np.sign(slope) == np.sign(q)),
                               'near_zero_batch_q': abs(q) <= 1e-8,
                               'batch_size': len(labels), 'batch_valid_attributes': int(stats['valid'].sum()),
                               'global_q_not_used_for_this_comparison': True})
    gates.reset()
    return output


def memory_record(torch):
    if not torch.cuda.is_available():
        return {}
    return {'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
            'current_allocated_mib': torch.cuda.memory_allocated() / 1024**2,
            'current_reserved_mib': torch.cuda.memory_reserved() / 1024**2}


def run(args, root):
    import numpy as np
    import pandas as pd
    import torch
    import lfwa
    import dataset
    from util import setseed
    torch.set_num_threads(2)
    setseed(42)
    assert sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256
    assert sha256(CHANNEL_CSV) == EXPECTED_CHANNEL_SHA256
    frames, metadata = lfwa.load_frames()
    assert len(frames['train']) == 5000 and len(frames['val']) == 1263
    assert set(frames['train'].split) == {'train'} and set(frames['val'].split) == {'val'}
    source, checkpoint = lfwa.load_checkpoint_model(CHECKPOINT, metadata, 'cpu')
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    channel_frame, representatives = source_audit(source, checkpoint, pd.read_csv(CHANNEL_CSV))
    frozen = state_hashes(source)
    pristine = copy.deepcopy(source)
    info = {'created_at': now(), 'python': sys.executable, 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda_build': torch.version.cuda,
            'source_checkpoint': str(CHECKPOINT), 'source_checkpoint_sha256': sha256(CHECKPOINT),
            'source_channel_csv': str(CHANNEL_CSV), 'source_channel_sha256': sha256(CHANNEL_CSV),
            'source_validation': checkpoint['validation'], 'source_sparsity': checkpoint['sparsity'],
            'manifest_sha256': metadata['manifest_sha256'], 'batch': args.batch,
            'workers': args.workers, 'epsilon': args.epsilon, 'sign_floor': args.sign_floor,
            'device': args.device, 'preflight': args.cpu_check,
            'cuda_cache_maxsize': os.environ['CUDA_CACHE_MAXSIZE'],
            'cuda_cache_path': os.environ['CUDA_CACHE_PATH'],
            'allocator_limit_gib': args.allocator_gib,
            'no_optimizer': True, 'no_bn_update': True, 'no_retraining': True,
            'train_preprocessing': 'existing deterministic evaluation transform; historical old phi used augmented train',
            'source_parameter_and_buffer_hashes': frozen,
            'dlib_decoder': getattr(dataset, 'dlib', None) is not None,
            'core_source_sha256': {name: sha256(ROOT / name) for name in
                                   ['lfwa.py', 'dataset.py', 'prune.py', 'util.py', 'train_and_val.py']}}
    dump(root / 'run.json', info)
    for name in list(info['core_source_sha256']) + ['scripts/run_lfwa_fairness_alignment.py']:
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    dump(root / 'toy_gradient_check.json', toy_gradient_check())
    if not args.cpu_check:
        assert torch.cuda.is_available()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.allocator_gib * 1024**3 / total, .95), 0)
        torch.cuda.reset_peak_memory_stats()
    source.to(args.device)
    first_batch = next(iter(loader_for(frames['train'], args)))
    check_labels(first_batch[1], frames['train'], 0)
    gates = ProjectionGates(source)
    sample = first_batch[0][:2 if args.cpu_check else 8].to(args.device)
    equivalence = equivalence_check(source, pristine, gates, sample, representatives)
    dump(root / 'gate_equivalence.json', {'input_images': len(sample), 'cases': equivalence,
                                         'atol': 1e-5, 'rtol': 1e-4})
    assert_frozen(source, frozen)
    del sample, pristine
    if args.cpu_check:
        gates.close()
        dump(root / 'complete.json', {'status': 'cpu_preflight_passed', 'gate_cases': len(equivalence),
                                      'no_gpu_used': True, 'source_state_unchanged': True})
        print('CPU_PREFLIGHT_PASSED', root, flush=True)
        return
    dump(root / 'status.json', {'status': 'running', 'stage': 'source_validation', 'updated_at': now()})
    source_prediction = predict(source, frames['val'], args, 'source_validation')
    source_summary = evaluate_arrays(frames['val'], source_prediction, root / 'source_validation')
    historical = checkpoint['validation']
    deltas = {key: source_summary[key] - historical[key] for key in ('accuracy', 'ba', 'eo')}
    matched = all(abs(value) <= 1e-10 for value in deltas.values())
    dump(root / 'historical_validation_comparison.json', {'deltas': deltas, 'matched_at_1e10': matched,
                                                          'historical': historical, 'observed': source_summary})
    assert matched, 'Source validation must reproduce the saved checkpoint before diagnosis'
    assert_frozen(source, frozen)
    dump(root / 'status.json', {'status': 'running', 'stage': 'train_global_means', 'updated_at': now()})
    train_prediction = predict(source, frames['train'], args, 'train_global_means')
    train_summary = evaluate_arrays(frames['train'], train_prediction, root / 'train_baseline')
    gradients, gradient_audit = gradient_pass(source, gates, frames['train'], train_prediction, args, root)
    gradient_audit['memory'] = memory_record(torch)
    dump(root / 'global_gradient_audit.json', gradient_audit)
    assert_frozen(source, frozen)
    panel, panel_manifest = choose_panel(channel_frame, gradients, args, root)
    dump(root / 'batch_finite_difference.json', batch_finite_difference(
        source, gates, first_batch, panel, args.device, args.epsilon))
    assert_frozen(source, frozen)
    full_train_checks = []
    if args.full_train_epsilon:
        representatives_full = []
        for sign in ('positive', 'negative'):
            representatives_full.extend([item for item in panel if item['numerical_sign'] == sign][:1])
        for item in representatives_full:
            block, channel = item['block'], item['channel']
            gates.set([(block, channel)], 1 - args.epsilon)
            shifted = predict(source, frames['train'], args, 'epsilon_' + block + '_' + str(channel))
            assert shifted['input_sha256'] == train_prediction['input_sha256']
            actual = soft_statistics(frames['train'][lfwa.TARGETS].to_numpy(),
                                     frames['train'].young.to_numpy(), shifted['probabilities'])['value']
            predicted_delta = -args.epsilon * item['q_soft_eo_gate']
            entry = {**item, 'epsilon': args.epsilon, 'soft_eo_before': train_summary['soft_eo'],
                     'soft_eo_after': actual, 'actual_soft_eo_delta': actual - train_summary['soft_eo'],
                     'first_order_predicted_delta': predicted_delta,
                     'absolute_first_order_error': abs(actual - train_summary['soft_eo'] - predicted_delta),
                     'same_input_sha256': shifted['input_sha256']}
            full_train_checks.append(entry)
            dump(root / 'full_train_finite_difference.json', full_train_checks)
            gates.reset()
            assert_frozen(source, frozen)
    results = []
    panel_digest = sha256(root / 'panel_manifest.json')
    for index, item in enumerate(panel):
        dump(root / 'status.json', {'status': 'running', 'stage': 'single_channel_validation',
                                   'panel_index': index, 'panel_size': len(panel),
                                   'channel': item, 'updated_at': now()})
        block, channel = item['block'], item['channel']
        gates.set([(block, channel)], 0.)
        prediction = predict(source, frames['val'], args, 'gate0_' + block + '_' + str(channel))
        assert prediction['input_sha256'] == source_prediction['input_sha256']
        summary = evaluate_arrays(frames['val'], prediction,
                                  root / 'channel_validation' / (block + '_ch' + str(channel)))
        delta = {key: summary[key] - source_summary[key] for key in
                 ['accuracy', 'ba', 'eo', 'soft_eo', 'constant_attributes', 'near_constant_99pct_attributes']}
        entry = {**item, 'validation': summary, 'delta_gate0_minus_source': delta,
                 'practical_ba_guard_passes': delta['ba'] >= -.005
                     and delta['near_constant_99pct_attributes'] <= 0,
                 'hard_eo_unchanged': abs(delta['eo']) < 1e-12,
                 'local_soft_gradient_is_not_full_removal_or_hard_eo_guarantee': True}
        results.append(entry)
        dump(root / 'panel_results.json', results)
        flat = [{**{k: v for k, v in row.items() if k not in ('validation', 'delta_gate0_minus_source')},
                 **{'delta_' + key: value for key, value in row['delta_gate0_minus_source'].items()}}
                for row in results]
        pd.DataFrame(flat).to_csv(root / 'panel_results.csv', index=False)
        print('CHANNEL_DONE', index + 1, len(panel), block, channel, json.dumps(delta), flush=True)
        gates.reset()
        assert_frozen(source, frozen)
        assert sha256(root / 'panel_manifest.json') == panel_digest
    gates.reset()
    assert_frozen(source, frozen)
    final = {'status': 'complete', 'completed_at': now(), 'source_validation_reproduced': True,
             'source_state_unchanged': True, 'gates_one_and_zero_equivalence_passed': True,
             'panel_manifest_sha256': panel_digest, 'evaluated_channels': len(results),
             'historically_removed_sign_counts': panel_manifest['historically_removed_sign_counts'],
             'train_soft_eo': train_summary['soft_eo'], 'validation_baseline': source_summary,
             'full_train_finite_difference': full_train_checks, 'memory': memory_record(torch),
             'limitations': ['Old phi uses historical augmented train; q uses deterministic train view.',
                            'q signs are numerical directions, not statistically confirmed signs.',
                            'Soft probability EO differs from hard argmax EO.',
                            'A local gate derivative need not predict full channel removal.',
                            'Frozen BN immediate effects are not retrained-model effects.',
                            'Panel was selected by train scores; no test predictions were used.']}
    dump(root / 'complete.json', final)
    dump(root / 'status.json', {'status': 'complete', 'updated_at': now()})
    gates.close()
    print('ALIGNMENT_COMPLETE', json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sign-floor', type=float, default=1e-8)
    parser.add_argument('--epsilon', type=float, default=.01)
    parser.add_argument('--allocator-gib', type=float, default=3.5)
    parser.add_argument('--full-train-epsilon', action='store_true')
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.sign_floor < 0 or not 0 < args.epsilon < .1:
        parser.error('Invalid batch, worker count, sign floor or epsilon')
    if not 1 <= args.allocator_gib < 4:
        parser.error('Allocator cap must be at least 1 GiB and less than 4 GiB')
    if args.cpu_check:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('CUDA_CACHE_MAXSIZE', '4294967296')
    os.environ.setdefault('CUDA_CACHE_PATH', '/tmp/fairgrape_lfwa_cuda_cache')
    Path(os.environ['CUDA_CACHE_PATH']).mkdir(parents=True, exist_ok=True)
    args.device = 'cpu' if args.cpu_check else 'cuda:0'
    root = args.run_dir.resolve()
    if (root / 'run.json').exists() or (root / 'complete.json').exists():
        raise FileExistsError('Refusing to overwrite an existing diagnosis: ' + str(root))
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    started = time.monotonic()
    try:
        run(args, root)
    except Exception as error:
        dump(root / 'status.json', {'status': 'failed', 'error': repr(error), 'updated_at': now(),
                                    'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()

