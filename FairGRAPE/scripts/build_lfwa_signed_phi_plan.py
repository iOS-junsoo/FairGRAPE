#!/usr/bin/env python3
"""CPU-only draft plan builder for a sign-only phi intervention at LFWA iter13.

Default mode verifies the archived unsigned controls only. --build-plan reads a
completed, independently reviewed alignment run and writes explicit selections;
it never loads a model, evaluates images, launches a runner, or trains anything.
"""
import argparse
import collections
import copy
import csv
import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_lfwa_channel_cap_diagnostic import read_plan, sha256, write_json
from run_lfwa_single_layer_cap_ablation import make_selections

SIGN_FLOOR = 1e-8
EPSILON = 1e-12
BLOCKS = tuple('features.' + str(i) for i in range(1, 18))


def phi_sign(q):
    """Near-zero q retains legacy phi; no confidence claim is attached."""
    if not math.isfinite(q):
        raise ValueError('Nonfinite gate derivative')
    return -1 if q < -SIGN_FLOOR else 1


def records_for(plan):
    with open(plan['source_channel_csv']) as handle:
        rows = list(csv.DictReader(handle))
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['block']].append(row)
    assert set(grouped) == set(BLOCKS)
    records = []
    for block, part in grouped.items():
        pm = max(float(row['perf_raw']) for row in part)
        fm = max(float(row['phi_raw']) for row in part)
        for row in part:
            p, f = float(row['perf_raw']), float(row['phi_raw'])
            assert math.isfinite(p) and math.isfinite(f) and p >= 0 and f >= 0
            pn, fn = p / (pm + EPSILON), f / (fm + EPSILON)
            records.append({**row, 'ch': int(row['ch']), 'weights': int(row['weights']),
                            'perf_norm_replayed': pn, 'phi_norm_replayed': fn,
                            'unsigned_score': plan['alpha'] * pn - (1 - plan['alpha']) * fn})
    keys = [(row['block'], row['ch']) for row in records]
    assert len(keys) == len(set(keys)) == 7136
    return records


def select(records, plan, cap, direction=False):
    active = collections.Counter(row['block'] for row in records if row['weights'] > 0)
    original = collections.Counter(row['block'] for row in records)
    floor = {block: math.ceil(original[block] * plan['layer_floor']) for block in BLOCKS}
    target = round(sum(row['weights'] for row in records) * (1 - plan['keep_per_iter']))
    candidates = [row for row in records if row['weights'] > 0 and row['status'] != 'protected']
    key = 'direction_score' if direction else 'unsigned_score'
    # Preserve original CSV/block/channel order for equal scores, as in replay.
    candidates.sort(key=lambda row: row[key])
    chosen, removed = [], 0
    counts = collections.Counter()
    for row in candidates:
        if removed >= target:
            break
        block = row['block']
        if active[block] - counts[block] - 1 < floor[block]:
            continue
        if cap and counts[block] >= math.floor(.1 * active[block]):
            continue
        chosen.append((block, row['ch']))
        counts[block] += 1
        removed += row['weights']
    if removed < target:
        raise ValueError('Permanent cap/floor cannot meet the nominal budget; do not relax either constraint')
    return {'selected_channels': chosen, 'expected_removed_weights': removed,
            'removed_channels_by_layer': dict(counts), 'nominal_removed_weights': target}


def verify_controls(plan, records):
    assert plan['iteration'] == 13 and plan['alpha'] == .7
    assert plan['normalization'] == 'max' and plan['protection_ratio'] == .005
    assert plan['layer_floor'] == .03 and plan['keep_per_iter'] == .975
    expected = {'baseline': (154, 32858), 'strict_cap10': (145, 33009)}
    # Independent existing helper and this builder must both reproduce controls.
    helper_caps, helper_base, _, _ = make_selections(plan, [BLOCKS])
    helper = {'baseline': helper_base, 'strict_cap10': helper_caps[0]}
    output = {}
    for name, cap in [('baseline', False), ('strict_cap10', True)]:
        selected = select(records, plan, cap)
        reference = next(arm for arm in plan['arms'] if arm['name'] == name)
        assert set(selected['selected_channels']) == set(reference['selected_channels']) == set(helper[name]['selected_channels'])
        assert selected['expected_removed_weights'] == reference['expected_removed_weights'] == helper[name]['expected_removed_weights']
        assert (len(selected['selected_channels']), selected['expected_removed_weights']) == expected[name]
        output[name] = {'selected_set_exact_match': True, 'selected_channels': expected[name][0],
                        'removed_weights': expected[name][1], 'nominal_removed_weights': selected['nominal_removed_weights']}
    # Scalar rule checks use synthetic q only, not a measured new policy.
    for q in (0., SIGN_FLOOR, -SIGN_FLOOR, 2 * SIGN_FLOOR):
        assert phi_sign(q) == 1
    assert phi_sign(-2 * SIGN_FLOOR) == -1
    p, f = .2, .3
    old = .7 * p - .3 * f
    assert math.isclose((.7 * p - .3 * f * phi_sign(-2 * SIGN_FLOOR)) - old, .6 * f)
    return output


def read_gradients(directory, plan, records):
    files = {name: directory / name for name in ('run.json', 'complete.json', 'global_gradient_audit.json', 'channel_gradients.csv')}
    if not all(path.is_file() for path in files.values()):
        raise ValueError('Alignment must be complete before any direction-on plan is built')
    meta, complete, audit = [json.loads(files[name].read_text()) for name in ('run.json', 'complete.json', 'global_gradient_audit.json')]
    assert complete['status'] == 'complete'
    assert complete['source_state_unchanged'] and complete['gates_one_and_zero_equivalence_passed']
    assert complete['source_validation_reproduced']
    assert meta['source_checkpoint_sha256'] == plan['source_checkpoint_sha256']
    assert meta['source_channel_sha256'] == plan['source_channel_csv_sha256']
    assert meta['sign_floor'] == SIGN_FLOOR
    assert meta['no_optimizer'] and meta['no_bn_update'] and meta['no_retraining']
    assert meta['train_preprocessing'] == 'existing deterministic evaluation transform; historical old phi used augmented train'
    assert audit['images'] == 5000 and audit['nonempty_valid_attributes'] == 39
    assert audit['max_second_pass_probability_error'] <= 1e-6
    assert audit['coefficient_batch_averaging'] is False
    assert audit['gradient_objective'] == 'full-train mean absolute probability group gaps conditional on label'
    with files['channel_gradients.csv'].open() as handle:
        q_rows = list(csv.DictReader(handle))
    q_map = {(row['block'], int(row['ch'])): row for row in q_rows}
    assert len(q_map) == len(q_rows) == len(records)
    assert set(q_map) == {(row['block'], row['ch']) for row in records}
    for row in records:
        measured = q_map[(row['block'], row['ch'])]
        assert int(measured['iter']) == plan['iteration']
        assert int(measured['weights']) == row['weights'] and measured['status'] == row['status']
        for raw in ('perf_raw', 'phi_raw'):
            assert math.isclose(float(measured[raw]), float(row[raw]), rel_tol=1e-12, abs_tol=0.)
        q = float(measured['q_soft_eo_gate'])
        factor = phi_sign(q)
        phi_effective = row['phi_norm_replayed'] * factor  # Never renormalize signed phi.
        score = plan['alpha'] * row['perf_norm_replayed'] - (1 - plan['alpha']) * phi_effective
        if factor == 1:
            assert score == row['unsigned_score']
        else:
            assert math.isclose(score - row['unsigned_score'], .6 * row['phi_norm_replayed'], rel_tol=1e-12, abs_tol=1e-15)
        row.update(q_soft_eo_gate=q, phi_sign_factor=factor, phi_effective=phi_effective,
                   direction_score=score, numerical_sign='negative' if q < -SIGN_FLOOR else 'positive' if q > SIGN_FLOOR else 'near_zero')
    return {'alignment_dir': str(directory), 'files_sha256': {name: sha256(path) for name, path in files.items()},
            'q_definition': 'd full-train softEO / d projection-input gate at gate=1',
            'q_training_images': 5000, 'sign_floor': SIGN_FLOOR,
            'old_phi_view': 'historical augmented train', 'q_view': 'deterministic train evaluation transform'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-plan', type=Path, default=ROOT / 'lfwa_diagnostics/alpha0p7_iter13_cap10_seed42_20260917_105048/plan.json')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--build-plan', action='store_true', help='Only invoke after root reviews the completed alignment evidence')
    parser.add_argument('--alignment-dir', type=Path)
    args = parser.parse_args()
    plan = read_plan(args.baseline_plan.resolve())
    records = records_for(plan)
    controls = verify_controls(plan, records)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    verification = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'controls': controls, 'scalar_sign_checks_passed': True,
                    'new_policy_constructed': False, 'model_loaded': False, 'gpu_used': False,
                    'baseline_plan': str(args.baseline_plan.resolve()),
                    'baseline_plan_sha256': sha256(args.baseline_plan), 'builder_sha256': sha256(__file__)}
    write_json(root / 'baseline_verification.json', verification)
    if not args.build_plan:
        print(json.dumps({'status': 'unsigned_controls_verified_only', **verification}, indent=2))
        return
    if args.alignment_dir is None:
        parser.error('--build-plan requires --alignment-dir')
    if (root / 'plan.json').exists():
        raise FileExistsError('Refusing to overwrite an existing policy plan')
    provenance = read_gradients(args.alignment_dir.resolve(), plan, records)
    proposed = copy.deepcopy(plan)
    proposed['arms'] = []
    for name, cap in [('direction_no_cap', False), ('direction_cap10', True)]:
        arm = select(records, plan, cap, direction=True)
        arm.update(name=name, reference_selection=False, strict_channel_cap_fraction=.1 if cap else None)
        chosen = set(arm['selected_channels'])
        arm['selected_numerical_sign_counts'] = dict(collections.Counter(row['numerical_sign'] for row in records if (row['block'], row['ch']) in chosen))
        arm['first_order_sum_minus_q_not_measured_delta'] = -sum(row['q_soft_eo_gate'] for row in records if (row['block'], row['ch']) in chosen)
        arm['selection_identical_to_existing_arms'] = [old['name'] for old in plan['arms'] if set(old['selected_channels']) == chosen]
        old = next(item for item in plan['arms'] if item['name'] == ('strict_cap10' if cap else 'baseline'))
        old_set = set(old['selected_channels'])
        arm['selected_channel_jaccard_with_unsigned_control'] = len(chosen & old_set) / len(chosen | old_set)
        proposed['arms'].append(arm)
    proposed['diagnostic'] = 'single_step_phi_sign_intervention_direction_by_cap_2x2'
    eligible = [row for row in records if row['weights'] > 0 and row['status'] != 'protected']
    proposed['direction_policy'] = {'eligible_q_sign_counts': dict(collections.Counter(row['numerical_sign'] for row in eligible)),
                                    'eligible_nonzero_phi_score_flips': sum(row['phi_sign_factor'] == -1 and row['phi_norm_replayed'] > 0 for row in eligible),
                                    'builder_sha256': sha256(__file__),
                                    'formula': 'phi_eff = old_phi_norm * sign(q), but abs(q)<=1e-8 retains old_phi_norm',
                                    'signed_phi_renormalized': False, 'provenance': provenance,
                                    'numerical_floor_is_confidence_threshold': False,
                                    'new_policy_executed': False, 'only_channel_plan_built': True,
                                    'controls': controls,
                                    'limitations': ['q is a local soft probability surrogate, not hard EO or full-removal certainty.',
                                                    'Legacy magnitude uses historical augmentation; q uses a deterministic train view.',
                                                    'Global budget is identical; indivisible channels cause realized count differences.',
                                                    'Direction changes layer allocation and can interact with cap and retraining.']}
    write_json(root / 'plan.json', proposed)
    with (root / 'channel_score_audit.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(json.dumps({'status': 'plan_built_only_no_execution', 'plan': str(root / 'plan.json')}, indent=2))


if __name__ == '__main__':
    main()
