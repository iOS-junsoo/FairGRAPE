#!/usr/bin/env python3
"""Build an explicit CPU rank-policy plan; never evaluate or train a model.

Uses the exact core _rank_percentile function on active raw CSV scores in float32.
Original gamma, floor, alpha and global budget are retained. Stable tie order is
computed only as an audit; the selected plan uses original default CPU argsort.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import ast
import collections
import csv
import math
from pathlib import Path

import torch


SOURCE_CSV = "lfwa_runs/alpha0p7_seed42_no_final_20260916_143706/channel_pruning/alpha0.7_iter13_channels.csv"


def load_rank_rows(root):
    """Return (all historical rows with rank fields, provenance metadata).

    Default fields are P_rank_default, phi_rank_default, S_rank_default.
    Stable sensitivity fields replace the suffix with _stable. Raw inputs and
    arithmetic use float32, matching the original torch percentile function.
    """
    root = Path(root)
    source = root / SOURCE_CSV
    prune = root / "prune.py"
    node = next(
        n for n in ast.parse(prune.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_rank_percentile"
    )
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(prune) + "::_rank_percentile", "exec"), scope)
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("ch", "weights"):
            row[key] = int(row[key])
        for key in ("perf_raw", "phi_raw"):
            row[key] = float(row[key])
        row["logged_score"] = float(row["score"])
    assert rows == sorted(rows, key=lambda row: (int(row["block"].split(".")[1]), row["ch"]))
    blocks = sorted({row["block"] for row in rows}, key=lambda block: int(block.split(".")[1]))
    ties = {}
    for block in blocks:
        current = [row for row in rows if row["block"] == block and row["weights"] > 0]
        ties[block] = {}
        for column, label in (("perf_raw", "P"), ("phi_raw", "phi")):
            values = torch.tensor([row[column] for row in current], dtype=torch.float32)
            normalized = scope["_rank_percentile"](values)
            stable_order = sorted(range(len(values)), key=lambda index: (float(values[index]), current[index]["ch"]))
            stable_ranks = torch.empty(len(values), dtype=torch.long)
            for rank, index in enumerate(stable_order):
                stable_ranks[index] = rank
            stable_normalized = stable_ranks.to(values.dtype) / (len(values) - 1) if len(values) > 1 else torch.full_like(values, 0.5)
            counts = collections.Counter(values.tolist())
            ties[block][column] = {
                "tied_value_groups": sum(count > 1 for count in counts.values()),
                "channels_in_ties": sum(count for count in counts.values() if count > 1),
                "zero_channels": counts.get(0.0, 0),
            }
            for index, row in enumerate(current):
                row[label + "_rank_default"] = float(normalized[index])
                row[label + "_rank_stable"] = float(stable_normalized[index])
                row[label + "_descending_rank_competition"] = 1 + int((values > values[index]).sum())
                row[label + "_tie_count"] = int((values == values[index]).sum())
                row[label + "_descending_position_default"] = len(values) - int(round(float(normalized[index]) * (len(values) - 1)))
                row[label + "_descending_position_stable"] = len(values) - int(stable_ranks[index])
        for mode in ("default", "stable"):
            performance = torch.tensor([row["P_rank_" + mode] for row in current], dtype=torch.float32)
            fairness = torch.tensor([row["phi_rank_" + mode] for row in current], dtype=torch.float32)
            scores = 0.7 * performance - (1.0 - 0.7) * fairness
            for index, row in enumerate(current):
                row["S_rank_" + mode] = float(scores[index])
    return rows, {
        "rank_function_source": ast.get_source_segment(prune.read_text(), node),
        "raw_value_ties_by_layer": ties,
        "dtype": "torch.float32",
        "torch_version": torch.__version__,
        "default_ties": "original torch.argsort default unstable CPU order",
        "stable_ties": "Python sort by float32 value then channel index",
    }


def select_rows(rows, score_key="S_rank_default", cap=False):
    """Return (audit dict, set of (block, channel)) for original budget/floor.

    cap=False means no cap; cap=True means strict floor(0.10*active_before)
    channel cap with no relaxation. Gamma protection is frozen to CSV status.
    """
    assert cap in (False, True)
    blocks = sorted({row["block"] for row in rows}, key=lambda block: int(block.split(".")[1]))
    original = {block: sum(row["block"] == block for row in rows) for block in blocks}
    active = {block: sum(row["block"] == block and row["weights"] > 0 for row in rows) for block in blocks}
    minimum = {block: math.ceil(original[block] * 0.03) for block in blocks}
    protected = {(row["block"], row["ch"]) for row in rows if row["status"] == "protected"}
    total_active_weights = sum(row["weights"] for row in rows)
    target = round(total_active_weights * (1.0 - 0.975))
    assert target == 32791
    candidates = sorted(
        [row for row in rows if row["weights"] > 0 and (row["block"], row["ch"]) not in protected],
        key=lambda row: row[score_key],
    )
    selected = []
    removed = 0
    counts = collections.Counter()
    weights = collections.Counter()
    floors = collections.Counter()
    caps = collections.Counter()
    for row in candidates:
        if removed >= target:
            break
        block = row["block"]
        if active[block] - counts[block] - 1 < minimum[block]:
            floors[block] += 1
            continue
        if cap and counts[block] >= math.floor(0.1 * active[block]):
            caps[block] += 1
            continue
        selected.append((block, row["ch"]))
        removed += row["weights"]
        counts[block] += 1
        weights[block] += row["weights"]
    audit = {
        "score_key": score_key,
        "strict_cap10": bool(cap),
        "selected_channels": len(selected),
        "removed_weights": removed,
        "nominal_removed_weights": target,
        "budget_overshoot": removed - target,
        "total_active_weights": total_active_weights,
        "selected_channel_ids": [block + ":" + str(channel) for block, channel in selected],
        "per_layer": {
            block: {
                "original_channels": original[block],
                "active_before": active[block],
                "protected": sum(key[0] == block for key in protected),
                "minimum_keep": minimum[block],
                "strict_cap10_limit": math.floor(0.1 * active[block]) if cap else None,
                "selected_channels": counts[block],
                "removed_weights": weights[block],
                "active_after": active[block] - counts[block],
                "floor_skips": floors[block],
                "cap_skips": caps[block],
            }
            for block in blocks
        },
    }
    assert not (set(selected) & protected)
    assert all(layer["active_after"] >= layer["minimum_keep"] for layer in audit["per_layer"].values())
    if cap:
        assert all(layer["selected_channels"] <= layer["strict_cap10_limit"] for layer in audit["per_layer"].values())
    return audit, set(selected)


import argparse
import copy
import datetime
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_lfwa_channel_cap_diagnostic import read_plan, sha256, write_json
from build_lfwa_signed_phi_plan import records_for, verify_controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-plan', type=Path, default=ROOT / 'lfwa_diagnostics/alpha0p7_iter13_cap10_seed42_20260917_105048/plan.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / 'plan.json').exists():
        raise FileExistsError('Refusing to overwrite an existing policy plan')
    plan = read_plan(args.baseline_plan.resolve())
    max_rows = records_for(plan)
    controls = verify_controls(plan, max_rows)
    assert Path(plan['source_channel_csv']).resolve() == (ROOT / SOURCE_CSV).resolve()
    assert plan['protection_ratio'] == .005 and plan['layer_floor'] == .03
    assert plan['alpha'] == .7 and plan['keep_per_iter'] == .975
    rows, rank_meta = load_rank_rows(ROOT)
    max_map = {(r['block'], r['ch']): r for r in max_rows}
    for row in rows:
        old = max_map[(row['block'], row['ch'])]
        row['P_max_replayed'] = old['perf_norm_replayed']
        row['phi_max_replayed'] = old['phi_norm_replayed']
        row['S_max_replayed'] = old['unsigned_score']
    assert len(rows) == len(max_rows) == 7136
    gamma_audit = {}
    for block in sorted({r['block'] for r in rows}, key=lambda b: int(b.split('.')[1])):
        active = [r for r in rows if r['block'] == block and r['weights'] > 0]
        count = math.ceil(len(active) * plan['protection_ratio'])
        # Match core P ranking float32, stable channel-index tie break.
        ordered = sorted(active, key=lambda r: (-float(torch.tensor(r['perf_raw'], dtype=torch.float32)), r['ch']))
        rebuilt = {r['ch'] for r in ordered[:count]}
        logged = {r['ch'] for r in active if r['status'] == 'protected'}
        assert rebuilt == logged
        gamma_audit[block] = {'active':len(active), 'ceil_protected_count':count, 'same_protected_set':True}
    proposed = copy.deepcopy(plan)
    proposed['normalization'] = 'rank'
    proposed['diagnostic'] = 'single_step_archived_raw_scores_rank_normalization_by_cap'
    proposed['score_replay_matches_historical_selection'] = False
    proposed['max_control_replay_matches_historical_selection'] = True
    proposed['arms'] = []
    audit = {}
    selected_by_arm = {}
    for name, cap in [('rank_no_cap', False), ('rank_cap10', True)]:
        current, selected = select_rows(rows, 'S_rank_default', cap)
        stable, stable_set = select_rows(rows, 'S_rank_stable', cap)
        assert current['removed_weights'] >= current['nominal_removed_weights'] == 32791
        # Parse in greedy order, preserving that order in the fixed selection plan.
        chosen = [(value.split(':')[0], int(value.split(':')[1])) for value in current['selected_channel_ids']]
        assert set(chosen) == selected and len(chosen) == len(selected)
        old = next(a for a in plan['arms'] if a['name'] == ('strict_cap10' if cap else 'baseline'))
        old_set = set(old['selected_channels'])
        arm = {'name': name, 'reference_selection':False,
               'selected_channels':chosen, 'expected_removed_weights':current['removed_weights'],
               'nominal_removed_weights':current['nominal_removed_weights'],
               'strict_channel_cap_fraction':.1 if cap else None,
               'removed_channels_by_layer': {b:v['selected_channels'] for b,v in current['per_layer'].items()},
               'selection_identical_to_existing_max_arms':[a['name'] for a in plan['arms'] if set(a['selected_channels']) == selected],
               'selected_channel_jaccard_with_max_control':len(selected & old_set)/len(selected | old_set)}
        proposed['arms'].append(arm)
        selected_by_arm[name] = selected
        current['stable_tie_order_same_selection'] = selected == stable_set
        current['stable_tie_removed_weights'] = stable['removed_weights']
        current['stable_tie_only_default_channels'] = sorted(selected-stable_set)
        current['stable_tie_only_stable_channels'] = sorted(stable_set-selected)
        current['selected_channel_jaccard_with_max_control'] = arm['selected_channel_jaccard_with_max_control']
        audit[name] = current
    same_rank_masks = selected_by_arm['rank_no_cap'] == selected_by_arm['rank_cap10']
    if same_rank_masks:
        shared = copy.deepcopy(proposed['arms'][1])
        assert proposed['arms'][0]['expected_removed_weights'] == shared['expected_removed_weights']
        shared['name'] = 'rank_shared'
        shared['represented_conditions'] = ['rank_no_cap', 'rank_cap10']
        shared['deduplicated_exact_selected_set'] = True
        shared['selected_channel_jaccard_with_max_controls'] = {
            a['name']: a['selected_channel_jaccard_with_max_control'] for a in proposed['arms']}
        shared.pop('selected_channel_jaccard_with_max_control')
        proposed['arms'] = [shared]
    target_review = []
    for key in [('features.1',4), ('features.2',20)]:
        row = next(r for r in rows if (r['block'],r['ch']) == key)
        target_review.append({k:row[k] for k in ('block','ch','weights','status','P_descending_rank_competition','phi_descending_rank_competition','P_max_replayed','phi_max_replayed','S_max_replayed','P_rank_default','phi_rank_default','S_rank_default')})
        target_review[-1]['selected_by_arm'] = {name:key in selection for name,selection in selected_by_arm.items()}
    for row in rows:
        for name, selection in selected_by_arm.items():
            row[name + '_selected'] = (row['block'],row['ch']) in selection
    policy = {'built_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'builder_sha256':sha256(__file__), 'core_prune_sha256':sha256(ROOT/'prune.py'),
              'core_rank_function_sha256':hashlib.sha256(rank_meta['rank_function_source'].encode()).hexdigest(),
              'baseline_plan_sha256':sha256(args.baseline_plan), 'controls':controls,
              'percentile_scope':'active channels per block, including protected channels in percentile calculation',
              'dtype':'torch.float32', 'device':'cpu', 'rank_metadata':rank_meta,
              'gamma_audit':gamma_audit, 'target_channel_review':target_review,
              'new_policy_executed':False, 'model_loaded_by_builder':False,
              'only_channel_plan_built':True,
              'condition_to_execution_arm': {name: 'rank_shared' if same_rank_masks else name for name in selected_by_arm},
              'deduplicated_rank_conditions': same_rank_masks,
              'notes':['Uses original torch.argsort default CPU tie behavior; stable-channel-index order is an audit only.',
                       'Original CUDA argsort ties may differ; this is a fixed CPU-generated plan, not a claim of CPU/GPU ordering identity.',
                       'Historical raw P and phi CSV are rounded; exact max control replay is verified but full-precision rank order cannot be recovered.',
                       'Rank removes magnitude-spacing information and can allocate distinct percentiles to tied zero values.',
                       'No manual protection of the two inspected channels and no alpha, gamma, floor or budget adjustment.',
                       'Same nominal weight budget; indivisible channel costs make realized removals differ.',
                       'Selecting the next method uses validation diagnostics; no test data are used.']}
    proposed['rank_policy'] = policy
    proposed['notes'] = ['Explicit rank selections from archived raw importance and original protected set.',
                         'Historical max controls reproduced exactly; new rank arms are not historical replay.',
                         'Same source12, alpha .7, seed42, gamma .005, floor .03, nominal budget32791.',
                         'CPU selection only at preparation; no GPU/evaluation/training is launched by this builder.',
                         'For subsequent paired training use the rank wrapper with the original five-epoch protocol.']
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'plan.json', proposed)
    write_json(output/'selection_audit.json', {'controls':controls,'rank_arms':audit,'target_channels':target_review,
                                             'rank_no_cap_and_cap_same_set':selected_by_arm['rank_no_cap']==selected_by_arm['rank_cap10']})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output/'channel_score_audit.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    layer_rows=[]
    for name, arm in audit.items():
        for block, values in arm['per_layer'].items():
            layer_rows.append({'arm':name,'block':block,**values})
    with (output/'layer_plan.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(layer_rows[0]));writer.writeheader();writer.writerows(layer_rows)
    source_dir=output/'source'
    for file in ['scripts/build_lfwa_rank_norm_plan.py','scripts/run_lfwa_rank_norm_diagnostic.py','scripts/build_lfwa_signed_phi_plan.py','scripts/run_lfwa_single_layer_cap_ablation.py','scripts/run_lfwa_channel_cap_diagnostic.py','prune.py']:
        destination=source_dir/file;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,destination)
    read_plan(output/'plan.json')
    print(json.dumps({'status':'rank_plan_built_no_execution','output':str(output),'arms':{n:{k:v for k,v in a.items() if k in ('selected_channels','removed_weights','budget_overshoot','stable_tie_order_same_selection')} for n,a in audit.items()},'target_channels':target_review},indent=2))


if __name__ == '__main__':
    main()
