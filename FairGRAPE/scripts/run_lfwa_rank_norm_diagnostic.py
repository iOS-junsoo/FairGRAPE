#!/usr/bin/env python3
"""Run an explicitly reviewed rank mask plan through the unchanged LFWA runner.

Only configure metadata is extended: after the original configuration, set
prune.IMPT2_NORM='rank'. Importance is not recomputed. A future GPU execution
recomputes only rank scores from saved vectors to validate selection equivalence.
--dry-run remains CPU-only with no image evaluation or training.
"""
import argparse
import ast
import collections
import datetime
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_lfwa_channel_cap_diagnostic as runner



def verify_gpu_rank_selection(plan):
    """Small-vector gate before any model loading in a future approved GPU run."""
    import torch
    from build_lfwa_signed_phi_plan import records_for, select
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA rank preflight requires CUDA for a non-dry-run execution')
    core=ROOT/'prune.py'
    if runner.sha256(core) != plan['rank_policy']['core_prune_sha256']:
        raise ValueError('Core source changed since the reviewed rank plan')
    source=core.read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_rank_percentile')
    scope={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(core)+'::_rank_percentile','exec'),scope)
    rows=records_for(plan)
    groups=collections.defaultdict(list)
    for row in rows:
        if row['weights']>0:
            groups[row['block']].append(row)
    for active in groups.values():
        performance=torch.tensor([float(row['perf_raw']) for row in active],dtype=torch.float32,device='cuda:0')
        fairness=torch.tensor([float(row['phi_raw']) for row in active],dtype=torch.float32,device='cuda:0')
        scores=(plan['alpha']*scope['_rank_percentile'](performance)-(1.0-plan['alpha'])*scope['_rank_percentile'](fairness)).cpu().tolist()
        for row,score in zip(active,scores):
            row['direction_score']=score  # Existing pure selector's alternate score field only; no q is read.
    planned={arm['name']:arm for arm in plan['arms']}
    mappings=plan['rank_policy']['condition_to_execution_arm']
    comparisons={}
    for condition,cap in [('rank_no_cap',False),('rank_cap10',True)]:
        selected=select(rows,plan,cap,direction=True)
        expected=planned[mappings[condition]]
        actual_set=set(selected['selected_channels'])
        expected_set=set(expected['selected_channels'])
        comparisons[condition]={
            'exact_selected_set_equal':actual_set==expected_set,
            'actual_removed_weights':selected['expected_removed_weights'],
            'expected_removed_weights':expected['expected_removed_weights'],
            'only_gpu_selected':sorted(actual_set-expected_set),
            'only_plan_selected':sorted(expected_set-actual_set)}
    return {'status':'passed' if all(v['exact_selected_set_equal'] and v['actual_removed_weights']==v['expected_removed_weights'] for v in comparisons.values()) else 'failed',
            'device':'cuda:0','torch':torch.__version__,'cuda':torch.version.cuda,
            'device_name':torch.cuda.get_device_name(0),'comparisons':comparisons,
            'model_loaded':False,'training_started':False,
            'raw_csv_precision_limitation_remains':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args=parser.parse_args()
    original_command=list(sys.argv)
    args.plan=args.plan.resolve()
    args.run_dir=args.run_dir.resolve()
    sys.argv=[sys.argv[0], '--plan', str(args.plan), '--run-dir', str(args.run_dir)] + (['--dry-run'] if args.dry_run else [])
    if args.dry_run:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    plan=runner.read_plan(args.plan.resolve())
    if plan.get('normalization') != 'rank':
        raise ValueError('Rank wrapper requires explicit normalization=rank')
    root=args.run_dir.resolve()
    marker=root/'rank_wrapper_metadata.json'
    if marker.exists() or (root/'complete.json').exists():
        raise FileExistsError('Refusing to overwrite an existing rank diagnostic')
    original_configure=runner.configure
    def rank_configure(plan, arm_root, metadata):
        original_configure(plan, arm_root, metadata)
        import prune
        prune.IMPT2_NORM='rank'
        assert prune.IMPT2_NORM == plan['normalization']
    runner.configure=rank_configure
    root.mkdir(parents=True,exist_ok=True)
    original=ROOT/'scripts/run_lfwa_channel_cap_diagnostic.py'
    support_files=['scripts/build_lfwa_signed_phi_plan.py', 'scripts/run_lfwa_single_layer_cap_ablation.py']
    runner.write_json(marker, {
        'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'wrapper_sha256':runner.sha256(__file__),
        'original_runner_sha256':runner.sha256(original),
        'support_source_sha256':{name:runner.sha256(ROOT/name) for name in support_files},
        'plan_sha256':runner.sha256(args.plan),
        'normalization':'rank', 'explicit_masks_only':True,
        'importance_recomputation':False,
        'rank_score_recomputation':'GPU vector preflight only; skipped for dry-run',
        'original_runner_score_recomputation':False, 'dry_run':args.dry_run,
        'configuration_change':'After original configure, set prune.IMPT2_NORM=rank for accurate policy metadata.',
        'original_runner_file_modified':False,
        'training_and_evaluation_implementation':'unchanged original runner',
        'command':original_command, 'canonical_runner_command':sys.argv, 'python':sys.executable})
    destination=root/'source/scripts/run_lfwa_rank_norm_diagnostic.py'
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,destination)
    for name in support_files:
        destination=root/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    if not args.dry_run:
        try:
            audit=verify_gpu_rank_selection(plan)
        except Exception as error:
            runner.write_json(root/'gpu_rank_preflight.json', {'status':'failed','error':repr(error),'training_started':False})
            raise
        runner.write_json(root/'gpu_rank_preflight.json',audit)
        if audit['status'] != 'passed':
            raise RuntimeError('GPU rank selection differs from the reviewed plan; stop before model loading and require review')
    runner.main()


if __name__ == '__main__':
    main()
