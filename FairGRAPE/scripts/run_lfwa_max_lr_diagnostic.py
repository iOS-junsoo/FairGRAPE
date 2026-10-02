#!/usr/bin/env python3
"""Replay the original LFWA max mask, verify identical initial tensors, change LR.

All evaluation/training comes from the unchanged original diagnostic runner.
--dry-run is CPU-only and never evaluates images or trains.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_lfwa_channel_cap_diagnostic as runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    original_command=list(sys.argv)
    args.plan=args.plan.resolve();args.run_dir=args.run_dir.resolve()
    if args.dry_run:os.environ['CUDA_VISIBLE_DEVICES']=''
    plan=runner.read_plan(args.plan)
    expected={'source_iteration':12,'iteration':13,'normalization':'max','alpha':.7,'protection_ratio':.005,
              'layer_floor':.03,'keep_per_iter':.975,'seed':42,'batch':256,'retrain_epochs':5,'retrain_lr':1e-4}
    for key,value in expected.items():
        if plan.get(key)!=value:raise ValueError('Unreviewed protocol: '+key)
    if len(plan['arms'])!=1 or plan['arms'][0]['name']!='max_lr1e4':raise ValueError('Require one max_lr1e4 arm')
    arm=plan['arms'][0]
    if not arm.get('reference_selection') or arm.get('strict_channel_cap_fraction') is not None:
        raise ValueError('Require exact historical baseline selection and no cap')
    if len(arm['selected_channels'])!=154 or arm['expected_removed_weights']!=32858:
        raise ValueError('Unexpected mask budget')
    provenance=plan['lr_intervention']
    for file,digest in provenance['original_control_source_sha256'].items():
        if runner.sha256(ROOT/file)!=digest:raise ValueError('Shared source differs from original control: '+file)
    for file,digest in provenance['original_control_file_sha256'].items():
        if runner.sha256(file)!=digest:raise ValueError('Original control artifact changed: '+file)
    reference=Path(provenance['baseline_post_prune_checkpoint'])
    if reference.resolve()==Path(plan['source_checkpoint']).resolve():raise ValueError('Source must be original pre-pruning source12')
    root=args.run_dir
    marker=root/'lr_wrapper_metadata.json'
    if marker.exists() or (root/'complete.json').exists():raise FileExistsError('Refusing to overwrite diagnostic')
    root.mkdir(parents=True,exist_ok=True)
    runner.write_json(marker,{'plan_sha256':runner.sha256(args.plan),'wrapper_sha256':runner.sha256(__file__),
                             'original_runner_sha256':runner.sha256(ROOT/'scripts/run_lfwa_channel_cap_diagnostic.py'),
                             'original_runner_file_modified':False,'normalization':'max','retrain_lr':1e-4,
                             'same_post_prune_state_required':True,'dry_run':args.dry_run,
                             'command':original_command,'python':sys.executable,
                             'baseline_post_prune_checkpoint':str(reference),'baseline_post_prune_checkpoint_sha256':runner.sha256(reference)})
    destination=root/'source/scripts/run_lfwa_max_lr_diagnostic.py';destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,destination)
    original_apply=runner.apply_selection
    def apply_same_initial_state(model,selected):
        import torch
        result=original_apply(model,selected)
        checkpoint=torch.load(reference,map_location='cpu')
        expected_state=checkpoint.get('model_state',checkpoint.get('model_state_dict'))
        actual=result.state_dict()
        if actual.keys()!=expected_state.keys():raise AssertionError('Post-prune tensor keys differ from original control')
        differences=[name for name,value in actual.items() if not torch.equal(value.detach().cpu(),expected_state[name])]
        audit={'all_tensor_values_exactly_equal':not differences,'tensor_count':len(actual),
               'mask_tensor_count':sum(name.endswith('.mask') for name in actual),
               'different_tensor_names':differences,'reference_post_prune_checkpoint_sha256':runner.sha256(reference),
               'source12_checkpoint_sha256':plan['source_checkpoint_sha256'],'training_started':False,
               'note':'Compare weights, biases, BN buffers and masks; optimizer state is not loaded.'}
        runner.write_json(root/'post_prune_initial_state_verification.json',audit)
        if differences:raise AssertionError('Initial tensors differ from original LR1e-5 control')
        return result
    runner.apply_selection=apply_same_initial_state
    sys.argv=[sys.argv[0],'--plan',str(args.plan),'--run-dir',str(root)]+(['--dry-run'] if args.dry_run else [])
    runner.main()


if __name__=='__main__':main()
