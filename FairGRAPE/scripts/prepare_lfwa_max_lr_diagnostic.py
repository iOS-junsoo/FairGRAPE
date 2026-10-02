#!/usr/bin/env python3
"""Prepare one LFWA source12 -> original max mask -> LR1e-4 five-epoch plan.

CPU metadata/CSV only. No checkpoint deserialization, model, GPU or training.
"""
import argparse
import copy
import csv
import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_lfwa_channel_cap_diagnostic import read_plan,sha256,write_json
CONTROL=ROOT/'lfwa_diagnostics/alpha0p7_iter13_cap10_seed42_20260917_105048'
ARM='max_lr1e4'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'lfwa_diagnostics/alpha0p7_iter13_max_lr1e4_seed42_20260917')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if (output/'plan.json').exists():raise FileExistsError('Refusing to overwrite an existing prepared plan')
    plan=read_plan(CONTROL/'plan.json')
    run=json.loads((CONTROL/'run/diagnostic_run.json').read_text())
    old=next(arm for arm in plan['arms'] if arm['name']=='baseline')
    archived=next(arm for arm in run['plan']['arms'] if arm['name']=='baseline')
    assert old['selected_channels']==[tuple(key) for key in archived['selected_channels']]
    assert len(old['selected_channels'])==154 and old['expected_removed_weights']==32858
    fields=['source_checkpoint','source_checkpoint_sha256','source_channel_csv','source_channel_csv_sha256',
            'iteration','alpha','seed','batch','retrain_epochs','retrain_lr','keep_per_iter','layer_floor',
            'normalization','protection_ratio','lfwa_csv','lfwa_image_root']
    for key in fields:assert plan[key]==run['plan'][key],key
    expected={'source_iteration':12,'iteration':13,'alpha':.7,'normalization':'max','protection_ratio':.005,
              'layer_floor':.03,'seed':42,'batch':256,'retrain_epochs':5,'retrain_lr':1e-5,'keep_per_iter':.975}
    for key,value in expected.items():assert plan[key]==value,(key,plan[key])
    for file,digest in run['source_sha256'].items():
        assert sha256(ROOT/file)==digest,file
        assert sha256(CONTROL/'run/source'/file)==digest,file
    with Path(plan['source_channel_csv']).open() as handle:rows=list(csv.DictReader(handle))
    historical={(row['block'],int(row['ch'])) for row in rows if row['status'].startswith('pruned_now')}
    weights={(row['block'],int(row['ch'])):int(row['weights']) for row in rows}
    assert set(old['selected_channels'])==historical
    assert sum(weights[key] for key in historical)==32858
    control_root=CONTROL/'run/baseline'
    control_complete=json.loads((control_root/'complete.json').read_text())
    assert control_complete['status']=='complete' and control_complete['selected_channels']==154
    assert control_complete['removed_target_weights']==32858 and control_complete['historical_masks_equal']
    assert control_complete['historical_mask_count']==53
    proposed=copy.deepcopy(plan)
    proposed['diagnostic']='same_source_and_original_max_mask_change_only_retraining_learning_rate'
    proposed['retrain_lr']=1e-4
    proposed.pop('layer_channel_cap_counts',None)
    arm=copy.deepcopy(old)
    arm.update(name=ARM,reference_selection=True,strict_channel_cap_fraction=None,
               reference_checkpoint=str(control_root/'post_prune.pt'))
    proposed['arms']=[arm]
    for key in fields:
        if key=='retrain_lr':assert proposed[key]==1e-4
        else:assert proposed[key]==plan[key],key
    assert Path(proposed['source_checkpoint']) not in (control_root/'post_prune.pt',control_root/'retrained.pt')
    control_files=[CONTROL/'run/diagnostic_run.json',CONTROL/'plan.json',control_root/'post_prune.pt',
                   control_root/'retrained.pt',control_root/'post_prune/summary.json',control_root/'post_retrain/summary.json',
                   control_root/'complete.json',control_root/'historical_mask_verification.json']
    provenance={str(file):sha256(file) for file in control_files}
    proposed['lr_intervention']={
        'changed_computational_parameter':'retrain_lr:1e-5 -> 1e-4',
        'baseline_lr':1e-5,'candidate_lr':1e-4,'retrain_epochs':5,
        'initialization':'Same source12 checkpoint, then original154-channel mask; not a recovered model warm start.',
        'fresh_optimizer':'Original train() constructs fresh Adam; no optimizer state is loaded.',
        'selection':'Original maximum validation accuracy, then lower EO/loss ties.',
        'baseline_control_root':str(control_root),
        'baseline_post_prune_checkpoint':str(control_root/'post_prune.pt'),
        'baseline_post_prune_checkpoint_sha256':sha256(control_root/'post_prune.pt'),
        'baseline_post_retrain_metrics':control_complete['post_retrain'],
        'baseline_post_prune_metrics':control_complete['post_prune'],
        'original_control_source_sha256':run['source_sha256'],
        'original_control_file_sha256':provenance,
        'original_control_runtime':run['runtime'],
        'baseline_control_retraining_repeated':False,
        'future_resource_gate':{'reference':'existing max LR1e-5 baseline at the same step13',
            'thresholds':{'delta_ba_min':.005,'delta_accuracy_min':-.002,'near_constant_delta_max':0},
            'scope':'utility-only resource-allocation gate; EO reported separately, not fairness success'},
        'gpu_execution_started':False,'queue_created':False}
    proposed['notes']=[
        'Original baseline selected set and order reused exactly:154 channels/32858weights.',
        'Start from source12 and apply the same mask before any new training.',
        'Only retraining LR changes;alpha/normalization/gamma/floor/mask/seed42/batch256/5epochs/selector unchanged.',
        'Existing LR1e-5 baseline is reused without duplicate training.',
        'Thin wrapper validates all post-prune tensors against the original control before any training.',
        'Validation only;EO and attribute collapse must be assessed separately from the utility resource gate.',
        'Historical uninterrupted-run RNG was not saved; comparison is to the completed paired diagnostic.']
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'plan.json',proposed)
    read_plan(output/'plan.json')
    source_files=list(run['source_sha256'])+['scripts/prepare_lfwa_max_lr_diagnostic.py','scripts/run_lfwa_max_lr_diagnostic.py']
    for file in source_files:
        destination=output/'source'/file;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,destination)
    verification={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'source_checkpoint_sha256':plan['source_checkpoint_sha256'],
                  'source12_matches_original_control':True,'selected_channel_order_and_set_match_original':True,
                  'selected_channels':154,'removed_weights':32858,'historical_csv_exact_match':True,
                  'shared_source_hashes_match_completed_control':True,'only_training_parameter_changed':'retrain_lr',
                  'old_retrain_lr':1e-5,'new_retrain_lr':1e-4,'starts_from_retrained_control':False,
                  'baseline_control_files_sha256':provenance,'builder_sha256':sha256(__file__),
                  'plan_sha256':sha256(output/'plan.json'),'model_loaded':False,'gpu_used':False,'queue_created':False}
    write_json(output/'plan_verification.json',verification)
    print(json.dumps({'status':'plan_prepared_only','output':str(output),'selected_channels':154,'removed_weights':32858},indent=2))


if __name__=='__main__':main()
