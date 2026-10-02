#!/usr/bin/env python3
"""Prepare/run one reviewed LFWA rank/no-cap continuation, iterations 14..25.

No original helper/core file is edited. The production selector adapter changes
only its normalization guard from max to rank and retains full-precision replay
and mask checks. --preflight never loads image pixels, scores, evaluates, trains,
or uses CUDA. Source12 is permitted only with --preflight-source12 in a separate
structural sandbox; the future actual continuation requires rank_shared iter13.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_lfwa_cap_trajectory as base

ARM = 'rank_no_cap'
FUTURE_SOURCE = ROOT / 'lfwa_diagnostics/alpha0p7_iter13_rank_norm_seed42_20260917/run/rank_shared/retrained.pt'
SOURCE12 = ROOT / 'lfwa_runs/alpha0p7_seed42_no_final_20260916_143706/prune_012.pt'
SOURCE12_SHA = '654e84357fd8ace65db1778cece5df0cee2733217853e284e3cd20e6852b0e80'
REFERENCE_RUN = ROOT / 'lfwa_diagnostics/alpha0p7_cap10_trajectory_seed42_20260917/run.json'
REFERENCE_BASELINE = ROOT / 'lfwa_diagnostics/alpha0p7_iter13_cap10_seed42_20260917_105048/run/baseline/retrained.pt'
GATE_PATH = ROOT / 'lfwa_diagnostics/alpha0p7_iter13_rank_norm_seed42_20260917/analysis/continuation_gate.json'
GATE_THRESHOLDS = {'delta_ba_min':.005, 'delta_accuracy_min':-.002, 'near_constant_delta_max':0}
GATE_SCOPE = 'utility-only resource-allocation gate; EO reported separately, not fairness success'
FILES = ['scripts/run_lfwa_rank_trajectory.py', 'scripts/run_lfwa_cap_trajectory.py',
         'prune.py', 'lfwa.py', 'lfwa_experiment.py', 'train_and_val.py',
         'dataset.py', 'util.py', 'config.py', 'adversarial_debiasing.py']


def rank_selector_adapter():
    """Reuse exact helper body with two audited string literals changed locally."""
    original = inspect.getsource(base.select_from_production)
    tree = ast.parse(original)
    constants = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    norm = [node for node in constants if node.value == 'max']
    message = [node for node in constants if node.value == 'This diagnostic requires production cap=None and normalization=max']
    if len(norm) != 1 or len(message) != 1:
        raise ValueError('Original selector normalization guard changed; review the adapter')
    norm[0].value = 'rank'
    message[0].value = 'This diagnostic requires production cap=None and normalization=rank'
    transformed = ast.fix_missing_locations(tree)
    namespace = dict(base.__dict__)
    exec(compile(transformed, __file__ + '::rank_selector_adapter', 'exec'), namespace)
    adapted = namespace['select_from_production']
    # Assert structural equality after undoing the two authorized literal edits.
    restored = copy.deepcopy(transformed)
    for node in ast.walk(restored):
        if isinstance(node, ast.Constant) and node.value == 'rank':
            node.value = 'max'
        elif isinstance(node, ast.Constant) and node.value == 'This diagnostic requires production cap=None and normalization=rank':
            node.value = 'This diagnostic requires production cap=None and normalization=max'
    if ast.dump(restored, include_attributes=False) != ast.dump(ast.parse(original), include_attributes=False):
        raise AssertionError('Selector adapter changed more than the normalization guard/message')
    audit = {'original_selector_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
             'adapted_selector_ast_sha256': hashlib.sha256(ast.dump(transformed, include_attributes=False).encode()).hexdigest(),
             'guard_change': 'payload.norm max -> rank', 'message_change_only': True,
             'selection_or_mask_logic_changed': False,
             'original_function': 'run_lfwa_cap_trajectory.select_from_production'}
    return adapted, audit


def install_adapters():
    import config
    import lfwa
    import prune
    original_configure = base.configure
    adapted_selector, audit = rank_selector_adapter()
    original_save = lfwa.save_checkpoint

    def configure(args, directory, metadata, iteration):
        original_configure(args, directory, metadata, iteration)
        prune.IMPT2_NORM = 'rank'
        if args.cap_fraction is not None or args.alpha != .7 or args.seed != 42:
            raise ValueError('This reviewed trajectory requires rank, no cap, alpha=.7 and seed42')
        expected = {'IMPT2_NORM':'rank', 'IMPT2_PROTECTION_RATIO':.005,
                    'IMPT2_MIN_KEEP_RATIO_PER_LAYER':.03, 'IMPT2_LAYER_CAP_MULTIPLIER':None,
                    'IMPT_TYPE2_IMPORTANCE_BATCH_SIZE':128, 'IMPT_TYPE2_ALPHA':.7}
        for name, value in expected.items():
            if getattr(prune, name) != value:
                raise AssertionError(f'Production protocol mismatch: {name}')
        if config.glo_impt_type != 2 or config.glo_keep_per_iter != .975:
            raise AssertionError('Unexpected pruning protocol')
        if config.glo_perf_only or config.glo_phi_gap_only or config.glo_phi_analysis or config.glo_use_grl:
            raise AssertionError('An unreviewed pruning option is enabled')

    def select(pruner, cfgs, directory, cap_fraction):
        if cap_fraction is not None or prune.IMPT2_NORM != 'rank':
            raise ValueError('Rank trajectory selector requires cap=None and rank normalization')
        return adapted_selector(pruner, cfgs, directory, cap_fraction)

    def save(model, path, metadata, **details):
        if details.get('stage') == 'cap_trajectory':
            if details.get('arm') != ARM or details.get('strict_cap_fraction') is not None:
                raise ValueError('Unexpected trajectory checkpoint configuration')
            details['stage'] = 'rank_trajectory'
            details['normalization'] = 'rank'
        return original_save(model, path, metadata, **details)

    base.configure = configure
    base.select_from_production = select
    lfwa.save_checkpoint = save
    audit['configure_change'] = 'Original configure followed by IMPT2_NORM=rank and protocol guards'
    audit['checkpoint_metadata_change'] = 'stage cap_trajectory -> rank_trajectory, normalization=rank'
    audit['reused_functions'] = ['run_arm', 'load_model', 'evaluate', 'completed_steps', 'select_channels']
    return audit


def fixed_args(args):
    # No sweep parameters are exposed: this is the one reviewed continuation.
    args.alpha = .7
    args.cap_fraction = None
    args.keep_per_iter = .975
    args.layer_floor = .03
    args.seed = 42
    args.batch = 256
    args.retrain_epochs = 5
    args.retrain_lr = 1e-5
    args.stop_batch = 10000
    return args


def verify_control_protocol(metadata):
    reference = json.loads(REFERENCE_RUN.read_text())
    expected = {'alpha':.7, 'keep_per_iter':.975, 'layer_floor':.03, 'seed':42,
                'batch':256, 'retrain_epochs':5, 'retrain_lr':1e-5, 'stop_batch':10000,
                'protection_ratio':.005, 'importance_batch':128, 'phi_batch':64,
                'evaluation_split':'val', 'final_retraining':False,
                'score_seed':'seed + iteration', 'training_seed':'seed + 10000 + iteration'}
    for key, value in expected.items():
        if reference['protocol'][key] != value:
            raise ValueError(f'Historical paired protocol mismatch: {key}')
    if reference['protocol']['experiment_metadata'] != metadata:
        raise ValueError('Historical control split/task metadata differs')
    # Shared helper/core/training code must match the completed paired controls.
    for file in FILES[1:]:
        if reference['source_sha256'].get(file) != base.sha256(ROOT / file):
            raise ValueError(f'Historical paired control implementation changed: {file}')
    run_arm_ast = ast.parse(inspect.getsource(base.run_arm))
    tuple_seed = [node for node in ast.walk(run_arm_ast) if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Tuple)
                  and [n.id for n in node.targets[0].elts] == ['score_seed','training_seed']]
    if len(tuple_seed) != 1 or ast.unparse(tuple_seed[0].value) != '(args.seed + iteration, args.seed + 10000 + iteration)':
        raise AssertionError('Shared run_arm seed schedule changed')
    return {'reference_run':str(REFERENCE_RUN), 'reference_run_sha256':base.sha256(REFERENCE_RUN),
            'shared_protocol_fields_equal':True, 'shared_source_files_equal':True,
            'score_seed_formula_verified':True, 'training_seed_formula_verified':True,
            'comparison_note':'Existing max baseline/cap trajectories are reused; no duplicate control training.'}



def audit_completed(arm_root, source_iteration, end_step):
    """Retain helper checksum/contiguity checks and reject stale completed tails."""
    completed=base.completed_steps(arm_root,source_iteration,end_step)
    contiguous={row['iteration'] for row in completed}
    all_completed={int(path.parent.name.split('_')[1]) for path in arm_root.glob('iteration_*/complete.json')
                   if re.fullmatch(r'iteration_[0-9]{3}',path.parent.name)}
    if all_completed != contiguous:
        raise ValueError('Completed rank steps exist beyond a gap or outside the requested continuation')
    for row in completed:
        iteration=row['iteration']
        if row.get('arm') != ARM or row.get('score_seed') != 42+iteration or row.get('training_seed') != 42+10000+iteration:
            raise ValueError('Completed rank step arm/seed metadata mismatch')
        if row['selection'].get('strict_cap_fraction') is not None or not row['selection'].get('production_selector_reproduced'):
            raise ValueError('Completed rank step selector/cap metadata mismatch')
        components=json.loads((arm_root/f'iteration_{iteration:03d}'/'score_components_full_precision.json').read_text())
        if components.get('normalization') != 'rank' or components.get('alpha') != .7 or components.get('gamma') != .005:
            raise ValueError('Completed rank step importance protocol mismatch')
    return completed


def resume_guard_preflight(run_dir):
    """Synthetic metadata only; dummy checkpoint bytes are never torch-loaded."""
    with tempfile.TemporaryDirectory(prefix='resume_guard_',dir=run_dir) as temporary:
        arm_root=Path(temporary)/ARM
        first=arm_root/'iteration_014';first.mkdir(parents=True)
        (first/'retrained.pt').write_bytes(b'CPU resume checksum fixture, not a model')
        row={'iteration':14,'arm':ARM,'score_seed':56,'training_seed':10056,
             'selection':{'strict_cap_fraction':None,'production_selector_reproduced':True},
             'checkpoint_sha256':base.sha256(first/'retrained.pt')}
        base.write_json(first/'complete.json',row)
        base.write_json(first/'score_components_full_precision.json',{'normalization':'rank','alpha':.7,'gamma':.005})
        (arm_root/'iteration_015').mkdir()  # An unfinished next step is allowed.
        assert len(audit_completed(arm_root,13,25))==1
        later=arm_root/'iteration_016';later.mkdir()
        base.write_json(later/'complete.json',{'iteration':16})
        try:
            audit_completed(arm_root,13,25)
        except ValueError:
            pass
        else:
            raise AssertionError('Completed step after a gap was accepted')
        (later/'complete.json').unlink()
        bad=dict(row,training_seed=42)
        base.write_json(first/'complete.json',bad)
        try:
            audit_completed(arm_root,13,25)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong completed training seed was accepted')
        base.write_json(first/'complete.json',row)
        (first/'retrained.pt').write_bytes(b'checksum corruption fixture')
        try:
            audit_completed(arm_root,13,25)
        except ValueError:
            pass
        else:
            raise AssertionError('Checkpoint hash mismatch was accepted')
    return {'contiguous_completed_accepted':True,'unfinished_next_step_allowed':True,
            'completed_after_gap_rejected':True,'wrong_seed_rejected':True,'checksum_mismatch_rejected':True,
            'dummy_checkpoint_torch_loaded':False}



def validate_gate_values(gate, source_path, source_sha, baseline_sha):
    required=['schema_version','passes','source_checkpoint','source_checkpoint_sha256',
              'reference_baseline_checkpoint_sha256','reference_metrics','rank_metrics',
              'deltas_rank_minus_baseline','thresholds','evaluated_at','scope']
    if any(key not in gate for key in required):
        raise ValueError('Continuation gate is missing a required field')
    if type(gate['schema_version']) is not int or gate['schema_version'] != 1 or gate['passes'] is not True:
        raise ValueError('Continuation gate must have schema_version1 and passes=true')
    named_source=Path(gate['source_checkpoint'])
    if not named_source.is_absolute() or named_source.resolve() != source_path.resolve() or gate['source_checkpoint_sha256'] != source_sha:
        raise ValueError('Gate does not identify the exact original rank13 source checkpoint')
    if gate['reference_baseline_checkpoint_sha256'] != baseline_sha:
        raise ValueError('Gate baseline checkpoint hash differs from the original paired baseline')
    if gate['thresholds'] != GATE_THRESHOLDS or gate['scope'] != GATE_SCOPE:
        raise ValueError('Continuation gate thresholds or scope changed')
    evaluated=base.datetime.datetime.fromisoformat(gate['evaluated_at'].replace('Z','+00:00'))
    if evaluated.tzinfo is None or evaluated.utcoffset() != base.datetime.timedelta(0):
        raise ValueError('Gate evaluated_at must be UTC')
    deltas={}
    for field in ('accuracy','ba','near_constant_99pct_attributes'):
        values=[]
        for name in ('reference_metrics','rank_metrics'):
            value=gate[name][field]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Gate utility metrics must be finite numbers')
            if field=='near_constant_99pct_attributes':
                if int(value)!=value or not 0<=value<=39:
                    raise ValueError('Invalid near-constant attribute count')
            elif not 0<=value<=1:
                raise ValueError('Invalid accuracy or BA')
            values.append(value)
        delta=values[1]-values[0]
        recorded=gate['deltas_rank_minus_baseline'][field]
        if isinstance(recorded,bool) or not isinstance(recorded,(int,float)) or not math.isfinite(recorded) or not math.isclose(delta,recorded,rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Gate metric deltas do not match the supplied metrics')
        deltas[field]=delta
    if deltas['ba'] < .005 or deltas['accuracy'] < -.002 or deltas['near_constant_99pct_attributes'] > 0:
        raise ValueError('Gate passes=true but the fixed utility criteria are not met')
    return deltas


def gate_input(args):
    if args.preflight:
        return {'status':'not_evaluated_cpu_preflight','evaluated':False,'passed':None,
                'path':str(args.gate_json),'required_for_actual_execution':True,
                'thresholds':GATE_THRESHOLDS,'scope':GATE_SCOPE}
    gate=json.loads(args.gate_json.read_text())
    delta=validate_gate_values(gate,args.source_checkpoint,base.sha256(args.source_checkpoint),base.sha256(REFERENCE_BASELINE))
    for name,path in [('reference_metrics',REFERENCE_BASELINE.parent/'post_retrain/summary.json'),
                      ('rank_metrics',args.source_checkpoint.parent/'post_retrain/summary.json')]:
        summary=json.loads(path.read_text())
        for field in ('accuracy','ba','near_constant_99pct_attributes'):
            if not math.isclose(gate[name][field],summary[field],rel_tol=1e-12,abs_tol=1e-14):
                raise ValueError('Gate utility metrics differ from their checkpoint evaluation summaries')
    return {'status':'passed','evaluated':True,'passed':True,'path':str(args.gate_json),
            'sha256':base.sha256(args.gate_json),'thresholds':GATE_THRESHOLDS,'scope':GATE_SCOPE,
            'computed_deltas':delta,'original_source_checkpoint_sha256':gate['source_checkpoint_sha256'],
            'reference_baseline_checkpoint_sha256':gate['reference_baseline_checkpoint_sha256']}


def gate_schema_preflight():
    """Synthetic schema/math checks do not evaluate any actual experiment gate."""
    source=Path('/tmp/rank_gate_fixture_source.pt')
    reference={'accuracy':.8,'ba':.7,'near_constant_99pct_attributes':5}
    rank={'accuracy':.81,'ba':.71,'near_constant_99pct_attributes':4}
    fixture={'schema_version':1,'passes':True,'source_checkpoint':str(source),
             'source_checkpoint_sha256':'fixture-source','reference_baseline_checkpoint_sha256':'fixture-baseline',
             'reference_metrics':reference,'rank_metrics':rank,
             'deltas_rank_minus_baseline':{key:rank[key]-reference[key] for key in reference},
             'thresholds':GATE_THRESHOLDS,'scope':GATE_SCOPE,'evaluated_at':'2026-09-17T00:00:00+00:00'}
    validate_gate_values(fixture,source,'fixture-source','fixture-baseline')
    for mutate in ('passes','source','ba','delta'):
        bad=copy.deepcopy(fixture)
        if mutate=='passes':bad['passes']=False
        elif mutate=='source':bad['source_checkpoint_sha256']='different-source'
        elif mutate=='ba':
            bad['rank_metrics']['ba']=.7
            bad['deltas_rank_minus_baseline']['ba']=0.
        else:bad['deltas_rank_minus_baseline']['accuracy']=.5
        try:
            validate_gate_values(bad,source,'fixture-source','fixture-baseline')
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid synthetic gate was accepted: '+mutate)
    return {'schema_checks_passed':True,'false_gate_rejected':True,'source_mismatch_rejected':True,
            'insufficient_ba_rejected':True,'inconsistent_delta_rejected':True,
            'actual_experiment_gate_evaluated':False}


def structural_preflight(args, model, checkpoint, metadata, adapter_audit, comparison):
    import random
    import numpy as np
    import torch
    import config
    import lfwa
    import prune
    from util import setseed
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or next(model.parameters()).device.type != 'cpu':
        raise AssertionError('Preflight must be CPU-only')
    before = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    frozen = lfwa.assert_masks(model)
    if len(frozen) != 53 or len(lfwa.TARGETS) != 39 or len(lfwa.OUTPUT_SLICES) != 39:
        raise AssertionError('Unexpected model mask/task structure')
    layout_original, layout_active = base.channel_layout(model)
    sparsity = lfwa.sparsity(model)
    if sparsity['model_parameters'] != 2323790 or sparsity['target_weights'] != 1779296:
        raise AssertionError('Unexpected parameter geometry')
    if model.classifier[-1].out_features != 78:
        raise AssertionError('Expected 39 binary output heads')
    base.configure(args, args.run_dir/'configuration_probe', metadata, 14)
    if prune.IMPT2_NORM != 'rank' or config.glo_prune_iter != 13:
        raise AssertionError('Configuration did not preserve rank/iteration protocol')
    score_seed, train_seed = args.seed + 14, args.seed + 10000 + 14
    def random_sample(seed):
        setseed(seed)
        return [random.random(), float(np.random.random()), float(torch.rand(()))]
    cold_or_warm_sample=random_sample(score_seed)  # Warm up existing imgaug global RNG only in this CPU probe.
    score_samples=[random_sample(score_seed),random_sample(score_seed)]
    training_samples=[random_sample(train_seed),random_sample(train_seed)]
    assert score_samples[0] == score_samples[1]
    assert training_samples[0] == training_samples[1]
    assert score_seed == 56 and train_seed == 10056
    # Base Prunner deep-copies and freezes loaded masks; no score/prune/forward call.
    pruner = prune.Prunner(model, torch.nn.CrossEntropyLoss(), None, lfwa.OUTPUT_SLICES)
    lfwa.assert_masks(pruner.model, frozen)
    if lfwa.sparsity(pruner.model) != sparsity:
        raise AssertionError('Prunner construction changed source sparsity')
    assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
    assert not torch.cuda.is_initialized()
    result = {'status':'preflight_passed', 'structural_source_iteration':int(checkpoint['iteration']),
              'intended_actual_source_iteration':13, 'intended_start_iteration':14,
              'source12_is_actual_rank13':False if args.preflight_source12 else True,
              'future_rank13_weights_or_performance_validated':False,
              'source_sparsity':sparsity, 'binary_frozen_mask_count':len(frozen),
              'original_channels':layout_original, 'active_channels':layout_active,
              'source_state_unchanged':True, 'Prunner_preserves_loaded_masks':True,
              'seed_schedule':{'iteration14_score_seed':score_seed,'iteration14_training_seed':train_seed,
                               'cpu_seed_reset_reproducible_after_first_initialization':True,
                               'sample_order':['python','numpy','torch'],
                               'first_probe_sample':cold_or_warm_sample,
                               'repeated_score_samples':score_samples,'repeated_training_samples':training_samples,
                               'first_sample_differs_from_initialized':cold_or_warm_sample != score_samples[0],
                               'warmup_only_in_structural_probe_not_added_to_training':True},
              'configuration':{'normalization':prune.IMPT2_NORM,'cap_fraction':None,
                               'gamma':prune.IMPT2_PROTECTION_RATIO,'floor':prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER,
                               'alpha':prune.IMPT_TYPE2_ALPHA,'keep_per_iter':config.glo_keep_per_iter},
              'adapter':adapter_audit,'control_comparison':comparison,'selector_invariant_checks':base.self_test(),
              'resume_guard_checks':resume_guard_preflight(args.run_dir),
              'continuation_gate':gate_input(args),'gate_schema_checks':gate_schema_preflight(),
              'scoring_performed':False,'images_loaded':False,'evaluation_performed':False,
              'training_performed':False,'gpu_used':False,'cuda_initialized':False,
              'limitations':['Source12 structural preflight does not validate the future rank13 checkpoint or any metric.',
                             'Rank production scoring and the adapted capture path still require an approved GPU execution.',
                             'Single-arm rank continuation is not launched by preflight.']}
    base.write_json(args.run_dir/'preflight.json',result)
    print(json.dumps({'status':'preflight_passed','output':str(args.run_dir/'preflight.json')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint',type=Path,default=FUTURE_SOURCE)
    parser.add_argument('--gate-json',type=Path,default=GATE_PATH)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--end-step',type=int,default=25)
    parser.add_argument('--lfwa-csv',type=Path,default=ROOT/'csv/LFWA.csv')
    parser.add_argument('--lfwa-image-root',type=Path,default=ROOT/'Images/LFWA')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--preflight-source12',action='store_true')
    args=fixed_args(parser.parse_args())
    if not 14 <= args.end_step <= 25:
        parser.error('Reviewed continuation is limited to iterations14..25')
    if args.preflight_source12 and not args.preflight:
        parser.error('--preflight-source12 is allowed only with --preflight')
    for field in ('source_checkpoint','gate_json','run_dir','lfwa_csv','lfwa_image_root'):
        setattr(args,field,getattr(args,field).resolve())
    if args.preflight:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.preflight_source12 and (args.source_checkpoint != SOURCE12 or base.sha256(args.source_checkpoint) != SOURCE12_SHA):
        raise ValueError('Structural source12 must be the recorded original source checkpoint')
    continuation_gate=gate_input(args)
    os.chdir(ROOT)
    import torch
    import numpy as np
    import lfwa
    torch.set_num_threads(2)
    frames,metadata=lfwa.load_frames(args.lfwa_csv,args.lfwa_image_root)
    model,checkpoint=base.load_model(args.source_checkpoint,metadata,'cpu')
    source_iteration=int(checkpoint['iteration'])
    if args.preflight_source12:
        if source_iteration != 12:
            raise ValueError('Structural preflight requires source12')
    elif source_iteration != 13 or checkpoint.get('arm') != 'rank_shared' or checkpoint.get('stage') != 'diagnostic_retrained':
        raise ValueError('Actual continuation requires the completed rank_shared iteration13 retrained checkpoint')
    comparison=verify_control_protocol(metadata)
    adapter_audit=install_adapters()
    protocol={field:getattr(args,field) for field in ('alpha','cap_fraction','keep_per_iter','layer_floor','seed','batch','retrain_epochs','retrain_lr','stop_batch')}
    protocol.update(normalization='rank',arm=ARM,source_iteration=source_iteration,
                    intended_source_iteration=13,intended_start_iteration=14,
                    structural_preflight_only=bool(args.preflight_source12),
                    experiment_metadata=metadata,source={'path':str(args.source_checkpoint),'sha256':base.sha256(args.source_checkpoint)},
                    protection_ratio=.005,importance_batch=128,phi_batch=64,evaluation_split='val',final_retraining=False,
                    score_seed='seed + iteration',training_seed='seed + 10000 + iteration',
                    scoring='Unmodified production fairness_grad at rank norm, full-precision capture/replay each step.',
                    continuation_gate=continuation_gate,
                    checkpoint_stage='rank_trajectory',optimizer='Original train() creates fresh Adam, LR1e-5, 5 epochs per step.',
                    rng_note='Same paired trajectory reset schedule; historical uninterrupted RNG not restored.')
    hashes={name:base.sha256(ROOT/name) for name in FILES}
    args.run_dir.mkdir(parents=True,exist_ok=True)
    marker=args.run_dir/'run.json'
    if marker.exists():
        if not args.resume:
            raise FileExistsError('Existing run requires --resume')
        previous=json.loads(marker.read_text())
        if previous['protocol'] != protocol or previous['source_sha256'] != hashes or previous['adapter'] != adapter_audit:
            raise ValueError('Resume protocol/source/adapter changed')
        if previous['preflight'] != args.preflight:
            raise ValueError('Use a separate output directory for structural preflight and actual trajectory')
    else:
        base.write_json(marker,{'created_at':base.now(),'protocol':protocol,'source_sha256':hashes,
                               'adapter':adapter_audit,'control_comparison':comparison,'initial_end_step':args.end_step,
                               'command':sys.argv,'python':sys.executable,'preflight':args.preflight,
                               'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,
                                          'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                                          'cuda_cache_maxsize':os.environ.get('CUDA_CACHE_MAXSIZE')}})
        for name in FILES:
            destination=args.run_dir/'source'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,destination)
        if not args.preflight:
            shutil.copy2(args.gate_json,args.run_dir/'source/continuation_gate.json')
    try:
        if args.preflight:
            structural_preflight(args,model,checkpoint,metadata,adapter_audit,comparison)
            return
        del model,checkpoint
        if not torch.cuda.is_available():
            raise RuntimeError('Actual rank trajectory requires CUDA; use --preflight for CPU structural checks')
        old_complete=args.run_dir/'complete.json'
        if old_complete.exists():
            previous_end=json.loads(old_complete.read_text())['end_step']
            if args.end_step < previous_end:
                raise ValueError('Cannot resume with end_step below an already completed run')
            if args.end_step > previous_end:
                old_complete.replace(args.run_dir/f'complete_through_{previous_end:03d}.json')
        audit_completed(args.run_dir/ARM,source_iteration,args.end_step)
        rows=base.run_arm(args,ARM,args.source_checkpoint,source_iteration,frames,metadata)
        audited=audit_completed(args.run_dir/ARM,source_iteration,args.end_step)
        if rows != audited:
            raise AssertionError('Completed trajectory differs from the resume integrity audit')
        base.write_json(args.run_dir/'results.json',{ARM:rows})
        if len(rows) != args.end_step-source_iteration:
            raise AssertionError('Continuation did not complete every intended step')
        base.write_json(args.run_dir/'complete.json',{'status':'complete','arm':ARM,'end_step':args.end_step,'completed_at':base.now()})
    except Exception as error:
        base.write_json(args.run_dir/'failure.json',{'status':'failed','error':repr(error),'failed_at':base.now()})
        raise


if __name__=='__main__':
    main()
