#!/usr/bin/env python3
"""Sequential, resumable source12 factorial experiment; whole-model60/70 endpoints."""
import argparse, copy, datetime, hashlib, json, math, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
import run_lfwa_channel_cap_diagnostic as one
import run_lfwa_cap_trajectory as trajectory
from build_lfwa_gamma_screen import selection

FILES=['scripts/run_lfwa_lr_protection_whole70.py','scripts/run_lfwa_channel_cap_diagnostic.py',
       'scripts/run_lfwa_cap_trajectory.py','scripts/build_lfwa_gamma_screen.py','prune.py','lfwa.py',
       'lfwa_experiment.py','train_and_val.py','dataset.py','util.py','config.py','adversarial_debiasing.py']
SCREEN=ROOT/'lfwa_diagnostics/alpha_gamma_screen_lr1e4_seed42_20260917'
OLD=ROOT/'lfwa_diagnostics/alpha0p7_iter13_cap10_seed42_20260917_105048/run'

def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,obj):one.write_json(p,obj)

def build(root):
    assert not (root/'manifest.json').exists()
    screen=read(SCREEN/'manifest.json')
    for f,h in screen['source_sha256'].items():assert one.sha256(ROOT/f)==h, f
    available={(c['alpha'],c['gamma']):c for c in screen['conditions']}
    # First the three explicitly requested conditions, each to70, then matched controls.
    order=[(.5,1e-5,.005),(.5,1e-5,.15),(.7,1e-5,.15),
           (.7,1e-5,.005),(.5,1e-4,.005),(.5,1e-4,.15),(.7,1e-4,.005),(.7,1e-4,.15)]
    conditions=[]
    for alpha,lr,gamma in order:
        name=f'a{alpha}_lr{lr:g}_g{gamma}'.replace('.','p')
        source_condition=available[alpha,gamma]
        template=read(source_condition['plan'])
        reuse=source_condition['reuse']
        high_run=Path(reuse['run_dir']) if reuse else SCREEN/'runs'/source_condition['name']
        high_arm=reuse['arm'] if reuse else source_condition['name']
        if lr==1e-4: reuse_run,reuse_arm=high_run,high_arm
        elif alpha==.7 and gamma==.005:reuse_run,reuse_arm=OLD,'baseline'
        else:reuse_run,reuse_arm=None,None
        plan=copy.deepcopy(template)
        plan.update(retrain_lr=lr,diagnostic='source12_lr_protection_factorial_whole70')
        plan['screen_condition'].update(retrain_lr=lr)
        plan['arms'][0]['name']=name
        plan_path=root/'plans'/name/'plan.json';write(plan_path,plan)
        record={'name':name,'alpha':alpha,'lr':lr,'gamma':gamma,'plan':str(plan_path),
                'plan_sha256':one.sha256(plan_path),'same_mask_reference':str(high_run/high_arm/'post_prune.pt')}
        if reuse_run:
            assert read(reuse_run/'complete.json')['status']=='complete'
            meta=read(reuse_run/'diagnostic_run.json')
            for k in ['source_checkpoint_sha256','source_channel_csv_sha256','alpha','protection_ratio','retrain_lr','retrain_epochs','seed','batch','keep_per_iter','layer_floor','normalization']:
                assert meta['plan'][k]==plan[k],(name,k)
            for f,h in meta['source_sha256'].items():assert one.sha256(ROOT/f)==h,(name,f)
            record['reuse']={'run':str(reuse_run),'arm':reuse_arm,'checkpoint':str(reuse_run/reuse_arm/'retrained.pt'),
                             'sha256':one.sha256(reuse_run/reuse_arm/'retrained.pt')}
        else:record['reuse']=None
        record['same_mask_reference_sha256']=one.sha256(record['same_mask_reference'])
        conditions.append(record)
    hashes={f:one.sha256(ROOT/f) for f in FILES}
    for f in FILES:
        dst=root/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
    write(root/'manifest.json',{'created_at':now(),'conditions':conditions,'source_sha256':hashes,
       'common_source_checkpoint':template['source_checkpoint'],'common_source_sha256':template['source_checkpoint_sha256'],
       'source_iteration':12,'endpoints':[.6,.7],'end_step_safety_limit':140,'normalization':'max',
       'layer_floor':.03,'keep_per_iter':.975,'retrain_epochs':5,'seed':42,'batch':256,
       'final_extra_epochs':0,'evaluation_split':'val',
       'seed_schedule':'step13=42; step>=14 scoring=42+iteration, training=10042+iteration',
       'scope':'Same alpha0.7 historical source12 intervention; not a dense-start experiment.',
       'endpoint_policy':'Default keep .975; reduce final removal budget to ceil(target*all_model_parameters)-already_removed at60 and70. Preserve channel rounding; record actual sparsity.',
       'selection_policy':'step13 exact archived float32 score replay to match completed controls; step14+ recompute original full-precision impt_type2 scores on each arm.'})
    print('PREPARED',root,flush=True)

def verify(root):
    m=read(root/'manifest.json')
    for f,h in m['source_sha256'].items():assert one.sha256(ROOT/f)==h,f
    assert one.sha256(m['common_source_checkpoint'])==m['common_source_sha256']
    for c in m['conditions']:
        assert one.sha256(c['plan'])==c['plan_sha256']
        assert one.sha256(c['same_mask_reference'])==c['same_mask_reference_sha256']
        if c['reuse']:assert one.sha256(c['reuse']['checkpoint'])==c['reuse']['sha256']
    return m

def setup_configure():
    orig_one=one.configure
    def conf_one(plan,directory,metadata):
        orig_one(plan,directory,metadata)
        import prune
        prune.IMPT2_PROTECTION_RATIO=plan['protection_ratio']
        assert prune.IMPT2_NORM=='max'
    one.configure=conf_one
    orig_trajectory=trajectory.configure
    def conf_trajectory(args,directory,metadata,iteration):
        orig_trajectory(args,directory,metadata,iteration)
        import prune
        prune.IMPT2_PROTECTION_RATIO=args.gamma
        assert prune.IMPT2_NORM=='max' and prune.IMPT_TYPE2_ALPHA==args.alpha
    trajectory.configure=conf_trajectory

def first_step(root,c,dry):
    plan=one.read_plan(Path(c['plan']))
    if c['reuse']:return Path(c['reuse']['checkpoint'])
    run=root/('dry_runs' if dry else 'step13')/c['name']
    marker=run/('dry_run.json' if dry else 'complete.json')
    if marker.exists():
        meta=read(run/('dry_run_metadata.json' if dry else 'diagnostic_run.json'))
        assert meta['plan']==read(c['plan'])
    else:
        if run.exists():raise RuntimeError('Review unfinished step13 directory before resume: '+str(run))
        check=selection(plan,device='cpu' if dry else 'cuda:0')
        assert check['selected_channels']==plan['arms'][0]['selected_channels']
        assert check['expected_removed_weights']==plan['arms'][0]['expected_removed_weights']
        assert set(check['protected_channels'])==set(map(tuple,plan['protected_channel_ids']))
        write(root/'preflight_selection'/f"{c['name']}_{'cpu' if dry else 'gpu'}.json",check)
        sys.argv=[sys.argv[0],'--plan',c['plan'],'--run-dir',str(run)]+(['--dry-run'] if dry else [])
        one.main()
    if not dry:
        import torch
        actual=trajectory.trusted_torch_load(run/c['name']/'post_prune.pt')['model_state']
        expected=trajectory.trusted_torch_load(c['same_mask_reference'])['model_state']
        assert actual.keys()==expected.keys()
        assert all(torch.equal(actual[k],expected[k]) for k in actual)
        write(run/'same_start_tensor_verification.json',{'all_tensors_equal':True,'tensor_count':len(actual),
              'reference':c['same_mask_reference'],'reference_sha256':c['same_mask_reference_sha256']})
    return run/c['name']/'retrained.pt'

def summarize(root,m):
    rows=[]
    for c in m['conditions']:
        for target in m['endpoints']:
            p=root/'endpoints'/c['name']/f'whole{round(target*100)}.json'
            if p.exists():
                endpoint=read(p);metrics=endpoint['post_retrain']
                rows.append({'condition':c['name'],'alpha':c['alpha'],'lr':c['lr'],'gamma':c['gamma'],
                    'target':target,'iteration':endpoint['iteration'],'path':endpoint['evaluation_dir'],
                    **{k:metrics[k] for k in ['accuracy','ba','eo','model_sparsity','near_constant_99pct_attributes']}})
    contrasts=[]
    lookup={(r['target'],r['alpha'],r['lr'],r['gamma']):r for r in rows}
    def add(label,ref,cand):
        if ref not in lookup or cand not in lookup:return
        a,b=lookup[ref],lookup[cand]
        contrasts.append({'comparison':label,'reference':a['condition'],'candidate':b['condition'],'target':ref[0],
                          'delta':{k:b[k]-a[k] for k in ['accuracy','ba','eo','model_sparsity','near_constant_99pct_attributes']},
                          'reference_iteration':a['iteration'],'candidate_iteration':b['iteration']})
    for t in m['endpoints']:
        for a in [.5,.7]:
            for g in [.005,.15]:add('learning_rate_1e4_minus_1e5',(t,a,1e-5,g),(t,a,1e-4,g))
            for lr in [1e-5,1e-4]:add('protection_15pct_minus_0p5pct',(t,a,lr,.005),(t,a,lr,.15))
        for lr in [1e-5,1e-4]:
            for g in [.005,.15]:add('alpha0p5_minus_alpha0p7',(t,.7,lr,g),(t,.5,lr,g))
    write(root/'analysis/progress.json',{'rows':rows,'contrasts':contrasts,'updated_at':now(),
       'scope':'Validation only, single seed, source12 intervention; actual channel-rounded whole sparsity and iteration counts reported.'})
    lines=['# LFWA 학습률 × 보호 비율: 전체60%·70%','','출발점: 공통 source12. 각 프루닝 단계5epoch, 추가8+1+1 없음. 아래에는 완료된 목표점만 표시한다.','',
           '| 목표 | α | LR | 보호 비율 | 단계 | 실제 전체 희소도 | Accuracy | BA | EO↓ | 근상수 속성 |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['target']:.0%} | {r['alpha']} | {r['lr']:g} | {r['gamma']:.1%} | {r['iteration']} | {r['model_sparsity']:.5%} | {r['accuracy']:.6f} | {r['ba']:.6f} | {r['eo']:.6f} | {r['near_constant_99pct_attributes']} |")
    lines+=['','전체 계획8조건×2목표점 중 '+str(len(rows))+'개 완료. 단일 seed의 점추정으로 통계적·데이터셋 일반화를 주장하지 않는다.']
    for d in contrasts:
        v=d['delta'];lines.append(f"\n{d['target']:.0%} {d['comparison']} ({d['candidate']} − {d['reference']}): ΔAcc {v['accuracy']:+.6f}, ΔBA {v['ba']:+.6f}, ΔEO {v['eo']:+.6f}.")
    (root/'analysis/progress.md').write_text('\n'.join(lines)+'\n')

def run(root,dry):
    m=verify(root);setup_configure()
    import torch,lfwa
    torch.set_num_threads(2)
    frames,metadata=lfwa.load_frames(ROOT/'csv/LFWA.csv',ROOT/'Images/LFWA')
    if dry:
        for c in m['conditions']:
            path=first_step(root,c,True)
            if c['reuse']:
                model,ck=trajectory.load_model(path,metadata,'cpu')
                assert ck['iteration']==13 and len(lfwa.assert_masks(model))==53
                assert .21<lfwa.sparsity(model)['model_sparsity']<.22
                del model,ck
        write(root/'preflight_complete.json',{'status':'passed','conditions':8,'new_step13_conditions':3,'time':now()});return
    assert (root/'preflight_complete.json').exists()
    for c in m['conditions']:
        verify(root)
        write(root/'status.json',{'status':'running','condition':c['name'],'stage':'step13','updated_at':now()})
        source=first_step(root,c,False)
        trajroot=root/'trajectories'/c['name']
        args=SimpleNamespace(run_dir=trajroot,end_step=m['end_step_safety_limit'],alpha=c['alpha'],gamma=c['gamma'],
             cap_fraction=None,keep_per_iter=.975,layer_floor=.03,seed=42,batch=256,retrain_epochs=5,retrain_lr=c['lr'],stop_batch=10000)
        existing=trajectory.completed_steps(trajroot/'policy',13,args.end_step)
        source_ck=trajectory.trusted_torch_load(source)
        current=existing[-1]['sparsity'] if existing else source_ck['sparsity']
        del source_ck
        # Restore endpoint markers if a prior process stopped after the iteration commit.
        for row in existing:
            for target in m['endpoints']:
                dest=root/'endpoints'/c['name']/f'whole{round(target*100)}.json'
                if row['sparsity']['model_sparsity']>=target and not dest.exists():
                    write(dest,{**row,'target_whole_sparsity':target,
                       'evaluation_dir':str(trajroot/'policy'/f"iteration_{row['iteration']:03d}"/'post_retrain')})
        for iteration in range(14+len(existing),args.end_step+1):
            pending=[t for t in m['endpoints'] if current['model_sparsity']<t]
            if not pending:break
            target=pending[0]
            active=current['target_weights']-current['target_removed']
            gap=math.ceil(target*current['model_parameters'])-current['model_removed']
            desired=min(round(active*.025),gap)
            assert desired>0
            args.keep_per_iter=1-desired/active
            write(root/'status.json',{'status':'running','condition':c['name'],'iteration':iteration,
                 'target':target,'current_whole_sparsity':current['model_sparsity'],'updated_at':now()})
            write(trajroot/'budget_plans'/f'iteration_{iteration:03d}.json',{'before':current,'target':target,
                  'nominal_removal':desired,'keep_per_iter':args.keep_per_iter})
            result=trajectory.run_arm(args,'policy',source,13,frames,metadata,stop_after=iteration)
            row=result[-1];current=row['sparsity']
            assert row['selection']['remove_target']==desired
            assert row['selection']['budget_deficit']==0,'Removal floor/protection blocked budget; review required'
            if current['model_sparsity']>=target:
                assert current['model_sparsity']-target<.002,'Endpoint rounding exceeds0.2 percentage points'
                write(root/'endpoints'/c['name']/f'whole{round(target*100)}.json',{**row,'target_whole_sparsity':target,
                     'evaluation_dir':str(trajroot/'policy'/f'iteration_{iteration:03d}'/'post_retrain')})
            summarize(root,m)
        assert current['model_sparsity']>=.7,'Safety iteration limit reached'
        write(root/'condition_complete'/f"{c['name']}.json",{'status':'complete','whole_sparsity':current['model_sparsity'],'time':now()})
    summarize(root,m)
    write(root/'complete.json',{'status':'complete','time':now(),'conditions':8,'endpoints_per_condition':2})
    write(root/'status.json',{'status':'complete','updated_at':now()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--dry-run',action='store_true')
    cli=parser.parse_args();root=cli.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    if cli.dry_run:os.environ['CUDA_VISIBLE_DEVICES']=''
    os.chdir(ROOT)
    try:
        if cli.prepare:build(root)
        else:run(root,cli.dry_run)
    except Exception as error:
        write(root/'failure.json',{'error':repr(error),'time':now(),'dry_run':cli.dry_run,'prepare':cli.prepare})
        raise
