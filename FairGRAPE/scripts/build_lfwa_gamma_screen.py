#!/usr/bin/env python3
"""CPU archived-score preparation for fixed-LR max-normalized gamma screening."""
import argparse,ast,collections,copy,csv,datetime,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_lfwa_channel_cap_diagnostic import read_plan,write_json,sha256
BLOCKS=[f'features.{i}' for i in range(1,18)]


def selection(plan,device='cpu'):
    import torch
    with open(plan['source_channel_csv']) as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        row['ch']=int(row['ch']);row['weights']=int(row['weights'])
    weights={(r['block'],r['ch']):r['weights'] for r in rows}
    code=(ROOT/'prune.py').read_text()
    node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='_compute_protected_channels')
    scope={'_get_impt_type2_block_layer_names':lambda b:(b,None,None),
           '_count_channel_weights':lambda model,b,c:weights[b,c]}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'prune.py')+'::_compute_protected_channels','exec'),scope)
    perf={};candidates=[];original={};active={};boundaries={}
    for block in BLOCKS:
        part=[r for r in rows if r['block']==block]
        assert [r['ch'] for r in part]==list(range(len(part)))
        p=torch.tensor([float(r['perf_raw']) for r in part],dtype=torch.float32,device=device)
        f=torch.tensor([float(r['phi_raw']) for r in part],dtype=torch.float32,device=device)
        perf[block]=p;original[block]=len(part);active[block]=sum(r['weights']>0 for r in part)
        scores=plan['alpha']*(p/(p.max()+1e-12))-(1-plan['alpha'])*(f/(f.max()+1e-12))
        for r,s in zip(part,scores):
            if r['weights']>0:candidates.append(((block,r['ch']),float(s),r['weights']))
    protected,counts,active_counts=scope['_compute_protected_channels'](None,perf,plan['protection_ratio'])
    historical={(r['block'],r['ch']) for r in rows if r['status']=='protected'}
    assert historical<=protected
    for b in BLOCKS:
        values=sorted([float(perf[b][c]) for c in range(original[b]) if weights[b,c]>0],reverse=True)
        k=counts[b];boundaries[b]={'protected_count':k,'active':len(values),'tie_at_boundary':values[k-1]==values[k] if 0<k<len(values) else False}
    target=round(sum(weights.values())*(1-plan['keep_per_iter']))
    selected=[];removed=0;per_layer=collections.Counter()
    for key,score,cost in sorted(candidates,key=lambda item:item[1]):
        if removed>=target:break
        if key in protected:continue
        b,c=key
        if active[b]-per_layer[b]-1<math.ceil(original[b]*plan['layer_floor']):continue
        selected.append(key);removed+=cost;per_layer[b]+=1
    assert removed>=target and not set(selected)&protected
    return {'selected_channels':selected,'expected_removed_weights':removed,'nominal_removed_weights':target,
            'protected_channels':sorted(protected),'protection_boundaries':boundaries,
            'removed_channels_by_layer':dict(per_layer),'device':device}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    out=args.output_dir.resolve();out.mkdir(exist_ok=False)
    base=read_plan(ROOT/'lfwa_diagnostics/alpha0p7_iter13_cap10_seed42_20260917_105048/plan.json')
    baseline=ROOT/'lfwa_diagnostics/alpha0p7_iter13_max_lr1e4_seed42_20260917/run'
    assert json.loads((baseline/'complete.json').read_text())['status']=='complete'
    original=selection(base)
    expected=next(a for a in base['arms'] if a['name']=='baseline')
    assert set(original['selected_channels'])==set(expected['selected_channels'])
    assert original['expected_removed_weights']==32858
    records=[];audit={};seen={}
    for alpha in [.5,.7]:
        for gamma in [.005,.05,.15]:
            name='a'+str(alpha).replace('.','p')+'_g'+str(gamma).replace('.','p')
            plan=copy.deepcopy(base);plan.update(alpha=alpha,protection_ratio=gamma,retrain_lr=1e-4,
                diagnostic='same_source_fixed_lr_max_gamma_screen')
            chosen=selection(plan);key=tuple(sorted(chosen['selected_channels']))
            arm={'name':name,'reference_selection':False,'strict_channel_cap_fraction':None,
                 'selected_channels':chosen['selected_channels'],'expected_removed_weights':chosen['expected_removed_weights']}
            plan['arms']=[arm];plan.pop('layer_channel_cap_counts',None)
            plan['protected_channel_ids']=chosen['protected_channels']
            plan['screen_condition']={'alpha':alpha,'gamma':gamma,'retrain_lr':1e-4,'normalization':'max',
                'final_additional_epochs':0,'selection_replay':'float32 archived raw CSV; original max control reproduced exactly',
                'raw_csv_precision_limit':True,'gpu_selection_replay_required':True}
            path=out/'plans'/name/'plan.json';write_json(path,plan)
            reuse=None
            if alpha==.7 and gamma==.005:
                assert key==tuple(sorted(expected['selected_channels']))
                reuse={'run_dir':str(baseline),'arm':'max_lr1e4','checkpoint_sha256':sha256(baseline/'max_lr1e4/retrained.pt')}
            elif key in seen:reuse={'deduplicated_condition':seen[key]}
            seen[key]=name
            records.append({'name':name,'alpha':alpha,'gamma':gamma,'plan':str(path),'plan_sha256':sha256(path),'reuse':reuse})
            audit[name]=chosen
    sources=['scripts/build_lfwa_gamma_screen.py','scripts/run_lfwa_gamma_screen.py','scripts/run_lfwa_channel_cap_diagnostic.py',
             'prune.py','lfwa.py','lfwa_experiment.py','train_and_val.py','dataset.py','util.py','config.py','adversarial_debiasing.py']
    hashes={f:sha256(ROOT/f) for f in sources}
    for f in sources:
        import shutil
        destination=out/'source'/f;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,destination)
    write_json(out/'manifest.json',{'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'conditions':records,'source_sha256':hashes,'normalization':'max','retrain_lr':1e-4,'seed':42,'epochs':5,
        'control_replayed':True,'final_additional_epochs':0,'source_checkpoint_sha256':base['source_checkpoint_sha256'],
        'next_action':'Review paired alpha/gamma utility and EO before any full dense-start sweep; no automatic promotion based only on accuracy.'})
    write_json(out/'selection_audit.json',audit)
    print(json.dumps({'conditions':[{k:v for k,v in r.items() if k!='plan_sha256'} for r in records]},indent=2))

if __name__=='__main__':main()
