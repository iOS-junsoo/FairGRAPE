#!/usr/bin/env python3
"""Compare same-mask five-epoch LFWA runs that differ only in learning rate."""
import json,hashlib
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'lfwa_diagnostics/alpha0p7_iter13_max_lr1e4_seed42_20260917'
OLD=ROOT/'lfwa_diagnostics/alpha0p7_iter13_cap10_seed42_20260917_105048/run'
RANK=ROOT/'lfwa_diagnostics/alpha0p7_iter13_rank_norm_seed42_20260917/run'

def read(p):return json.loads(p.read_text())

def main():
    assert read(NEW/'run/complete.json')['status']=='complete'
    assert read(NEW/'run/post_prune_initial_state_verification.json')['all_tensor_values_exactly_equal']
    oldmeta,newmeta=read(OLD/'diagnostic_run.json'),read(NEW/'run/diagnostic_run.json')
    assert oldmeta['source_sha256']==newmeta['source_sha256']
    for field in ['source_checkpoint_sha256','source_channel_csv_sha256','iteration','alpha','seed','batch','retrain_epochs','layer_floor','keep_per_iter','normalization','protection_ratio']:
        assert oldmeta['plan'][field]==newmeta['plan'][field],field
    assert oldmeta['plan']['retrain_lr']==1e-5 and newmeta['plan']['retrain_lr']==1e-4
    paths={'max_lr1e5':OLD/'baseline','max_lr1e4':NEW/'run/max_lr1e4','rank_lr1e5':RANK/'rank_shared'}
    metrics={a:read(p/'post_retrain/summary.json') for a,p in paths.items()}
    for key in ['accuracy','ba','eo','model_sparsity']:
        assert abs(read(paths['max_lr1e5']/'post_prune/summary.json')[key]-read(paths['max_lr1e4']/'post_prune/summary.json')[key])<1e-10,key
    keys=['accuracy','ba','eo','model_sparsity','constant_prediction_attributes','near_constant_99pct_attributes']
    delta={a:{k:metrics['max_lr1e4'][k]-m[k] for k in keys} for a,m in metrics.items() if a!='max_lr1e4'}
    primary=delta['max_lr1e5']
    gate={'ba':primary['ba']>=.005,'accuracy':primary['accuracy']>=-.002,'near_constant':primary['near_constant_99pct_attributes']<=0}
    out=NEW/'analysis';out.mkdir(exist_ok=True)
    pd.DataFrame([{'arm':a,**{k:m[k] for k in keys}} for a,m in metrics.items()]).to_csv(out/'metrics.csv',index=False)
    result={'metrics':metrics,'lr1e4_minus_comparators':delta,'same_post_prune_tensors':True,'shared_implementation_hashes_identical':True,
            'utility_resource_gate':{'passes':all(gate.values()),'criteria':gate},
            'scope':'Same-mask LR contrast is primary. Rank reference has a different mask. Five epochs per arm, no final8+1+1. Single source and seed, validation only.'}
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    source=read(OLD/'before_prune/summary.json')
    for a,color,label in [('max_lr1e5','#bc4b38','Same max mask, LR1e-5'),('max_lr1e4','#286bb0','Same max mask, LR1e-4')]:
        frame=pd.read_csv(paths[a]/'epochs.csv');frame=frame[frame.phase=='val'];assert len(frame)==5
        post=read(paths[a]/'post_prune/summary.json')
        for ax,k in zip(axes,['accuracy','ba','eo']):ax.plot([-1,0,1,2,3,4,5],[source[k],post[k]]+frame[k].tolist(),'-o',label=label,color=color,markersize=4)
    for ax,k in zip(axes,['accuracy','ba','eo']):
        ax.set_ylabel(k);ax.set_xticks([-1,0,1,2,3,4,5],['Source','Pruned','1','2','3','4','5']);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('LFWA step13: exact same starting tensors and masks; five epochs only')
    for ext in ['png','pdf']:fig.savefig(out/f'same_mask_lr.{ext}',dpi=180)
    plt.close(fig)
    lines=['**동일 마스크·동일5epoch: 학습률 대조 결과**','','| 조건 | Accuracy | BA | EO↓ | 근상수 속성 |','|---|---:|---:|---:|---:|']
    for a,m in metrics.items():lines.append(f"| {a} | {m['accuracy']:.6f} | {m['ba']:.6f} | {m['eo']:.6f} | {m['near_constant_99pct_attributes']} |")
    lines+=['','주 비교는 max 두 학습률이며 rank는 다른 마스크의 보조 참고다. 추가8+1+1은 없다.','',f"사전 성능 진행 기준: {all(gate.values())}. EO 성공과 별도로 판단한다."]
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'delta':delta,'gate':gate},indent=2))

if __name__=='__main__':main()
