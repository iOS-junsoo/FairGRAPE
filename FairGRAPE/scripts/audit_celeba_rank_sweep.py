#!/usr/bin/env python3
"""Audit archived CelebA rank summaries without evaluating or modifying models."""
import json,re,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'retrain_epoch_results/Activate_diff/⭐️옵션3(상한선 3% +보호구역 0.5%)/CelebA'
SOURCE=BASE/'정규화: 랭크/42'
OUT=ROOT/'lfwa_diagnostics/celeba_rank_sweep_audit_20260917'
OUT.mkdir(exist_ok=True)

def match(pattern,text):
    m=re.search(pattern,text)
    if not m:raise ValueError(pattern)
    return m.group(1)

rows=[]; attributes=[]; hashes={}; final_paths={}
for alpha in ['0.3','0.5','0.7','0.9']:
    files=sorted((SOURCE/alpha).rglob('BEST_MODEL_SUMMARY*'))
    grouped={}
    for path in files:
        text=path.read_text(); step=int(match(r'Prune Iteration:\s*(\d+)',text))
        assert match(r'Norm:\s*(\w+)',text)=='rank'
        assert float(match(r'Alpha:\s*([\d.]+)',text))==float(alpha)
        assert match(r'Seed:\s*(\d+)',text)=='42'
        assert match(r'GRL Enabled:\s*(\w+)',text)=='False'
        row={'alpha':float(alpha),'iteration':step,'timestamp':match(r'(\d{8}_\d{6})',path.name),
             'epochs':int(match(r'총 재학습 에폭 수:\s*(\d+)',text)),
             'accuracy':float(match(r'Best Accuracy:\s*([\d.]+)',text)),
             'ba':float(match(r'Best BA:\s*([\d.]+)',text)),
             'eo':float(match(r'Best EO:\s*([\d.]+)',text)),
             'model_sparsity_percent':float(match(r'전체 파라미터:.*?제거율 ([\d.]+)',text)),
             'path':str(path)}
        grouped.setdefault(step,[]).append(row)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(grouped)==set(range(1,23))
    for step,group in grouped.items():
        group.sort(key=lambda x:x['timestamp'])
        assert [r['epochs'] for r in group]==([5,8,1,1] if step==22 else [5])
        for i,row in enumerate(group):
            row['stage']=['prune_retrain','final_8epochs','final_1epoch_a','final_1epoch_b'][i]
            rows.append(row)
    latest=max((SOURCE/alpha).rglob('Total_info*'),key=lambda p:match(r'output_(\d+_\d+)',p.name))
    text=latest.read_text();final_paths[alpha]=str(latest)
    final=grouped[22][-1]
    for key,pattern in [('accuracy',r'전체 클래스 평균 정확도: ([\d.]+)'),('ba',r'전체 클래스 평균 Balanced Accuracy: ([\d.]+)'),('eo',r'전체 클래스 평균 EO: ([\d.]+)')]:
        assert final[key]==float(match(pattern,text))
    cells={}
    for name,g,c,correct,total in re.findall(r'^(.+?) \(group=(\d), class=(\d)\): (\d+)/(\d+) =',text,re.M):
        cells.setdefault(name,{})[int(g),int(c)]=(int(correct),int(total))
    assert len(cells)==39
    for name,cell in cells.items():
        assert len(cell)==4
        rates={k:(correct/total if total else np.nan) for k,(correct,total) in cell.items()}
        eo=(abs(rates[0,1]-rates[1,1])+abs(rates[0,0]-rates[1,0]))/2
        n=sum(total for correct,total in cell.values())
        pos=sum(cell[g,1][0]+cell[g,0][1]-cell[g,0][0] for g in [0,1])
        attributes.append({'alpha':float(alpha),'attribute':name,'accuracy':sum(correct for correct,total in cell.values())/n,
                           'ba':np.nanmean(list(rates.values())),'eo':eo,'n':n,'minimum_cell':min(total for correct,total in cell.values()),
                           'predicted_positive_fraction':pos/n,'constant':pos in [0,n],
                           'near_constant':pos/n<=.01 or pos/n>=.99})
    hashes[str(latest)]=hashlib.sha256(latest.read_bytes()).hexdigest()
frame=pd.DataFrame(rows).sort_values(['alpha','iteration','timestamp']);frame.to_csv(OUT/'all_summaries.csv',index=False)
attrs=pd.DataFrame(attributes);attrs.to_csv(OUT/'final_attributes_recomputed.csv',index=False)
checks={}
for alpha in [.3,.5,.7,.9]:
    part=attrs[attrs.alpha==alpha];summary=frame[(frame.alpha==alpha)&(frame.stage=='final_1epoch_b')].iloc[0]
    for key in ['accuracy','ba','eo']:
        assert abs(part[key].mean()-summary[key])<=.00005001,(alpha,key,part[key].mean(),summary[key])
    checks[str(alpha)]={'eo_valid_attributes':int(part.eo.notna().sum()),'undefined_eo_attributes':part.loc[part.eo.isna(),'attribute'].tolist(),
                       'constant_attributes':part.loc[part.constant,'attribute'].tolist(),'near_constant_attributes':part.loc[part.near_constant,'attribute'].tolist(),
                       'metrics':{key:float(part[key].mean()) for key in ['accuracy','ba','eo']}}
comparison=attrs[attrs.alpha==.3].set_index('attribute').add_suffix('_03').join(attrs[attrs.alpha==.9].set_index('attribute').add_suffix('_09'))
for key in ['accuracy','ba','eo']:comparison[key+'_03_minus_09']=comparison[key+'_03']-comparison[key+'_09']
comparison.to_csv(OUT/'final_alpha03_vs09_attributes.csv')
common_rare=comparison.minimum_cell_03<10
extra={'alpha03_vs09':{'eo_change_all_defined':float(comparison.eo_03_minus_09.mean()),
                     'eo_change_cells_at_least_10':float(comparison.loc[~common_rare,'eo_03_minus_09'].mean()),
                     'n_attributes_cells_at_least_10':int((~common_rare).sum())},
       'checks':checks,'final_prediction_files':final_paths,
       'precision':'Summary metrics rounded to four decimals. Final metrics independently recomputed from integer group/class counts.',
       'scope':'Historical single-seed audit, not new GPU experiments. No checkpoint/source/environment manifest in supplied rank folders. Legacy test phase corresponds to validation in current main_test.py; historical split hash not available.'}
(OUT/'audit.json').write_text(json.dumps(extra,indent=2)+'\n');(OUT/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(13.8,4.1),constrained_layout=True)
for stage,label,style in [('prune_retrain','After pruning + 5 epochs','o--'),('final_1epoch_b','After final 8+1+1 epochs','s-')]:
    part=frame[(frame.iteration==22)&(frame.stage==stage)].sort_values('alpha')
    for ax,key in zip(axes,['accuracy','ba','eo']):ax.plot(part.alpha,part[key],style,label=label)
for ax,label in zip(axes,['Accuracy','Group/class balanced accuracy','EO gap (lower is better)']):
    ax.set_xlabel('Alpha');ax.set_xticks([.3,.5,.7,.9]);ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('CelebA rank, seed42: whole-model sparsity about 69.09%')
for ext in ['png','pdf']:fig.savefig(OUT/f'alpha_before_after_final_retraining.{ext}',dpi=180)
plt.close(fig)
print(json.dumps(extra,indent=2))
