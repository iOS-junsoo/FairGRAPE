#!/usr/bin/env python3
"""Extend the frozen gamma15/LR1e-5/91iter experiment to alpha .3 and .9."""
import argparse,copy,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import run_lfwa_protection15_91 as base
PREVIOUS=ROOT/'lfwa_diagnostics/protection15_alpha05_07_91iter_lr1e5_20260918'

def prepare(root):
 assert not (root/'manifest.json').exists()
 previous=base.verify(PREVIOUS)
 assert base.read(PREVIOUS/'complete.json')['status']=='complete'
 source=previous['conditions'][0]
 template=source['arguments'];conditions=[]
 for a in [.3,.9]:
  name='alpha'+str(a).replace('.','p')
  args=dict(template,impt2_alpha=a,lfwa_run_dir=str(root/'runs'/name))
  assert args['retrain_lr']==1e-5 and args['retrain_iter']==5 and args['no_retrain'] and args['defer_test']
  c={'name':name,'alpha':a,'arguments':args,'checkpoint':source['checkpoint'],'checkpoint_sha256':source['checkpoint_sha256'],
     'baseline':str(PREVIOUS/'runs/alpha0p5'),'baseline_manifest_sha256':base.sha(PREVIOUS/'runs/alpha0p5/run.json'),
     'baseline_field_role':'Verified argument template from completed gamma15 run; not an alpha-matched gamma0.5 comparator.'}
  conditions.append(c)
 old_results=PREVIOUS/'analysis/results.json';old_rows=[r for r in base.read(old_results) if r['gamma']==.15]
 assert {(r['alpha'],r['point']) for r in old_rows}=={(a,p) for a in [.5,.7] for p in ['near60','iter91']}
 files=list(previous['source_sha256'])+['scripts/run_lfwa_protection15_alpha03_09.py']
 hashes={f:base.sha(ROOT/f) for f in files}
 for f in files:
  dst=root/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
 base.write(root/'manifest.json',{'created_at':base.now(),'conditions':conditions,'source_sha256':hashes,
  'protection_ratio':.15,'retrain_lr':1e-5,'iterations':91,'retrain_epochs_per_iteration':5,'final_extra_epochs':0,
  'seed':42,'source':'same dense SHA256 as completed alpha0.5/0.7 runs','previous_root':str(PREVIOUS),
  'previous_results':str(old_results),'previous_results_sha256':base.sha(old_results),
  'reused_rows':old_rows,'reused_checkpoints_sha256':{r['checkpoint']:base.sha(r['checkpoint']) for r in old_rows},
  'comparison':'gamma15 alpha0.3/0.5/0.7/0.9 at nearest whole sparsity0.60 and iteration91; no final8+1+1',
  'seed_policy':previous['seed_policy']})
 print('PREPARED',root,flush=True)

def verify(root):
 m=base.verify(root)
 assert base.sha(m['previous_results'])==m['previous_results_sha256']
 for p,h in m['reused_checkpoints_sha256'].items():assert base.sha(p)==h
 return m

def preflight(root):
 m=verify(root)
 import torch,lfwa,lfwa_experiment
 torch.set_num_threads(2)
 _,metadata=lfwa.load_frames(ROOT/'csv/LFWA.csv',ROOT/'Images/LFWA')
 model,ck=lfwa.load_checkpoint_model(m['conditions'][0]['checkpoint'],metadata,'cpu')
 assert lfwa.sparsity(model)['model_removed']==0
 assert metadata==base.read(PREVIOUS/'runs/alpha0p5/run.json')['experiment_metadata']
 prior=base.read(PREVIOUS/'runs/alpha0p5/run.json')['arguments']
 for c in m['conditions']:
  assert {k for k,v in c['arguments'].items() if v!=prior[k]}=={'impt2_alpha','lfwa_run_dir'}
  assert lfwa_experiment.planned_iterations(c['arguments']['keep_per_iter'],c['arguments']['prune_rate'])==91
 base.write(root/'preflight_complete.json',{'status':'passed','time':base.now(),'only_argument_changes':['impt2_alpha','lfwa_run_dir'],
    'same_dense_sha256':m['conditions'][0]['checkpoint_sha256'],'metadata_matches_previous':True,'reused_comparison_checkpoints_verified':4})
 print('PREFLIGHT_PASSED',flush=True)

def compare(root):
 m=verify(root)
 import torch,lfwa,pandas as pd
 torch.set_num_threads(2)
 frames,metadata=lfwa.load_frames(ROOT/'csv/LFWA.csv',ROOT/'Images/LFWA');rows=copy.deepcopy(m['reused_rows'])
 for c in m['conditions']:
  records=[base.read(root/'progress'/c['name']/f'iteration_{i:03d}.json') for i in range(1,92)]
  points={'near60':min(records,key=lambda r:(abs(r['sparsity']['model_sparsity']-.6),r['iteration'])),'iter91':records[-1]}
  for label,row in points.items():
   dest=root/'analysis/evaluations'/c['name']/label
   model,ck=lfwa.load_checkpoint_model(row['checkpoint'],metadata,'cuda:0')
   assert ck['iteration']==row['iteration'] and ck['stage']=='prune'
   summary=lfwa.evaluate_model(model,frames['val'],dest,256)
   preds=pd.read_csv(dest/'predictions.csv');rates=preds[[n+'_pred' for n in lfwa.TARGETS]].mean()
   summary['near_constant_99pct_attributes']=int(((rates<=.01)|(rates>=.99)).sum())
   summary['constant_prediction_attributes']=int(((rates==0)|(rates==1)).sum())
   base.write(dest/'summary.json',summary)
   rows.append({'alpha':c['alpha'],'gamma':.15,'point':label,'iteration':row['iteration'],'checkpoint':row['checkpoint'],'evaluation':str(dest),**summary})
   del model,ck;torch.cuda.empty_cache()
   base.write(root/'analysis/results.json',rows)
 rows=sorted(rows,key=lambda r:(r['point'],r['alpha']))
 pd.DataFrame(rows).to_csv(root/'analysis/metrics.csv',index=False)
 lines=['# LFWA 보호15%: α0.3·0.5·0.7·0.9 비교','','동일 dense·max·LR1e-5·seed42·단계당5epoch·91iter. 추가8+1+1 없음. 기존 완료된 α0.5/0.7 결과를 해시 검증 후 재사용한다.','',
 '| 지점 | α | 단계 | 전체 희소도 | Accuracy | BA | EO↓ | 근상수 속성 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['point']} | {r['alpha']} | {r['iteration']} | {r['model_sparsity']:.4%} | {r['accuracy']:.6f} | {r['ba']:.6f} | {r['eo']:.6f} | {r['near_constant_99pct_attributes']} |")
 contrasts=[]
 for label in ['near60','iter91']:
  subset=[r for r in rows if r['point']==label]
  for low,high in zip(subset[:-1],subset[1:]):
   delta={k:low[k]-high[k] for k in ['accuracy','ba','eo','model_sparsity','near_constant_99pct_attributes']}
   contrasts.append({'point':label,'lower_alpha':low['alpha'],'higher_alpha':high['alpha'],'delta_lower_minus_higher':delta})
   lines.append(f"\n{label}, α{low['alpha']}−α{high['alpha']}: ΔAcc={delta['accuracy']:+.6f}, ΔBA={delta['ba']:+.6f}, ΔEO={delta['eo']:+.6f}, Δ근상수속성={delta['near_constant_99pct_attributes']:+d}.")
 lines+=['','α별 EO 감소는 BA와 근상수 예측 변화를 함께 해석한다. 단일seed의 validation 결과이며 유의성·일반화 또는 다른 희소도에서의 단조성을 주장하지 않는다.']
 (root/'analysis/report.md').write_text('\n'.join(lines)+'\n');base.write(root/'analysis/contrasts.json',contrasts)

def queue(root):
 m=verify(root);assert (root/'preflight_complete.json').exists()
 for c in m['conditions']:
  run=Path(c['arguments']['lfwa_run_dir'])
  if (run/'complete.json').exists():
   info=base.read(run/'run.json');assert info['arguments']==c['arguments'] and info['protection_ratio']==.15
   assert (root/'progress'/c['name']/'iteration_091.json').exists();continue
  assert not (run/'run.json').exists(),'Interrupted run requires RNG-aware recovery; no overwrite'
  base.write(root/'status.json',{'status':'starting','condition':c['name'],'time':base.now()})
  with (root/f"{c['name']}.log").open('ab',buffering=0) as log:
   subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--train',c['name']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 base.write(root/'status.json',{'status':'evaluating_comparisons','time':base.now()});compare(root)
 base.write(root/'complete.json',{'status':'complete','time':base.now(),'new_alphas':[.3,.9],'all_alphas':[.3,.5,.7,.9],'report':str(root/'analysis/report.md')})
 base.write(root/'status.json',{'status':'complete','time':base.now()})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--train');a=p.parse_args()
 root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);os.chdir(ROOT)
 if a.preflight:os.environ['CUDA_VISIBLE_DEVICES']=''
 try:
  if a.prepare:prepare(root)
  elif a.preflight:preflight(root)
  elif a.train:verify(root);base.train_one(root,a.train)
  else:queue(root)
 except Exception as error:
  base.write(root/('failure_'+a.train+'.json' if a.train else 'failure.json'),{'error':repr(error),'time':base.now()});raise
