#!/usr/bin/env python3
"""Approved LR-only experiment: alpha0.7 then0.5, gamma0.005, LR1e-4,91iters."""
import argparse,ast,copy,inspect,json,os,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import run_lfwa_protection15_91 as base
from run_lfwa_protection15_91 import read,write,sha,now
PREVIOUS=ROOT/'lfwa_diagnostics/protection15_alpha05_07_91iter_lr1e5_20260918'

def verify(root):
 m=base.verify(root)
 assert sha(m['previous_results'])==m['previous_results_sha256']
 for p,h in m['reused_checkpoints_sha256'].items():assert sha(p)==h
 for c in m['conditions']:
  assert c['arguments']['retrain_lr']==1e-4 and c['arguments']['retrain_iter']==5
 assert m['protection_ratio']==.005
 return m

def prepare(root):
 assert not (root/'manifest.json').exists()
 prior=base.verify(PREVIOUS);assert read(PREVIOUS/'complete.json')['status']=='complete'
 conditions=[]
 for alpha in [.7,.5]:
  old=next(c for c in prior['conditions'] if c['alpha']==alpha)
  args=dict(old['arguments'],retrain_lr=1e-4,lfwa_run_dir=str(root/'runs'/old['name']))
  c={k:old[k] for k in ['name','alpha','baseline','baseline_manifest_sha256','checkpoint','checkpoint_sha256']}
  c['arguments']=args;conditions.append(c)
 result_file=PREVIOUS/'analysis/results.json';rows=read(result_file)
 assert len(rows)==8
 for r in rows:
  r['lr']=1e-5;r['setting']='original' if r['gamma']==.005 else 'protection15'
 files=list(prior['source_sha256'])+['scripts/run_lfwa_lr1e4_gamma005_91.py']
 hashes={f:sha(ROOT/f) for f in files}
 for f in files:
  dst=root/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
 write(root/'manifest.json',{'created_at':now(),'conditions':conditions,'source_sha256':hashes,
  'protection_ratio':.005,'retrain_lr':1e-4,'iterations':91,'retrain_epochs_per_iteration':5,'final_extra_epochs':0,
  'previous_results':str(result_file),'previous_results_sha256':sha(result_file),'reused_rows':rows,
  'reused_checkpoints_sha256':{r['checkpoint']:sha(r['checkpoint']) for r in rows},
  'source':'same dense SHA256 as previous runs','seed_policy':prior['seed_policy'],
  'comparison':'LR-only vs original at same gamma0.005; protection-only reference gamma0.15/LR1e-5. Nearest actual whole60 and iteration91.',
  'scope':'Only two new runs, alpha0.7 then0.5; no gamma0.15/LR1e-4 combined intervention, so interaction is not identified.'})
 print('PREPARED',root,flush=True)

def preflight(root):
 m=verify(root)
 # The validated training adapter differs only in its three gamma constants.
 expected=inspect.getsource(base.train_one)
 assert expected.count('.15')==3
 expected=expected.replace('.15','.005')
 assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(inspect.getsource(train_one)))
 import torch,lfwa,lfwa_experiment
 torch.set_num_threads(2)
 _,metadata=lfwa.load_frames(ROOT/'csv/LFWA.csv',ROOT/'Images/LFWA')
 model,ck=lfwa.load_checkpoint_model(m['conditions'][0]['checkpoint'],metadata,'cpu')
 assert lfwa.sparsity(model)['model_removed']==0
 changes={}
 for c in m['conditions']:
  old=read(Path(c['baseline'])/'run.json');assert old['experiment_metadata']==metadata
  args=c['arguments'];diff={k:{'old':old['arguments'][k],'new':v} for k,v in args.items() if old['arguments'][k]!=v}
  assert set(diff)<= {'retrain_lr','lfwa_run_dir','checkpoint','no_retrain','defer_test'}
  assert diff['retrain_lr']=={'old':1e-5,'new':1e-4}
  assert old['alpha']==c['alpha'] and old['protection_ratio']==.005 and old['normalization']=='max'
  assert args['no_retrain'] and args['defer_test'] and args['keep_per_iter']==.975
  assert lfwa_experiment.planned_iterations(.975,args['prune_rate'])==91
  changes[c['name']]=diff
 write(root/'preflight_complete.json',{'status':'passed','time':now(),'argument_differences':changes,
  'training_adapter_identical_except_gamma_constants':True,'same_dense_sha256':m['conditions'][0]['checkpoint_sha256'],
  'comparison_checkpoints_verified':8,'split_metadata_matches':True})
 print('PREFLIGHT_PASSED',flush=True)

def train_one(root,name):
 m=verify(root);assert (root/'preflight_complete.json').exists()
 c=next(c for c in m['conditions'] if c['name']==name)
 import torch,lfwa,lfwa_experiment,prune,random,numpy as np
 torch.set_num_threads(2)
 original_config=lfwa_experiment._configure
 def configure(args,directory,metadata):
  original_config(args,directory,metadata);prune.IMPT2_PROTECTION_RATIO=.005
  assert prune.IMPT2_NORM=='max' and prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER==.03 and prune.IMPT2_LAYER_CAP_MULTIPLIER is None
 lfwa_experiment._configure=configure
 original_save=lfwa.save_checkpoint
 def save(model,path,metadata,**details):
  ck=original_save(model,path,metadata,**details)
  if details.get('stage')=='prune' and Path(path).name.startswith('prune_'):
   assert prune.IMPT2_PROTECTION_RATIO==.005 and prune.IMPT_TYPE2_ALPHA==c['alpha']
   i=details['iteration'];dest=root/'progress'/name
   write(dest/f'iteration_{i:03d}.json',{'iteration':i,'checkpoint':str(path),'sparsity':ck['sparsity'],
       'validation':ck['validation'],'seconds':details.get('seconds'),'time':now()})
   # State capture does not consume RNG or add evaluations to the original training path.
   rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all(),'iteration':i}
   p=Path(path).with_suffix('.rng.pt');q=p.with_suffix('.part');torch.save(rng,q);q.replace(p)
   write(root/'status.json',{'status':'running','condition':name,'completed_iteration':i,'whole_sparsity':ck['sparsity']['model_sparsity'],'time':now()})
  return ck
 lfwa.save_checkpoint=save
 lfwa_experiment.run(SimpleNamespace(**c['arguments']))
 assert read(Path(c['arguments']['lfwa_run_dir'])/'run.json')['protection_ratio']==.005


def compare(root):
 m=verify(root)
 import torch,lfwa,pandas as pd
 torch.set_num_threads(2)
 frames,metadata=lfwa.load_frames(ROOT/'csv/LFWA.csv',ROOT/'Images/LFWA');rows=copy.deepcopy(m['reused_rows'])
 for c in m['conditions']:
  records=[read(root/'progress'/c['name']/f'iteration_{i:03d}.json') for i in range(1,92)]
  points={'near60':min(records,key=lambda r:(abs(r['sparsity']['model_sparsity']-.6),r['iteration'])),'iter91':records[-1]}
  for label,row in points.items():
   dest=root/'analysis/evaluations'/c['name']/label
   model,ck=lfwa.load_checkpoint_model(row['checkpoint'],metadata,'cuda:0')
   assert ck['iteration']==row['iteration'] and ck['stage']=='prune'
   summary=lfwa.evaluate_model(model,frames['val'],dest,256)
   preds=pd.read_csv(dest/'predictions.csv');rates=preds[[n+'_pred' for n in lfwa.TARGETS]].mean()
   summary['near_constant_99pct_attributes']=int(((rates<=.01)|(rates>=.99)).sum())
   summary['constant_prediction_attributes']=int(((rates==0)|(rates==1)).sum())
   write(dest/'summary.json',summary)
   rows.append({'alpha':c['alpha'],'gamma':.005,'lr':1e-4,'setting':'lr1e4','point':label,
    'iteration':row['iteration'],'checkpoint':row['checkpoint'],'evaluation':str(dest),**summary})
   del model,ck;torch.cuda.empty_cache();write(root/'analysis/results.json',rows)
 order={'original':0,'protection15':1,'lr1e4':2}
 rows.sort(key=lambda r:(r['point'],r['alpha'],order[r['setting']]))
 pd.DataFrame(rows).to_csv(root/'analysis/metrics.csv',index=False)
 lines=['# LFWA 학습률 단독 변경: α0.7·0.5,91iter','','동일 dense, max, 단계당5epoch. 추가8+1+1 없음. 기본 설정 대비 보호 비율만 변경한 결과와 학습률만 변경한 결과를 각각 비교한다.','',
 '| 지점 | α | 설정 | 보호 | LR | 단계 | 전체 희소도 | Accuracy | BA | EO↓ | 근상수 속성 |',
 '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['point']} | {r['alpha']} | {r['setting']} | {r['gamma']:.1%} | {r['lr']:g} | {r['iteration']} | {r['model_sparsity']:.4%} | {r['accuracy']:.6f} | {r['ba']:.6f} | {r['eo']:.6f} | {r['near_constant_99pct_attributes']} |")
 lookup={(r['point'],r['alpha'],r['setting']):r for r in rows};contrasts=[]
 for point in ['near60','iter91']:
  for a in [.5,.7]:
   original=lookup[point,a,'original']
   for setting in ['protection15','lr1e4']:
    candidate=lookup[point,a,setting]
    delta={k:candidate[k]-original[k] for k in ['accuracy','ba','eo','model_sparsity','near_constant_99pct_attributes']}
    contrasts.append({'point':point,'alpha':a,'comparison':setting+'_minus_original','delta':delta})
    lines.append(f"\n{point}, α={a}, {setting}−original: ΔAcc={delta['accuracy']:+.6f}, ΔBA={delta['ba']:+.6f}, ΔEO={delta['eo']:+.6f}.")
  for setting in order:
   low,high=lookup[point,.5,setting],lookup[point,.7,setting]
   delta={k:low[k]-high[k] for k in ['accuracy','ba','eo','model_sparsity','near_constant_99pct_attributes']}
   contrasts.append({'point':point,'setting':setting,'comparison':'alpha0p5_minus_alpha0p7','delta':delta})
 lines+=['','LR1e-4·보호15% 결합 조건은 실행하지 않았으므로 두 변경의 상호작용은 이번 실험으로 추정하지 않는다. 단일seed validation 결과이며 EO는 분류 성능·근상수 속성과 함께 해석한다. 기존 과거 실험은 소스 전체 스냅샷이 없어 당시 코드의 비트 단위 동일성을 주장하지 않는다.']
 write(root/'analysis/contrasts.json',contrasts);(root/'analysis/report.md').write_text('\n'.join(lines)+'\n')

def queue(root):
 m=verify(root);assert (root/'preflight_complete.json').exists()
 for c in m['conditions']:
  run=Path(c['arguments']['lfwa_run_dir'])
  if (run/'complete.json').exists():
   info=read(run/'run.json');assert info['arguments']==c['arguments'] and info['protection_ratio']==.005
   assert (root/'progress'/c['name']/'iteration_091.json').exists();continue
  assert not (run/'run.json').exists(),'Interrupted run requires RNG-aware recovery; no overwrite'
  write(root/'status.json',{'status':'starting','condition':c['name'],'time':now()})
  with (root/f"{c['name']}.log").open('ab',buffering=0) as log:
   subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--train',c['name']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 write(root/'status.json',{'status':'evaluating_comparisons','time':now()});compare(root)
 write(root/'complete.json',{'status':'complete','time':now(),'alphas':[.7,.5],'new_runs':2,'report':str(root/'analysis/report.md')})
 write(root/'status.json',{'status':'complete','time':now()})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--train');a=p.parse_args()
 root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);os.chdir(ROOT)
 if a.preflight:os.environ['CUDA_VISIBLE_DEVICES']=''
 try:
  if a.prepare:prepare(root)
  elif a.preflight:preflight(root)
  elif a.train:train_one(root,a.train)
  else:queue(root)
 except Exception as error:
  write(root/('failure_'+a.train+'.json' if a.train else 'failure.json'),{'error':repr(error),'time':now()});raise
