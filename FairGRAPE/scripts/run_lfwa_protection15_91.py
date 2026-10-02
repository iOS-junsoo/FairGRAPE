#!/usr/bin/env python3
"""Two approved dense-start runs: alpha .5 then .7, gamma .15, LR1e-5,91iters."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASELINES={'.5':ROOT/'lfwa_runs/seed42_alpha0p5_20260916_022453/pruned',
           '.7':ROOT/'lfwa_runs/alpha0p7_seed42_no_final_20260916_143706'}
FILES=['scripts/run_lfwa_protection15_91.py','lfwa_experiment.py','lfwa.py','prune.py','train_and_val.py','dataset.py','util.py','config.py','adversarial_debiasing.py']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.part');q.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');q.replace(p)
def verify(root):
 m=read(root/'manifest.json')
 for f,h in m['source_sha256'].items():assert sha(ROOT/f)==h,f
 for c in m['conditions']:
  assert sha(c['checkpoint'])==c['checkpoint_sha256']
  assert sha(Path(c['baseline'])/'run.json')==c['baseline_manifest_sha256']
 return m

def prepare(root):
 assert not (root/'manifest.json').exists()
 conditions=[]
 for label,old in BASELINES.items():
  info=read(old/'run.json');args=info['arguments'];checkpoint=Path(args['checkpoint']).resolve()
  assert info['alpha']==float(label) and info['protection_ratio']==.005 and info['normalization']=='max'
  assert info['layer_floor']==.03 and info['layer_cap'] is None
  assert args['retrain_lr']==1e-5 and args['retrain_iter']==5 and args['keep_per_iter']==.975 and info['prune_iterations']==91
  name='alpha'+label.replace('.','0p');out=root/'runs'/name
  changed=dict(args,lfwa_run_dir=str(out),checkpoint=str(checkpoint),no_retrain=True,defer_test=True)
  conditions.append({'name':name,'alpha':float(label),'baseline':str(old),'baseline_manifest_sha256':sha(old/'run.json'),
    'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'arguments':changed})
 assert len({c['checkpoint_sha256'] for c in conditions})==1
 hashes={f:sha(ROOT/f) for f in FILES}
 for f in FILES:
  dst=root/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
 write(root/'manifest.json',{'created_at':now(),'conditions':conditions,'source_sha256':hashes,
  'protection_ratio':.15,'retrain_lr':1e-5,'iterations':91,'retrain_epochs_per_iteration':5,
  'final_extra_epochs':0,'source':'identical dense checkpoint','seed_policy':'original lfwa_experiment.run; setseed42 once per run, no per-iteration reseeding',
  'comparison':'nearest actual whole-model sparsity .60 and iteration91 (~.70); historical prune checkpoints only, no final8+1+1',
  'historical_limit':'Stored settings and checkpoint metadata checked; historical run.json has no complete source-code snapshot, so bitwise historical implementation equivalence is not asserted.'})
 print('PREPARED',root,flush=True)

def preflight(root):
 m=verify(root)
 import torch,lfwa,lfwa_experiment
 torch.set_num_threads(2)
 frames,meta=lfwa.load_frames(ROOT/'csv/LFWA.csv',ROOT/'Images/LFWA')
 index={}
 for c in m['conditions']:
  model,ck=lfwa.load_checkpoint_model(c['checkpoint'],meta,'cpu')
  assert lfwa.sparsity(model)['model_removed']==0
  assert meta==read(Path(c['baseline'])/'run.json')['experiment_metadata']
  assert lfwa_experiment.planned_iterations(.975,.9)==91
  rows=[]
  for i in range(1,92):
   p=Path(c['baseline'])/f'prune_{i:03d}.pt';ck=torch.load(p,map_location='cpu')
   lfwa.validate_checkpoint(ck,meta);assert ck['iteration']==i and ck['stage']=='prune'
   rows.append({'iteration':i,'checkpoint':str(p),'sha256':sha(p),'sparsity':ck['sparsity'],'validation':ck['validation']})
  index[c['name']]=rows
  del model,ck
 write(root/'baseline_checkpoint_index.json',index)
 write(root/'preflight_complete.json',{'status':'passed','time':now(),'same_dense_hash':m['conditions'][0]['checkpoint_sha256'],
  'historical_prune_checkpoints_checked':182,'metadata_matches':True})
 print('PREFLIGHT_PASSED',flush=True)

def train_one(root,name):
 m=verify(root);assert (root/'preflight_complete.json').exists()
 c=next(c for c in m['conditions'] if c['name']==name)
 import torch,lfwa,lfwa_experiment,prune,random,numpy as np
 torch.set_num_threads(2)
 original_config=lfwa_experiment._configure
 def configure(args,directory,metadata):
  original_config(args,directory,metadata);prune.IMPT2_PROTECTION_RATIO=.15
  assert prune.IMPT2_NORM=='max' and prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER==.03 and prune.IMPT2_LAYER_CAP_MULTIPLIER is None
 lfwa_experiment._configure=configure
 original_save=lfwa.save_checkpoint
 def save(model,path,metadata,**details):
  ck=original_save(model,path,metadata,**details)
  if details.get('stage')=='prune' and Path(path).name.startswith('prune_'):
   assert prune.IMPT2_PROTECTION_RATIO==.15 and prune.IMPT_TYPE2_ALPHA==c['alpha']
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
 assert read(Path(c['arguments']['lfwa_run_dir'])/'run.json')['protection_ratio']==.15


def compare(root):
 m=verify(root)
 import torch,lfwa,pandas as pd
 torch.set_num_threads(2)
 frames,meta=lfwa.load_frames(ROOT/'csv/LFWA.csv',ROOT/'Images/LFWA')
 old=read(root/'baseline_checkpoint_index.json');rows=[]
 for c in m['conditions']:
  new=[read(root/'progress'/c['name']/f'iteration_{i:03d}.json') for i in range(1,92)]
  for gamma,records in [(.005,old[c['name']]),(.15,new)]:
   choices={'near60':min(records,key=lambda r:(abs(r['sparsity']['model_sparsity']-.6),r['iteration'])),'iter91':records[-1]}
   for label,row in choices.items():
    p=Path(row['checkpoint'])
    if 'sha256' in row:assert sha(p)==row['sha256']
    dest=root/'analysis/evaluations'/c['name']/f'g{gamma}_{label}'
    model,ck=lfwa.load_checkpoint_model(p,meta,'cuda:0')
    assert ck['stage']=='prune'
    summary=lfwa.evaluate_model(model,frames['val'],dest,256)
    pred=pd.read_csv(dest/'predictions.csv');rates=pred[[x+'_pred' for x in lfwa.TARGETS]].mean()
    summary['near_constant_99pct_attributes']=int(((rates<=.01)|(rates>=.99)).sum())
    summary['constant_prediction_attributes']=int(((rates==0)|(rates==1)).sum())
    write(dest/'summary.json',summary)
    rows.append({'alpha':c['alpha'],'gamma':gamma,'point':label,'iteration':row['iteration'],'checkpoint':str(p),'evaluation':str(dest),**summary})
    del model,ck;torch.cuda.empty_cache()
    write(root/'analysis/results.json',rows)
 lines=['# LFWA 보호 비율 비교: LR1e-5,91iter','','같은 dense 출발점, max 정규화, 각단계5epoch. 추가8+1+1 이전 프루닝 체크포인트만 평가. 약60% 지점은 전체 파라미터 희소도로 가장 가까운 checkpoint, 마지막은91iter 고정.','',
 '| 비교점 | α | 보호 | 단계 | 전체 희소도 | Accuracy | BA | EO↓ | 근상수 속성 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in sorted(rows,key=lambda r:(r['point'],r['alpha'],r['gamma'])):
  lines.append(f"| {r['point']} | {r['alpha']} | {r['gamma']:.1%} | {r['iteration']} | {r['model_sparsity']:.4%} | {r['accuracy']:.6f} | {r['ba']:.6f} | {r['eo']:.6f} | {r['near_constant_99pct_attributes']} |")
 lookup={(r['point'],r['alpha'],r['gamma']):r for r in rows};contrasts=[]
 for point in ['near60','iter91']:
  for a in [.5,.7]:
   lo,hi=lookup[point,a,.005],lookup[point,a,.15]
   delta={k:hi[k]-lo[k] for k in ['accuracy','ba','eo','model_sparsity','near_constant_99pct_attributes']}
   contrasts.append({'point':point,'alpha':a,'contrast':'protection15pct_minus_0p5pct','delta':delta})
   lines.append(f"\n{point}, α={a}: 보호15%−0.5% ΔAcc={delta['accuracy']:+.6f}, ΔBA={delta['ba']:+.6f}, ΔEO={delta['eo']:+.6f}.")
  for gamma in [.005,.15]:
   lo,hi=lookup[point,.5,gamma],lookup[point,.7,gamma]
   delta={k:lo[k]-hi[k] for k in ['accuracy','ba','eo','model_sparsity','near_constant_99pct_attributes']}
   contrasts.append({'point':point,'gamma':gamma,'contrast':'alpha0p5_minus_alpha0p7','delta':delta})
 lines+=['','단일seed의 관측값이며 통계적 유의성이나 타 데이터셋 일반화를 주장하지 않는다. 기존 실험의 설정·dense 모델·split은 확인했으나 당시 모든 소스 버전의 동일성을 입증한 것은 아니다.']
 write(root/'analysis/contrasts.json',contrasts)
 (root/'analysis/report.md').write_text('\n'.join(lines)+'\n')
 pd.DataFrame(rows).to_csv(root/'analysis/metrics.csv',index=False)


def queue(root):
 m=verify(root);assert (root/'preflight_complete.json').exists()
 for c in m['conditions']:
  run=Path(c['arguments']['lfwa_run_dir'])
  if (run/'complete.json').exists():
   info=read(run/'run.json');assert info['arguments']==c['arguments'] and info['protection_ratio']==.15
   assert (root/'progress'/c['name']/'iteration_091.json').exists()
   continue
  assert not (run/'run.json').exists(),'Interrupted original run requires explicit RNG-aware resume; no automatic overwrite'
  write(root/'status.json',{'status':'starting','condition':c['name'],'time':now()})
  with (root/f"{c['name']}.log").open('ab',buffering=0) as log:
   subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--train',c['name']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 write(root/'status.json',{'status':'evaluating_comparisons','time':now()})
 compare(root)
 write(root/'complete.json',{'status':'complete','time':now(),'new_runs':2,'iterations_per_run':91,'report':str(root/'analysis/report.md')})
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
