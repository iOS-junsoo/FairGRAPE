#!/usr/bin/env python3
"""Isolated, paired DogsCats continuation using the optional production cap.

Default protocol: each alpha's saved iteration 25 -> iterations 26..28,
full scoring data, Adam 1e-5 / 5 epochs, validation only. No test evaluation.
--preflight reads masks on CPU only. Completed iterations can be resumed.
"""
import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCES={
    .7:'save_models/20260919_123145_DogsCats_alpha0.7_max_impt2_seed42',
    .9:'save_models/20260918_145720_DogsCats_alpha0.9_max_impt2_seed42',
}
CODE=['channel_cap.py','prune.py','main_test.py','config.py','train_and_val.py',
      'dataset.py','util.py','adversarial_debiasing.py','baseline_results.py',
      'scripts/run_dogscats_layer_cap_smoke.py']


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(args,stage,iteration=None,**extra):
    row={'alpha':args.alpha,'cap':args.cap,'stage':stage,'iteration':iteration,'time':now(),**extra}
    write_json(args.run_dir/'status.json',row)
    print('SMOKE_STATUS '+json.dumps(row),flush=True)


def load_model(path):
    import torch
    import prune
    from util import make_model
    ck=torch.load(path,map_location='cpu')
    if not isinstance(ck,dict) or 'model_state_dict' not in ck:
        raise ValueError('Expected a local BEST_MODEL or smoke checkpoint')
    model=make_model(network='mobilenetv2',n_classes=2,pretrained=False)
    modules=dict(model.named_modules())
    for name,value in ck['model_state_dict'].items():
        if name.endswith('.mask'):
            layer=modules[name[:-5]]
            layer.mask=torch.nn.Parameter(torch.ones_like(value),requires_grad=False)
            layer.forward=types.MethodType(prune.forward_mapping_dict[type(layer).__name__],layer)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    check_masks(model)
    return model,ck


def check_masks(model,previous=None):
    import torch
    masks={}
    for name,layer in model.named_modules():
        if hasattr(layer,'mask'):
            mask=layer.mask.detach()
            if mask.shape!=layer.weight.shape or not bool(((mask==0)|(mask==1)).all()):
                raise AssertionError(f'Non-binary or mismatched mask: {name}')
            if layer.mask.requires_grad:raise AssertionError('Masks must stay frozen')
            masks[name]=mask.clone()
            if previous is not None and not torch.equal(mask,previous[name].to(mask.device)):
                raise AssertionError(f'Mask changed during retraining: {name}')
    if len(masks)!=53:raise AssertionError(f'Expected 53 convolution/linear masks, got {len(masks)}')
    return masks


def layout(model):
    import prune
    modules=dict(model.named_modules());rows={}
    for i in range(1,18):
        block=f'features.{i}';first,_,_=prune._get_impt_type2_block_layer_names(block)
        m=modules[first].mask.detach();width=m.shape[0]
        rows[block]={'original':width,'active':int((m.reshape(width,-1).sum(1)>0).sum())}
    return rows


def sparsity(model):
    from baseline_results import _mask_sparsity
    return _mask_sparsity(model,'blocks',None)


def prepare_frames(args):
    from dataset import make_frame,apply_fscl_skew,make_balanced_eval
    frames=make_frame(str(ROOT/'csv/DogsCats.csv'),'Images/DogsCats',seven_races=False,
                      stratify=True,strat_cols=['species','color'],split_csv_out=None)
    train=apply_fscl_skew(frames['train'],10,seed=42,skew_col='color',group_col='species')
    train=train.sample(frac=1,random_state=42).reset_index(drop=True)
    val=make_balanced_eval(frames['val'],seed=42,cols=('species','color'))
    assert len(train)==8841 and len(val)==1236
    assert list(val.groupby(['species','color']).size())==[309]*4
    for name,frame in [('train',train),('val',val)]:frame.to_csv(args.run_dir/f'{name}_frame.csv',index=False)
    return {'train':train,'val':val}


@contextlib.contextmanager
def preserve_rng():
    import numpy as np
    import torch
    import imgaug as ia
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
           torch.cuda.get_rng_state_all(),copy.deepcopy(ia.random.get_global_rng().state))
    try:yield
    finally:
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
        torch.cuda.set_rng_state_all(state[3]);ia.random.get_global_rng().state=state[4]


def evaluate(model,frame,directory,batch=256):
    import torch
    import pandas as pd
    from dataset import make_datasets
    from baseline_results import dogscats_metrics
    from util import safe_forward_with_cudnn_fallback
    directory.mkdir(parents=True,exist_ok=True)
    was_training=model.training
    with preserve_rng():
        _,loader=make_datasets(frame,frame,True,batch,['species','color'])
        truth=[];colors=[];pred=[];probs=[]
        model.eval()
        with torch.no_grad(),torch.backends.cudnn.flags(enabled=False):
            for images,labels in loader:
                out=safe_forward_with_cudnn_fallback(model,images.to('cuda',dtype=torch.float).contiguous())
                pred.extend(out.argmax(1).cpu().tolist());probs.extend(out.softmax(1)[:,1].cpu().tolist())
                truth.extend(labels[:,0].tolist());colors.extend(labels[:,1].tolist())
        data=pd.DataFrame({'face_name_align':frame['face_name_align'].tolist(),
                           'species':truth,'color':colors,'species_preds_fair':pred,'dog_probability':probs})
        metrics=dogscats_metrics(data)
        data.to_csv(directory/'predictions.csv',index=False)
        write_json(directory/'metrics.json',metrics)
    model.train(was_training)
    return metrics


def configure(args,directory,iteration):
    import config
    import prune
    for key,value in {'glo_dataset':'DogsCats','glo_seed':42,'glo_impt_type':2,
        'glo_n_groups':2,'glo_use_grl':False,'glo_pretrained':False,'glo_skew_beta':10,
        'glo_keep_per_iter':.975,'glo_perf_only':False,'glo_phi_gap_only':False,
        'glo_phi_analysis':False,'glo_imp_rate':args.alpha,'glo_prune_iter':iteration-1,
        'glo_impt2_layer_max_remove_ratio':args.cap,'glo_epoch_observer':None,
        'glo_collapse_notes':[],'glo_is_checked':False,'glo_fairness':{},
        'glo_results_run_dir':str(directory/'history'),'glo_model_run_dir':str(directory/'selected_models'),
        'glo_channel_log_run_dir':str(directory/'channels')}.items():setattr(config,key,value)
    for sub in ['history','selected_models','channels']:(directory/sub).mkdir(parents=True,exist_ok=True)
    prune.IMPT_TYPE2_ALPHA=args.alpha
    assert prune.IMPT2_NORM=='max' and prune.IMPT2_PROTECTION_RATIO==.005
    assert prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER==.03 and prune.IMPT2_LAYER_CAP_MULTIPLIER is None


def select(args,pruner,directory,iteration,frames):
    import prune
    from channel_cap import channel_removal_limits
    captured=[];logger=prune._save_channel_pruning_log
    def capture(**kw):
        captured.append(kw)
        logger(**kw)
    before=layout(pruner.model)
    try:
        prune._save_channel_pruning_log=capture
        masks=prune.fairness_grad(pruner.model,min(.9,1-.975**iteration),frames['train'],
            new_img_dir=None,sensitive_classes=2,masked_grads=True,output_cols_each_task=[(0,2)],
            col_names=['species','color'],para_batch=1,impt_type=2,stop_batch=10000,
            delta_p=False,network='mobilenetv2',sensitive_group='color')
    finally:prune._save_channel_pruning_log=logger
    assert len(captured)==1
    payload=captured[0];keys=payload['selected_channels'];candidates=payload['score_by_channel_dict']
    assert not set(keys).intersection(payload['protected_set'])
    original={b:v['original'] for b,v in before.items()};active={b:v['active'] for b,v in before.items()}
    floors={b:math.ceil(v*.03) for b,v in original.items()}
    limits=channel_removal_limits(original,active,floors,args.cap)
    rows=[]
    for b,v in before.items():
        count=sum(k[0]==b for k in keys)
        assert active[b]-count>=floors[b]
        if limits is not None:assert count<=limits[b]
        rows.append({'block':b,**v,'removed':count,'active_after':active[b]-count,
                     'limit':None if limits is None else limits[b],
                     'removed_fraction':count/active[b] if active[b] else 0})
    assert payload['cap_relaxed_removed']==0
    audit={'layers':rows,'selected_channels':keys,'remove_target':payload['remove_target'],
           'removed_weights':payload['accum_removed'],
           'deficit':max(0,payload['remove_target']-payload['accum_removed']),
           'cap':args.cap,'protected':sorted(payload['protected_set']),
           'no_cap_relaxation':True}
    write_json(directory/'selection.json',audit)
    write_json(directory/'scores_full_precision.json',{
        'candidates':[{'block':b,'channel':c,'score':score,'weights':w} for (b,c),(score,w) in candidates.items()],
        **{field:{k:v.detach().cpu().tolist() for k,v in payload[field].items()}
           for field in ['phi_by_layer','perf_by_layer','phi_scaled_by_layer','perf_scaled_by_layer']}})
    return masks,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha',type=float,choices=[.9,.7],required=True)
    parser.add_argument('--cap',default='off',choices=['off','0.20','0.10'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--iterations',type=int,default=3,choices=[3])
    args=parser.parse_args();args.cap=None if args.cap=='off' else float(args.cap)
    args.run_dir=args.run_dir.resolve();args.run_dir.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(4)  # Avoid oversubscription during CPU checkpoint/mask validation.
    import config
    import prune
    from dataset import make_datasets
    from util import setseed
    from train_and_val import train
    source=next((ROOT/SOURCES[args.alpha]).glob('BEST_MODEL_prune_25_*.pt'))
    manifest={'alpha':args.alpha,'cap':args.cap,'source':str(source),'source_sha256':sha256(source),
              'source_iteration':25,'iterations':3,'batch':256,'retrain_epochs':5,'retrain_lr':1e-5,
              'cpu_threads':4,'seed':42,'score_seed':'42 + iteration','training_seed':'10042 + iteration',
              'norm':'max','keep_per_iter':.975,'floor':.03,'gamma':.005,'beta':10,
              'evaluation':'validation only','test_evaluated':False,
              'source_sha256s':{p:sha256(ROOT/p) for p in CODE}}
    mf=args.run_dir/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Existing run manifest differs')
    write_json(mf,manifest)
    model,ck=load_model(source)
    assert ck['prune_iteration']==25 and abs(float(ck['alpha'])-args.alpha)<1e-8
    assert not ck['grl_enabled']
    write_json(args.run_dir/'preflight.json',{'source_layout':layout(model),'sparsity':sparsity(model),
              'checkpoint_acc':ck['best_acc'],'checkpoint_eo':ck['best_eo'],'masks':len(check_masks(model))})
    if args.preflight:
        print('CPU_PREFLIGHT_PASSED',args.alpha,args.cap,flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('GPU required for the approved smoke run')
    torch.backends.cudnn.enabled=False
    setseed(42)
    frames=prepare_frames(args)
    model=model.to('cuda')
    if not (args.run_dir/'initial/metrics.json').exists():
        status(args,'initial_validation',25)
        initial=evaluate(model,frames['val'],args.run_dir/'initial')
        assert abs(initial['ACC']-float(ck['best_acc']))<1e-6,(initial,ck['best_acc'])
        assert abs(initial['EO']-float(ck['best_eo']))<1e-6,(initial,ck['best_eo'])
    results=[]
    for iteration in range(26,29):
        directory=args.run_dir/f'iteration_{iteration:03d}'
        complete=directory/'complete.json'
        if complete.exists():
            row=json.loads(complete.read_text())
            assert sha256(directory/'retrained.pt')==row['checkpoint_sha256']
            del model;torch.cuda.empty_cache()
            model,_=load_model(directory/'retrained.pt');model=model.to('cuda');results.append(row);continue
        if directory.exists():directory.rename(directory.with_name(directory.name+'_interrupted_'+str(time.time_ns())))
        directory.mkdir()
        configure(args,directory,iteration)
        started=time.monotonic()
        for name,param in model.named_parameters():param.requires_grad_(not name.endswith('.mask'))
        old_masks=check_masks(model);before=sparsity(model)
        setseed(42+iteration)
        status(args,'scoring_and_pruning',iteration)
        pruner=prune.Prunner(model,torch.nn.CrossEntropyLoss(),None,[(0,2)])
        masks,audit=select(args,pruner,directory,iteration,frames)
        pruner.get_mask=lambda unused:masks
        model=pruner.prune(None,False)
        del pruner,masks
        frozen=check_masks(model)
        assert all(not bool((frozen[k]>old_masks[k]).any()) for k in frozen)
        after=sparsity(model)
        assert after['pruned_target_weights']-before['pruned_target_weights']==audit['removed_weights']
        actual_layout=layout(model)
        assert all(actual_layout[row['block']]['active']==row['active_after'] for row in audit['layers'])
        del old_masks
        status(args,'post_prune_validation',iteration)
        post=evaluate(model,frames['val'],directory/'post_prune')
        epochs=[]
        def observe(current,epoch,metrics):
            record={k:v for k,v in metrics.items() if k!='text'}
            record.update(epoch=epoch,iteration=iteration)
            record['cells']=[{'color':int(g),'species':int(y),'correct':int(n),'count':int(total),
                              'accuracy':int(n)/int(total)}
                for g,y,n,total in re.findall(r'species \(group=(\d), class=(\d)\): (\d+)/(\d+)',metrics['text'])]
            check_masks(current,frozen)
            epochs.append(record)
            write_json(directory/'epochs.json',epochs)
            torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in current.state_dict().items()},
                        'alpha':args.alpha,'cap':args.cap,'prune_iteration':iteration,'epoch':epoch,
                        'grl_enabled':False,'metrics':record},directory/f'epoch_{epoch:02d}.pt')
            status(args,'retraining',iteration,epoch=epoch,accuracy=record['accuracy'],eo=record['eo'])
        config.glo_epoch_observer=observe
        setseed(10042+iteration)
        loaders=make_datasets(frames['train'],frames['val'],True,256,['species','color'])
        cfgs=['DogsCats','FairGRAPE','species',.9,frames['val'],None,2,'mobilenetv2',
              ['species','color'],[(0,2)],'color',0,42]
        status(args,'retraining',iteration,epoch=0)
        try:
            model,_=train(model,torch.nn.CrossEntropyLoss(),{'train':loaders[0],'test':loaders[1]},
                          [1e-5],[5],['species'],[(0,2)],cfgs=cfgs,use_debiasing=False)
        finally:config.glo_epoch_observer=None
        check_masks(model,frozen)
        del frozen,loaders
        best=evaluate(model,frames['val'],directory/'best')
        expected=min(epochs,key=lambda r:(-r['accuracy'],r['eo'],r['loss']))
        assert abs(best['ACC']-expected['accuracy'])<1e-6
        assert abs(best['EO']-expected['eo'])<1e-6
        torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'alpha':args.alpha,'cap':args.cap,'prune_iteration':iteration,'grl_enabled':False,
                    'best_acc':best['ACC'],'best_eo':best['EO'],'best_loss':expected['loss']},directory/'retrained.pt')
        row={'alpha':args.alpha,'cap':args.cap,'iteration':iteration,'post_prune':post,'best':best,
             'best_epoch':expected['epoch'],'before':before,'after':sparsity(model),
             'selection':audit,'elapsed_seconds':time.monotonic()-started,
             'checkpoint_sha256':sha256(directory/'retrained.pt')}
        write_json(complete,row);results.append(row)
        status(args,'iteration_complete',iteration,accuracy=best['ACC'],eo=best['EO'],elapsed=row['elapsed_seconds'])
        if audit['removed_weights']==0:raise RuntimeError('No eligible channel remains; stop without relaxing cap')
    write_json(args.run_dir/'results.json',results)
    status(args,'complete',28)


if __name__=='__main__':main()
