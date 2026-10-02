#!/usr/bin/env python3
"""Identity-paired sample uncertainty for two fixed LFWA validation models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=2000)
    args=parser.parse_args()
    roots=[args.reference.resolve(),args.candidate.resolve()]
    frames=[pd.read_csv(root/'predictions.csv') for root in roots]
    summaries=[json.loads((root/'summary.json').read_text()) for root in roots]
    names=pd.read_csv(roots[0]/'attributes.csv').attribute.tolist()
    assert names==pd.read_csv(roots[1]/'attributes.csv').attribute.tolist()
    assert len(names)==39 and all(summary['split']=='val' for summary in summaries)
    for column in ['source_index','identity','young']+[name+'_true' for name in names]:
        assert np.array_equal(frames[0][column],frames[1][column]),column
    truth=frames[0][[name+'_true' for name in names]].to_numpy()
    group=frames[0].young.to_numpy()
    codes,identities=pd.factorize(frames[0].identity,sort=True)
    n=len(identities)
    sizes=np.bincount(codes,minlength=n)
    sufficient=np.zeros((n,2,2,4,len(names)),dtype=np.float64)
    for arm,frame in enumerate(frames):
        pred=frame[[name+'_pred' for name in names]].to_numpy()
        indicators=[(truth==1)&(pred==1),(truth==1)&(pred==0),
                    (truth==0)&(pred==1),(truth==0)&(pred==0)]
        for g in [0,1]:
            for cell,indicator in enumerate(indicators):
                np.add.at(sufficient[:,arm,g,cell,:],codes,indicator*(group==g)[:,None])
    def compute(weights):
        counts=(weights@sufficient.reshape(n,-1)).reshape(len(weights),2,2,4,len(names))
        tp,fn,fp,tn=[counts[:,:,:,i,:] for i in range(4)]
        with np.errstate(divide='ignore',invalid='ignore'):
            tpr,fpr=tp/(tp+fn),fp/(fp+tn)
            ba=np.nanmean(np.nanmean(np.stack([tpr,1-fpr],axis=3),axis=(2,3)),axis=2)
            eo_attr=(np.abs(tpr[:,:,0,:]-tpr[:,:,1,:])+np.abs(fpr[:,:,0,:]-fpr[:,:,1,:]))/2
            eo=np.nanmean(eo_attr,axis=2)
            acc=(tp+tn).sum(axis=(2,3))/((weights@sizes)[:,None]*len(names))
        return np.stack([acc,ba,eo],axis=2),np.isfinite(eo_attr).sum(axis=2)
    point,_=compute(np.ones((1,n)))
    metrics=['accuracy','ba','eo']
    for arm,summary in enumerate(summaries):
        for j,key in enumerate(metrics):
            assert abs(point[0,arm,j]-summary[key])<1e-12,(arm,key)
    rng=np.random.default_rng(20260917)
    weights=rng.multinomial(n,np.full(n,1/n),size=args.replicates)
    bootstrap,valid=compute(weights)
    delta=bootstrap[:,1,:]-bootstrap[:,0,:]
    intervals={key:{'delta_candidate_minus_reference':summaries[1][key]-summaries[0][key],
                    'percentile_95':np.percentile(delta[:,j],[2.5,97.5]).tolist()}
               for j,key in enumerate(metrics)}
    result={'reference':str(roots[0]),'candidate':str(roots[1]),'evaluation_split':'val',
            'replicates':args.replicates,'seed':20260917,'sampling_unit':'identity',
            'n_identities':n,'n_images':len(frames[0]),'n_attributes':len(names),
            'eo_valid_attributes_min':int(valid.min()),'eo_valid_attributes_max':int(valid.max()),
            'intervals':intervals,'point_metrics_reproduced':True,
            'prediction_sha256':{str(root):hashlib.sha256((root/'predictions.csv').read_bytes()).hexdigest() for root in roots},
            'scope':'Exploratory validation-sample uncertainty conditional on two fixed fitted checkpoints. Does not include training-seed variation, adaptive validation reuse, or across-dataset uncertainty.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'intervals':intervals,'eo_valid_range':[int(valid.min()),int(valid.max())]},indent=2))


if __name__=='__main__':
    main()
