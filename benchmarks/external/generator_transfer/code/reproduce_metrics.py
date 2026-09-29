"""Recalculate HS-positive AUROC/AP from the released continuous prediction scores.

Usage:
    python code/reproduce_metrics.py
    python code/reproduce_metrics.py --bootstrap 2000 --output recomputed_metrics.csv
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

MODELS=['GESA','SAScore','SCScore','SYBA','BR-SAScore','RAscore-XGB','GASA','SynFrag','DeepSA']
FUNCTIONS={'AUROC':roc_auc_score,'AP':average_precision_score}


def summarize(frame, generator, subset, reps):
    frame=frame.reset_index(drop=True)
    groups=[g.to_numpy() for _,g in frame.groupby('pocket_id').groups.items()]
    y=frame.target_hard.to_numpy(dtype=int)
    scores=frame[['HS_score_'+m.replace('-','_') for m in MODELS]].to_numpy(dtype=float)
    if len(np.unique(y))!=2:return []
    point={metric:np.asarray([fn(y,s) for s in scores.T]) for metric,fn in FUNCTIONS.items()}
    draws={metric:[] for metric in FUNCTIONS}
    rng=np.random.default_rng(6621)
    for _ in range(reps):
        idx=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        if len(np.unique(y[idx]))!=2:continue
        for metric,fn in FUNCTIONS.items():
            draws[metric].append([fn(y[idx],s[idx]) for s in scores.T])
    rows=[]
    for metric in FUNCTIONS:
        for i,model in enumerate(MODELS):
            r=dict(generator=generator,subset=subset,model=model,metric=metric,
                n=len(y),ES=int(sum(y==0)),HS=int(sum(y==1)),pockets=len(groups),point=point[metric][i])
            if draws[metric]:
                dist=np.asarray(draws[metric]);lo,hi=np.percentile(dist[:,i],[2.5,97.5])
                dl,dh=np.percentile(dist[:,0]-dist[:,i],[2.5,97.5])
                r.update(lower=lo,upper=hi,paired_lower=dl,paired_upper=dh)
            rows.append(r)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--bootstrap',type=int,default=0)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    data=pd.read_csv(args.root/'data/evaluation_molecules.csv')
    rows=[]
    for name,cohort in data.groupby('generator',sort=False):
        masks={'all_unfiltered':np.ones(len(cohort),dtype=bool),
            'without_shared_scaffolds':cohort.shared_released_train_scaffold.eq(0).to_numpy(),
            'SAScore_3_to_4':(cohort.SAScore.ge(3)&cohort.SAScore.lt(4)).to_numpy()}
        for subset,mask in masks.items():
            rows.extend(summarize(cohort.loc[mask],name,subset,args.bootstrap))
    actual=pd.DataFrame(rows)
    expected=pd.read_csv(args.root/'metrics/all_metrics.csv')
    keys=['generator','subset','model','metric']
    check=actual.merge(expected,on=keys,suffixes=('_actual','_expected'),validate='one_to_one')
    assert len(check)==len(expected)==len(actual)
    for col in ['point','n','ES','HS','pockets']:
        assert np.allclose(check[col+'_actual'],check[col+'_expected'],atol=1e-12,rtol=0),col
    if args.bootstrap==2000:
        for col in ['lower','upper','paired_lower','paired_upper']:
            assert np.allclose(check[col+'_actual'],check[col+'_expected'],atol=1e-12,rtol=0),col
    if args.output:actual.to_csv(args.output,index=False)
    print(f'Verified {len(actual)} metric rows across {data.generator.nunique()} generators and {len(data)} evaluable molecules.')
    print(actual.loc[actual.model.eq('GESA') & actual.metric.eq('AUROC'),['generator','subset','n','ES','HS','point']].to_string(index=False))


if __name__=='__main__':main()
