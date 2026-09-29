"""Reproduce TS1–TS4 point estimates, paired intervals and proximity sensitivity."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
ARMS=['historical','generative_replacement']

class WeightedMetrics:
    def __init__(self,y,score):
        self.y=np.asarray(y,dtype=np.int8)
        _,self.group=np.unique(np.asarray(score,dtype=float),return_inverse=True)
        self.n_groups=int(self.group.max())+1
    def __call__(self,w):
        p=np.bincount(self.group,weights=w*self.y,minlength=self.n_groups)
        n=np.bincount(self.group,weights=w*(1-self.y),minlength=self.n_groups)
        P,N=p.sum(),n.sum()
        auc=np.sum(p*(np.cumsum(n)-.5*n))/(P*N)
        dp,dn=p[::-1],n[::-1];cp=np.cumsum(dp);den=cp+np.cumsum(dn)
        precision=np.divide(cp,den,out=np.zeros_like(cp),where=den>0)
        return float(auc),float(np.sum(dp/P*precision))

def seed(base,*parts):
    h=hashlib.sha256('|'.join(map(str,parts)).encode()).digest()
    return (int.from_bytes(h[:8],'little')+base)%(2**32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--output',type=Path,default=ROOT/'results');a=ap.parse_args()
    if a.bootstrap<1:raise ValueError('At least one bootstrap draw is required')
    a.output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ROOT/'data/predictions_TS1_TS4.csv.gz')
    nn=pd.read_csv(ROOT/'data/test_nearest_neighbors.csv.gz')
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    per=[];paired=[];near=[]
    for rep in protocol['sampling_seeds']:
        for split in ['TS1','TS2','TS3','TS4']:
            frames=[df[(df.replicate==rep)&(df.split==split)&(df.arm==arm)].sort_values('molecule_id').reset_index(drop=True) for arm in ARMS]
            h,g=frames
            for f in frames:
                assert f.molecule_id.is_unique and np.isfinite(f.score).all()
            for key in ['molecule_id','canonical_smiles','target_hard']:assert np.array_equal(h[key],g[key])
            y=h.target_hard.to_numpy('int8');es,hs=np.flatnonzero(y==0),np.flatnonzero(y==1)
            point=np.array([[roc_auc_score(y,f.score),average_precision_score(y,f.score)] for f in frames])
            rng=np.random.default_rng(seed(protocol['uncertainty']['analysis_seed'],rep,split))
            fun=[WeightedMetrics(y,f.score) for f in frames];boot=np.empty((a.bootstrap,2,2))
            for i in range(a.bootstrap):
                w=np.zeros(len(y),dtype=np.int32)
                w[es]=rng.multinomial(len(es),np.repeat(1/len(es),len(es)))
                w[hs]=rng.multinomial(len(hs),np.repeat(1/len(hs),len(hs)))
                for j,metric in enumerate(fun):boot[i,j]=metric(w)
            for j,arm in enumerate(ARMS):
                row={'replicate':rep,'split':split,'arm':arm,'N':len(y),'ES':len(es),'HS':len(hs)}
                for k,name in enumerate(['AUROC','AP']):
                    row[name]=point[j,k];row[name+'_CI_low'],row[name+'_CI_high']=np.quantile(boot[:,j,k],[.025,.975])
                per.append(row)
            for k,name in enumerate(['AUROC','AP']):
                delta=boot[:,1,k]-boot[:,0,k];lo,hi=np.quantile(delta,[.025,.975])
                paired.append({'replicate':rep,'split':split,'metric':name,'historical':point[0,k],'generative_replacement':point[1,k],'delta':point[1,k]-point[0,k],'CI_low':lo,'CI_high':hi,'N':len(y),'bootstrap_seed':seed(protocol['uncertainty']['analysis_seed'],rep,split)})
            x=nn[nn.cohort==split].set_index('molecule_id').loc[h.molecule_id]
            H=x[f'seed_{rep}__historical_only'].to_numpy();G=x[f'seed_{rep}__historical_plus_generative'].to_numpy()
            mask=np.abs(H-G)<=protocol['uncertainty']['equal_proximity_abs_nn_difference']
            row={'replicate':rep,'cohort':split,'N_full':len(y),'N_equal_proximity':int(mask.sum()),'ES':int((y[mask]==0).sum()),'HS':int((y[mask]==1).sum())}
            for j,arm in enumerate(ARMS):
                row[arm+'_AUROC']=roc_auc_score(y[mask],frames[j].score.to_numpy()[mask])
                row[arm+'_AP']=average_precision_score(y[mask],frames[j].score.to_numpy()[mask])
            row['delta_AUROC']=row['generative_replacement_AUROC']-row['historical_AUROC']
            row['delta_AP']=row['generative_replacement_AP']-row['historical_AP'];near.append(row)
    per=pd.DataFrame(per);per.to_csv(a.output/'per_run_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(a.output/'paired_differences.csv',index=False)
    pd.DataFrame(near).to_csv(a.output/'equal_proximity.csv',index=False)
    summary=per.groupby(['split','arm'])[['AUROC','AP']].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(a.output/'summary.csv',index=False)
    print(summary.to_string())

if __name__=='__main__':main()
