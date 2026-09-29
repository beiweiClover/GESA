"""Verify the released matched lists, property balance, scaffold coverage and pairs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def smd(a,b):
    den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
    return float((b.mean()-a.mean())/den) if den else 0.
def main():
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    train=pd.read_csv(ROOT/'data/training_subsets.csv.gz');pairs=pd.read_csv(ROOT/'data/matched_pairs.csv.gz')
    pred=pd.read_csv(ROOT/'data/predictions_TS1_TS4.csv.gz')
    # Canonical isomeric identity keys were computed after removing atom maps.
    from rdkit import Chem
    def identity(s):
        mol=Chem.MolFromSmiles(s)
        assert mol is not None and len(Chem.GetMolFrags(mol))==1
        for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    test=pred[['molecule_id','canonical_smiles']].drop_duplicates()
    test_keys=set(map(identity,test.canonical_smiles))
    ref=set(pd.read_csv(ROOT/'data/historical_reference_scaffolds.csv.gz').scaffold_key)
    diagnostics=[];balance=[];maxdiff={f:0. for f in p['matching_features']}
    for rep in p['sampling_seeds']:
        arms=[train[(train.replicate==rep)&(train.arm==arm)] for arm in ['historical','generative_replacement']]
        h,g=arms
        for d in arms:
            assert len(d)==12000 and d.identity_key.nunique()==12000
            assert d.target_hard.value_counts().to_dict()=={0:6000,1:6000}
            assert not (set(d.identity_key)&test_keys)
        common=[set(d.loc[d.experiment_role=='shared_historical_background','identity_key']) for d in arms]
        assert common[0]==common[1] and len(common[0])==6000
        assert h.scaffold_key.nunique()==g.scaffold_key.nunique()
        for label in [0,1]:
            for f in p['matching_features']:
                balance.append({'replicate':rep,'label':label,'feature':f,'SMD':smd(h.loc[h.target_hard==label,f],g.loc[g.target_hard==label,f])})
        pair=pairs[pairs.replicate==rep]
        H=h.set_index('molecule_id').loc[pair.historical_molecule_id]
        G=g.set_index('molecule_id').loc[pair.generative_molecule_id]
        assert np.array_equal(H.target_hard,G.target_hard)
        for f in p['matching_features']:
            v=float(np.abs(H[f].to_numpy()-G[f].to_numpy()).max());maxdiff[f]=max(maxdiff[f],v)
            assert v<=p['absolute_pairwise_calipers'][f]+1e-10
        generated=g[g.experiment_role=='generative_replacement']
        row={'replicate':rep,'scaffolds_per_arm':h.scaffold_key.nunique(),'generated_novel_scaffold_fraction_vs_historical_arm':float((~generated.scaffold_key.isin(set(h.scaffold_key))).mean()),'generated_novel_scaffold_fraction_vs_reference_pool':float((~generated.scaffold_key.isin(ref)).mean())}
        for arm,d in zip(['historical','generative_replacement'],arms):
            counts=d.scaffold_key.value_counts();freq=counts.to_numpy()/len(d)
            row[arm+'_normalized_scaffold_entropy']=float(-(freq*np.log(freq)).sum()/np.log(len(counts)))
        diagnostics.append(row)
    result={'status':'passed','scope':'six fixed training lists and TS1–TS4 evaluation data','N_per_arm':12000,'ES_HS_per_arm':[6000,6000],'max_abs_class_specific_SMD':max(abs(r['SMD']) for r in balance),'maximum_pairwise_feature_differences':maxdiff,'class_specific_property_balance':balance,'replicate_scaffold_diagnostics':diagnostics,'scaffold_identities_matched':False,'scaffold_richness_matched':True,'exact_training_TS1_TS4_identity_overlaps':0}
    (ROOT/'results/matching_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Matching audit passed; maximum class-specific |SMD|:',result['max_abs_class_specific_SMD'])

if __name__=='__main__':main()
