"""Verify source-stratified TS4 AUROC/AP from the existing frozen predictions."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parent


def main():
    d=pd.read_csv(ROOT/'TS4_source_predictions.csv')
    assert len(d)==3584 and d.molecule_id.is_unique
    expected=pd.read_csv(ROOT/'TS4_GESA_within_source_metrics.csv').set_index('source')
    for source,g in d.groupby('source_dataset'):
        r=expected.loc[source]
        assert len(g)==r.n
        assert int(g.target_hard.eq(0).sum())==r.ES
        assert int(g.target_hard.eq(1).sum())==r.HS
        auc=roc_auc_score(g.target_hard,g.GESA_score)
        ap=average_precision_score(g.target_hard,g.GESA_score)
        assert abs(auc-r.AUROC)<1e-12 and abs(ap-r.AP)<1e-12
        print(f'{source}: N={len(g)}, AUROC={auc:.6f}, AP={ap:.6f}')
    print('All source-stratified point estimates verified.')


if __name__=='__main__':main()
