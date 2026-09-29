"""Summarize existing label records and route lengths; does not invoke a planner or train a model."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    needed=['canonical_smiles','source_dataset','route_found','route_steps','target_hard']
    def load(folder):
        out=[]
        for split in ['train','validation','test']:
            d=pd.read_csv(folder/(split+'.csv'),usecols=needed,dtype=str).fillna('')
            d['split']=split;out.append(d)
        d=pd.concat(out,ignore_index=True)
        assert d.canonical_smiles.is_unique
        return d
    current=load(args.project/'GESA_Data/retrostar_target')
    historical=load(args.project/'target_original')
    excluded=historical.loc[~historical.canonical_smiles.isin(set(current.canonical_smiles))].copy()
    assert len(historical)==587021 and len(current)==582227 and len(excluded)==4794
    assert excluded.canonical_smiles.str.contains('.',regex=False).all()
    assert not current.canonical_smiles.str.contains('.',regex=False).any()
    pd.crosstab(excluded.split,excluded.target_hard).rename(columns={'0':'ES','1':'HS'}).reset_index().to_csv(
        args.output/'multicomponent_removals_by_split.csv',index=False)
    summary=[];distribution=[]
    for source,d in current.groupby('source_dataset'):
        steps=pd.to_numeric(d.route_steps,errors='coerce')
        success=d.route_found.str.lower().eq('true') & steps.notna()
        s=steps[success]
        summary.append({'source_combination':source,'N':len(d),'ES':int(d.target_hard.eq('0').sum()),
            'HS':int(d.target_hard.eq('1').sum()),'complete_route_with_exact_steps':int(success.sum()),
            'complete_route_gt10':int(s.gt(10).sum()),'completed_no_route':int(d.route_found.str.lower().eq('false').sum()),
            'route_steps_min':float(s.min()) if len(s) else None,
            'route_steps_median':float(s.median()) if len(s) else None,
            'route_steps_p90':float(s.quantile(.9)) if len(s) else None,
            'route_steps_max':float(s.max()) if len(s) else None})
        for step,n in s.astype(int).value_counts().sort_index().items():
            distribution.append({'source_combination':source,'route_steps':step,'N_complete_routes':int(n)})
    stats=pd.DataFrame(summary)
    stats.to_csv(args.output/'source_label_and_route_summary.csv',index=False)
    pd.DataFrame(distribution).to_csv(args.output/'complete_route_length_distribution.csv',index=False)
    collapsed=stats[['source_combination','N','ES','HS']].copy()
    collapsed['group']=collapsed.source_combination.map(lambda s:'Shared across sources' if ';' in s else s+' only')
    collapsed.groupby('group',sort=False)[['N','ES','HS']].sum().reset_index().to_csv(
        args.output/'source_label_counts_disjoint_groups.csv',index=False)
    explicit=current.loc[~current.source_dataset.eq('GASA_RetroStar')]
    assert len(explicit)==432386
    x=pd.to_numeric(explicit.route_steps,errors='coerce');assert int(x.gt(10).sum())==1161
    report={'before_single_component_cleanup':len(historical),'current_single_component':len(current),
        'removed_multicomponent':len(excluded),'explicit_generated_source_planning_records':len(explicit),
        'historical_GASA_labels_only':149841,'count_difference_587021_minus_432386':154635,
        'complete_routes_above_10_steps':int(x.gt(10).sum()),
        'note':'Historical GASA ES/HS labels are retained as published.'}
    (args.output/'count_reconciliation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    settings=[]
    for source,base in [('TarPass',args.project/'Data_old/sa_baseline_transfer_v1/retrostar_tarpass_v1/outputs'),
                        ('MolGenBench',args.project/'retro_star/task_1_codex_three_partition_production/production_outputs')]:
        for partition in ['task0','task1','task2']:
            log=base/partition/'run.log';config=None
            with log.open(errors='replace') as handle:
                for _ in range(25):
                    line=handle.readline()
                    if 'Config:' in line:config=json.loads(line.split('Config:',1)[1]);break
            assert config is not None
            keys=['iterations','expansion_topk','fp_dim','timeout_per_molecule','use_value_fn']
            r={'source':source,'partition':partition,**{k:config[k] for k in keys}}
            for k in ['starting_molecules','mlp_templates','mlp_model_dump','value_model']:
                r[k]=Path(config[k]).name
            settings.append(r)
    fields=[k for k in settings[0] if k not in ['source','partition']]
    assert all(tuple(r[k] for k in fields)==tuple(settings[0][k] for k in fields) for r in settings)
    pd.DataFrame(settings).to_csv(args.output/'planning_settings_from_primary_run_logs.csv',index=False)
    # Confirm that only primary run partitions contributed to the construction aggregate.
    aggregate=args.project/'retro_star/task_1_codex_three_partition_production/final_summary/retrostar_200k_aggregate_with_difficulty_proxy.csv'
    a=pd.read_csv(aggregate,usecols=['raw_partition_name','raw_status'],dtype=str).fillna('')
    report['retry_rows_in_construction_aggregate']=int(a.raw_partition_name.str.contains('retry').sum())
    assert report['retry_rows_in_construction_aggregate']==0
    (args.output/'count_reconciliation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print(collapsed.groupby('group')[['N','ES','HS']].sum().to_string(),flush=True)


if __name__=='__main__':main()
