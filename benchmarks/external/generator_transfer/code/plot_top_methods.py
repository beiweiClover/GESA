"""Visualize actual full versus fixed-SA-subset results for the three external generators."""
from pathlib import Path
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=pd.read_csv(ROOT/'metrics/all_metrics.csv')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
        'axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,132/25.4))
    fig.subplots_adjust(left=.12,right=.98,top=.80,bottom=.12,wspace=.78,hspace=1.02)
    fig.text(.03,.97,'Evaluation on held-out generators',weight='bold',fontsize=10,va='top',color='#243744')
    fig.text(.03,.91,'AUROC point estimates: full cohorts and fixed SAScore [3,4) subsets',fontsize=7,va='top',color='#53616B')
    plotted=[]
    for row,subset in enumerate(['all_unfiltered','SAScore_3_to_4']):
        for col,name in enumerate(['NEAT-POCKET','FLOWR','ResGen']):
            ax=axes[row,col]
            group=data.loc[data.generator.eq(name)&data.subset.eq(subset)&data.metric.eq('AUROC')].copy()
            ax.text(-.40,1.33,string.ascii_lowercase[row*3+col],transform=ax.transAxes,fontsize=8,weight='bold')
            ax.text(0,1.33,name,transform=ax.transAxes,fontsize=8,weight='bold',color='#243744')
            ax.text(0,1.13,'Full cohort' if row==0 else 'SAScore [3,4)',transform=ax.transAxes,fontsize=6.5,color='#53616B')
            if group.empty:
                ax.text(.5,.5,'Not estimable',ha='center',va='center',transform=ax.transAxes)
                ax.axis('off');continue
            group=group.sort_values(['point','model'],ascending=[False,True])
            cutoff=group.point.iloc[2]
            selected=group.loc[(group.point>=cutoff)|np.isclose(group.point,cutoff,atol=1e-12,rtol=0)|group.model.eq('GESA')].copy()
            selected=selected.sort_values(['point','model'],ascending=[False,True])
            ranks=group.set_index('model').point.rank(method='min',ascending=False).astype(int)
            y=np.arange(len(selected))
            ax.barh(y,selected.point,color=['#147B83' if x=='GESA' else '#BBC4CC' for x in selected.model],height=.62,zorder=2)
            ax.axvline(.5,color='#ADB7BF',lw=.7,ls=(0,(2,2)),zorder=1)
            ax.set_xlim(0,1);ax.set_ylim(len(selected)-.45,-.65)
            ax.set_yticks(y,[f'{ranks[m]}. {m}' for m in selected.model],fontsize=6.2)
            ax.set_xticks([0,.5,1],['0','0.5','1.0']);ax.tick_params(axis='x',labelsize=6,length=2)
            ax.tick_params(axis='y',length=0,pad=3)
            if row:ax.set_facecolor('#F3F6F7')
            first=group.iloc[0]
            cohort_label='Full cohort' if row==0 else 'SAScore [3,4)'
            ax.texts[2].set_text(f'{cohort_label} | {int(first.pockets)} pockets')
            caution=' †' if min(int(first.ES),int(first.HS))<20 else ''
            ax.text(0,.995,f'n={int(first.n)}  |  {int(first.ES)} ES / {int(first.HS)} HS{caution}',
                    transform=ax.transAxes,fontsize=5.8,color='#53616B')
            for yy,r in zip(y,selected.itertuples()):
                ax.text(r.point-.02,yy,f'{r.point:.3f}',ha='right',va='center',fontsize=6,
                    color='white' if r.model=='GESA' else '#243744',weight='bold' if r.model=='GESA' else 'normal')
                plotted.append(r._asdict())
            for label in ax.get_yticklabels():
                if 'GESA' in label.get_text():label.set_color('#147B83');label.set_weight('bold')
    fig.text(.03,.048,'Top 3 shown, including cutoff ties. HS is the positive class.',fontsize=5.7,color='#53616B')
    fig.text(.03,.025,'Point estimates shown; confidence intervals are supplied in the source tables. † Fewer than 20 in one class.',fontsize=5.7,color='#53616B')
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    stem=out/'three_generators_full_vs_SA_top_methods'
    fig.savefig(stem.with_suffix('.png'),dpi=250)
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    pd.DataFrame(plotted).to_csv(out/'figure_source_data.csv',index=False,encoding='utf-8-sig')
    plt.close(fig)


if __name__=='__main__':main()
