"""Submission view: controlled training-data contribution on TS1–TS4.

Reuses the completed, verified experiment without retraining or changing statistics.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
COLORS={'historical':'#6887A8','generative_replacement':'#208F87'}
SPLITS=['TS1','TS2','TS3','TS4']
ARMS=['historical','generative_replacement']

def main():
    df=pd.read_csv(ROOT/'results/per_run_metrics.csv')
    proto=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    assert proto['N_per_arm']==12000 and proto['generative_fraction_in_mixed_arm']==.5
    d=df[df.split.isin(SPLITS)].copy()
    assert len(d)==24 and (d.groupby(['split','arm']).size()==3).all()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,
        'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig=plt.figure(figsize=(7,4.8))
    gs=fig.add_gridspec(2,1,height_ratios=[1.25,2.9],left=.17,right=.99,bottom=.11,top=.94,hspace=.48)
    a=fig.add_subplot(gs[0]);b=fig.add_subplot(gs[1])
    grey='#E5E9ED'
    for y,arm in [(1,'historical'),(0,'generative_replacement')]:
        a.barh(y,6000,height=.58,color=grey,edgecolor='white',linewidth=.7)
        a.barh(y,6000,left=6000,height=.58,color=COLORS[arm],edgecolor='white',linewidth=.7)
        a.text(3000,y,'Same historical molecules\n6,000',ha='center',va='center',fontsize=7,color='#26313B')
        a.text(9000,y,('Additional historical\n6,000' if arm=='historical' else 'Generated molecules\n6,000'),ha='center',va='center',fontsize=7,color='white')
    a.set_yticks([1,0],['Historical Retro* only','Historical + generated'])
    a.set_xlim(0,12000);a.set_ylim(-.65,1.55)
    a.set_xticks([]);a.spines['bottom'].set_visible(False);a.spines['left'].set_visible(False)
    a.tick_params(axis='y',length=0,pad=8)
    a.set_title('Controlled training-data replacement',loc='left',pad=3)
    a.text(.5,-.05,'Both arms: 12,000 molecules; ES/HS = 6,000/6,000.\nMatched properties and scaffold counts; scaffold identities allowed to differ.',
        transform=a.transAxes,ha='center',va='top',fontsize=6.7,color='#333333')
    a.text(-.035,1.12,'a',transform=a.transAxes,fontweight='bold',fontsize=11,va='top')
    seeds=sorted(d.replicate.unique());offsets=np.linspace(-.035,.035,len(seeds))
    b.axvspan(2.53,3.47,color=COLORS['generative_replacement'],alpha=.065,zorder=0)
    for i,split in enumerate(SPLITS):
        for j,arm in enumerate(ARMS):
            vals=d[(d.split==split)&(d.arm==arm)].set_index('replicate').loc[seeds,'AUROC'].to_numpy()
            x=i+(-.18 if j==0 else .18);mean=vals.mean();sd=vals.std(ddof=1)
            b.bar(x,mean-.5,bottom=.5,width=.32,color=COLORS[arm],alpha=.82,zorder=2)
            b.errorbar(x,mean,yerr=sd,fmt='none',color='#28323B',lw=.7,capsize=2,zorder=4)
            b.text(x,mean+sd+.009,f'{mean:.4f}',ha='center',va='bottom',fontsize=7.5,color=COLORS[arm])
    b.set_xticks(range(len(SPLITS)),['TS1','TS2','TS3','TS4'])
    b.set_xlim(-.62,3.62);b.set_ylim(.5,1.075);b.set_yticks(np.arange(.5,1.01,.1))
    b.set_ylabel('GESA AUROC');b.set_xlabel('Fixed evaluation cohort',labelpad=6)
    b.set_title('Performance of the same GESA architecture after training',loc='left',pad=10)
    b.legend(handles=[Patch(facecolor=COLORS[arm],alpha=.82,label=label) for arm,label in zip(ARMS,['Trained on historical Retro* only','Trained on historical + generated'])],loc='upper center',ncol=2,fontsize=7)
    ts4=d[d.split=='TS4'].pivot(index='replicate',columns='arm',values='AUROC')
    delta=(ts4.generative_replacement-ts4.historical).mean()
    b.text(3,.89,f'TS4: ΔAUROC = +{delta:.4f}',ha='center',va='center',fontsize=7.5,fontweight='bold',color='#146B65')
    b.text(-.035,1.075,'b',transform=b.transAxes,fontweight='bold',fontsize=11,va='top')
    base=ROOT/'figures/Fig_S5_TS1_TS4_training_data_effect'
    for ext in ['pdf','svg','png','tiff']:
        kw={'dpi':600} if ext in ['png','tiff'] else {}
        if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(base.with_suffix('.'+ext),**kw)
    fig.savefig(base.parent/(base.name+'_preview.png'),dpi=150)
    plt.close(fig)
    d.to_csv(ROOT/'figures/figure_source_data_TS1_TS4_performance.csv',index=False)
    pd.DataFrame([{'arm':arm,'shared_historical':6000,'additional_historical':6000 if arm=='historical' else 0,'generated':6000 if arm=='generative_replacement' else 0,'N':12000,'ES':6000,'HS':6000} for arm in ARMS]).to_csv(ROOT/'figures/figure_source_data_TS1_TS4_design.csv',index=False)
    en='''**Figure S5. Effect of generative-molecule supervision under controlled training-data replacement.** (a) Training-data composition. Both arms shared the same 6,000 historical GASA-Retro* molecules. The remaining 6,000 molecules were either additional historical GASA-Retro* molecules or property-matched molecules generated by TarPass/MolGenBench models. Each arm contained 6,000 ES and 6,000 HS molecules. Seven molecular properties were matched within classes, and distinct non-chiral scaffold-key counts were matched within each replicate while allowing scaffold identities to differ. (b) The same GESA architecture was trained separately on each arm using identical pretrained encoder weights, a fixed training seed, and the same training and validation-selection settings, then evaluated on the same TS1–TS4 molecules. Bars and error bars show the mean and sample standard deviation over three independently sampled training-data pairs; AUROC is plotted from 0.5, the random-ranking reference, and numerical labels show means. HS was the positive class. Test-cohort uncertainty was assessed separately with 2,000 paired class-stratified bootstrap replicates per fitted pair and is provided in the source data.\n'''
    zh='''**补充图S5．受控训练数据替换下生成分子监督的影响。**（a）训练数据组成。两组共用同一批6,000个历史GASA-Retro*分子，其余6,000个分别为额外历史GASA-Retro*分子，或与之进行性质匹配的TarPass／MolGenBench生成分子。每组均含6,000个ES和6,000个HS分子；在类别内匹配七项分子性质，并在每次重复中匹配非手性骨架键数量，同时保留骨架身份差异。（b）使用完全相同的预训练编码器权重、固定训练随机种子、相同训练设置及验证集选模规则，分别训练同一种GESA架构，再在相同TS1–TS4测试分子上评价。柱高和误差线为三次独立训练数据抽样的均值及样本标准差。AUROC纵轴从随机排序参照值0.5开始，数字标签表示均值。HS为阳性类。测试队列不确定性另采用每对模型2,000次按类别分层的配对Bootstrap估计，数值保存在源数据中。\n'''
    (ROOT/'figures/caption_en_zh.md').write_text(en+'\n'+zh,encoding='utf-8')
    print('Rendered TS1–TS4 mean/SD submission figure; individual predictions and all original cohorts retained in the source data.')

if __name__=='__main__':main()
