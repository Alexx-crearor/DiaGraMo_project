"""Static paper-oriented XAI figures, with source hashes and output units."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from xai_common import OUT,sha,write_json
from report_xai import load_attributions
from diagramo.explain import aggregate_shap


def main():
    if not json.loads((OUT/'validation.json').read_text()).get('numerical_validation_passed'): raise ValueError('Run numerical report first')
    mapping=pd.read_csv(OUT/'feature_mapping.csv'); data=load_attributions()
    exp=shap.Explanation(values=data['raw_path'],base_values=data['base_raw_path'],data=data['feature_values'],feature_names=mapping.full_name.tolist())
    features=pd.read_csv(OUT/'feature_importance.csv'); tasks=pd.read_csv(OUT/'task_importance.csv'); groups=pd.read_csv(OUT/'group_importance.csv')
    fusion=pd.read_csv(OUT/'fusion_participants.csv'); cases=json.loads((OUT/'local_cases.json').read_text())
    figdir=OUT/'figures'; figdir.mkdir(exist_ok=True); manifest=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'diagramo-xai-42'})
    attribution_sources=['feature_mapping.csv',*[f'attributions/kinematic_fold_{f}.npz' for f in range(6)]]
    def save(fig,name,description,sources,units,n_participants=258):
        for ext in ('png','svg'): fig.savefig(figdir/(name+'.'+ext),dpi=300,facecolor='white')
        manifest.append(dict(name=name,description=description,sources=sources,units=units,n_participants=n_participants,source_cohort_size=258,
            uncertainty='SD between six outer fold models/test samples when shown; not CI',
            source_sha256={p:sha(OUT/p) for p in sources},
            output_sha256={ext:sha(figdir/(name+'.'+ext)) for ext in ('png','svg')}))
        plt.close(fig)
    np.random.seed(42)
    shap.summary_plot(exp.values,exp.data,feature_names=exp.feature_names,max_display=20,show=False,plot_size=(14,9),rng=np.random.default_rng(42))
    fig=plt.gcf(); fig.subplots_adjust(left=.43,bottom=.12); plt.gca().set_xlabel('SHAP value for group=1 (raw log-odds)')
    save(fig,'shap_summary','Legacy summary of fixed OOF TreeSHAP; same evidence as modern beeswarm.',attribution_sources,'log-odds')
    shap.plots.beeswarm(exp,max_display=20,show=False,plot_size=(14,9))
    fig=plt.gcf(); fig.subplots_adjust(left=.43,bottom=.12); plt.gca().set_xlabel('SHAP value for group=1 (raw log-odds)')
    save(fig,'shap_beeswarm','Modern beeswarm, 258 held-out rows from six fixed models.',attribution_sources,'log-odds; color = raw input value')
    shap.plots.bar(exp,max_display=20,show=False)
    fig=plt.gcf(); fig.set_size_inches(14,9); fig.subplots_adjust(left=.43,bottom=.12)
    plt.gca().set_xlabel('Mean absolute SHAP (log-odds)')
    save(fig,'shap_bar','Global mean absolute feature contributions; not causal importance.',attribution_sources,'mean absolute log-odds')
    j=int(features.feature_index.iloc[0]); shap.plots.scatter(exp[:,j],show=False)
    fig=plt.gcf(); fig.set_size_inches(11,6); fig.subplots_adjust(left=.12,right=.97,bottom=.21)
    plt.gca().set_ylabel('SHAP (raw log-odds)'); plt.gca().set_title('Observed dependence across six outer models; not an intervention')
    save(fig,'shap_dependence','Highest mass feature dependence scatter, descriptive post-hoc feature choice.',attribution_sources+['feature_importance.csv'],'raw feature units versus log-odds')
    ids=data['participant_ids'].astype(str)
    for case in cases:
        i=int(np.flatnonzero(ids==case['participant_id'])[0]); row=fusion[(fusion.scenario=='adaptive__KC')&(fusion.participant_id==ids[i])].iloc[0]
        shap.plots.waterfall(exp[i],max_display=12,show=False)
        fig=plt.gcf(); fig.set_size_inches(15,8); fig.subplots_adjust(left=.45,right=.96,top=.83,bottom=.15)
        fig.suptitle(f"{case['outcome']} | {ids[i]} | outer fold {case['fold']+1}\npK={row.kinematic_probability:.3f}; pFusion={row.fusion_probability:.3f}; waterfall units=log-odds",fontsize=12,y=.98)
        save(fig,'waterfall_'+case['outcome'],'First lexicographic participant in outcome; explanation of Kinematic, not feature SHAP of Fusion.',attribution_sources+['local_cases.json','fusion_participants.csv'],'raw log-odds',1)
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained'); pos=np.arange(len(tasks))
    ax.barh(pos,tasks.sum_mean_abs_shap,xerr=tasks.fold_sd_mass,color='#0072B2',capsize=3)
    ax.set(yticks=pos,yticklabels=tasks.name,xlabel='Sum of mean |SHAP| (log-odds); error bars = fold SD',title='Task attribution mass, all 14 tasks')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    save(fig,'task_importance','Each task has 74 features; six-fold SD is not a confidence interval.',['task_importance.csv'],'mean absolute log-odds mass')
    fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained'); pos=np.arange(len(groups))
    labels=[f'{r.name} (n={r.n_features})' for r in groups.itertuples()]
    axes[0].barh(pos,groups.sum_mean_abs_shap,xerr=groups.fold_sd_mass,capsize=3,color='#0072B2')
    axes[1].barh(pos,groups.per_feature_mean_abs,xerr=groups.fold_sd_mass/groups.n_features,capsize=3,color='#E69F00')
    for ax in axes: ax.set(yticks=pos,yticklabels=labels); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    axes[0].set(xlabel='Attribution mass (log-odds)',title='Total; groups have different feature counts')
    axes[1].set(xlabel='Mass / number of features',title='Per-feature average; same group order')
    save(fig,'feature_groups','Source-prefix groups; compare total and per-feature average to expose count effects.',['group_importance.csv'],'log-odds mass or mean per feature')
    heat=pd.read_csv(OUT/'task_group_importance.csv').pivot(index='task',columns='group',values='sum_mean_abs_shap').reindex(index=tasks.name,columns=groups.name)
    fig,ax=plt.subplots(figsize=(13,8),layout='constrained'); im=ax.imshow(heat,aspect='auto',cmap='viridis',vmin=0)
    ax.set(xticks=np.arange(len(heat.columns)),xticklabels=heat.columns,yticks=np.arange(len(heat)),yticklabels=heat.index,title='Task × source feature group: attribution mass')
    ax.tick_params(axis='x',rotation=45); fig.colorbar(im,ax=ax,label='Mean sum |SHAP| (log-odds)')
    save(fig,'task_group_heatmap','All task/group cells, no omission of low attribution cells.',['task_group_importance.csv'],'mean absolute log-odds mass')
    ranks=pd.read_csv(OUT/'task_rank_stability.csv').set_index('name').loc[tasks.name,[f'fold_{f}_rank' for f in range(6)]]
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained'); im=ax.imshow(ranks,cmap='viridis_r',vmin=1,vmax=14,aspect='auto')
    for i in range(14):
        for f in range(6): ax.text(f,i,f'{ranks.iloc[i,f]:.0f}',ha='center',va='center',color='white' if ranks.iloc[i,f]<7 else 'black')
    ax.set(xticks=np.arange(6),xticklabels=np.arange(1,7),yticks=np.arange(14),yticklabels=ranks.index,xlabel='Outer fold',title='Task rank stability (1 = largest attribution mass)')
    fig.colorbar(im,ax=ax,label='Rank'); save(fig,'task_fold_ranks','Six models/test samples; dependent training folds, not independent replications.',['task_rank_stability.csv'],'rank')
    perm=pd.read_csv(OUT/'permutation_importance.csv'); perm=perm[perm.level=='task'].set_index('block').loc[tasks.name]
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained'); pos=np.arange(len(perm))
    ax.errorbar(perm.mean_drop_f1_macro_pp,pos,xerr=perm.fold_sd_drop_f1_macro_pp,fmt='o',capsize=3,color='#0072B2')
    ax.axvline(0,color='black',linestyle='--'); ax.set(yticks=pos,yticklabels=perm.index,xlabel='Baseline F1 − permuted F1 (pp); mean ± fold SD',title='Joint task permutation on held-out folds; 30 repeats/fold')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    save(fig,'grouped_permutation_tasks','All task blocks ordered by SHAP mass; retain negative score drops.',['permutation_importance.csv','task_importance.csv'],'F1 macro percentage points')
    styles={'corrected':('#0072B2','o',55),'harmed':('#D55E00','X',65),'unchanged_correct':('#999999','.',15),'unchanged_wrong':('#333333','^',30)}
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained',sharey=True)
    for ax,scenario in zip(axes,['late__KC','adaptive__KC']):
        d=fusion[fusion.scenario==scenario]
        for status,(color,marker,size) in styles.items():
            frame=d[d.outcome==status]; ax.scatter(frame.kinematic_probability,frame.delta_cwt,color=color,marker=marker,s=size,label=f'{status} (n={len(frame)})',alpha=.8)
        ax.axhline(0,color='black',linewidth=.8); ax.axvline(.5,color='black',linestyle='--',linewidth=.8)
        ax.set(xlabel='Kinematic p(group=1)',ylabel='pFusion − pK',title=scenario,xlim=(0,1)); ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig,'fusion_probability_correction','Exact probability delta versus baseline confidence; participant outcomes are descriptive, not used to tune gates.',['fusion_participants.csv'],'probability difference')
    counts=fusion.groupby(['scenario','outcome']).size().unstack(fill_value=0).reindex(columns=list(styles))
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained'); pos=np.arange(4)
    for j,(scenario,row) in enumerate(counts.iterrows()):
        ax.barh(pos+(j-.5)*.33,row.values,height=.3,label=scenario,color=['#0072B2','#E69F00'][j],hatch=['//','xx'][j])
        for i,val in enumerate(row): ax.text(val+1,pos[i]+(j-.5)*.33,str(val),va='center',fontsize=9)
    ax.set(yticks=pos,yticklabels=counts.columns,xlabel='Participants (n=258 per method)',title='Fusion outcomes relative to Kinematic'); ax.invert_yaxis(); ax.legend(); ax.grid(axis='x',alpha=.2)
    save(fig,'fusion_corrected_harmed','All four outcomes, including unchanged errors and corrections that harm baseline.',['fusion_participants.csv'],'participant counts')
    components=pd.read_csv(OUT/'cwt_component_importance.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained')
    sources=['cwt_component_importance.csv']; displayed_folds=[]
    for ax,scenario in zip(axes,['late__KC','adaptive__KC']):
        d=components[(components.scenario==scenario)&components.reducer.str.startswith('pca')]
        f=int(d.fold.min()); displayed_folds.append(f); d=d[d.fold==f].sort_values('rank').head(15)
        ax.barh(np.arange(len(d)),d.mean_abs_logit_contribution,color='#0072B2')
        ax.set(yticks=np.arange(len(d)),yticklabels=d.component,xlabel='Mean |linear contribution| (CWT log-odds)',title=f'{scenario}: first PCA fold {f+1}')
        ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    save(fig,'cwt_pca_components','Fold-specific PCA coordinates, 43 held-out participants per panel; PC indices not pooled across folds or mapped to biology/tasks.',sources,'CWT classifier log-odds',43*len(set(displayed_folds)))
    names,signed,_=aggregate_shap(exp.values,mapping,'task')
    case=next(c for c in cases if c['outcome']=='corrected'); i=int(np.flatnonzero(ids==case['participant_id'])[0])
    task_exp=shap.Explanation(values=signed[i],base_values=exp.base_values[i],feature_names=names)
    shap.plots.waterfall(task_exp,max_display=14,show=False)
    fig=plt.gcf(); fig.set_size_inches(13,9); fig.subplots_adjust(left=.37,right=.96,bottom=.13,top=.87)
    fig.suptitle(f'{ids[i]}: signed sums of individual SHAP by task\nNot a joint-masker Group SHAP computation',fontsize=12)
    save(fig,'task_waterfall','Task sums reconstruct margin, unlike global absolute attribution mass.',attribution_sources+['local_cases.json'],'signed raw log-odds',1)
    write_json(figdir/'manifest.json',manifest)
    print('Exported',len(manifest),'PNG/SVG figure pairs',flush=True)


if __name__=='__main__': main()
