"""Paper-draft figures from validated, participant-held-out Kinematic tables."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from task_transformer_common import ROOT,OUT,FAMILIES,sha,write

T=OUT/'tables';D=OUT/'figures';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','svg.hashsalt':'diagramo42',
 'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
manifest=[]
def read(name):return pd.read_csv(T/(name+'.csv'))
def save(name,fig,sources,caption):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(D/(name+'.'+ext),dpi=300,bbox_inches='tight',metadata={'Date':None}if ext=='svg'else None)
    manifest.append(dict(name=name,caption=caption,dpi=300,unit='participant; n=258, six outer folds',
       uncertainty='Fold SD (not a confidence interval); three seeds are sensitivity runs on the same cohort',
       sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
       outputs={e:sha(D/(name+'.'+e))for e in ('png','svg')}));plt.close(fig)
def heat(ax,df,title,cmap='viridis',center=False):
    A=df.to_numpy(float);limit=np.nanmax(np.abs(A)) or 1.
    im=ax.imshow(A,aspect='auto',cmap=cmap,vmin=-limit if center else None,vmax=limit if center else None)
    ax.set_xticks(range(len(df.columns)),df.columns,rotation=40,ha='right');ax.set_yticks(range(len(df)),df.index)
    ax.set_title(title);plt.colorbar(im,ax=ax,shrink=.8)

def main():
    fig,ax=plt.subplots(figsize=(11,3.3));ax.set_axis_off()
    labels=['14 tasks × 74 features','Train-fit imputation\n+ standardization','Shared 74 → d projection\n+ optional task-ID embedding',
      '1–2 masked attention blocks\n2/4 heads; d=32/64','Mean / attention pooling\n→ binary logit']
    for i,label in enumerate(labels):
        x=.005+i*.2
        ax.add_patch(FancyBboxPatch((x,.49),.18,.38,boxstyle='round,pad=.007',fc='#e8f1f8',ec='#255a78'))
        ax.text(x+.09,.68,label,ha='center',va='center',fontsize=8)
        if i<4:ax.annotate('',(x+.2,.68),(x+.18,.68),arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.27,'No chronological positional encoding. Missing tokens are masked in attention and pooling.\nTask-MLP: shared token feed-forward blocks, mean pooling, no attention.',ha='center',va='center')
    ax.text(.5,.05,'Outer: fixed 6 folds (43 test each)  |  Inner: 3 folds for configuration + early stopping\nFinal epochs: median inner best epoch; outer test is never monitored during fitting.',ha='center',va='center')
    save('01_architecture',fig,[ROOT/'configs/task_transformer.json',ROOT/'src/diagramo/task_transformer.py'],'Actual architecture and nested selection protocol.')
    primary=read('ablation_primary').set_index('model');names=['lightgbm',*FAMILIES,'nested_selected']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric in zip(axes.flat,['f1_macro','accuracy','auc','balanced_accuracy']):
        d=primary.loc[names];ax.barh(names,d['mean_'+metric],xerr=d['std_'+metric],color='#357b9c',capsize=2,label='Fold mean ± SD')
        ax.scatter(d[metric],names,c='#d67c27',s=20,zorder=4,label='Pooled OOF');ax.set_xlim(0,1);ax.set_xlabel(metric);ax.invert_yaxis()
    axes.flat[0].legend(fontsize=8,loc='lower left');save('02_predictive_ablation',fig,[T/'ablation_primary.csv'],'Fold mean±SD and pooled OOF; no outer-test winner selection.')
    d=read('matched_ablation');fig,ax=plt.subplots(figsize=(9,3.8));ax.barh(d.comparison,d.mean_delta_f1_macro,xerr=d.std_delta_f1_macro,capsize=3,color='#378e83');ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Paired fold Δ macro-F1 (mean ± SD)');ax.invert_yaxis()
    save('03_matched_ablation',fig,[T/'matched_ablation.csv'],'Matched architecture controls; epochs selected independently in inner CV.')
    seeds=read('seed_sensitivity');fig,ax=plt.subplots(figsize=(10,4.5))
    for i,name in enumerate(names):
        d=seeds[seeds.model==name].sort_values('seed');ax.scatter(i+np.linspace(-.16,.16,len(d)),d.f1_macro,s=25,label=name if False else None)
    ax.set_xticks(range(len(names)),names,rotation=30,ha='right');ax.set_ylim(0,1);ax.set_ylabel('OOF macro-F1; seeds 42, 43, 44')
    save('04_seed_sensitivity',fig,[T/'seed_sensitivity.csv'],'Fixed selected configurations, three training seeds; same participants and splits.')
    curves=read('learning_curves');fig,axes=plt.subplots(2,3,figsize=(12,6))
    for ax,name in zip(axes.flat,FAMILIES):
        d=curves[(curves.scope=='inner')&(curves.model==name)].groupby('epoch')[['train_loss','validation_loss']].mean()
        ax.plot(d.index,d.train_loss,label='Train');ax.plot(d.index,d.validation_loss,label='Inner validation');ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('Binary cross-entropy')
    axes.flat[0].legend();save('05_learning_curves',fig,[T/'learning_curves.csv'],'Descriptive mean over inner searches; later epochs contain fewer runs due to early stopping.')
    cost=read('inference_cost').groupby('model').median(numeric_only=True).reindex(['lightgbm','lightgbm_without_text',*FAMILIES]);fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].barh(cost.index,1000*cost.median_inference_seconds/43);axes[0].set_xlabel('Median cached-feature inference (ms / participant)');axes[0].invert_yaxis()
    p=primary.loc[FAMILIES];axes[1].barh(p.index,p.parameters_mean);axes[1].set_xlabel('Neural trainable parameters (mean across folds)');axes[1].invert_yaxis()
    save('06_compute_cost',fig,[T/'inference_cost.csv',T/'ablation_primary.csv'],'CPU LightGBM vs GPU neural timing includes preprocessing and transfer; excludes raw extraction; hardware differs.')
    robust=read('robustness_summary');fig,axes=plt.subplots(1,2,figsize=(13,5))
    heat(axes[0],robust.pivot(index='model',columns='condition',values='delta_f1_macro_mean'),'Controlled nuisance: fold mean Δ macro-F1','RdBu',True)
    heat(axes[1],robust.pivot(index='model',columns='condition',values='mean_absolute_probability_shift_mean'),'Mean absolute probability shift')
    save('07_nuisance_robustness',fig,[T/'robustness_summary.csv'],'Fixed post-fit raw-signal stresses, not external device or language validation.')
    masks=read('masking_summary');fig,ax=plt.subplots(figsize=(12,5));heat(ax,masks.pivot(index='model',columns='kind',values='drop_f1_macro_mean'),'Missing tasks: fold mean macro-F1 drop','RdBu_r',True)
    save('08_missing_tasks',fig,[T/'masking_summary.csv'],'Native neural token masks; tree Train-median replacement; draws averaged within each fold.')
    imp=read('attribution_importance');fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,level in zip(axes,['task','group']):
        d=imp[(imp.level==level)&imp.model.isin(['lightgbm','transformer','motor_normalized'])]
        heat(ax,d.pivot(index='block',columns='model',values='share_percent'),level+' attribution share (%)')
    save('09_attribution_patterns',fig,[T/'attribution_importance.csv'],'Absolute held-out IG / SHAP mass shares; different references; no causal interpretation.')
    st=read('attribution_stability');fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,col in zip(axes,['spearman','top3_jaccard']):
        d=st[st.level=='task'].groupby('model')[col].agg(['mean','std']);ax.barh(d.index,d['mean'],xerr=d['std'],capsize=2);ax.set_xlabel('Task rank '+col+'; 15 fold pairs, mean ± SD');ax.set_xlim(-1 if col=='spearman'else 0,1)
    save('10_attribution_stability',fig,[T/'attribution_stability.csv'],'Fold pairs are dependent; descriptive SD, not inferential evidence.')
    faithful=read('faithfulness');fig,ax=plt.subplots(figsize=(13,5));heat(ax,faithful[faithful.level=='task'].pivot(index='model',columns='comparison',values='spearman'),'Task attribution vs independent held-out perturbation (Spearman)','RdBu',True)
    save('11_faithfulness',fig,[T/'faithfulness.csv'],'Permutation and zero-feature ablation are independent checks; correlations do not prove causal validity.')
    att=read('attention');d=att.groupby(['task','model']).incoming_attention.mean().unstack('model');fig,ax=plt.subplots(figsize=(11,6));heat(ax,d,'Supplementary incoming attention (not attribution)')
    save('12_attention_supplementary',fig,[T/'attention.csv'],'Mean incoming layer/head attention; model-specific missing tokens have zero weight.')
    cm=read('confusion_matrices');fig,axes=plt.subplots(1,4,figsize=(12,3))
    for ax,name in zip(axes,['lightgbm','transformer','motor_normalized','nested_selected']):
        d=cm[cm.model==name].pivot(index='label',columns='prediction',values='n');ax.imshow(d,vmin=0,vmax=143,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(d.loc[i,j]),ha='center',va='center')
        ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel('Prediction');ax.set_ylabel('True group');ax.set_title(name)
    save('13_confusion_matrices',fig,[T/'confusion_matrices.csv'],'Seed 42 pooled OOF counts, each participant predicted once.')
    paired=read('paired_comparisons').set_index('model').loc[names[1:]];fig,ax=plt.subplots(figsize=(11,4.5));ax.barh(paired.index,paired.corrected,label='Corrected',color='#318875');ax.barh(paired.index,-paired.harmed,label='Harmed',color='#c65c53');ax.axvline(0,color='black',lw=.8);ax.legend();ax.set_xlabel('Participants relative to original LightGBM (n=258)')
    save('14_paired_outcomes',fig,[T/'paired_comparisons.csv'],'Same-participant corrected/harmed counts; no independent-cohort claim.')
    write(D/'manifest.json',manifest);print('FIGURES',len(manifest))

if __name__=='__main__':main()
