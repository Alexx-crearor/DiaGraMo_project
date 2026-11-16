"""Aggregate nested CV, paired OOF, robustness and Kinematic XAI comparisons."""
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from task_transformer_common import ROOT,OUT,CONFIG,FAMILIES,METRICS,data,score,write
from diagramo.kinematic_xai import feature_mapping,aggregate,prediction_changes

TABLES=OUT/'tables'


def save(name,frame):
    TABLES.mkdir(parents=True,exist_ok=True);frame.to_csv(TABLES/(name+'.csv'),index=False);return frame


def read_outer():
    frames,metadata=[],[]
    for parent in [OUT/'folds'/f'fold_{f}'/'outer' for f in range(6)]+[OUT/'controls'/f'fold_{f}' for f in range(6)]:
        for p in sorted(parent.glob('*.oof.csv')):
            frames.append(pd.read_csv(p,float_precision='round_trip'))
            info=json.loads(p.with_name(p.name.replace('.oof.csv','.json')).read_text())
            metadata.append(info)
    original=pd.concat(frames,ignore_index=True)
    selected=[]
    for f in range(6):
        decision=json.loads((OUT/'folds'/f'fold_{f}'/'selection.json').read_text())
        d=original[(original.fold==f)&(original.model==decision['nested_selected_family'])].copy()
        d['source_model']=d.model;d['model']='nested_selected';selected.append(d)
    return pd.concat([original,*selected],ignore_index=True),metadata


def main():
    X,y,ids,schema,assignments,baseline,splits=data()
    oof,metadata=read_outer()
    save('oof_predictions',oof)
    fold_rows=[]
    for (model,seed,f),d in oof.groupby(['model','seed','fold'],sort=False):
        fold_rows.append(dict(model=model,seed=seed,fold=f,**score(d.label,d.probability,d.prediction)))
    folds=save('fold_results',pd.DataFrame(fold_rows))
    rows=[]
    for (model,seed),d in oof.groupby(['model','seed'],sort=False):
        assert len(d)==258 and d.participant_id.is_unique
        df=folds[(folds.model==model)&(folds.seed==seed)]
        meta=[r for r in metadata if r['model']==model and r['seed']==seed]
        row=dict(model=model,seed=seed,**score(d.label,d.probability,d.prediction),
              **{'mean_'+k:df[k].mean() for k in METRICS},**{'std_'+k:df[k].std(ddof=1) for k in METRICS})
        if meta:
            params=[r['parameters'] for r in meta if 'parameters' in r]
            row.update(parameters_mean=np.mean(params) if params else np.nan,
                       parameters_min=min(params) if params else np.nan,parameters_max=max(params) if params else np.nan,
                       train_seconds_mean=np.mean([r['train_seconds'] for r in meta if 'train_seconds'in r]) if params else np.nan,
                       task_count_mean=np.mean([np.asarray(r['present']).sum() if 'present'in r else len(r['columns'])/74 for r in meta]))
        rows.append(row)
    summary=save('performance_summary',pd.DataFrame(rows))
    primary=summary[summary.seed==42]
    save('ablation_primary',primary)
    save('seed_sensitivity',summary[summary.model.isin([*FAMILIES,'lightgbm','lightgbm_without_text','nested_selected'])])
    seed_summary=summary[summary.model.isin([*FAMILIES,'lightgbm','lightgbm_without_text','nested_selected'])].groupby('model')[list(METRICS)].agg(['mean','std']).reset_index()
    seed_summary.columns=['_'.join(filter(None,c)) if isinstance(c,tuple) else c for c in seed_summary.columns]
    save('seed_sensitivity_summary',seed_summary)
    paired,paired_summary=[],[]
    base=baseline.set_index('participant_id')
    for model in primary.model:
        d=oof[(oof.model==model)&(oof.seed==42)].set_index('participant_id').loc[ids]
        changes=prediction_changes(y,base.loc[ids].prediction.to_numpy(),d.prediction.to_numpy())
        for i,pid in enumerate(ids):
            paired.append(dict(model=model,participant_id=pid,label=int(y[i]),fold=int(d.iloc[i].fold),
                baseline_prediction=int(base.loc[pid,'prediction']),prediction=int(d.iloc[i].prediction),
                baseline_probability=base.loc[pid,'probability'],probability=d.iloc[i].probability,
                probability_delta=d.iloc[i].probability-base.loc[pid,'probability'],change=changes[i]))
        row=dict(model=model,corrected=int((changes=='corrected').sum()),harmed=int((changes=='harmed').sum()),
                 unchanged_correct=int((changes=='unchanged_correct').sum()),unchanged_wrong=int((changes=='unchanged_wrong').sum()))
        for metric in METRICS:
            b=folds[(folds.model=='lightgbm')&(folds.seed==42)].set_index('fold')[metric]
            t=folds[(folds.model==model)&(folds.seed==42)].set_index('fold')[metric]
            row['mean_delta_'+metric]=(t-b).mean();row['std_delta_'+metric]=(t-b).std(ddof=1)
        paired_summary.append(row)
    paired=save('paired_participant_changes',pd.DataFrame(paired));save('paired_comparisons',pd.DataFrame(paired_summary))
    confusion=[]
    for model in primary.model:
        d=oof[(oof.model==model)&(oof.seed==42)]
        for actual in (0,1):
            for pred in (0,1):
                confusion.append(dict(model=model,label=actual,prediction=pred,n=int(((d.label==actual)&(d.prediction==pred)).sum())))
    save('confusion_matrices',pd.DataFrame(confusion))
    searches=[];curves=[];selected=[]
    for f in range(6):
        searches.append(pd.read_csv(OUT/'folds'/f'fold_{f}'/'inner_search.csv'))
        decision=json.loads((OUT/'folds'/f'fold_{f}'/'selection.json').read_text())
        for family,c in decision['families'].items():selected.append(dict(fold=f,family=family,**c))
        selected.append(dict(fold=f,family='nested_selected',source_model=decision['nested_selected_family'],candidate=decision['nested_selected_candidate']))
        for p in (OUT/'folds'/f'fold_{f}').rglob('*_history.json'):
            history=json.loads(p.read_text())
            source=p.with_name(p.name.replace('_history.json','.json'))
            meta=json.loads(source.read_text())
            for row in history:curves.append(dict(fold=f,scope='inner' if 'inner'in p.parts else 'outer_train',
                 model=meta.get('model',meta.get('family')),candidate=meta['candidate'],inner=meta.get('inner',-1),seed=meta['seed'],**row))
    search=save('inner_search',pd.concat(searches,ignore_index=True));save('selected_configurations',pd.DataFrame(selected))
    curves=save('learning_curves',pd.DataFrame(curves))
    overfit=[]
    for row in search.itertuples():
        history=curves[(curves.scope=='inner')&(curves.fold==row.fold)&(curves.model==row.family)&(curves.candidate==row.candidate)&(curves.inner==row.inner)]
        point=history[history.epoch==row.best_epoch].iloc[0]
        overfit.append(dict(fold=row.fold,family=row.family,candidate=row.candidate,inner=row.inner,best_epoch=row.best_epoch,
              train_loss=point.train_loss,validation_loss=point.validation_loss,loss_gap=point.validation_loss-point.train_loss,
              train_accuracy=point.train_accuracy,validation_accuracy=point.validation_accuracy))
    save('overfitting_diagnostics',pd.DataFrame(overfit))
    matched=[]
    comparisons=[('matched_transformer_no_type','matched_task_mlp','self_attention'),
                 ('matched_transformer_type','matched_transformer_no_type','task_type_embedding'),
                 ('matched_motor_normalized','matched_transformer_no_type','motor_normalization'),
                 ('matched_train_augmented','matched_transformer_no_type','training_augmentation'),
                 ('matched_xai_selected','matched_transformer_type','train_only_task_selection')]
    for a,b,label in comparisons:
        row=dict(comparison=label,model=a,reference=b)
        for metric in METRICS:
            row['oof_delta_'+metric]=float(primary.set_index('model').loc[a,metric]-primary.set_index('model').loc[b,metric])
            left=folds[(folds.model==a)&(folds.seed==42)].set_index('fold')[metric]
            right=folds[(folds.model==b)&(folds.seed==42)].set_index('fold')[metric]
            row['mean_delta_'+metric]=(left-right).mean();row['std_delta_'+metric]=(left-right).std(ddof=1)
        matched.append(row)
    save('matched_ablation',pd.DataFrame(matched))
    analysis={name:pd.concat([pd.read_csv(OUT/'analysis'/f'fold_{f}'/(name+'.csv')) for f in range(6)],ignore_index=True)
              for name in ('robustness_oof','robustness_folds','permutation','masking_oof','attention','inference_cost','feature_ablation_oof')}
    for name,df in analysis.items():save(name,df)
    robust=analysis['robustness_folds'].groupby(['model','condition','fold'],sort=False).first().reset_index()
    save('robustness_by_fold',robust)
    columns=[*METRICS,*['delta_'+k for k in METRICS],'mean_absolute_probability_shift','attribution_rank_spearman']
    robustness=robust.groupby(['model','condition'])[columns].agg(['mean','std']).reset_index()
    robustness.columns=['_'.join(filter(None,c)) if isinstance(c,tuple)else c for c in robustness.columns]
    save('robustness_summary',robustness)
    masking=[]
    for (model,kind,removed,repeat,f),d in analysis['masking_oof'].groupby(['model','kind','removed','repeat','fold'],sort=False):
        s=score(d.label,d.probability,d.prediction)
        original_metrics=folds[(folds.model==model)&(folds.seed==42)&(folds.fold==f)].iloc[0]
        masking.append(dict(model=model,kind=kind,removed=removed,repeat=repeat,fold=f,**s,
                           **{'drop_'+k:original_metrics[k]-s[k] for k in METRICS}))
    masks=save('masking_by_fold',pd.DataFrame(masking))
    # Average repeated missing-task draws within fold, then describe six folds.
    masksummary=masks.groupby(['model','kind','fold'])[['drop_'+k for k in METRICS]].mean().groupby(['model','kind']).agg(['mean','std']).reset_index()
    masksummary.columns=['_'.join(filter(None,c)) if isinstance(c,tuple)else c for c in masksummary.columns]
    save('masking_summary',masksummary)
    zero=[]
    for (model,task,f),d in analysis['feature_ablation_oof'].groupby(['model','task','fold'],sort=False):
        s=score(d.label,d.probability,d.prediction)
        original_metrics=folds[(folds.model==model)&(folds.seed==42)&(folds.fold==f)].iloc[0]
        zero.append(dict(model=model,task=task,fold=f,**s,**{'drop_'+k:original_metrics[k]-s[k] for k in METRICS}))
    zero=save('feature_ablation_by_fold',pd.DataFrame(zero))
    perm=analysis['permutation'].groupby(['model','level','block','fold'])[['drop_'+k for k in METRICS]].mean().reset_index()
    save('permutation_by_fold',perm)
    psum=perm.groupby(['model','level','block'])[['drop_'+k for k in METRICS]].agg(['mean','std']).reset_index()
    psum.columns=['_'.join(filter(None,c)) if isinstance(c,tuple)else c for c in psum.columns]
    save('permutation_summary',psum)
    mapping=feature_mapping(schema['tasks'],schema['features_per_task'])
    mapping.to_csv(OUT/'feature_mapping.csv',index=False)
    importance=[];per_fold=[];stability=[];faithful=[];group_task=[];participant=[]
    for model in ['lightgbm',*FAMILIES]:
        phi=np.zeros_like(X,dtype=float)
        for f,(_,te) in enumerate(splits):
            path=OUT/'analysis'/f'fold_{f}'/('lightgbm_shap.npz'if model=='lightgbm'else model+'_ig.npz')
            phi[te]=np.load(path)['phi']
        for level in ('task','group'):
            mass=aggregate(phi,mapping,level)
            signed=aggregate(phi,mapping,level,absolute=False)
            mean=mass.mean();rank=mean.rank(ascending=False,method='average')
            by_fold=mass.groupby(assignments.fold.to_numpy()).mean()
            for name in mass:
                n_features=int((mapping[level]==name).sum())
                importance.append(dict(model=model,level=level,block=name,n_features=n_features,
                    mean_absolute_mass=mean[name],share_percent=100*mean[name]/mean.sum(),mass_per_feature=mean[name]/n_features,
                    rank=rank[name],fold_sd=by_fold[name].std(ddof=1),mean_signed=signed[name].mean()))
            for f in range(6):
                ranks=by_fold.loc[f].rank(ascending=False,method='average')
                for name in mass:per_fold.append(dict(model=model,level=level,block=name,fold=f,mass=by_fold.loc[f,name],rank=ranks[name]))
            for a,b in itertools.combinations(range(6),2):
                ta=set(by_fold.loc[a].nlargest(3).index);tb=set(by_fold.loc[b].nlargest(3).index)
                stability.append(dict(model=model,level=level,fold_a=a,fold_b=b,spearman=float(spearmanr(by_fold.loc[a],by_fold.loc[b]).statistic),
                                      top3_jaccard=len(ta&tb)/len(ta|tb)))
            p=psum[(psum.model==model)&(psum.level==level)].set_index('block').loc[mean.index]
            faithful.append(dict(model=model,level=level,comparison='attribution_vs_permutation',spearman=float(spearmanr(mean,p.drop_f1_macro_mean).statistic)))
            if level=='task':
                z=zero[zero.model==model].groupby('task').drop_f1_macro.mean().loc[mean.index]
                faithful.append(dict(model=model,level=level,comparison='attribution_vs_zero_feature_ablation',spearman=float(spearmanr(mean,z).statistic)))
                for i,pid in enumerate(ids):
                    for task in mass:participant.append(dict(model=model,participant_id=pid,fold=int(assignments.fold[i]),task=task,
                          absolute_mass=mass.loc[i,task],signed_logit=signed.loc[i,task]))
        for (task,group),frame in mapping.groupby(['task','group'],sort=False):
            group_task.append(dict(model=model,task=task,group=group,mean_absolute_mass=np.abs(phi[:,frame.column]).sum(1).mean()))
    imp=save('attribution_importance',pd.DataFrame(importance));save('attribution_by_fold',pd.DataFrame(per_fold))
    save('participant_attributions',pd.DataFrame(participant));save('task_group_importance',pd.DataFrame(group_task))
    stability=save('attribution_stability',pd.DataFrame(stability));save('faithfulness',pd.DataFrame(faithful))
    correlations=[]
    for level in ('task','group'):
        reference=imp[(imp.model=='lightgbm')&(imp.level==level)].set_index('block')
        for model in FAMILIES:
            sample=imp[(imp.model==model)&(imp.level==level)].set_index('block').loc[reference.index]
            correlations.append(dict(model=model,level=level,reference='lightgbm_shap',spearman=float(spearmanr(reference.mean_absolute_mass,sample.mean_absolute_mass).statistic)))
    save('xai_baseline_rank_comparison',pd.DataFrame(correlations))
    attention=analysis['attention'].groupby(['model','task']).incoming_attention.mean().reset_index()
    attention_agreement=[]
    for model in attention.model.unique():
        m=attention[attention.model==model].set_index('task')
        reference=imp[(imp.model==model)&(imp.level=='task')].set_index('block').loc[m.index]
        attention_agreement.append(dict(model=model,comparison='incoming_attention_vs_IG',spearman=float(spearmanr(m.incoming_attention,reference.mean_absolute_mass).statistic)))
    save('attention_agreement',pd.DataFrame(attention_agreement))
    local=[]
    for change in ('corrected','harmed','unchanged_correct','unchanged_wrong'):
        d=paired[(paired.model=='transformer')&(paired.change==change)].sort_values('participant_id')
        if len(d):
            row=d.iloc[0].to_dict();row['selection']='lexicographically first ID per paired outcome; post-hoc illustration';local.append(row)
    save('local_cases',pd.DataFrame(local))
    write(OUT/'validation.json',dict(complete=False,stage='tables_generated',outer_folds=6,
         inner_neural_fits=len(search),oof_rows=len(oof),analysis_models=8))
    print('TABLES GENERATED',len(oof),'OOF rows;',len(search),'inner fits',flush=True)


if __name__=='__main__':main()
