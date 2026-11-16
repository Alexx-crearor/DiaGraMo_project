"""Descriptive faithfulness/stability; never choose a model from held-out XAI."""
import itertools
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from raw_temporal_common import *
T=OUT/'tables'
def save(name,df):T.mkdir(exist_ok=True);df.to_csv(T/(name+'.csv'),index=False);return df
def rho(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(spearmanr(a,b).statistic)if np.ptp(a)>0 and np.ptp(b)>0 else float('nan')

def describe_examples(cases,task_occlusion,schema):
    """Describe the exact deterministic intervals displayed in the raw-signal plots."""
    from diagramo.temporal.preprocessing import preprocess
    descriptions=[]
    for row in cases.itertuples():
        q=task_occlusion[(task_occlusion.participant_id==row.participant_id)&(task_occlusion.model=='hierarchical')&(task_occlusion.seed==42)]
        task=q.iloc[q.logit_delta.abs().to_numpy().argmax()].task;ti=schema['tasks'].index(task)
        with np.load(OUT/'cache'/(row.participant_id+'.npz'))as cache:
            offset=sum(len(cache[f'n{j}'])for j in range(ti));inventory=cache[f'm{ti}']
        with np.load(OUT/'xai/examples'/(row.participant_id+'_seed_42_zero_joint.npz'))as attribution:
            channel_ig=attribution['x'][offset:offset+len(inventory)]
        window=int(np.abs(channel_ig).sum((1,2)).argmax());start,end=inventory[window,:2].astype(int)
        records,_,_=recording_map(row.participant_id,schema['tasks']);raw=records[task][1];pre=preprocess(raw)
        speed=pre['physical'][start:end,10][pre['velocity_valid'][start:end]]
        acceleration=np.hypot(pre['physical'][start:end,11],pre['physical'][start:end,12])[pre['acceleration_valid'][start:end]]
        pressure=np.asarray(raw['pressure'])[start:end];source=raw_svc_path(row.participant_id,records[task][0])
        descriptions.append(dict(participant_id=row.participant_id,category=row.category,task=task,window_index=window,
            raw_start=int(start),raw_end_exclusive=int(end),samples=int(end-start),pen_state=int(pre['pen'][start]),
            time_start=float(pre['time'][start]),time_last=float(pre['time'][end-1]),
            elapsed_start=float(pre['time'][start]-pre['time'][0]),elapsed_last=float(pre['time'][end-1]-pre['time'][0]),
            duration=float(pre['time'][end-1]-pre['time'][start]),valid_speed_samples=len(speed),
            speed_median=float(np.median(speed))if len(speed)else np.nan,speed_max=float(speed.max())if len(speed)else np.nan,
            valid_acceleration_samples=len(acceleration),acceleration_median=float(np.median(acceleration))if len(acceleration)else np.nan,
            acceleration_max=float(acceleration.max())if len(acceleration)else np.nan,
            pressure_min=float(pressure.min()),pressure_median=float(np.median(pressure)),pressure_max=float(pressure.max()),
            boundaries=int(pre['boundary'][start:end].sum()),invalid_time_samples=int((pre['physical'][start:end,15]==0).sum()),
            source_file=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),
            selection='Task: max absolute occlusion effect; window: max absolute sample-channel IG, seed42 zero_joint'))
    return pd.DataFrame(descriptions)
def main():
    ids,y,schema,assignment,splits=cohort();frames={}
    for name in ('task_occlusion','segment_occlusion','stroke_occlusion','temporal_occlusion','perturbation_oof','embeddings','attention_supplementary','latency'):
        frames[name]=pd.concat([pd.read_csv(OUT/'xai'/f'fold_{f}'/(name+'.csv'),float_precision='round_trip')for f in range(6)],ignore_index=True)
        save(name,frames[name])
    task=frames['task_occlusion'];task['absolute_logit_delta']=task.logit_delta.abs()
    importance=task.groupby(['model','seed','task']).agg(mean_absolute_logit_delta=('absolute_logit_delta','mean'),mean_signed_logit_delta=('logit_delta','mean'),mean_absolute_probability_delta=('probability_delta',lambda x:x.abs().mean())).reset_index()
    importance['rank']=importance.groupby(['model','seed']).mean_absolute_logit_delta.rank(ascending=False,method='average');save('task_importance',importance)
    fold=task.groupby(['model','seed','fold','task']).absolute_logit_delta.mean().reset_index();save('task_importance_by_fold',fold)
    stability=[]
    for model in task.model.unique():
        d=fold[(fold.model==model)&(fold.seed==42)].pivot(index='task',columns='fold',values='absolute_logit_delta')
        for a,b in itertools.combinations(range(6),2):
            aa=set(d[a].nlargest(3).index);bb=set(d[b].nlargest(3).index)
            # A constant importance vector has no identifiable top tasks.
            jaccard=len(aa&bb)/len(aa|bb) if np.ptp(d[a])>0 and np.ptp(d[b])>0 else float('nan')
            stability.append(dict(model=model,kind='fold_rank',fold_a=a,fold_b=b,spearman=rho(d[a],d[b]),top3_jaccard=jaccard))
    save('fold_xai_stability',pd.DataFrame(stability))
    seedstab=[]
    for(model,pid),d in task.groupby(['model','participant_id']):
        a=d.pivot(index='task',columns='seed',values='absolute_logit_delta')
        for seed in (43,44):seedstab.append(dict(model=model,participant_id=pid,fold=int(d.fold.iloc[0]),seed=seed,spearman=rho(a[42],a[seed])))
    save('seed_xai_stability',pd.DataFrame(seedstab))
    geometry=[];embed=frames['embeddings'];cols=['z'+str(i)for i in range(32)]
    for(model,f),d in embed.groupby(['model','fold']):
        a=d[d.seed==42].sort_values('participant_id');reference=pdist(a[cols].to_numpy())
        for seed in (43,44):
            b=d[d.seed==seed].set_index('participant_id').loc[a.participant_id]
            geometry.append(dict(model=model,fold=f,seed=seed,participants=43,pairwise_distances=903,spearman=rho(reference,pdist(b[cols].to_numpy())),note='Rotation/translation invariant distance rank comparison within held-out subjects; pair distances are dependent'))
    save('representation_geometry_stability',pd.DataFrame(geometry))
    diagnostic=[];taskfold=[]
    for(model,seed,f,name),d in task.groupby(['model','seed','fold','task']):
        clean=metrics(d.label,d.probability);changed=metrics(d.label,d.removed_probability)
        taskfold.append(dict(model=model,seed=seed,fold=f,task=name,**{'drop_'+k:clean[k]-changed[k]for k in clean}))
    taskfold=save('task_occlusion_fold_metrics',pd.DataFrame(taskfold))
    for model in task.model.unique():
        a=importance[(importance.model==model)&(importance.seed==42)].set_index('task')
        b=taskfold[(taskfold.model==model)&(taskfold.seed==42)].groupby('task').drop_f1_macro.mean().loc[a.index]
        diagnostic.append(dict(model=model,comparison='mean_absolute_logit_delta_vs_task_macroF1_drop',spearman=rho(a.mean_absolute_logit_delta,b)))
    save('task_faithfulness',pd.DataFrame(diagnostic))
    old=pd.read_csv(ROOT/'results/kinematic_xai/exploratory/tables/task_importance.csv').set_index('task');comparison=[]
    for model in task.model.unique():
        a=importance[(importance.model==model)&(importance.seed==42)].set_index('task').loc[old.index]
        comparison.append(dict(model=model,reference='Kinematic TreeSHAP, descriptive only; different method/reference',spearman=rho(a.mean_absolute_logit_delta,old.mean_absolute_shap_mass)))
    save('kinematic_xai_rank_comparison',pd.DataFrame(comparison))
    perturb=[]
    for(kind,f,repeat),d in frames['perturbation_oof'].groupby(['kind','fold','repeat']):
        clean=metrics(d.label,d.clean_probability);changed=metrics(d.label,d.probability)
        perturb.append(dict(kind=kind,fold=f,repeat=repeat,**changed,**{'delta_'+k:changed[k]-clean[k]for k in changed},mean_absolute_probability_shift=float((d.probability-d.clean_probability).abs().mean())))
    pf=save('temporal_perturbation_folds',pd.DataFrame(perturb));cols=list(metrics(y,np.ones(len(y))*.55))+['delta_'+k for k in metrics(y,np.ones(len(y))*.55)]+['mean_absolute_probability_shift']
    ps=pf.groupby(['kind','fold'])[cols].mean().groupby('kind').agg(['mean','std']).reset_index();ps.columns=['_'.join(filter(None,c))if isinstance(c,tuple)else c for c in ps.columns];save('temporal_perturbation_summary',ps)
    examples=OUT/'xai/examples';selected=pd.read_csv(examples/'selected_cases.csv');sensitivity=[]
    save('behavioral_examples',describe_examples(selected,task,schema))
    for row in selected.itertuples():
        reference=np.load(examples/(row.participant_id+'_seed_42_zero_joint.npz'))
        for seed in (43,44):
            other=np.load(examples/(row.participant_id+f'_seed_{seed}_zero_joint.npz'))
            sensitivity.append(dict(participant_id=row.participant_id,comparison='seed_'+str(seed),task_spearman=rho(reference['task_mass'],other['task_mass']),patch_spearman=rho(np.abs(reference['x']).sum((1,2)),np.abs(other['x']).sum((1,2)))))
        for kind in ('structure_preserving','random_head','random_all'):
            path=examples/(row.participant_id+('_seed_42_'+kind if kind=='structure_preserving'else'_'+kind)+'.npz');other=np.load(path)
            sensitivity.append(dict(participant_id=row.participant_id,comparison=kind,task_spearman=rho(reference['task_mass'],other['task_mass']),patch_spearman=rho(np.abs(reference['x']).sum((1,2)),np.abs(other['x']).sum((1,2)))))
    save('ig_baseline_seed_randomization',pd.DataFrame(sensitivity))
    diagnostics=pd.read_csv(examples/'ig_validation.csv');components=[]
    for row in diagnostics.itertuples():
        filename=(row.participant_id+f'_seed_{row.seed}_{row.baseline}' if row.kind=='trained' else row.participant_id+'_'+row.kind)+'.npz'
        with np.load(examples/filename)as attribution:
            components.append(dict(participant_id=row.participant_id,category=row.category,seed=row.seed,baseline=row.baseline,kind=row.kind,
                reference=float(attribution['reference']),logit=float(attribution['logit']),
                sample_channel_sum=float(attribution['x'].sum()),patch_time_sum=float(attribution['patch_meta'].sum()),segment_time_sum=float(attribution['segment_meta'].sum()),
                residual=float(attribution['residual'])))
    save('ig_components',pd.DataFrame(components))
    for name in ('ig_validation','faithfulness','deletion_curves','selected_cases'):save(name,pd.read_csv(examples/(name+'.csv')))
    write(OUT/'xai/summary_complete.json',dict(complete=True,participants=258,cases=len(selected),segment_rows=len(frames['segment_occlusion']),stroke_rows=len(frames['stroke_occlusion'])))
    print('XAI TABLES COMPLETE',flush=True)

if __name__=='__main__':main()
