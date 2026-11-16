"""Participant OOF metrics and paired/nested/multiscale tables from saved fits."""
import itertools
from raw_temporal_common import *
TABLES=OUT/'tables'
def save(name,df):TABLES.mkdir(exist_ok=True);df.to_csv(TABLES/(name+'.csv'),index=False);return df
def main():
    ids,y,schema,assignment,splits=cohort();config=json.loads((ROOT/'configs/raw_temporal.json').read_text());frames=[];metadata=[];search=[];histories=[];choices=[]
    for f in range(6):
        folder=OUT/'folds'/f'fold_{f}';assert json.loads((folder/'complete.json').read_text())['complete']
        for p in (folder/'outer').glob('*.oof.csv'):
            frames.append(pd.read_csv(p,float_precision='round_trip'));metadata.append(json.loads(p.with_name(p.name.replace('.oof.csv','.json')).read_text()))
        decision=json.loads((folder/'selection.json').read_text());d=pd.concat(frames,ignore_index=True)
        alias=d[(d.fold==f)&(d.model==decision['nested_winner'])].copy();alias['source_model']=alias.model;alias['model']='nested_selected';frames.append(alias)
        for name,c in decision['choices'].items():choices.append(dict(fold=f,model=name,**c,nested_winner=decision['nested_winner']))
        search.append(pd.read_csv(folder/'inner_search.csv'))
        for p in folder.rglob('*_history.json'):
            meta=json.loads(p.with_name(p.name.replace('_history.json','.json')).read_text())
            for row in json.loads(p.read_text()):histories.append(dict(fold=f,model=meta['model'],candidate=meta['candidate'],seed=meta['seed'],inner=meta.get('inner',-1),selected_epoch=meta['best_epoch'],scope='inner'if'inner'in p.parts else'outer_train',**row))
    majority=[]
    for f,(tr,te)in enumerate(splits):
        label=int(np.bincount(y[tr]).argmax())
        majority.append(pd.DataFrame(dict(participant_id=ids[te],label=y[te],fold=f,model='train_majority_reference',seed=42,probability=float(label),prediction=label)))
    frames.extend(majority)
    base=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv',float_precision='round_trip');base['model']='kinematic_lightgbm_reference';base['seed']=42;frames.append(base)
    oof=save('oof_predictions',pd.concat(frames,ignore_index=True));foldrows=[]
    for(model,seed,f),d in oof.groupby(['model','seed','fold']):foldrows.append(dict(model=model,seed=seed,fold=f,**metrics(d.label,d.probability)))
    folds=save('fold_results',pd.DataFrame(foldrows));rows=[]
    decisions=[]
    for(model,seed,f),d in oof.groupby(['model','seed','fold']):
        decisions.append(dict(model=model,seed=seed,fold=f,n=len(d),predicted_negative=int((d.prediction==0).sum()),
            predicted_positive=int((d.prediction==1).sum()),single_class_predictions=d.prediction.nunique()==1,
            probability_min=d.probability.min(),probability_max=d.probability.max(),probability_std=d.probability.std(ddof=1),threshold=.5))
    save('decision_diagnostics',pd.DataFrame(decisions))
    for(model,seed),d in oof.groupby(['model','seed']):
        assert len(d)==258 and d.participant_id.is_unique
        fs=folds[(folds.model==model)&(folds.seed==seed)];row=dict(model=model,seed=seed,**metrics(d.label,d.probability))
        for key in metrics(d.label,d.probability):row['mean_'+key]=fs[key].mean();row['std_'+key]=fs[key].std(ddof=1)
        meta=[r for r in metadata if r['model']==model and r['seed']==seed]
        if meta:row.update(parameters=min(r['parameters']for r in meta),parameters_max=max(r['parameters']for r in meta),train_seconds_mean=np.mean([r['train_seconds']for r in meta]),inference_seconds_mean=np.mean([r['inference_seconds']for r in meta]),peak_gpu_bytes=max(r['peak_gpu_bytes']for r in meta))
        rows.append(row)
    summary=save('performance_summary',pd.DataFrame(rows));save('primary_ablation',summary[summary.seed==42]);save('inner_search',pd.concat(search));save('selected_configurations',pd.DataFrame(choices));save('learning_curves',pd.DataFrame(histories))
    cost_records=pd.concat([pd.concat(search).assign(scope='inner_search'),pd.DataFrame(metadata).assign(scope='outer_train')],ignore_index=True)
    costs=cost_records.groupby(['model','scope']).agg(n_fits=('train_seconds','size'),train_seconds_sum=('train_seconds','sum'),train_seconds_mean=('train_seconds','mean'),evaluation_seconds_sum=('inference_seconds','sum'),peak_tensor_bytes=('peak_gpu_bytes','max')).reset_index()
    assert costs.n_fits.sum()==312;save('computational_cost_by_scope',costs)
    seedsummary=summary[summary.model.isin(config['families']+['nested_selected'])].groupby('model')[list(metrics(y,np.full(258,.55)))].agg(['mean','std']).reset_index()
    seedsummary.columns=['_'.join(filter(None,c))if isinstance(c,tuple)else c for c in seedsummary.columns];save('seed_sensitivity',seedsummary)
    paired=[];changes=[];confusion=[];ablation=[]
    for model in summary[summary.seed==42].model:
        d=oof[(oof.model==model)&(oof.seed==42)].set_index('participant_id').loc[ids];b=base.set_index('participant_id').loc[ids]
        old=b.prediction.to_numpy()==y;new=d.prediction.to_numpy()==y
        c=np.select([~old&new,old&~new,old&new],['corrected','harmed','unchanged_correct'],default='unchanged_wrong')
        row=dict(model=model,corrected=int((c=='corrected').sum()),harmed=int((c=='harmed').sum()))
        for metric in ('f1_macro','accuracy','auc','balanced_accuracy','sensitivity','specificity'):
            f=folds[(folds.model==model)&(folds.seed==42)].set_index('fold')[metric];reference=folds[folds.model=='kinematic_lightgbm_reference'].set_index('fold')[metric]
            row['mean_delta_'+metric]=(f-reference).mean();row['std_delta_'+metric]=(f-reference).std(ddof=1)
        paired.append(row)
        for i,pid in enumerate(ids):changes.append(dict(model=model,participant_id=pid,fold=int(d.loc[pid,'fold']),label=int(y[i]),probability=float(d.loc[pid,'probability']),baseline_probability=float(b.loc[pid,'probability']),outcome=c[i]))
        for a,p in itertools.product((0,1),repeat=2):confusion.append(dict(model=model,label=a,prediction=p,n=int(((d.label==a)&(d.prediction==p)).sum())))
        if model in config['ablations']:
            r=dict(model=model,reference='hierarchical')
            for metric in ('f1_macro','accuracy','auc','balanced_accuracy'):
                left=folds[(folds.model==model)&(folds.seed==42)].set_index('fold')[metric];right=folds[(folds.model=='hierarchical')&(folds.seed==42)].set_index('fold')[metric]
                r['mean_delta_'+metric]=(left-right).mean();r['std_delta_'+metric]=(left-right).std(ddof=1)
            ablation.append(r)
    save('paired_comparisons',pd.DataFrame(paired));save('participant_changes',pd.DataFrame(changes));save('confusion_matrices',pd.DataFrame(confusion));save('temporal_multiscale_ablation',pd.DataFrame(ablation))
    h=pd.DataFrame(histories);diagnostics=[]
    for(f,m,c,j),d in h[h.scope=='inner'].groupby(['fold','model','candidate','inner']):
        d=d.sort_values('epoch');best=d[d.epoch==int(d.selected_epoch.iloc[0])].iloc[0];diagnostics.append(dict(fold=f,model=m,candidate=c,inner=j,best_epoch=int(best.epoch),loss_gap=best.validation_loss-best.train_loss,max_epochs_reached=int(d.epoch.max()==config['training']['maximum_epochs'])))
    save('overfitting',pd.DataFrame(diagnostics));write(OUT/'validation.json',dict(complete=False,stage='evaluation_tables_generated'))
    print('RAW TEMPORAL TABLES GENERATED',len(oof),flush=True)

if __name__=='__main__':main()
