"""Artifact-only paired participant evaluation, bootstrap, calibration and errors."""
from scipy.stats import spearmanr
from raw_temporal_v2_common import *


def error_categories(y,kinematic,raw):
    kin=(np.asarray(kinematic)>.5)==y;other=(np.asarray(raw)>.5)==y
    return np.where(kin&other,'both_correct',np.where(kin,'kin_only',np.where(other,'raw_only','both_wrong')))


def collect():
    ids,y,schema,assignments,splits=cohort();parts=[]
    old=pd.read_csv(OLD/'tables/oof_predictions.csv',float_precision='round_trip')
    selected=old[old.model.isin(['cnn','tcn','transformer','hierarchical','nested_selected'])]
    selected=selected.copy();selected['model']='v1_'+selected.model
    parts.append(selected[['participant_id','label','fold','model','seed','probability']])
    baseline=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv',float_precision='round_trip')
    baseline['model']='kinematic_lightgbm';baseline['seed']=42
    parts.append(baseline[['participant_id','label','fold','model','seed','probability']])
    for name in ('nested_oof.csv','fusion_oof.csv'):
        frame=pd.read_csv(OUT/'phase4_complementarity'/name,float_precision='round_trip')
        parts.append(frame[['participant_id','label','fold','model','seed','probability']])
    result=pd.concat(parts,ignore_index=True)
    for (_,seed),group in result.groupby(['model','seed']):
        assert len(group)==258 and not group.participant_id.duplicated().any()
        aligned=group.set_index('participant_id').loc[ids]
        np.testing.assert_array_equal(aligned.label,y);np.testing.assert_array_equal(aligned.fold,assignments.fold)
        assert np.isfinite(aligned.probability).all() and aligned.probability.between(0,1).all()
    result.to_csv(OUT/'phase4_complementarity/all_comparison_oof.csv',index=False)
    return result


def fast_metrics(y,p):
    predicted=p>.5;positive=y==1;negative=~positive
    tp=np.sum(predicted&positive);tn=np.sum(~predicted&negative);fp=np.sum(predicted&negative);fn=np.sum(~predicted&positive)
    return dict(accuracy=float((tp+tn)/len(y)),f1_macro=float((2*tp/max(2*tp+fp+fn,1)+2*tn/max(2*tn+fp+fn,1))/2),
        auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),balanced_accuracy=float((tp/max(tp+fn,1)+tn/max(tn+fp,1))/2),
        sensitivity=float(tp/max(tp+fn,1)),specificity=float(tn/max(tn+fp,1)),brier=float(np.mean((p-y)**2)))


def main():
    cfg=json.loads((ROOT/'configs/raw_temporal_v2/phase4.json').read_text());ids,y,schema,assignments,splits=cohort();all_oof=collect()
    folds=[];rows=[];predictions={}
    for (model,seed),group in all_oof.groupby(['model','seed']):
        p=group.set_index('participant_id').loc[ids].probability.to_numpy();predictions[(model,int(seed))]=p
        fold=[]
        for f,(train,test) in enumerate(splits):
            row=dict(model=model,seed=int(seed),fold=f,participants=len(test),**metrics(y[test],p[test]));fold.append(row);folds.append(row)
        summary=dict(model=model,seed=int(seed),n_participants=len(ids),**metrics(y,p))
        for metric in cfg['metrics']:
            values=[r[metric] for r in fold];summary['mean_'+metric]=float(np.mean(values));summary['sd_'+metric]=float(np.std(values,ddof=1))
        rows.append(summary)
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'tables/phase4_nested_comparison.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'tables/phase4_fold_results.csv',index=False)
    # Identical participant resamples for all methods produce paired deltas.
    rng=np.random.default_rng(cfg['bootstrap']['seed']);indices=rng.integers(0,len(y),(cfg['bootstrap']['replicates'],len(y)))
    baseline=predictions[('kinematic_lightgbm',42)];calibrated=predictions.get(('kinematic_calibrated',42))
    reference_samples={};bootstrap=[];deltas=[]
    reference_probabilities={'kinematic_lightgbm':baseline,'kinematic_calibrated':calibrated,
        'raw_v2':predictions.get(('raw_v2',42)),'v1_nested_selected':predictions.get(('v1_nested_selected',42)),
        'random_frozen_v2':predictions.get(('random_frozen_v2',42))}
    for name,p in reference_probabilities.items():
        if p is None:continue
        reference_samples[name]={metric:[] for metric in cfg['metrics']}
        for selected in indices:
            value=fast_metrics(y[selected],p[selected])
            for metric in cfg['metrics']:reference_samples[name][metric].append(value[metric])
        reference_samples[name]={metric:np.asarray(values) for metric,values in reference_samples[name].items()}
    for (model,seed),p in predictions.items():
        samples={metric:[] for metric in cfg['metrics']}
        for selected in indices:
            value=fast_metrics(y[selected],p[selected])
            for metric in cfg['metrics']:samples[metric].append(value[metric])
        for metric in cfg['metrics']:
            values=np.asarray(samples[metric]);lo,hi=np.quantile(values,[.025,.975])
            bootstrap.append(dict(model=model,seed=seed,metric=metric,estimate=metrics(y,p)[metric],lower95=float(lo),upper95=float(hi),
                unit='participant, fixed trained OOF models',replicates=len(indices)))
            for reference,baseline_values in reference_samples.items():
                delta=values-baseline_values[metric];a,b=np.quantile(delta,[.025,.975])
                ref=reference_probabilities[reference]
                deltas.append(dict(model=model,seed=seed,reference=reference,metric=metric,
                    delta=metrics(y,p)[metric]-metrics(y,ref)[metric],lower95=float(a),upper95=float(b)))
        print('BOOTSTRAP',model,seed,flush=True)
    pd.DataFrame(bootstrap).to_csv(OUT/'tables/phase4_participant_bootstrap.csv',index=False)
    pd.DataFrame(deltas).to_csv(OUT/'tables/phase4_paired_deltas.csv',index=False)
    metadata=pd.read_csv(ROOT/'data/derived/participants.csv').set_index('participant_id').loc[ids]
    overlap=[];changes=[];calibration=[]
    for (model,seed),p in predictions.items():
        categories=error_categories(y,baseline,p)
        changes.extend(dict(participant_id=pid,label=int(label),model=model,seed=seed,
            kinematic_probability=float(a),candidate_probability=float(b),delta=float(b-a),category=category)
            for pid,label,a,b,category in zip(ids,y,baseline,p,categories))
        scopes=[('all','all',np.ones(len(ids),dtype=bool))]
        for field in cfg['subgroup_analysis']['fields']:
            for value in metadata[field].dropna().unique():
                mask=metadata[field].to_numpy()==value
                if mask.sum()>=cfg['subgroup_analysis']['minimum_n']:scopes.append((field,str(value),mask))
        for field,value,mask in scopes:
            counts={category:int(np.sum(categories[mask]==category)) for category in ('both_correct','kin_only','raw_only','both_wrong')}
            union=counts['kin_only']+counts['raw_only']+counts['both_wrong']
            corr=float(np.corrcoef(baseline[mask],p[mask])[0,1]) if np.std(baseline[mask])>0 and np.std(p[mask])>0 else None
            overlap.append(dict(model=model,seed=seed,subgroup_field=field,subgroup=value,n=int(mask.sum()),**counts,
                error_jaccard=counts['both_wrong']/union if union else None,probability_pearson=corr,
                probability_spearman=float(spearmanr(baseline[mask],p[mask]).statistic) if corr is not None else None))
        bin_id=np.minimum((p*10).astype(int),9)
        for b in range(10):
            mask=bin_id==b;calibration.append(dict(model=model,seed=seed,bin=b,lower=b/10,upper=(b+1)/10,n=int(mask.sum()),
                mean_probability=float(p[mask].mean()) if mask.any() else None,positive_fraction=float(y[mask].mean()) if mask.any() else None))
    pd.DataFrame(overlap).to_csv(OUT/'tables/phase4_complementarity.csv',index=False)
    pd.DataFrame(changes).to_csv(OUT/'tables/phase4_participant_changes.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'tables/phase4_calibration.csv',index=False)
    write(OUT/'phase4_complementarity/analysis_validation.json',dict(complete=True,participants=258,
        model_seed_pairs=len(predictions),participant_bootstrap_replicates=len(indices),paired_resampling=True,
        folds_not_independent_cohorts=True,bootstrap_does_not_retrain_or_remove_adaptivity=True))


if __name__=='__main__':main()
