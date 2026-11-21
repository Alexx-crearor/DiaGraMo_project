"""Replay stored OOF, source/label sensitivities and validation feasibility, without fitting."""
from __future__ import annotations
from typing import Any
from common import *


def wilson(success: int,total: int,z: float=1.959963984540054) -> tuple[float,float]:
    if total==0:return np.nan,np.nan
    p=success/total;denominator=1+z*z/total;centre=(p+z*z/(2*total))/denominator
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return float(centre-half),float(centre+half)


def calibration(y: np.ndarray,p: np.ndarray,bins: int=10) -> tuple[pd.DataFrame,float]:
    index=np.minimum((p*bins).astype(int),bins-1);rows=[]
    for number in range(bins):
        at=index==number;n=int(at.sum());success=int(y[at].sum());lo,hi=wilson(success,n)
        rows.append(dict(bin=number,left=number/bins,right=(number+1)/bins,n=n,group1=success,
            mean_probability=float(p[at].mean()) if n else np.nan,observed_rate=success/n if n else np.nan,
            wilson_low=lo,wilson_high=hi))
    table=pd.DataFrame(rows);gap=(table.mean_probability-table.observed_rate).abs()
    return table,float((table.n*gap.fillna(0)).sum()/len(y))


def error_summary(frame: pd.DataFrame) -> dict[str,Any]:
    y=frame.label.to_numpy();k=frame.pK.to_numpy()>.5;t=frame.pT.to_numpy()>.5;f=frame.pF.to_numpy()>.5
    return dict(n=len(frame),group0=int((y==0).sum()),group1=int((y==1).sum()),
        both_correct=int(((k==y)&(t==y)).sum()),kinematic_only_correct=int(((k==y)&(t!=y)).sum()),
        temporal_only_correct=int(((k!=y)&(t==y)).sum()),both_wrong=int(((k!=y)&(t!=y)).sum()),
        fusion_corrected=int(((k!=y)&(f==y)).sum()),fusion_harmed=int(((k==y)&(f!=y)).sum()),
        probability_correlation=float(np.corrcoef(frame.pK,frame.pT)[0,1]) if len(frame)>1 and frame.pK.std()>0 and frame.pT.std()>0 else np.nan)


def small_scores(y: np.ndarray,p: np.ndarray) -> dict[str,float]:
    pred=p>.5;tp=((y==1)&pred).sum();tn=((y==0)&~pred).sum();fp=((y==0)&pred).sum();fn=((y==1)&~pred).sum()
    return dict(f1_macro=float(.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))),
        auc=float(roc_auc_score(y,p)),brier=float(np.mean((y-p)**2)))


def main() -> None:
    assignment=folds();cohort=read('audit_cohort.csv').set_index('participant_id').loc[assignment.participant_id]
    baseline_path=ROOT/'results/cv/kinematic__LightGBM.oof.csv'
    baseline=pd.read_csv(baseline_path,float_precision='round_trip').set_index('participant_id').loc[assignment.participant_id]
    if not baseline.index.is_unique or len(baseline)!=len(assignment):raise AssertionError('Invalid baseline OOF')
    np.testing.assert_array_equal(baseline.label,assignment.label);np.testing.assert_array_equal(baseline.fold,assignment.fold)
    parts=[];fit_checks=[]
    for fold in range(6):
        directory=ROOT/'results/raw_temporal_v2/phase4_lite/fits'/f'stroke_transformer_fold{fold}'
        result=json.loads((directory/'result.json').read_text());part=pd.read_csv(directory/'oof_predictions.csv',float_precision='round_trip')
        train=set(assignment.loc[assignment.fold!=fold,'participant_id']);test=set(assignment.loc[assignment.fold==fold,'participant_id'])
        assert set(result['fit_ids'])==set(result['normalizer_fit_ids'])==train
        assert set(result['evaluation_ids'])==set(part.participant_id)==test and not train&test
        assert result['fixed_epochs']==28 and result['seed']==42 and not result['validation_ids']
        np.testing.assert_allclose(part.probability,1/(1+np.exp(-part.logit)),rtol=0,atol=1e-15)
        assert (part.fold==fold).all() and (part.seed==42).all() and part.participant_id.is_unique
        # Hash only, never deserialize/retrain checkpoints in this audit.
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
        fit_checks.append(dict(fold=fold,n_train=len(train),n_test=len(test),normalizer_fit_only_train=True,
            checkpoint_hash_matches=True,stored_prior_checkpoint_replay_max_logit_error=result['checkpoint_replay_max_logit_error'],
            audit_scope='Read OOF and saved fit-ID lists; hash checkpoint; no new checkpoint inference or model fitting'))
        parts.append(part)
    temporal=pd.concat(parts).set_index('participant_id')
    if not temporal.index.is_unique or set(temporal.index)!=set(assignment.participant_id):raise AssertionError('Missing/duplicate Temporal OOF')
    temporal=temporal.loc[assignment.participant_id];np.testing.assert_array_equal(temporal.label,assignment.label);np.testing.assert_array_equal(temporal.fold,assignment.fold)
    frame=assignment.copy();frame['pK']=baseline.probability.to_numpy();frame['pT']=temporal.probability.to_numpy();frame['pF']=.5*(frame.pK+frame.pT)
    frame=frame.merge(cohort.reset_index()[['participant_id','group','diagnosis','flag_to_omit_total','id_city','id_school_ppp','age','sex_female']],on='participant_id',validate='one_to_one')
    for field,prediction in [('pK',baseline.prediction),('pT',temporal.prediction)]:np.testing.assert_array_equal(frame[field]>.5,prediction.to_numpy())
    saved=pd.read_csv(ROOT/'results/raw_temporal_v2/locked_study/participant_complementarity.csv',float_precision='round_trip').set_index('participant_id').loc[frame.participant_id]
    np.testing.assert_allclose(frame.pK,saved.kinematic_probability,rtol=0,atol=1e-15)
    np.testing.assert_allclose(frame.pT,saved.probability,rtol=0,atol=1e-15);np.testing.assert_allclose(frame.pF,saved.fixed_average_probability,rtol=0,atol=1e-15)
    frame['fusion_effect']=np.where(((frame.pK>.5)!=frame.label)&((frame.pF>.5)==frame.label),'corrected',
        np.where(((frame.pK>.5)==frame.label)&((frame.pF>.5)!=frame.label),'harmed',np.where((frame.pK>.5)==frame.label,'unchanged_correct','unchanged_wrong')))
    frame['fusion_probability_change']=frame.pF-frame.pK
    csv('matched_oof_replay.csv',frame);csv('fusion_corrected_harmed.csv',frame);csv('fit_and_fold_checks.csv',pd.DataFrame(fit_checks))
    model_fields={'Kinematic_LightGBM':'pK','Stroke_Temporal':'pT','Fixed50':'pF'}
    all_metrics=[];source_metrics=[];calib=[];calib_bins=[];sensitivities=[];effects=[];fold_metrics=[]
    for name,field in model_fields.items():
        all_metrics.append(dict(model=name,**metrics(frame.label.to_numpy(),frame[field].to_numpy())))
        for fold,part in frame.groupby('fold'):fold_metrics.append(dict(model=name,fold=fold,**metrics(part.label.to_numpy(),part[field].to_numpy())))
        for source_field in ['id_city','id_school_ppp']:
            for source,part in frame.groupby(source_field):
                result=metrics(part.label.to_numpy(),part[field].to_numpy());lo_s,hi_s=wilson(result['tp'],result['group1']);lo_c,hi_c=wilson(result['tn'],result['group0'])
                source_metrics.append(dict(model=name,source_field=source_field,source=source,**result,
                    sensitivity_wilson_low=lo_s,sensitivity_wilson_high=hi_s,specificity_wilson_low=lo_c,specificity_wilson_high=hi_c,
                    inference='Post-hoc internal OOF subset; not source-held-out or external validation'))
        for scope,part in [('all_258',frame),('within_city1',frame[frame.id_city=='city1'])]:
            table,ece=calibration(part.label.to_numpy(),part[field].to_numpy());table.insert(0,'scope',scope);table.insert(0,'model',name);calib_bins.append(table)
            scores=metrics(part.label.to_numpy(),part[field].to_numpy())
            calib.append(dict(model=name,scope=scope,n=len(part),group1=int(part.label.sum()),brier=scores['brier'],log_loss=scores['log_loss'],ece10=ece,
                calibration_fit=False,bins='10 prespecified equal-width bins; empty bins preserved',ci='Wilson binomial intervals; participant units; ignores source clustering/selection'))
        for scope,mask,label_field in [('original_258',np.ones(len(frame),bool),'label'),('consistent_labels',frame.label.eq(frame.diagnosis).to_numpy(),'label'),
            ('no_omission_flag',frame.flag_to_omit_total.eq(0).to_numpy(),'label'),('diagnosis_relabelled_same_OOF',np.ones(len(frame),bool),'diagnosis')]:
            part=frame[mask];sensitivities.append(dict(model=name,cohort=scope,target=label_field,**metrics(part[label_field].to_numpy(),part[field].to_numpy()),
                post_hoc=True,probabilities_trained_for='group, never retrained for changed target',model_selection_allowed=False))
    for scope,groups in [('all',[('all_258',frame)]),('fold',frame.groupby('fold')),('city',frame.groupby('id_city')),('site',frame.groupby('id_school_ppp'))]:
        for key,part in groups:effects.append(dict(scope=scope,stratum=key,**error_summary(part)))
    all_metrics=pd.DataFrame(all_metrics);csv('matched_model_metrics.csv',all_metrics);csv('model_metrics_by_fold.csv',pd.DataFrame(fold_metrics))
    csv('source_stratified_model_metrics.csv',pd.DataFrame(source_metrics));csv('error_overlap_by_source.csv',pd.DataFrame(effects))
    csv('calibration_diagnostics.csv',pd.DataFrame(calib));csv('calibration_bins.csv',pd.concat(calib_bins,ignore_index=True));csv('label_sensitivity.csv',pd.DataFrame(sensitivities))
    previous=pd.read_csv(ROOT/'results/raw_temporal_v2/locked_study/matched_oof_metrics.csv').set_index('model')
    for name,old in [('Kinematic_LightGBM','Kinematic'),('Stroke_Temporal','Temporal'),('Fixed50','Fixed50')]:
        current=all_metrics.set_index('model').loc[name]
        for metric in ['accuracy','f1_macro','auc','balanced_accuracy','sensitivity','specificity','brier','log_loss']:
            np.testing.assert_allclose(current[metric],previous.loc[old,metric],rtol=0,atol=1e-12)
    rng=np.random.default_rng(SEED);groups=[np.flatnonzero(frame.label.to_numpy()==label) for label in (0,1)];deltas=[]
    y=frame.label.to_numpy();pK=frame.pK.to_numpy();pF=frame.pF.to_numpy();pointK=small_scores(y,pK);pointF=small_scores(y,pF)
    for _ in range(2000):
        at=np.concatenate([rng.choice(group,len(group),replace=True) for group in groups]);a=small_scores(y[at],pK[at]);b=small_scores(y[at],pF[at]);deltas.append({key:b[key]-a[key] for key in a})
    boot=pd.DataFrame(deltas);ci=[]
    for key in boot:
        lo,hi=np.quantile(boot[key],[.025,.975]);ci.append(dict(metric=key,delta=pointF[key]-pointK[key],lower95=lo,upper95=hi,replicates=2000,
            method='Paired participant bootstrap within label; fixed models, not selection- or cluster-adjusted'))
    csv('paired_model_uncertainty.csv',pd.DataFrame(ci))
    csv('research_decisions.csv',pd.DataFrame([
        dict(direction='Temporal encoder expansion on current cohort',decision='NO-GO for confirmatory claims; CONDITIONAL GO for explicitly exploratory methodological controls',prerequisite='Resolve labels/source provenance and independent balanced validation; no architecture race now'),
        dict(direction='Train-only calibrated fusion',decision='CONDITIONAL GO as separately preregistered exploratory control, not clinical validation',prerequisite='Train-only calibration/cross-fitting and independent/source-aware evaluation; current OOF cannot select parameters'),
        dict(direction='SSL',decision='NO-GO as current priority',prerequisite='SSL may preserve/amplify source information; first establish source/clinical identifiability'),
        dict(direction='Static Visual branch',decision='NO-GO as current priority',prerequisite='Images can also encode source/protocol; require data validity and a separately justified hypothesis'),
        dict(direction='Source invariance / cross-domain / clinical biomarkers',decision='NO-GO claims',prerequisite='Independent multi-source balanced data, verified reference labels and acquisition metadata')]))
    write_json('EF_STATUS.json',dict(complete=True,participants=len(frame),all_fold_and_label_joins_exact=True,
        all_metrics_match_previous_reports=True,model_training=False,calibration_fit=False,threshold_tuning=False,
        task_attribution_identifiable_from_participant_OOF=False,checkpoint_replay='OOF arithmetic/logit/metrics replay only; saved checkpoints hash-checked; no fresh network inference',
        clinical_validation=False,source_independent_validation=False))
    print('PHASE E-F OOF REPLAY COMPLETE',len(frame),flush=True)


if __name__=='__main__':main()
