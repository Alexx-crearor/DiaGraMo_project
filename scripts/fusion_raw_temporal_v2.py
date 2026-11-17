"""Strict two-level cross-fitting for minimal conditional-value experiments.

Primary nested-CV models use inner validation for early stopping, so their inner
predictions are deliberately not recycled as fusion training features. These
base models instead use a fully fixed predeclared training recipe.
"""
from raw_temporal_v2_common import *
from raw_temporal_v2_evaluation import (config,key,kinematic,fixed_raw,probability,
    weight_fit,meta_fit,logit_features,CONFIG_PATH,SOURCES)


def assert_block(pool,fit,validation,kin,raw):
    ids,y,_,_,_=cohort();pool=np.asarray(pool);fit=np.asarray(fit);validation=np.asarray(validation)
    assert set(fit)<=set(pool) and set(validation)<=set(pool) and not set(fit)&set(validation)
    assert set(fit)|set(validation)==set(pool)
    for result in (kin,raw):
        assert set(result['fit_ids'])==set(ids[fit]) and set(result['evaluation_ids'])==set(ids[validation])
    if 'architecture' in raw:
        assert raw['fixed_epochs'] is not None and raw['validation_ids']==[], 'Early-stopped validation is not a strict fusion OOF input'
        assert set(raw['normalizer_fit_ids'])==set(ids[fit])
    else:
        feature=json.loads((ROOT/raw['feature_path']).with_suffix('.json').read_text())
        assert set(feature['fingerprint']['fit_ids'])==set(ids[fit])
    return dict(pool_ids=ids[pool].tolist(),fit_ids=ids[fit].tolist(),validation_ids=ids[validation].tolist(),
        kinematic_experiment=kin['experiment_id'],raw_experiment=raw['experiment_id'],
        no_validation_labels_for_base_parameters=True)


def cross_fitted(pool,name):
    ids,y,_,_,_=cohort();pool=np.sort(pool);p=np.full((len(pool),2),np.nan);proof=[]
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(pool,y[pool])):
        fit=pool[a];validation=pool[b];kin=kinematic(fit,validation);raw=fixed_raw(fit,validation,42)
        proof.append(dict(inner_fold=fold,**assert_block(pool,fit,validation,kin,raw)))
        p[b,0]=probability(kin,validation);p[b,1]=probability(raw,validation)
    assert np.isfinite(p).all()
    path=OUT/'phase4_complementarity'/'fusion_crossfit'/name;path.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(participant_id=ids[pool],label=y[pool],kinematic_probability=p[:,0],raw_probability=p[:,1])).to_csv(path/'train_oof.csv',index=False)
    write(path/'provenance.json',dict(complete=True,pool_ids=ids[pool].tolist(),blocks=proof,
        no_in_sample_base_predictions=True,fixed_base_training_recipe=True))
    return p,proof


def remaining_neural_runs():
    rows=pd.read_csv(OUT/'EXPERIMENT_LEDGER.csv',keep_default_na=False)
    return RESOURCE['maximum_training_runs']-int(rows.training_status.isin(['running','completed','failed']).sum())


def sensitivity_plan():
    cfg=config();path=OUT/'phase4_complementarity/fusion_resource_plan.json'
    if path.exists():return json.loads(path.read_text())
    if cfg['fusion']['source']=='raw':primary=78;secondary=6
    elif cfg['preferred_ssl_method']=='finetune':primary=132;secondary=6
    else:primary=54;secondary=0
    available=remaining_neural_runs();enabled=available>=primary+secondary
    plan=dict(primary_neural_upper_bound=primary,optional_neural_runs=secondary,
        remaining_neural_runs_at_start=available,seed43_sensitivity=enabled,
        cpu_probe_limit=RESOURCE['maximum_cpu_probe_fits'],other_cpu_fit_limit=cfg['other_cpu_maximum_fits'],
        reason='Reserve every remaining primary fusion base fit before optional initialization sensitivity')
    write(path,plan);return plan


def run_outer(outer,seed43):
    cfg=config();ids,y,schema,assignments,splits=cohort();train,test=splits[outer]
    folder=OUT/'phase4_complementarity'/f'fusion_outer_{outer}';folder.mkdir(parents=True,exist_ok=True)
    outer_oof=np.full((len(train),2),np.nan);inner_metrics=[];outer_blocks=[]
    for j,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(train,y[train])):
        inner_train=train[a];inner_validation=train[b]
        inner_oof,proof=cross_fitted(inner_train,f'outer{outer}_inner{j}_train')
        kin=kinematic(inner_train,inner_validation);raw=fixed_raw(inner_train,inner_validation,42)
        outer_blocks.append(dict(inner_fold=j,**assert_block(train,inner_train,inner_validation,kin,raw)))
        valid_base=np.column_stack([probability(kin,inner_validation),probability(raw,inner_validation)])
        outer_oof[b]=valid_base
        weight=weight_fit(y[inner_train],inner_oof)
        weighted=(1-weight)*valid_base[:,0]+weight*valid_base[:,1]
        inner_metrics.append(dict(outer_fold=outer,inner_fold=j,method='train_selected_weight',raw_weight=weight,**metrics(y[inner_validation],weighted)))
        inner_metrics.append(dict(outer_fold=outer,inner_fold=j,method='fixed_average',raw_weight=.5,**metrics(y[inner_validation],valid_base.mean(1))))
        for kind,kin_only in [('stacking',False),('kinematic_calibrated',True)]:
            result=meta_fit(f'p4_meta_outer{outer}_inner{j}_{kind}',inner_train,inner_validation,inner_oof,valid_base,kin_only)
            inner_metrics.append(dict(outer_fold=outer,inner_fold=j,method=kind,raw_weight=None,**result['metrics']))
        pd.DataFrame(inner_metrics).to_csv(folder/'inner_combiner_validation.csv',index=False)
    assert np.isfinite(outer_oof).all()
    pd.DataFrame(dict(participant_id=ids[train],label=y[train],kinematic_probability=outer_oof[:,0],raw_probability=outer_oof[:,1])).to_csv(folder/'outer_train_oof.csv',index=False)
    write(folder/'outer_train_crossfit_provenance.json',dict(complete=True,fit_pool_ids=ids[train].tolist(),
        excluded_test_ids=ids[test].tolist(),blocks=outer_blocks,strict_fixed_base_recipe=True))
    kin=kinematic(train,test);raw=fixed_raw(train,test,42)
    test_base=np.column_stack([probability(kin,test),probability(raw,test)]);weight=weight_fit(y[train],outer_oof)
    meta=meta_fit(f'p4_meta_outer{outer}_final_stacking',train,test,outer_oof,test_base,False)
    calibrated=meta_fit(f'p4_meta_outer{outer}_final_kinematic_calibrated',train,test,outer_oof,test_base,True)
    stack_state=__import__('joblib').load(ROOT/meta['checkpoint'])
    predictions={'raw_fusion_base':test_base[:,1],'fixed_average':test_base.mean(1),
        'train_selected_weight':(1-weight)*test_base[:,0]+weight*test_base[:,1],
        'stacking':probability(meta,test),'kinematic_calibrated':probability(calibrated,test)}
    records=[]
    def append(values,seed,raw_result):
        for method,p in values.items():
            records.extend(dict(participant_id=pid,label=int(label),fold=outer,model=method,seed=seed,
                probability=float(value),prediction=int(value>.5),kinematic_probability=float(kinprob),
                raw_probability=float(rawprob),raw_weight=weight if method=='train_selected_weight' else .5 if method=='fixed_average' else None,
                raw_experiment=raw_result['experiment_id'],raw_checkpoint=raw_result['checkpoint'],
                raw_checkpoint_sha256=raw_result['checkpoint_sha256'])
                for pid,label,value,kinprob,rawprob in zip(ids[test],y[test],p,test_base[:,0],
                    probability(raw_result,test)))
    append(predictions,42,raw)
    if seed43:
        changed=fixed_raw(train,test,43);p=probability(changed,test);base43=np.column_stack([test_base[:,0],p])
        sensitivity={'raw_fusion_base':p,'fixed_average':base43.mean(1),
            'train_selected_weight':(1-weight)*base43[:,0]+weight*base43[:,1],
            'stacking':stack_state['pipeline'].predict_proba(logit_features(base43))[:,1]}
        append(sensitivity,43,changed)
    pd.DataFrame(records).to_csv(folder/'fusion_oof.csv',index=False)
    write(folder/'combiner.json',dict(raw_weight=weight,weight_fit_ids=ids[train].tolist(),
        outer_test_ids=ids[test].tolist(),stacking_checkpoint=meta['checkpoint'],calibration_checkpoint=calibrated['checkpoint'],
        meta_C=cfg['fusion']['meta_C'],source=cfg['fusion']['source'],all_combiner_inputs_cross_fitted=True,
        seed43_interpretation='Final base initialization varied with seed42-trained combiner fixed; no independently selected seed43 meta pipeline'))
    print('FUSION OUTER COMPLETE',outer,'weight',weight,flush=True)


def main():
    assert (OUT/'phase4_complementarity/nested_oof.csv').exists()
    plan=sensitivity_plan()
    for outer in range(6):run_outer(outer,plan['seed43_sensitivity'])
    frames=[pd.read_csv(OUT/'phase4_complementarity'/f'fusion_outer_{outer}/fusion_oof.csv') for outer in range(6)]
    frame=pd.concat(frames,ignore_index=True);assert not frame.duplicated(['participant_id','model','seed']).any()
    assert (frame[frame.seed==42].groupby('model').size()==258).all()
    frame.to_csv(OUT/'phase4_complementarity/fusion_oof.csv',index=False)
    print('LIMITED FUSION FITS FINISHED; FULL VALIDATION AND XAI PENDING',flush=True)


if __name__=='__main__':
    with gpu_worker():main()
