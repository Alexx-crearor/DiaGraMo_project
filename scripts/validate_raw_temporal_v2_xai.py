"""Validate held-out explanation artifacts and quantify task/reference stability."""
from scipy.special import expit
from raw_temporal_v2_common import *
from explain_raw_temporal_v2 import SOURCES,source_mapping,choose_cases,rho
from diagramo.temporal.dataset import load_person


def main():
    cfg=json.loads((ROOT/'configs/raw_temporal_v2/phase4.json').read_text());active=json.loads((OUT/'phase4_complementarity/xai/active_run.json').read_text())
    run=ROOT/active['run'];signature=json.loads((run/'signature.json').read_text())
    for path in SOURCES:assert signature[path]==sha(ROOT/path)
    assert signature['config']==sha(ROOT/'configs/raw_temporal_v2/phase4.json')
    assert signature['models']==sha(OUT/'phase4_complementarity/model_validation.json')
    ids,y,schema,assignments,splits=cohort();tasks=pd.read_csv(run/'task_occlusion.csv',float_precision='round_trip')
    method_count=tasks.model.nunique()
    assert len(tasks)==258*14*method_count*2 and not tasks.duplicated(['participant_id','model','seed','task_index']).any()
    assert np.isfinite(tasks[['raw_logit','probability','logit_delta','absolute_logit_delta']]).all().all()
    np.testing.assert_allclose(tasks.probability,expit(tasks.raw_logit),atol=1e-12,rtol=0)
    assert (tasks.task==tasks.task_index.map(dict(enumerate(schema['tasks'])))).all()
    replay=json.loads((run/'task_replay.json').read_text());assert len(replay)==6*method_count*2
    for record in replay:
        train,test=splits[record['fold']]
        assert set(record['fit_ids'])==set(ids[train]) and set(record['evaluation_ids'])==set(ids[test])
        assert record['maximum_probability_replay_error']<1e-4 and record['cache_mask_restoration_passed']
        path=run/'tasks'/f"fold{record['fold']}_{record['model']}_seed{record['seed']}.csv"
        assert sha(path)==record['csv_sha256']
    frame=pd.read_csv(OUT/'phase4_complementarity/nested_oof.csv',float_precision='round_trip');expected,target=choose_cases(frame,cfg)
    cases=pd.read_csv(run/'selected_cases.csv')
    assert cases.participant_id.tolist()==expected.participant_id.tolist()
    ig_checks=[];stability=[];segments=[];deletions=[];ig_tasks=[];ig_channels=[];duration=[];artifacts={}
    for case in cases.to_dict('records'):
        pid=case['participant_id'];person=load_person(str(CACHE),pid)
        for seed in cfg['attribution']['seeds']:
            folder=run/'cases'/f'{pid}_seed{seed}';saved=json.loads((folder/'validation.json').read_text());assert saved['complete'] and saved['test_only']
            mapping=pd.read_csv(folder/'source_mapping.csv');actual=source_mapping(pid)
            pd.testing.assert_frame_equal(mapping,actual,check_dtype=False,rtol=1e-10,atol=1e-10)
            assert saved['source_map_sha256']==sha(folder/'source_mapping.csv')
            for check in saved['ig_validations']:
                row=dict(participant_id=pid,seed=seed,reference=check['reference'],status=check['status'],valid=check['complete'],
                    steps=check['steps'],absolute_completeness_error=check.get('absolute_completeness_error'))
                if 'attribution_path' in check:
                    path=ROOT/check['attribution_path'];assert sha(path)==check['attribution_sha256']
                    with np.load(path) as data:attribution=data['attribution']
                    assert attribution.shape==(len(mapping),21,128) and np.isfinite(attribution).all()
                    total=float(attribution.astype('float64').sum())
                    assert abs(total-check['attribution_sum'])<1e-5
                    if check['complete']:
                        assert abs(total-check['raw_logit_difference'])<=cfg['attribution']['ig_completeness_atol']+cfg['attribution']['ig_completeness_rtol']*abs(check['raw_logit_difference'])
                    sample=np.arange(128)[None,:]<person['length'][:,None]
                    assert not attribution.transpose(0,2,1)[~sample].any()
                    assert not attribution[:,5:8,:].any() and not attribution[:,13:,:].any()
                    velocity=person['x'][:,:,13]>.5;acceleration=person['x'][:,:,14]>.5
                    assert not attribution[:,8:11,:].transpose(0,2,1)[~velocity].any()
                    assert not attribution[:,11:13,:].transpose(0,2,1)[~acceleration].any()
                    artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
                ig_checks.append(row)
            for name,destination in [('ig_tasks.csv',ig_tasks),('ig_channels.csv',ig_channels),('segment_occlusion.csv',segments),
                ('deletion_faithfulness.csv',deletions),('duration_readout_stress.csv',duration)]:
                path=folder/name;table=pd.read_csv(path)
                if len(table):destination.extend(table.to_dict('records'))
            for path in folder.iterdir():
                if path.is_file():artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
            assert saved['tested_segments']<=64 and saved['tested_segments']<=saved['total_chosen_task_segments']
            stability.append(json.loads((folder/'stability.json').read_text()))
    assert len(ig_checks)==len(cases)*2*2
    tables={'ig_validation':ig_checks,'reference_seed_randomization':stability,'segment_occlusion':segments,
        'deletion_faithfulness':deletions,'ig_tasks':ig_tasks,'ig_channels':ig_channels,'duration_readout_stress':duration}
    for name,records in tables.items():pd.DataFrame(records).to_csv(run/(name+'.csv'),index=False)
    fold_importance=tasks.groupby(['model','seed','fold','task_index','task'],as_index=False).absolute_logit_delta.mean()
    fold_importance['rank']=fold_importance.groupby(['model','seed','fold']).absolute_logit_delta.rank(ascending=False,method='average')
    fold_importance.to_csv(run/'task_importance_by_fold.csv',index=False)
    fold_stability=[];seed_stability=[]
    for (model,seed),group in fold_importance.groupby(['model','seed']):
        pivot=group.pivot(index='task_index',columns='fold',values='rank').sort_index()
        for a in range(6):
            for b in range(a+1,6):fold_stability.append(dict(model=model,seed=seed,fold_a=a,fold_b=b,spearman=rho(pivot[a],pivot[b])))
    for (pid,model),group in tasks.groupby(['participant_id','model']):
        a=group[group.seed==42].sort_values('task_index').absolute_logit_delta.to_numpy()
        b=group[group.seed==43].sort_values('task_index').absolute_logit_delta.to_numpy()
        seed_stability.append(dict(participant_id=pid,model=model,spearman=rho(a,b)))
    pd.DataFrame(fold_stability).to_csv(run/'fold_task_stability.csv',index=False)
    pd.DataFrame(seed_stability).to_csv(run/'seed_task_stability.csv',index=False)
    # Quality gates are descriptive diagnostics, never clinical validation.
    from raw_temporal_v2_common import metrics
    gates=[]
    for model in sorted(frame.model.unique()):
        p=frame[(frame.model==model)&(frame.seed==42)].set_index('participant_id').loc[ids].probability.to_numpy()
        performance=metrics(y,p);seed_values=pd.DataFrame(seed_stability);median=seed_values[seed_values.model==model].spearman.median()
        gate=cfg['attribution']['interpretation_gate']
        suitable=performance['auc']>=gate['minimum_auc'] and performance['balanced_accuracy']>=gate['minimum_balanced_accuracy'] and pd.notna(median) and median>=gate['minimum_seed_task_rank_median_spearman']
        gates.append(dict(model=model,auc=performance['auc'],balanced_accuracy=performance['balanced_accuracy'],
            median_seed_task_rank_spearman=float(median) if pd.notna(median) else None,
            descriptive_stability_gate_passed=bool(suitable),clinical_or_causal_validation=False))
    write(run/'interpretation_gates.json',gates)
    ig_frame=pd.DataFrame(ig_checks);failures=int((~ig_frame.valid).sum()) if len(ig_frame) else 0
    write(run/'validation.json',dict(complete=True,explanation_execution_complete=True,
        tasks_rows=len(tasks),checkpoint_replay_passed=True,train_test_scopes_verified=True,
        source_mapping_verified=True,ig_reference_cases=len(ig_checks),ig_checks_failed=failures,
        ig_failed_cases_not_used_for_substantive_interpretation=True,interpretation_gates=gates,
        detailed_case_participants=len(cases),detailed_case_selection_posthoc_not_representative=True,
        conditional_motor_ig_time_and_flags_held_fixed=True,verified_artifacts=artifacts))
    write(OUT/'phase4_complementarity/xai/validation.json',dict(complete=True,active_run=active['run'],
        run_validation_sha256=sha(run/'validation.json'),ig_checks_failed=failures,
        checkpoint_and_mapping_checks_passed=True,figures_pending=True))
    print('XAI VALIDATED; failed IG numerical checks retained:',failures,flush=True)


if __name__=='__main__':main()
