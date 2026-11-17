"""Held-out task/stroke/motor explanations with replay and source-time mapping."""
import hashlib
from scipy.special import expit
from scipy.stats import spearmanr
from raw_temporal_v2_common import *
from raw_temporal_v2_explain import (load_model,tensor,collate_people,task_occlusion,motor_reference,
    integrated_motor_gradients,cached_local,randomized_head)
from diagramo.temporal.dataset import load_person
from diagramo.temporal.preprocessing import CHANNELS
from analyze_raw_temporal_v2_phase4 import error_categories

SOURCES=['scripts/explain_raw_temporal_v2.py','scripts/raw_temporal_v2_explain.py']


def rho(a,b):
    return float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None


def source_mapping(pid):
    person=load_person(str(CACHE),pid);rows=[];offset=0
    for task,inventory in enumerate(person['inventories']):
        for j,row in enumerate(inventory):
            patch=offset+j;start,end=int(row[0]),int(row[1]);assert end-start==person['length'][patch]
            rows.append(dict(patch_id=patch,task_index=task,segment_id=int(person['patch_segment'][patch]),
                raw_start_index=start,raw_end_index=end,source_start_seconds=float(row[5]),source_end_seconds=float(row[6]),
                valid_samples=int(person['length'][patch])))
        offset+=len(inventory)
    assert offset==len(person['length']);return pd.DataFrame(rows)


def task_jobs(run,cfg,frame):
    ids,y,schema,assignments,splits=cohort();tasknames=schema['tasks'];rows=[];replay=[]
    for fold,(train,test) in enumerate(splits):
        for method in sorted(frame.model.unique()):
            for seed in cfg['attribution']['seeds']:
                name=f'fold{fold}_{method}_seed{seed}';path=run/'tasks'/f'{name}.csv';meta=path.with_suffix('.json')
                group=frame[(frame.fold==fold)&(frame.model==method)&(frame.seed==seed)].set_index('participant_id').loc[ids[test]]
                r=json.loads((OUT/'phase4_complementarity'/group.experiment_id.iloc[0]/'result.json').read_text())
                if path.exists() and meta.exists():
                    saved=json.loads(meta.read_text());assert saved['checkpoint_sha256']==r['checkpoint_sha256'] and saved['csv_sha256']==sha(path)
                    rows.extend(pd.read_csv(path).to_dict('records'));replay.append(saved);continue
                model,normalizer=load_model(r,train,seed);local=[];errors=[]
                for index in test:
                    pid=ids[index];batch=collate_people([tensor(pid,normalizer)])
                    contribution,logit=task_occlusion(model,batch);p=float(expit(logit));error=abs(p-float(group.loc[pid,'probability']))
                    assert error<1e-4,'Differentiable/raw wrapper does not replay held-out model'
                    errors.append(error)
                    local.extend(dict(participant_id=pid,label=int(y[index]),fold=fold,model=method,seed=seed,
                        task_index=task,task=tasknames[task],raw_logit=logit,probability=p,
                        logit_delta=float(contribution[task]),absolute_logit_delta=float(abs(contribution[task])),
                        checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256']) for task in range(14))
                path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(local).to_csv(path,index=False)
                saved=dict(complete=True,fold=fold,model=method,seed=seed,checkpoint_sha256=r['checkpoint_sha256'],
                    csv_sha256=sha(path),maximum_probability_replay_error=max(errors),test_participants=len(test),
                    fit_ids=ids[train].tolist(),evaluation_ids=ids[test].tolist(),cache_mask_restoration_passed=True)
                write(meta,saved);rows.extend(local);replay.append(saved)
                print('XAI TASKS',name,flush=True)
    table=pd.DataFrame(rows);table.to_csv(run/'task_occlusion.csv',index=False)
    write(run/'task_replay.json',replay);return table


def choose_cases(frame,cfg):
    ids,y,schema,assignments,splits=cohort()
    priority={'raw_v2':3,'ssl_v2':2,'random_frozen_v2':1}
    scores=[]
    for name,group in frame[frame.seed==42].groupby('model'):
        p=group.set_index('participant_id').loc[ids].probability.to_numpy();score=metrics(y,p)
        scores.append((score['f1_macro'],score['auc'],priority[name],name))
    target=max(scores)[-1]  # Descriptive XAI target only; no predictive refitting.
    candidate=frame[(frame.model==target)&(frame.seed==42)].set_index('participant_id').loc[ids].copy()
    kin=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv',float_precision='round_trip').set_index('participant_id').loc[ids]
    candidate['category']=error_categories(y,kin.probability.to_numpy(),candidate.probability.to_numpy())
    candidate['confidence']=(candidate.probability-.5).abs();candidate=candidate.reset_index();rows=[]
    for fold in range(6):
        for category in ('both_correct','kin_only','raw_only','both_wrong'):
            options=candidate[(candidate.fold==fold)&(candidate.category==category)]
            if len(options):rows.append(options.sort_values(['confidence','participant_id'],ascending=[False,True]).iloc[0].to_dict())
    assert len(rows)<=cfg['attribution']['maximum_detailed_cases']
    return pd.DataFrame(rows),target


def detail_case(run,cfg,case,target,seed,frame,tasks):
    ids,y,schema,assignments,splits=cohort();lookup={pid:i for i,pid in enumerate(ids)}
    pid=case['participant_id'];fold=int(case['fold']);index=lookup[pid];train,test=splits[fold]
    group=frame[(frame.fold==fold)&(frame.model==target)&(frame.seed==seed)].set_index('participant_id')
    result=json.loads((OUT/'phase4_complementarity'/group.loc[pid,'experiment_id']/'result.json').read_text())
    folder=run/'cases'/f'{pid}_seed{seed}';done=folder/'validation.json'
    if done.exists():
        prior=json.loads(done.read_text());assert prior['checkpoint_sha256']==result['checkpoint_sha256'];return
    folder.mkdir(parents=True,exist_ok=True);mapping=source_mapping(pid);mapping.to_csv(folder/'source_mapping.csv',index=False)
    model,normalizer=load_model(result,train,seed);batch=collate_people([tensor(pid,normalizer)])
    with torch.no_grad():base=float(model(batch).sum())
    assert abs(float(expit(base))-float(group.loc[pid,'probability']))<1e-4
    taskrows=tasks[(tasks.participant_id==pid)&(tasks.model==target)&(tasks.seed==42)].sort_values('task_index')
    chosen_task=int(taskrows.absolute_logit_delta.to_numpy().argmax());pred_sign=1 if base>0 else -1
    segments=batch['segment_task'].cpu().numpy();patch_segment=batch['patch_segment'].cpu().numpy();patch_task=batch['patch_task'].cpu().numpy()
    igrows=[];channelrows=[];validations=[];segment_importance=None;task_ig={}
    attrs_dir=LOCAL/'xai_attrs'/run.name/f'{pid}_seed{seed}';attrs_dir.mkdir(parents=True,exist_ok=True)
    for reference_name,kind in [('train_mean','train_mean'),('window_mean','window_mean')]:
        reference=motor_reference(batch,kind)
        attribution,validation=integrated_motor_gradients(model,batch,reference,tuple(cfg['attribution']['ig_steps']),
            cfg['attribution']['ig_completeness_atol'],cfg['attribution']['ig_completeness_rtol'])
        validation.update(reference=reference_name,participant_id=pid,seed=seed)
        if attribution is not None:
            path=attrs_dir/(reference_name+'.npz');np.savez_compressed(path,attribution=attribution.cpu().numpy())
            validation.update(attribution_path=path.relative_to(ROOT).as_posix(),attribution_sha256=sha(path))
            signed=attribution.double().sum((1,2)).cpu().numpy();absolute=attribution.double().abs().sum((1,2)).cpu().numpy()
            signed_segments=np.bincount(patch_segment,weights=signed,minlength=len(segments))
            abs_segments=np.bincount(patch_segment,weights=absolute,minlength=len(segments))
            task_values=[]
            for task in range(14):
                at=patch_task==task;number=int(batch['length'].cpu().numpy()[at].sum());magnitude=float(absolute[at].sum())
                task_values.append(magnitude)
                igrows.append(dict(participant_id=pid,seed=seed,reference=reference_name,task_index=task,task=schema['tasks'][task],
                    signed_logit_attribution=float(signed[at].sum()),absolute_logit_attribution=magnitude,
                    mean_abs_per_source_sample=magnitude/max(number,1),valid_source_samples=number,ig_valid=validation['complete']))
            if validation['complete']:task_ig[reference_name]=np.asarray(task_values)
            for channel,name in enumerate(CHANNELS):channelrows.append(dict(participant_id=pid,seed=seed,reference=reference_name,
                channel=name,signed_attribution=float(attribution[:,channel,:].sum()),absolute_attribution=float(attribution[:,channel,:].abs().sum()),ig_valid=validation['complete'],
                interpretation='Motor contrast only; zero time/flag attribution means held fixed, not model unimportance'))
            if reference_name=='train_mean' and validation['complete']:
                segment_importance=(signed_segments,abs_segments)
        validations.append(validation)
    pd.DataFrame(igrows,columns=['participant_id','seed','reference','task_index','task','signed_logit_attribution','absolute_logit_attribution','mean_abs_per_source_sample','valid_source_samples','ig_valid']).to_csv(folder/'ig_tasks.csv',index=False)
    pd.DataFrame(channelrows,columns=['participant_id','seed','reference','channel','signed_attribution','absolute_attribution','ig_valid','interpretation']).to_csv(folder/'ig_channels.csv',index=False)
    eligible=np.flatnonzero(segments==chosen_task);assert len(eligible)>0
    uniform=eligible[np.unique(np.linspace(0,len(eligible)-1,min(32,len(eligible))).astype(int))]
    if segment_importance is not None:
        top=eligible[np.argsort(-segment_importance[1][eligible],kind='stable')[:32]];selected=np.unique(np.r_[top,uniform])
    else:selected=eligible[np.unique(np.linspace(0,len(eligible)-1,min(64,len(eligible))).astype(int))]
    assert len(selected)<=cfg['attribution']['stroke_occlusion_maximum_segments'];occlusion=[];deletion=[]
    with cached_local(model,batch):
        with torch.no_grad():
            for segment in selected:
                present=batch['patch_segment']!=int(segment);logit=float(model(batch,patch_present=present).sum())
                source=mapping[mapping.segment_id==segment]
                occlusion.append(dict(participant_id=pid,seed=seed,task_index=chosen_task,task=schema['tasks'][chosen_task],
                    segment_id=int(segment),logit_delta=base-logit,prediction_support=pred_sign*(base-logit),
                    probability_delta=float(expit(base)-expit(logit)),raw_start_index=int(source.raw_start_index.min()),
                    raw_end_index=int(source.raw_end_index.max()),source_start_seconds=float(source.source_start_seconds.min()),
                    source_end_seconds=float(source.source_end_seconds.max()),tested_segments=len(selected),total_task_segments=len(eligible)))
            if segment_importance is not None:
                candidates=eligible;ranking=candidates[np.argsort(-pred_sign*segment_importance[0][candidates],kind='stable')];rank_type='signed_ig_support'
            else:
                candidates=selected;order=pd.DataFrame(occlusion).sort_values(['prediction_support','segment_id'],ascending=[False,True]);ranking=order.segment_id.to_numpy();rank_type='occlusion_tested_subset_support'
            rng=np.random.default_rng(42+index*1009)
            for fraction in cfg['attribution']['deletion_fractions']:
                number=max(1,int(np.ceil(fraction*len(candidates))))
                for repeat in range(-1,cfg['attribution']['random_deletion_repeats']):
                    removed=ranking[:number] if repeat==-1 else rng.choice(candidates,number,replace=False)
                    present=~torch.isin(batch['patch_segment'],torch.as_tensor(removed,device=DEVICE))
                    changed=float(model(batch,patch_present=present).sum())
                    deletion.append(dict(participant_id=pid,seed=seed,task_index=chosen_task,fraction=fraction,
                        method=rank_type if repeat==-1 else 'matched_random',attribution_rank_type=rank_type,repeat=repeat,removed_segments=number,
                        eligible_segments=len(candidates),original_prediction_confidence_logit_drop=pred_sign*(base-changed),
                        raw_logit=base,perturbed_raw_logit=changed))
            assert abs(float(model(batch).sum())-base)<1e-5
    pd.DataFrame(occlusion).to_csv(folder/'segment_occlusion.csv',index=False)
    pd.DataFrame(deletion).to_csv(folder/'deletion_faithfulness.csv',index=False)
    # Duration gate contrast changes the fitted readout only, never retrains it.
    duration=[]
    if getattr(model,'aggregation_stage',None)=='task_to_person' and getattr(model,'aggregation_method',None)=='duration':
        values=[]
        for participant in ids[train]:
            with np.load(LOCAL/'physical_targets'/f'{participant}.npz') as data:values.append(data['task_duration'])
        median=np.median(values,axis=0)
        with cached_local(model,batch):
            with torch.no_grad():
                for name,value in [('uniform',np.ones(14)),('outer_train_task_median',median)]:
                    changed=float(model(dict(batch,task_duration=torch.tensor(value,device=DEVICE,dtype=torch.float32))).sum())
                    duration.append(dict(participant_id=pid,seed=seed,reference=name,original_raw_logit=base,
                        counterfactual_raw_logit=changed,probability_delta=float(expit(changed)-expit(base)),
                        interpretation='Fitted-parameter readout stress, not retrained ablation or causal effect'))
    pd.DataFrame(duration,columns=['participant_id','seed','reference','original_raw_logit','counterfactual_raw_logit','probability_delta','interpretation']).to_csv(folder/'duration_readout_stress.csv',index=False)
    # Head randomization is an explanation diagnostic, not a new trained model.
    with randomized_head(model,cfg['attribution']['randomization']['seed']+fold):
        random_task,_=task_occlusion(model,batch)
        random_attr,random_validation=integrated_motor_gradients(model,batch,motor_reference(batch,'train_mean'),
            tuple(cfg['attribution']['ig_steps']),cfg['attribution']['ig_completeness_atol'],cfg['attribution']['ig_completeness_rtol'])
        random_task_ig=None
        if random_attr is not None and random_validation['complete']:
            absolute=random_attr.abs().sum((1,2)).cpu().numpy();random_task_ig=np.bincount(patch_task,weights=absolute,minlength=14)
    with torch.no_grad():assert abs(float(model(batch).sum())-base)<1e-5,'Randomization changed the restored fitted model'
    original_task=tasks[(tasks.participant_id==pid)&(tasks.model==target)&(tasks.seed==seed)].sort_values('task_index').absolute_logit_delta.to_numpy()
    stability=dict(participant_id=pid,seed=seed,task_head_randomization_spearman=rho(original_task,np.abs(random_task)),
        ig_head_randomization_spearman=rho(task_ig['train_mean'],random_task_ig) if random_task_ig is not None and 'train_mean' in task_ig else None,
        ig_reference_spearman=rho(task_ig['train_mean'],task_ig['window_mean']) if len(task_ig)==2 else None,
        random_ig_valid=random_validation['complete'])
    write(folder/'stability.json',stability)
    write(done,dict(complete=True,participant_id=pid,fold=fold,seed=seed,target=target,
        checkpoint_sha256=result['checkpoint_sha256'],test_only=True,chosen_task=chosen_task,chosen_task_name=schema['tasks'][chosen_task],
        original_probability=float(expit(base)),label=int(y[index]),case_category_seed42=case['category'],
        ig_validations=validations,ig_all_numerical_checks_passed=all(v['complete'] for v in validations),
        failed_ig_not_interpreted=not all(v['complete'] for v in validations),source_map_sha256=sha(folder/'source_mapping.csv'),
        tested_segments=len(selected),total_chosen_task_segments=len(eligible)))
    print('XAI CASE',pid,seed,'IG pass',all(v['complete'] for v in validations),flush=True)


def main():
    seed_all(42);cfg=json.loads((ROOT/'configs/raw_temporal_v2/phase4.json').read_text())
    validation=json.loads((OUT/'phase4_complementarity/model_validation.json').read_text());assert validation['complete']
    for path,digest in validation['verified_artifacts'].items():assert sha(ROOT/path)==digest
    signature_value={p:sha(ROOT/p) for p in SOURCES};signature_value['config']=sha(ROOT/'configs/raw_temporal_v2/phase4.json')
    signature_value['models']=sha(OUT/'phase4_complementarity/model_validation.json')
    token=hashlib.sha256(json.dumps(signature_value,sort_keys=True).encode()).hexdigest()[:16]
    run=OUT/'phase4_complementarity/xai/runs'/('run_'+token);run.mkdir(parents=True,exist_ok=True)
    write(run/'signature.json',signature_value)
    frame=pd.read_csv(OUT/'phase4_complementarity/nested_oof.csv',float_precision='round_trip')
    tasks=task_jobs(run,cfg,frame);cases,target=choose_cases(frame,cfg);cases.to_csv(run/'selected_cases.csv',index=False)
    for case in cases.to_dict('records'):
        for seed in cfg['attribution']['seeds']:detail_case(run,cfg,case,target,seed,frame,tasks)
    write(OUT/'phase4_complementarity/xai/active_run.json',dict(run=run.relative_to(ROOT).as_posix(),signature=token,
        tasks_complete=True,detailed_cases_complete=True,target=target,case_count=len(cases),validation_pending=True))
    print('HELD-OUT XAI COMPUTATION FINISHED; VALIDATION/PLOTS PENDING',flush=True)


if __name__=='__main__':
    with gpu_worker():main()
