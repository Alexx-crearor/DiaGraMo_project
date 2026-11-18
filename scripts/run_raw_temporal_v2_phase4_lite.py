"""Fixed-recipe six-fold fits with optimizer-step checkpoints and session caps."""
import argparse
import gc
from raw_temporal_v2_lite_common import *


def gradient_update(model,normalizer,ids,y,group,cfg):
    losses=[]
    for start in range(0,len(group),cfg['training']['micro_batch']):
        indices=group[start:start+cfg['training']['micro_batch']]
        batch=collate_people([tensor(ids[i],normalizer) for i in indices])
        logits=model(batch)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(y[indices],device=DEVICE,dtype=torch.float32))
        assert torch.isfinite(loss)
        (loss*len(indices)/len(group)).backward();losses.extend([float(loss.detach())]*len(indices))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip'])
    assert torch.isfinite(norm)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return losses,float(norm)


def benchmark(cfg,budget):
    path=LITE/'runtime_forecast.json';sig=model_signature()
    if path.exists():
        data=json.loads(path.read_text());assert data['signature']==sig['signature'];return data
    ids,y,schema,fit,val,test=development();normalizer=normalizer_for(fit)
    ranked=sorted(fit,key=lambda i:len(load_person(str(CACHE),str(ids[i]))['length']))
    chosen=np.asarray([ranked[i] for i in np.rint(np.linspace(0,len(ranked)-1,8)).astype(int)])
    selected_windows=sum(len(load_person(str(CACHE),str(ids[i]))['length']) for i in chosen)
    train_windows=sum(len(load_person(str(CACHE),str(ids[i]))['length']) for i in fit)
    rows={}
    for name,candidate in cfg['candidates'].items():
        assert budget.permits(180,reserve=120),'Not enough budget for benchmark'
        seed_all(cfg['seed']);model=build_model(candidate['architecture']).to(DEVICE);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
        torch.cuda.reset_peak_memory_stats();sync();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        losses,norm=gradient_update(model,normalizer,ids,y,chosen,cfg);optimizer.step();sync();seconds=time.perf_counter()-started
        sync();started=time.perf_counter();predict(model,normalizer,ids[chosen]);sync();prediction_seconds=time.perf_counter()-started
        # 215/143 scales the representative development pool to outer Train.
        epoch=seconds*train_windows/selected_windows*215/len(fit)
        forecast=(epoch*candidate['fixed_epochs']+prediction_seconds*43/len(chosen))*cfg['compute']['forecast_safety_factor']
        rows[name]=dict(pilot_participants=len(chosen),pilot_ids=ids[chosen].tolist(),normalizer_fit_ids=ids[fit].tolist(),
            pilot_windows=selected_windows,training_seconds=seconds,prediction_seconds=prediction_seconds,
            estimated_outer_train_epoch_seconds=epoch,estimated_fold_seconds_with_safety=forecast,
            estimated_six_folds_seconds_with_safety=6*forecast,gradient_norm=norm,mean_online_bce=float(np.mean(losses)),
            peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            historical_development_seconds_per_epoch=candidate['development_seconds_per_epoch'],
            method='One measured eight-person accumulated update across full recordings, window scaling to outer Train215,1.5 safety factor; no outer Test features/labels loaded')
        print('LITE BENCHMARK',name,rows[name],flush=True)
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    data=dict(signature=sig['signature'],candidate_forecasts=rows,full_suite_seconds=sum(r['estimated_six_folds_seconds_with_safety'] for r in rows.values()),
        outer_scores_not_used_for_admission=True,initial_seed=cfg['seed'])
    write(path,data);return data


def fit_fold(name,fold,cfg,budget,forecast):
    ids,y,schema,assignments,splits=cohort();train,test=splits[fold];candidate=cfg['candidates'][name];seed_all(cfg['seed'])
    dest=LITE/'fits'/f'{name}_fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    checkpoint=LOCAL/'phase4_lite'/f'{name}_fold{fold}.pt';resume=checkpoint.with_suffix('.resume.pt')
    sig=model_signature();result_path=dest/'result.json'
    if result_path.exists():
        r=json.loads(result_path.read_text());assert r['signature']==sig['signature'] and sha(ROOT/r['checkpoint'])==r['checkpoint_sha256'];return True
    model=build_model(candidate['architecture']).to(DEVICE);normalizer=normalizer_for(train)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    generator=np.random.default_rng(cfg['seed']);epoch=0;cursor=0;order=None;elapsed=0.;history=[];losses=[];peak=0.;reserved=0.;step_count=0;norm=0.
    if resume.exists():
        state=torch.load(resume,map_location=DEVICE,weights_only=False)
        assert state['signature']==sig['signature'] and state['fit_ids']==ids[train].tolist() and state['test_ids']==ids[test].tolist()
        assert state['fixed_epochs']==candidate['fixed_epochs']
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'],generator)
        epoch=state['epoch'];cursor=state['cursor'];order=state['order'];elapsed=state['training_seconds']
        history=state['history'];losses=state['epoch_losses'];peak=state['peak_gpu_allocated_mib'];reserved=state['peak_gpu_reserved_mib'];step_count=state['optimizer_steps']
        norm=state['last_gradient_norm']
    torch.cuda.reset_peak_memory_stats();sync();segment_started=time.perf_counter();maximum_step_seconds=0.
    def persist(status):
        nonlocal peak,reserved
        sync();peak=max(peak,torch.cuda.max_memory_allocated()/1024**2);reserved=max(reserved,torch.cuda.max_memory_reserved()/1024**2)
        duration=elapsed+time.perf_counter()-segment_started
        state=dict(signature=sig['signature'],model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(generator),
            epoch=epoch,cursor=cursor,order=order,training_seconds=duration,history=history,epoch_losses=losses,
            fit_ids=ids[train].tolist(),test_ids=ids[test].tolist(),fixed_epochs=candidate['fixed_epochs'],normalizer=normalizer,
            peak_gpu_allocated_mib=peak,peak_gpu_reserved_mib=reserved,optimizer_steps=step_count,last_gradient_norm=norm)
        atomic_checkpoint(state,resume)
        pd.DataFrame(history).to_csv(dest/'learning_curve.csv',index=False)
        write(dest/'status.json',dict(status=status,epochs_completed=epoch,next_group_cursor=cursor,
            fixed_epochs=candidate['fixed_epochs'],training_seconds=duration,optimizer_steps=step_count,
            resume_checkpoint=resume.relative_to(ROOT).as_posix(),resume_sha256=sha(resume),signature=sig['signature']))
        return duration
    write(dest/'admission.json',dict(session=budget.path.relative_to(ROOT).as_posix(),remaining_session_seconds=budget.remaining(),
        estimated_fold_seconds=forecast['estimated_fold_seconds_with_safety'],remaining_epoch_estimate=(candidate['fixed_epochs']-epoch)*forecast['estimated_outer_train_epoch_seconds'],
        may_pause_at_optimizer_boundary=True,signature=sig,fit_ids=ids[train].tolist(),test_ids=ids[test].tolist(),
        normalizer_fit_ids=normalizer.fit_ids,validation_ids=[],architecture=candidate['architecture'],fixed_epochs=candidate['fixed_epochs']))
    while epoch<candidate['fixed_epochs']:
        model.train()
        if order is None:order=generator.permutation(train);cursor=0;losses=[]
        while cursor<len(order):
            if not budget.permits(max(5,maximum_step_seconds*2),reserve=120):
                persist('paused_session_budget');print('LITE PAUSED',name,fold,'epoch',epoch,'cursor',cursor,flush=True);return False
            update_started=time.perf_counter();group=order[cursor:cursor+cfg['training']['effective_batch']]
            optimizer.zero_grad(set_to_none=True);values,norm=gradient_update(model,normalizer,ids,y,group,cfg)
            optimizer.step();sync();maximum_step_seconds=max(maximum_step_seconds,time.perf_counter()-update_started)
            losses.extend(values);cursor+=len(group);step_count+=1;persist('training')
        epoch+=1;history.append(dict(epoch=epoch,online_train_bce=float(np.mean(losses)),last_gradient_norm=norm))
        order=None;cursor=0;losses=[];persist('training')
        if epoch%5==0 or epoch==candidate['fixed_epochs']:print('LITE',name,'fold',fold,'epoch',epoch,'/',candidate['fixed_epochs'],flush=True)
    if not budget.permits(max(10,forecast['prediction_seconds']*43/8*2),reserve=90):
        persist('trained_evaluation_pending');return False
    duration=persist('trained_evaluation_pending');sync();evaluation_started=time.perf_counter()
    # Held-out features and labels are reached only after the complete fixed fit.
    logits,_=predict(model,normalizer,ids[test]);p=1/(1+np.exp(-logits))
    atomic_checkpoint(dict(state=model.state_dict(),architecture=candidate['architecture'],normalizer=normalizer,
        fit_ids=ids[train].tolist(),evaluation_ids=ids[test].tolist(),normalizer_fit_ids=normalizer.fit_ids,
        validation_ids=[],fixed_epochs=candidate['fixed_epochs'],signature=sig['signature'],seed=cfg['seed']),checkpoint)
    loaded=torch.load(checkpoint,map_location=DEVICE,weights_only=False)
    replay=build_model(loaded['architecture']).to(DEVICE);replay.load_state_dict(loaded['state'])
    replay_logits,_=predict(replay,loaded['normalizer'],ids[test]);error=float(np.max(np.abs(replay_logits-logits)))
    assert error<1e-7 and set(normalizer.fit_ids)==set(ids[train]) and not set(ids[test])&set(normalizer.fit_ids)
    digest=sha(checkpoint)
    frame=pd.DataFrame(dict(participant_id=ids[test],label=y[test],fold=fold,model=name,seed=cfg['seed'],
        logit=logits,probability=p,prediction=(p>.5).astype(int),checkpoint_sha256=digest))
    frame.to_csv(dest/'oof_predictions.csv',index=False)
    sync();evaluation_seconds=time.perf_counter()-evaluation_started
    result=dict(complete=True,signature=sig['signature'],model=name,fold=fold,seed=cfg['seed'],architecture=candidate['architecture'],
        fixed_epochs=candidate['fixed_epochs'],fit_ids=ids[train].tolist(),evaluation_ids=ids[test].tolist(),normalizer_fit_ids=normalizer.fit_ids,
        validation_ids=[],training_seconds=duration,evaluation_and_replay_seconds=evaluation_seconds,
        peak_gpu_allocated_mib=max(peak,torch.cuda.max_memory_allocated()/1024**2),peak_gpu_reserved_mib=max(reserved,torch.cuda.max_memory_reserved()/1024**2),
        checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=digest,checkpoint_replay_max_logit_error=error,
        parameters=sum(v.numel() for v in model.parameters()),metrics=metrics(y[test],p),optimizer_steps=step_count,
        no_outer_test_model_selection=True,no_early_stopping=True,inference_scope=cfg['inference_scope'])
    write(result_path,result);write(dest/'status.json',dict(status='complete',epochs_completed=epoch,checkpoint_sha256=digest))
    print('LITE FOLD COMPLETE',name,fold,result['metrics'],flush=True)
    del model,replay,optimizer;gc.collect();torch.cuda.empty_cache();return True


def main(minutes=110):
    cfg=json.loads(CONFIG.read_text());assert DEVICE=='cuda'
    assert sha(ROOT/'results/cv/fold_assignments.csv')==cfg['existing_fold_assignments_sha256']
    for path,digest in cfg['protocol_dependencies'].items():assert sha(ROOT/path)==digest,path
    budget=SessionBudget(minutes);forecast=benchmark(cfg,budget)
    reason='all_planned_folds_fitted'
    try:
        for name in cfg['execution_order']:
            for fold in range(6):
                if (LITE/'fits'/f'{name}_fold{fold}/result.json').exists():continue
                row=forecast['candidate_forecasts'][name]
                if not budget.permits(max(10,row['estimated_outer_train_epoch_seconds']),reserve=150):
                    reason='session_budget';break
                if not fit_fold(name,fold,cfg,budget,row):reason='session_budget';break
            if reason=='session_budget':break
        budget.finish('fits_finished' if reason!='session_budget' else 'paused_session_budget',reason=reason)
    except Exception as error:
        budget.finish('failed',reason=repr(error));raise
    # Reports read saved folds only, including after a clean budget pause.
    from report_raw_temporal_v2_lite import main as report
    report_started=time.perf_counter();report()
    budget.finish('fits_finished' if reason!='session_budget' else 'paused_session_budget',reason=reason,
                  reporting_seconds=time.perf_counter()-report_started,includes_reporting=True)
    print('LITE SESSION FINISHED',reason,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=float,default=110);args=parser.parse_args()
    with gpu_worker(),awake_during_session():main(args.minutes)
