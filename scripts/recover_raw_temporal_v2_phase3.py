"""Real CUDA smoke and one small SSL epoch; never launches full Phase3."""
from raw_temporal_v2_lite_common import *
from diagramo.temporal_v2.masked_motor import MaskedMotorModel,corrupt,window_loss,TARGET_CHANNELS
from diagramo.temporal_v2.multiscale import OriginalRetainedLocal,MultiScaleLocal


def coverage(ids,indices):
    rows=[]
    for index in indices:
        item=load_person(str(CACHE),str(ids[index]));valid=np.arange(128)[None,:]<item['length'][:,None]
        pen=valid&(item['x'][:,:,7]>.5)
        rows.append(dict(participant_id=ids[index],index=int(index),windows=len(item['length']),
                         eligible_windows=int(pen.any(1).sum()),samples=int(item['length'].sum())))
    return rows


def main(minutes=15):
    assert DEVICE=='cuda','Real CUDA verification required'
    assert 0<minutes<=15
    cfg=json.loads(CONFIG.read_text());budget=SessionBudget(minutes,RECOVERY);seed_all(42)
    started=time.perf_counter()
    fix=ROOT/'src/diagramo/temporal_v2/masked_motor.py'
    committed=subprocess.check_output(['git','show','05b3e28:src/diagramo/temporal_v2/masked_motor.py'],cwd=ROOT)
    import hashlib
    assert hashlib.sha256(committed).hexdigest()==sha(fix),'Verify exact committed fix'
    ids,y,schema,fit,val,test=development();norm=normalizer_for(fit)
    train_rows=coverage(ids,fit);val_rows=coverage(ids,val)
    ordered=sorted(train_rows,key=lambda row:row['eligible_windows'])
    chosen=[ordered[i]['index'] for i in np.rint(np.linspace(0,len(ordered)-1,8)).astype(int)]
    assert len(set(chosen))==8 and set(chosen)<=set(fit) and not set(chosen)&set(test)
    torch.cuda.reset_peak_memory_stats();smoke=[]
    first=norm.transform(load_person(str(CACHE),str(ids[chosen[-1]])),DEVICE,'cnn')
    selected=torch.nonzero(((first['x'][:,7,:]>.5)&first['sample_mask']).any(1)).flatten()[:64]
    x=first['x'][selected];length=first['length'][selected]
    corrupted,mask,_=corrupt(x,length,torch.Generator(device=DEVICE).manual_seed(42))
    for name,local,stride in [('original',OriginalRetainedLocal(),8),('multiscale',MultiScaleLocal(),4)]:
        model=MaskedMotorModel(local,stride).to(DEVICE);gradients=[];outputs=[]
        for repeat in range(2):
            model.zero_grad(set_to_none=True);prediction=model(corrupted,length)
            loss,defined=window_loss(prediction,x[:,TARGET_CHANNELS,:],mask)
            loss[defined].mean().backward()
            gradient=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            assert torch.isfinite(prediction).all() and torch.isfinite(gradient).all() and gradient.abs().sum()>0
            gradients.append(gradient.detach().clone());outputs.append(prediction.detach().clone())
        assert torch.equal(gradients[0],gradients[1]) and torch.equal(outputs[0],outputs[1])
        smoke.append(dict(encoder=name,windows=len(selected),finite_gradients=True,
            repeated_gradients_bitwise_equal=True,repeated_forward_bitwise_equal=True,loss=float(loss[defined].mean().detach()),
            gradient_l2=float(gradients[-1].norm())))
        del model,gradients,outputs
    del first,x,corrupted,mask
    sync();smoke_seconds=time.perf_counter()-started
    write(RECOVERY/'cuda_smoke.json',dict(complete=True,device=torch.cuda.get_device_name(),
        strict_deterministic=torch.are_deterministic_algorithms_enabled(),committed_fix_sha256=sha(fix),
        normalizer_fit_ids=ids[fit].tolist(),used_ids=[ids[chosen[-1]]],excluded_outer_test_ids=ids[test].tolist(),
        records=smoke,setup_and_smoke_seconds=smoke_seconds))
    print('REAL CUDA SMOKE PASSED',smoke,flush=True)
    settings=json.loads((ROOT/'configs/raw_temporal_v2/phase3.json').read_text())['pretraining']
    seed_all(42);model=MaskedMotorModel(OriginalRetainedLocal(),8).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    rng=np.random.default_rng(42);order=rng.permutation(chosen);losses=[];count=0;rows=[]
    torch.cuda.reset_peak_memory_stats();sync();epoch_started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    try:
        for index in order:
            if not budget.permits(5,reserve=30):raise TimeoutError('Small SSL epoch exceeded pilot budget')
            person_started=time.perf_counter();item=norm.transform(load_person(str(CACHE),str(ids[index])),DEVICE,'cnn')
            selected=torch.nonzero(((item['x'][:,7,:]>.5)&item['sample_mask']).any(1)).flatten()
            selected=selected[torch.as_tensor(rng.permutation(len(selected)),device=DEVICE)]
            generator=torch.Generator(device=DEVICE).manual_seed(42+100000+int(index)*1009)
            for position in range(0,len(selected),settings['batch_windows']):
                if not budget.permits(2,reserve=30):raise TimeoutError('Small SSL epoch exceeded pilot budget')
                at=selected[position:position+settings['batch_windows']];x=item['x'][at];length=item['length'][at]
                corrupted,mask,_=corrupt(x,length,generator,settings['masked_span'],settings['halo'])
                prediction=model(corrupted,length);loss,defined=window_loss(prediction,x[:,TARGET_CHANNELS,:],mask)
                assert defined.all() and torch.isfinite(loss).all()
                (loss.sum()/len(selected)/len(order)).backward();losses.extend(loss.detach().cpu().tolist());count+=len(at)
            sync();rows.append(dict(participant_id=ids[index],eligible_windows=len(selected),seconds=time.perf_counter()-person_started))
        gradients=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]);assert torch.isfinite(gradients).all()
        gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']));optimizer.step();sync()
    except TimeoutError as error:
        budget.finish('pilot_incomplete',reason=str(error));write(RECOVERY/'status.json',dict(phase3_complete=False,pilot_complete=False,reason=str(error)));return
    seconds=time.perf_counter()-epoch_started
    # Time inference over the same full recordings as a separate estimate of
    # reconstruction validation cost; no labels are read by the objective.
    sync();evaluation_started=time.perf_counter();model.eval()
    with torch.no_grad():
        for index in order:
            item=norm.transform(load_person(str(CACHE),str(ids[index])),DEVICE,'cnn')
            selected=torch.nonzero(((item['x'][:,7,:]>.5)&item['sample_mask']).any(1)).flatten()
            for position in range(0,len(selected),128):
                at=selected[position:position+128];model(item['x'][at],item['length'][at])
    sync();evaluation_seconds=time.perf_counter()-evaluation_started
    ratio=sum(row['eligible_windows'] for row in train_rows)/count
    val_ratio=sum(row['eligible_windows'] for row in val_rows)/count
    full_epoch_train=seconds*ratio;full_epoch_val=evaluation_seconds*val_ratio
    # Baseline/normalization overhead and ragged-size uncertainty are explicit.
    pretrain_estimate=2*settings['epochs']*(full_epoch_train+full_epoch_val)*1.5
    p1=json.loads((OUT/'phase1_bottleneck/p1_hierarchical_task_to_person_duration/result.json').read_text())
    neural_estimate=12*p1['training_seconds']*1.5
    full_estimate=pretrain_estimate+neural_estimate
    result=dict(pilot_complete=True,phase3_complete=False,fit_ids=ids[order].tolist(),normalizer_fit_ids=ids[fit].tolist(),
        eligible_windows=count,epochs_completed=1,optimizer_updates=1,training_seconds=seconds,
        forward_only_seconds=evaluation_seconds,windows_per_second=count/seconds,mean_masked_mse=float(np.mean(losses)),
        gradient_norm=gradient_norm,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        setup_and_smoke_seconds=smoke_seconds,full_train_windows=sum(r['eligible_windows'] for r in train_rows),
        full_validation_windows=sum(r['eligible_windows'] for r in val_rows),estimated_full_train_epoch_seconds=full_epoch_train,
        estimated_validation_forward_seconds=full_epoch_val,estimated_two_seed_pretraining_upper_seconds=pretrain_estimate,
        estimated_full_phase3_seconds=full_estimate,estimate_method='Window throughput scaling, 1.5 safety factor; downstream12 fits proxied by measured duration model cost.38 CPU heads/extraction not included, so full-suite estimate may be low.',
        configured_full_phase3_budget_seconds=cfg['compute']['full_phase3_budget_seconds'],
        full_suite_estimate_within_budget=full_estimate<=cfg['compute']['full_phase3_budget_seconds'],
        full_ssl_launched=False,reason='Recovery pilot only; prioritize fixed-recipe held-out validation within session budget',
        small_epoch_scope='Eight development-Train participants, all eligible windows, one participant-balanced optimizer update; no claim of downstream learning')
    checkpoint=LOCAL/'phase3_recovery/pilot_epoch1.pt'
    atomic_checkpoint(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),normalizer=norm,
        fit_ids=ids[order].tolist(),rng=rng_state(rng),result=result),checkpoint)
    result.update(checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint))
    pd.DataFrame(rows).to_csv(RECOVERY/'pilot_participant_runtime.csv',index=False)
    pd.DataFrame(train_rows+val_rows).to_csv(RECOVERY/'development_window_inventory.csv',index=False)
    write(RECOVERY/'pilot_result.json',result)
    write(RECOVERY/'status.json',dict(recovery_checks_complete=True,phase3_complete=False,status='pilot_only_full_ssl_deferred'))
    budget.finish('pilot_complete_full_ssl_deferred',pilot_training_seconds=seconds)
    print('SSL PILOT',result,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=float,default=15);args=parser.parse_args()
    with gpu_worker():main(args.minutes)
