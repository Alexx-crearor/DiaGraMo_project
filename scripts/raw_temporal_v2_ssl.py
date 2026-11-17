"""Train-participant-only masked motor pretraining, with recoverable checkpoints."""
import time
from raw_temporal_v2_common import *
from diagramo.temporal.dataset import TrainNormalizer,load_person
from diagramo.temporal_v2.multiscale import MultiScaleLocal,OriginalRetainedLocal
from diagramo.temporal_v2.masked_motor import MaskedMotorModel,corrupt,window_loss,TARGET_CHANNELS

SOURCES=['scripts/raw_temporal_v2_ssl.py','src/diagramo/temporal_v2/masked_motor.py',
    'src/diagramo/temporal_v2/multiscale.py','src/diagramo/temporal_v2/models.py']


def person(pid,normalizer):return normalizer.transform(load_person(str(CACHE),str(pid)),DEVICE,'cnn')


def eligible(item):return torch.nonzero(((item['x'][:,7,:]>.5)&item['sample_mask']).any(1)).flatten()


def interpolation(x,length,hidden,constant):
    b,c,l=x.shape;first=hidden.to(torch.int64).argmax(1);last=l-1-hidden.flip(1).to(torch.int64).argmax(1)
    left=(first-1).clamp_min(0);right=torch.minimum(last+1,length-1);at=torch.arange(b,device=x.device)
    targets=x[:,TARGET_CHANNELS,:];a=targets[at,:,left];z=targets[at,:,right]
    fraction=((torch.arange(l,device=x.device)[None,:]-left[:,None])/(right-left).clamp_min(1)[:,None]).clamp(0,1)
    output=a[:,:,None]+(z-a)[:,:,None]*fraction[:,None,:]
    available=(first>0)&(last<length-1)&hidden.any(1)
    allowed=available[:,None].expand(-1,4).clone()
    pen=(x[at,7,left]>.5)&(x[at,7,right]>.5);allowed&=pen[:,None]
    allowed[:,3]&=(x[at,13,left]>.5)&(x[at,13,right]>.5)
    return torch.where(allowed[:,:,None],output,constant[None,:,None])


def evaluate(model,normalizer,ids,indices,constant,seed):
    model.eval();rows=[]
    with torch.no_grad():
        for index in indices:
            item=person(ids[index],normalizer);selected=eligible(item);totals=np.zeros(3);count=0
            generator=torch.Generator(device=DEVICE).manual_seed(seed+int(index)*1009)
            for position in range(0,len(selected),128):
                at=selected[position:position+128];x=item['x'][at];length=item['length'][at]
                corrupted,mask,hidden=corrupt(x,length,generator)
                prediction=model(corrupted,length);target=x[:,TARGET_CHANNELS,:]
                baselines=[prediction,constant[None,:,None].expand_as(prediction),interpolation(x,length,hidden,constant)]
                for j,out in enumerate(baselines):
                    loss,defined=window_loss(out,target,mask);totals[j]+=float(loss[defined].sum())
                count+=int(defined.sum())
            assert count>0
            rows.append(dict(participant_id=ids[index],windows=len(item['length']),eligible_windows=len(selected),
                masked_windows=count,model_mse=float(totals[0]/count),train_constant_mse=float(totals[1]/count),
                visible_endpoint_interpolation_mse=float(totals[2]/count)))
    return rows


def pretrain(experiment,config_path,settings,train,validation=None,fixed_epochs=None,seed=42,extra_sources=(),phase='3'):
    seed_all(seed);ids,labels,schema,folds,splits=cohort();train=np.asarray(train)
    validation=np.asarray(validation) if validation is not None else np.asarray([],dtype=int)
    assert len(train) and not set(train)&set(validation)
    assert (fixed_epochs is None and len(validation)>0) or (fixed_epochs is not None and len(validation)==0)
    assert str(phase) in ('3','4')
    sig=signature(config_path,SOURCES+list(extra_sources));dest=OUT/('phase3_ssl' if str(phase)=='3' else 'phase4_complementarity')/experiment;dest.mkdir(parents=True,exist_ok=True)
    result_path=dest/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['signature']==sig['signature'] and set(result['fit_ids'])==set(ids[train])
        assert set(result['validation_ids'])==set(ids[validation])
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256'];return result
    begin(experiment,str(phase),'masked_motor_local',sig,hypothesis='H3 self-supervised local motor representation',
        input_representation='Original21 channels; contiguous span masking and dependent-channel halo',
        ablation='Masked source-coordinate/pressure/speed; no clinical label objective',
        split=dict(fit=ids[train].tolist(),validation=ids[validation].tolist()),seed=seed)
    write(dest/'signature.json',sig)
    with np.load(CACHE/'statistics.npz') as data:statistics=data['statistics']
    normalizer=TrainNormalizer().fit(statistics,train,ids[train]);means=[];coverage=[]
    for index in train:
        item=person(ids[index],normalizer);selected=eligible(item)
        valid=item['sample_mask']&(item['x'][:,7,:]>.5);mask=valid[:,None,:].expand(-1,4,-1).clone();mask[:,3,:]&=item['x'][:,13,:]>.5
        target=item['x'][:,TARGET_CHANNELS,:];means.append(((target*mask).sum((0,2))/mask.sum((0,2)).clamp_min(1)).cpu().numpy())
        coverage.append(dict(participant_id=ids[index],all_windows=len(item['length']),eligible_windows=len(selected),
            windows_without_pen_down_targets=len(item['length'])-len(selected)))
    constant=torch.tensor(np.mean(means,0),device=DEVICE,dtype=torch.float32)
    local=MultiScaleLocal() if settings.get('local_encoder','multiscale')=='multiscale' else OriginalRetainedLocal()
    model=MaskedMotorModel(local,4 if settings.get('local_encoder','multiscale')=='multiscale' else 8).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    checkpoint=LOCAL/f'phase{phase}'/(experiment+'.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True);resume=checkpoint.with_suffix('.resume.pt')
    rng=np.random.default_rng(seed);history=[];best=None;best_loss=1e99;best_epoch=0;stale=0;start_epoch=1;elapsed=0.
    if resume.exists():
        saved=torch.load(resume,map_location=DEVICE,weights_only=False);assert saved['signature']==sig['signature']
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer']);history=saved['history']
        best=saved['best'];best_loss=saved['best_loss'];best_epoch=saved['best_epoch'];stale=saved['stale']
        start_epoch=saved['epoch']+1;elapsed=saved['elapsed'];rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if DEVICE=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    maximum=int(fixed_epochs) if fixed_epochs is not None else settings['epochs'];epoch=start_epoch-1
    already_stopped=bool(len(validation) and epoch>=settings['minimum_epochs'] and stale>=settings['patience'])
    if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
    sync();began=time.perf_counter()
    for epoch in ([] if already_stopped else range(start_epoch,maximum+1)):
        model.train();order=rng.permutation(train);losses=[];gradient=[]
        for position in range(0,len(order),8):
            group=order[position:position+8];optimizer.zero_grad(set_to_none=True)
            for index in group:
                item=person(ids[index],normalizer);selected=eligible(item)
                permutation=torch.as_tensor(rng.permutation(len(selected)),device=DEVICE);selected=selected[permutation]
                generator=torch.Generator(device=DEVICE).manual_seed(seed+epoch*100000+int(index)*1009)
                assert len(selected)>0
                for micro in range(0,len(selected),settings['batch_windows']):
                    at=selected[micro:micro+settings['batch_windows']];x=item['x'][at];length=item['length'][at]
                    corrupted,mask,hidden=corrupt(x,length,generator,settings['masked_span'],settings['halo'])
                    prediction=model(corrupted,length);loss,defined=window_loss(prediction,x[:,TARGET_CHANNELS,:],mask)
                    assert defined.all() and torch.isfinite(loss).all()
                    (loss.sum()/len(selected)/len(group)).backward();losses.extend(loss.detach().cpu().numpy().tolist())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);assert torch.isfinite(norm)
            gradient.append(float(norm));optimizer.step()
        row=dict(epoch=epoch,online_train_masked_mse=float(np.mean(losses)),gradient_norm=float(np.mean(gradient)))
        if len(validation):
            scored=evaluate(model,normalizer,ids,validation,constant,seed+700000)
            objective=float(np.mean([r['model_mse'] for r in scored]));row.update(validation_masked_mse=objective,
                validation_train_constant_mse=float(np.mean([r['train_constant_mse'] for r in scored])),
                validation_interpolation_mse=float(np.mean([r['visible_endpoint_interpolation_mse'] for r in scored])))
            if objective<best_loss-1e-5:
                best_loss=objective;best_epoch=epoch;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            else:stale+=1
        else:best_epoch=epoch
        history.append(row);sync();duration=elapsed+time.perf_counter()-began
        torch.save(dict(signature=sig['signature'],state=model.state_dict(),optimizer=optimizer.state_dict(),
            history=history,epoch=epoch,best=best,best_loss=best_loss,best_epoch=best_epoch,stale=stale,elapsed=duration,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if DEVICE=='cuda' else None),resume)
        pd.DataFrame(history).to_csv(dest/'learning_curve.csv',index=False);resource_check()
        print('SSL',experiment,epoch,row,flush=True)
        if len(validation) and epoch>=settings['minimum_epochs'] and stale>=settings['patience']:break
    if len(validation):model.load_state_dict(best)
    validation_rows=evaluate(model,normalizer,ids,validation,constant,seed+700000) if len(validation) else []
    if validation_rows:pd.DataFrame(validation_rows).to_csv(dest/'validation_reconstruction.csv',index=False)
    pd.DataFrame(coverage).to_csv(dest/'train_window_coverage.csv',index=False)
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},
        local_state={k:v.detach().cpu() for k,v in model.local.state_dict().items()},normalizer=normalizer,
        fit_ids=ids[train].tolist(),validation_ids=ids[validation].tolist(),constant=constant.cpu(),
        local_encoder=settings.get('local_encoder','multiscale'),signature=sig['signature'],seed=seed,
        best_epoch=best_epoch),checkpoint)
    sync();result=dict(complete=True,experiment_id=experiment,signature=sig['signature'],seed=seed,
        fit_ids=ids[train].tolist(),validation_ids=ids[validation].tolist(),best_epoch=best_epoch,epochs_run=epoch,
        training_seconds=elapsed+time.perf_counter()-began,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2 if DEVICE=='cuda' else 0.,
        checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint),
        no_clinical_label_objective=True,all_eligible_train_windows_each_epoch=True,
        target_units='Train-normalized signed-log source-relative coordinates, pressure and speed; no calibrated biomechanical units',
        coverage_exclusions='Pen-up-only windows have no pen-down reconstruction target, counted explicitly; classification retains all windows',
        validation_reconstruction={key:float(np.mean([r[key] for r in validation_rows])) for key in ('model_mse','train_constant_mse','visible_endpoint_interpolation_mse')} if validation_rows else {})
    write(result_path,result)
    ledger(experiment,training_status='completed',completion_status='completed',best_checkpoint=result['checkpoint'],
        validation_metrics=result['validation_reconstruction'],training_cost=result['training_seconds'],
        gpu_peak_memory=result['peak_gpu_mib'],parameter_count=sum(p.numel() for p in model.parameters()),
        scientific_interpretation='Masked reconstruction does not establish clinical feature quality; downstream controls required')
    return result
