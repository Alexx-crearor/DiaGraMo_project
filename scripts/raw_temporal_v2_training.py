"""Reusable supervised fitter with explicit validation versus final-Test roles.

Final fits use a fixed epoch budget chosen elsewhere on Train; evaluation
participants are not loaded until fitting has ended. Callers own the GPU lock.
"""
import copy
import time
from functools import lru_cache
from raw_temporal_v2_common import *
from diagramo.temporal.dataset import TrainNormalizer,load_person,collate_people
from diagramo.temporal_v2.multiscale import MultiScaleModel
from diagramo.temporal_v2.models import RecoveryModel

SOURCES=['scripts/raw_temporal_v2_training.py','src/diagramo/temporal_v2/multiscale.py',
         'src/diagramo/temporal_v2/models.py','scripts/prepare_raw_temporal_v2_relations.py',
         'scripts/prepare_raw_temporal_v2_targets.py',
         'results/raw_temporal_v2/phase1_bottleneck/physical_targets_manifest.json',
         'results/raw_temporal_v2/phase2_temporal/relationship_manifest.json']


@lru_cache(maxsize=1)
def metadata_hashes():
    physical=json.loads((OUT/'phase1_bottleneck/physical_targets_manifest.json').read_text())
    relation=json.loads((OUT/'phase2_temporal/relationship_manifest.json').read_text())
    assert physical['complete'] and relation['complete']
    return ({r['participant_id']:r['sha256'] for r in physical['records']},
            {r['participant_id']:r['sha256'] for r in relation['records']})


@lru_cache(maxsize=258)
def metadata(pid):
    physical_hashes,relation_hashes=metadata_hashes()
    path=LOCAL/'physical_targets'/f'{pid}.npz';relationship=LOCAL/'relationships'/f'{pid}.npy'
    assert sha(path)==physical_hashes[pid] and sha(relationship)==relation_hashes[pid]
    with np.load(path) as data:result={key:data[key].copy() for key in ('token_meta','patch_duration','segment_duration','task_duration')}
    result['segment_relation']=np.load(relationship);return result


def build_model(architecture):
    config=dict(architecture);kind=config.pop('model_kind','multiscale')
    assert kind in ('multiscale','recovery')
    return RecoveryModel(**config) if kind=='recovery' else MultiScaleModel(**config)


def tensor(pid,normalizer):
    item=normalizer.transform(load_person(str(CACHE),str(pid)),DEVICE,'cnn')
    item.update({key:torch.as_tensor(value,device=DEVICE) for key,value in metadata(str(pid)).items()})
    return item


def predict(model,normalizer,ids,stress=None,occlusion=None):
    model.eval();outputs=[];representations=[]
    with torch.no_grad():
        for position in range(0,len(ids),2):
            batch=collate_people([tensor(pid,normalizer) for pid in ids[position:position+2]])
            kwargs={}
            if occlusion is not None:kwargs=occlusion(batch)
            if stress is not None:kwargs['stress']=stress
            logits,details=model(batch,return_details=True,**kwargs)
            outputs.extend(logits.cpu().numpy());representations.extend(details['person'].cpu().numpy())
    return np.asarray(outputs,dtype='float64'),np.asarray(representations)


def fit(experiment,phase,config_path,settings,architecture,train,validation=None,
        evaluation=None,fixed_epochs=None,seed=42,extra_sources=(),pretrained_local=None,
        normalization_train=None):
    seed_all(seed);ids,y,schema,folds,splits=cohort();train=np.asarray(train)
    validation=np.asarray(validation) if validation is not None else np.asarray([],dtype=int)
    evaluation=np.asarray(evaluation) if evaluation is not None else validation
    normalization_train=train if normalization_train is None else np.asarray(normalization_train)
    assert len(train) and len(evaluation) and not set(train)&set(evaluation)
    assert not set(train)&set(validation)
    assert set(train)<=set(normalization_train)
    assert not set(normalization_train)&(set(validation)|set(evaluation))
    assert (fixed_epochs is None and len(validation)>0) or (fixed_epochs is not None and len(validation)==0)
    sig=signature(config_path,SOURCES+list(extra_sources))
    dest=OUT/f'phase{phase}_temporal' if str(phase)=='2' else OUT/f'phase{phase}_ssl' if str(phase)=='3' else OUT/'phase4_complementarity'
    dest=dest/experiment;dest.mkdir(parents=True,exist_ok=True);target=dest/'result.json'
    if target.exists():
        result=json.loads(target.read_text());assert result['signature']==sig['signature'] and result['architecture']==architecture
        assert set(result['fit_ids'])==set(ids[train]) and set(result['evaluation_ids'])==set(ids[evaluation])
        assert set(result['normalizer_fit_ids'])==set(ids[normalization_train])
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256'];print('REUSE',experiment,flush=True);return result
    model=build_model(architecture).to(DEVICE)
    if pretrained_local is not None:
        state=torch.load(pretrained_local,map_location=DEVICE,weights_only=False)
        assert set(state['fit_ids'])<=set(ids[normalization_train]),'SSL participants extend beyond allowed Train domain'
        local_state=state['local_state']
        if architecture.get('model_kind')=='recovery':
            assert state['local_encoder']=='original'
            assert all(key.startswith('readout.backbone.') for key in local_state)
            model.local.backbone.load_state_dict({key.removeprefix('readout.backbone.'):value for key,value in local_state.items()})
        else:model.local.load_state_dict(local_state)
    with np.load(CACHE/'statistics.npz') as data:statistics=data['statistics']
    normalizer=TrainNormalizer().fit(statistics,normalization_train,ids[normalization_train],index_only=architecture.get('index_only',False))
    if pretrained_local is not None:
        np.testing.assert_allclose(normalizer.mean,state['normalizer'].mean,atol=1e-8,rtol=0)
        np.testing.assert_allclose(normalizer.scale,state['normalizer'].scale,atol=1e-8,rtol=0)
    begin(experiment,str(phase),architecture.get('design',architecture.get('family')),sig,hypothesis='Controlled raw temporal representation study',
        input_representation='All original chronological cached samples; multiple local tokens',
        ablation=architecture,split=dict(train=ids[train].tolist(),validation=ids[validation].tolist(),
        evaluation=ids[evaluation].tolist(),unlabeled_normalization_pool=ids[normalization_train].tolist(),fixed_final_epochs=fixed_epochs),seed=seed)
    write(dest/'signature.json',sig)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    checkpoint=LOCAL/f'phase{phase}'/(experiment+'.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True)
    resume=checkpoint.with_suffix('.resume.pt');history=[];best=None;best_loss=1e99;best_epoch=0;stale=0;elapsed=0.;start_epoch=1
    rng=np.random.default_rng(seed)
    if resume.exists():
        saved=torch.load(resume,map_location=DEVICE,weights_only=False);assert saved['signature']==sig['signature']
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];best_loss=saved['best_loss'];best_epoch=saved['best_epoch']
        stale=saved['stale'];elapsed=saved['elapsed'];start_epoch=saved['epoch']+1
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if DEVICE=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    maximum=int(fixed_epochs) if fixed_epochs is not None else settings['epochs']
    if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
    sync();start=time.perf_counter()
    epoch=start_epoch-1
    already_stopped=bool(len(validation) and epoch>=settings['minimum_epochs'] and stale>=settings['patience'])
    for epoch in ([] if already_stopped else range(start_epoch,maximum+1)):
        model.train();order=rng.permutation(train);losses=[];gradients=[];local_gradients=[]
        for position in range(0,len(order),RESOURCE['effective_batch_participants']):
            group=order[position:position+RESOURCE['effective_batch_participants']];optimizer.zero_grad(set_to_none=True)
            for micro in range(0,len(group),RESOURCE['micro_batch_participants']):
                at=group[micro:micro+RESOURCE['micro_batch_participants']]
                logits=model(collate_people([tensor(ids[i],normalizer) for i in at]))
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(y[at],device=DEVICE,dtype=torch.float32))
                assert torch.isfinite(loss);(loss*len(at)/len(group)).backward();losses.extend([float(loss.detach())]*len(at))
            local_gradients.append(float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.local.parameters() if p.grad is not None))))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);assert torch.isfinite(norm)
            gradients.append(float(norm));optimizer.step()
        row=dict(epoch=epoch,online_train_bce=float(np.mean(losses)),gradient_norm=float(np.mean(gradients)),local_gradient_norm=float(np.mean(local_gradients)))
        if len(validation):
            logits,_=predict(model,normalizer,ids[validation]);probability=1/(1+np.exp(-logits))
            loss=float(np.mean(np.logaddexp(0,logits)-y[validation]*logits));row.update(validation_bce=loss,**metrics(y[validation],probability))
            if loss<best_loss-1e-4:
                best_loss=loss;best_epoch=epoch;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            else:stale+=1
        else:best_epoch=epoch
        history.append(row)
        stopping=len(validation)>0 and epoch>=settings['minimum_epochs'] and stale>=settings['patience']
        if epoch%RESOURCE['checkpoint_every_epochs']==0 or stopping or epoch==maximum:
            sync();duration=elapsed+time.perf_counter()-start
            torch.save(dict(signature=sig['signature'],state=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                history=history,best=best,best_loss=best_loss,best_epoch=best_epoch,stale=stale,elapsed=duration,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if DEVICE=='cuda' else None),resume)
            pd.DataFrame(history).to_csv(dest/'learning_curve.csv',index=False);resource_check()
            print('FIT',experiment,'epoch',epoch,'trainBCE',round(row['online_train_bce'],4),'valBCE',row.get('validation_bce'),flush=True)
        if stopping:break
    if len(validation):model.load_state_dict(best)
    # Final evaluation is deliberately below the entire optimization loop.
    logits,z=predict(model,normalizer,ids[evaluation]);p=1/(1+np.exp(-logits))
    trainlogits,_=predict(model,normalizer,ids[train])
    pd.DataFrame(dict(participant_id=ids[evaluation],label=y[evaluation],logit=logits,probability=p)).to_csv(dest/'evaluation_predictions.csv',index=False)
    pd.DataFrame(dict(participant_id=ids[train],label=y[train],probability=1/(1+np.exp(-trainlogits)))).to_csv(dest/'training_predictions.csv',index=False)
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=architecture,
        normalizer=normalizer,fit_ids=ids[train].tolist(),validation_ids=ids[validation].tolist(),
        normalizer_fit_ids=ids[normalization_train].tolist(),
        evaluation_ids=ids[evaluation].tolist(),best_epoch=best_epoch,signature=sig['signature'],
        pretrained_sha256=sha(pretrained_local) if pretrained_local else None),checkpoint)
    sync();duration=elapsed+time.perf_counter()-start
    result=dict(complete=True,experiment_id=experiment,phase=str(phase),signature=sig['signature'],architecture=architecture,
        seed=seed,fit_ids=ids[train].tolist(),validation_ids=ids[validation].tolist(),evaluation_ids=ids[evaluation].tolist(),
        normalizer_fit_ids=ids[normalization_train].tolist(),
        metrics=metrics(y[evaluation],p),train_metrics=metrics(y[train],1/(1+np.exp(-trainlogits))),
        best_epoch=best_epoch,epochs_run=epoch,fixed_epochs=fixed_epochs,parameters=sum(p.numel() for p in model.parameters()),
        training_seconds=duration,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2 if DEVICE=='cuda' else 0.,
        embedding_variance=float(z.var(0).mean()),checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint),
        no_evaluation_labels_during_fixed_epoch_training=fixed_epochs is not None,
        pretrained_sha256=sha(pretrained_local) if pretrained_local else None)
    write(target,result)
    ledger(experiment,training_status='completed',completion_status='completed',best_checkpoint=result['checkpoint'],
        validation_metrics=result['metrics'] if len(validation) else {},held_out_metrics=result['metrics'] if not len(validation) else {},
        parameter_count=result['parameters'],gpu_peak_memory=result['peak_gpu_mib'],training_cost=duration,
        artifact_paths=dest.relative_to(ROOT).as_posix(),scientific_interpretation='Exploratory internal participant evaluation')
    print('FIT COMPLETE',experiment,result['metrics'],flush=True)
    return result
